"""Local demo UI: python -m promofinder serve  ->  http://127.0.0.1:8765

Flow: upload a scripts .zip -> extracted locally (only .txt, safe) -> parsed -> candidates -> shortlisted clips sent
to the LLM judge (progress streamed) -> ranked promos, posting plan, evaluation, learning loop.
Stdlib only, binds to localhost. Demo clicks go to runs/feedback.demo.jsonl; the demo never saves or promotes a model.
"""
import argparse
import json
import os
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import evaluate as E
from . import learn as L
from . import model as M
from . import pipeline as P
from .ingest import extract_scripts
from .judge import get_judge, load_env

HERE = os.path.dirname(__file__)
PLAIN = {  # feature -> how an editor would say it
    "hook_arousal": "a shocking opening", "hook_dialogue": "opening on a spoken line", "clean_entry": "a clean starting point",
    "cliff_proximity": "ending where the episode ends", "cliff_content": "ending on a threat or demand",
    "cliff_open_end": "ending on an unanswered line", "resolution": "endings that wrap things up",
    "conflict_density": "lots of conflict", "emotion_intensity": "high emotion", "speech_ratio": "lots of dialogue",
    "turn_rate": "fast back-and-forth", "standalone": "being easy to follow cold", "dur_fit": "the clip's length",
    "j_hook": "the AI judge's hook score", "j_cliff": "the AI judge's cliffhanger score",
    "j_standalone": "the AI judge's standalone score", "j_emotion": "the AI judge's emotion score",
    "hook_and_payoff": "having both a hook and a payoff"}


def fb_path(out):
    return os.path.join(out, "feedback.demo.jsonl")


def plain_shifts(old, new, n=4):
    """The biggest weight changes, as sentences."""
    shifts = sorted(((new.w[k] - old.w[k], k) for k in M.FEATURES if k in PLAIN), key=lambda t: -abs(t[0]))[:n]
    return [f"{'Cares more' if d > 0 else 'Cares less'} about {PLAIN[k]}" for d, k in shifts if abs(d) >= 0.05] or ["Hardly anything changed yet"]
UPLOADS = "runs/uploads"
MAX_UPLOAD = 100 * 1024 * 1024
# Free tier = 50 requests/day: 10 best clips + 30 reserved episode endings + 2 stitches (~42, fewer after overlap and
# cache) + pairwise on the top 4 (12). New calls are capped at the remaining quota anyway; cached clips are free.
LLM_BUDGET = {"top_n": 10, "reserve_endings": 30, "stitch_n": 2, "pairwise_top": int(os.environ.get("PROMO_PAIRWISE_TOP", 4))}
LOCK = threading.Lock()


class Job:
    def __init__(self):
        self.reset()

    def reset(self, **kw):
        self.state = {"status": "idle", "step": "", "log": [], "judged": 0, "to_judge": 0, "cached": 0,
                      "pairs": 0, "pairs_total": 0, "error": None, "started": None, "finished": None, **kw}

    def log(self, msg):
        self.state["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        self.state["step"] = msg


JOB = Job()


def state(out, n_eps=None):
    with open(os.path.join(out, "promos.json"), encoding="utf-8") as f:
        d = json.load(f)
    rows = {r["id"]: r for r in P.load_rows(os.path.join(out, "rows.jsonl"))}
    promos = d["promos"]
    for p in promos:
        r = rows.get(p["id"], {})
        p["script"] = r.get("view", {}).get("lines", [])
        j = r.get("judge") or {}
        p["scores"] = {k: v for k, v in j.items() if k in ("hook", "cliffhanger", "standalone", "emotion")} if M.real_judge(j) else {}
    n_eps = n_eps or max([s["episode"] for p in promos for s in p["segments"]] or [1])
    waves = [{"title": t, "why": w, "items": [p["rank"] for p in items]} for t, items, w in P.posting_plan(promos, n_eps)]
    ev = None
    if os.path.exists(os.path.join(out, "eval.json")):
        with open(os.path.join(out, "eval.json"), encoding="utf-8") as f:
            ev = json.load(f)
    fb = L.load_feedback(fb_path(out))
    return {"manifest": d["manifest"], "promos": promos, "waves": waves, "eval": ev,
            "feedback": {r["id"]: r["y"] for r in fb}, "out": out, "model": M.load().version}


def budget():
    """Which judge is configured (re-read from .env each time) and, for OpenRouter, free requests left today."""
    load_env(override=True)
    key, base = os.environ.get("JUDGE_API_KEY"), os.environ.get("JUDGE_BASE_URL", "")
    host = urllib.parse.urlparse(base).hostname if base else None
    provider = {"integrate.api.nvidia.com": "NVIDIA", "openrouter.ai": "OpenRouter"}.get(host, host)
    info = {"model": os.environ.get("JUDGE_MODEL"), "provider": provider, "llm_available": bool(key and os.environ.get("JUDGE_MODEL")),
            "max_calls": LLM_BUDGET["top_n"] + LLM_BUDGET["reserve_endings"] + LLM_BUDGET["stitch_n"] + LLM_BUDGET["pairwise_top"] * (LLM_BUDGET["pairwise_top"] - 1)}
    if key and "openrouter.ai" in base:
        try:
            req = urllib.request.Request(base.rstrip("/") + "/key", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=10) as r:
                info["free_requests"] = json.load(r)["data"].get("free_model_daily_requests")
        except Exception as ex:
            info["free_requests_error"] = type(ex).__name__
    return info


def run_job(scripts_dir, out, offline, k):
    st = JOB.state
    try:
        judge = get_judge(offline=offline, reload_env=True)
        if hasattr(judge, "max_new"):
            left = (budget().get("free_requests") or {}).get("remaining")
            if left is not None:
                judge.max_new = max(0, left - 1)  # keep one request in reserve
                JOB.log(f"{left} free requests left today; new judgements capped at {judge.max_new}, cached ones are free")
        model = M.load()
        prefilters = [M.load("v0000")] + ([model] if model.version != "v0000" else [])
        full = {"top_n": 200, "reserve_endings": "all", "stitch_n": 30, "pairwise_top": 0}
        if offline:
            params = full
        elif "openrouter.ai" in os.environ.get("JUDGE_BASE_URL", ""):
            params = LLM_BUDGET  # 50 requests/day: cut-down shortlist
        else:
            params = dict(full, pairwise_top=LLM_BUDGET["pairwise_top"])  # e.g. NVIDIA: rate-limited, not day-capped
        JOB.log(f"judge: {judge.name}")

        def on_judged(i, n, by):
            st["judged"], st["to_judge"] = i, n
            st["cached"] = getattr(judge, "stats", {}).get("cache_hits", 0)
            st["step"] = f"judging clip {i}/{n} with {judge.name}" + (f" ({st['cached']} from cache)" if st["cached"] else "")

        rows, ctx, stats = P.featurize(scripts_dir, judge, prefilters, top_n=params["top_n"], aliases="auto",
                                       stitch_n=params["stitch_n"], reserve_endings=params["reserve_endings"],
                                       log=JOB.log, progress=on_judged)
        P.save_rows(rows, os.path.join(out, "rows.jsonl"))
        promos = P.rank(rows, model, k=k)
        if params["pairwise_top"] and hasattr(judge, "compare"):
            n = min(params["pairwise_top"], len(promos))
            left = (budget().get("free_requests") or {}).get("remaining")
            if left is not None and left < n * (n - 1):
                JOB.log(f"skipping pairwise re-rank: needs up to {n * (n - 1)} requests, {left} free requests left today")
                n = 0
            params = dict(params, pairwise_top=n)
        if params["pairwise_top"] and hasattr(judge, "compare"):
            n = params["pairwise_top"]
            st["pairs_total"] = n * (n - 1) // 2

            def on_pair(a, b):
                st["pairs"] += 1
                st["step"] = f"pairwise: #{a} vs #{b} ({st['pairs']}/{st['pairs_total']}, both orders)"
            promos = P.pairwise_rerank(promos, rows, judge, top=params["pairwise_top"], log=JOB.log, progress=on_pair)
        stats.update(getattr(judge, "stats", {}))
        stats["tag_coverage"] = ctx.get("tag_coverage")
        man = P.manifest(model, judge, scripts_dir, {"k": k, "offline": offline, **params}, stats)
        man["show"] = ctx.get("show")
        P.write_outputs(promos, out, man, ctx["n_eps"])
        JOB.log(f"done: {len(promos)} promos; {stats.get('calls', 0)} judge calls, {stats.get('cache_hits', 0)} from cache")
        H.out, H.scripts = out, scripts_dir
        st["status"] = "done"
    except P.JudgeUnavailable as ex:
        st["status"], st["error"], st["fix"] = "error", f"{ex} Nothing was mixed or saved.", "offline"
        JOB.log(f"stopped: {ex}")
    except (ValueError, FileNotFoundError) as ex:
        st["status"], st["error"], st["fix"] = "error", str(ex), "upload"
        JOB.log(f"stopped: {ex}")
    except urllib.error.URLError as ex:
        st["status"], st["error"], st["fix"] = "error", "Couldn't reach the LLM service. Check the internet connection or run offline.", "offline"
        JOB.log(f"stopped: network error ({ex})")
    except Exception as ex:  # anything unexpected: plain message in the UI, full trace in the server log
        traceback.print_exc()
        st["status"], st["error"], st["fix"] = "error", f"Something went wrong ({type(ex).__name__}). The details are in the server log.", "retry"
        JOB.log(f"failed: {type(ex).__name__}: {ex}")
    st["finished"] = time.time()


class H(BaseHTTPRequestHandler):
    out = None  # the page starts empty; results appear only after a run in this session (or `serve --show DIR`)
    scripts = "data/scripts"

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "demo.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path.startswith("/api/state"):
            if not H.out:
                return self._send(404, {"error": "no run yet"})
            try:
                return self._send(200, state(H.out))
            except (FileNotFoundError, RuntimeError):
                return self._send(404, {"error": "no run yet"})
        if self.path.startswith("/api/job"):
            return self._send(200, JOB.state)
        if self.path.startswith("/api/budget"):
            return self._send(200, budget())
        self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            if self.path == "/api/upload":
                return self._send(200, self.upload())
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                raise ValueError("Bad request.")
            if self.path in ("/api/feedback", "/api/retrain", "/api/eval") and not H.out:
                raise ValueError("Run the pipeline on a scripts zip first.")
            if self.path == "/api/start":
                return self._send(200, self.start(body))
            with LOCK:
                if self.path == "/api/feedback":
                    return self._send(200, self.feedback(body))
                if self.path == "/api/retrain":
                    return self._send(200, self.retrain())
                if self.path == "/api/eval":
                    return self._send(200, self.eval())
        except zipfile.BadZipFile:
            return self._send(400, {"error": "That file isn't a valid zip."})
        except ValueError as ex:
            return self._send(400, {"error": str(ex)})
        except Exception as ex:
            traceback.print_exc()
            return self._send(500, {"error": f"Something went wrong ({type(ex).__name__}). The details are in the server log."})
        self._send(404, {"error": "not found"})

    def upload(self):
        """Body = raw zip bytes. Saved to a fresh folder, then only .txt members are extracted (see ingest.py)."""
        if JOB.state["status"] == "running":
            raise ValueError("A run is already in progress.")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            raise ValueError("The upload was empty.")
        if n > MAX_UPLOAD:
            raise ValueError(f"The zip must be under {MAX_UPLOAD // 1024 // 1024} MB.")
        job_id = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(UPLOADS, job_id)
        os.makedirs(base, exist_ok=True)
        zpath = os.path.join(base, "upload.zip")
        with open(zpath, "wb") as f:
            left = n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        written, skipped = extract_scripts(zpath, os.path.join(base, "scripts"))
        name = os.path.basename(self.headers.get("X-Filename", "upload.zip"))
        JOB.reset(status="uploaded", upload=job_id)
        JOB.log(f"received {name} ({n / 1024 / 1024:.1f} MB)")
        JOB.log(f"extracted {len(written)} scripts locally to {base}/scripts" + (f"; skipped {len(skipped)} non-script files" if skipped else ""))
        return {"upload": job_id, "files": [os.path.basename(p) for p in written], "skipped": skipped[:20]}

    def start(self, body):
        if JOB.state["status"] == "running":
            raise ValueError("A run is already in progress.")
        up = JOB.state.get("upload") or body.get("upload")
        if not up:
            raise ValueError("Upload a scripts zip first.")
        base = os.path.join(UPLOADS, os.path.basename(up))
        offline = bool(body.get("offline"))
        if not offline:
            b = budget()
            if not b["llm_available"]:
                raise ValueError("No LLM key is set in .env. Run offline, or add a key and try again.")
            # no blanket refusal: cached clips are free; new calls are capped at today's remaining quota in run_job
        log_so_far = JOB.state["log"]
        JOB.reset(status="running", upload=up, started=time.time(), log=log_so_far)
        threading.Thread(target=run_job, args=(os.path.join(base, "scripts"), os.path.join(base, "out"), offline,
                                               int(body.get("k", 25))), daemon=True).start()
        return {"ok": True}

    def feedback(self, body):
        """accept / reject / clear one promo. Labels live next to the run, so every new run starts unreviewed."""
        path = fb_path(H.out)
        promos = {p["id"]: p for p in state(H.out)["promos"]}
        p = promos.get(body.get("id"))
        if p is None:
            raise ValueError("That clip isn't in the current run.")
        others = [r for r in L.load_feedback(path) if r["id"] != p["id"]]  # latest click wins
        keep = others
        if body.get("decision") in ("accept", "reject"):
            keep = others + [{"id": p["id"], "episode": p["segments"][0]["episode"], "x": p["features"], "ts": time.time(),
                              "y": 1 if body["decision"] == "accept" else 0, "editor": "demo", "src": "ui", "schema": p.get("schema")}]
        with open(path, "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        return {"ok": True, "labels": len(keep)}

    def retrain(self):
        """Trial retrain from the editor's choices, explained in plain words. Never saved or promoted."""
        from .cli import _ctx
        champ = M.load()
        chall, note = L.retrain(champ, min_labels=4, path=fb_path(H.out))
        if chall is None:
            return {"message": "Review at least 4 clips (accept or reject) first."}
        ctx = _ctx(H.scripts)
        gold = [g for g in E.load_gold("gold/gold.jsonl") if g.get("show") == ctx["show"]]
        learned = plain_shifts(champ, chall)
        if not gold:
            return {"verdict": "There are no editor labels for this show yet, so there's no fair way to check whether the "
                               "retrained ranker is better. It is kept as a trial only.", "promote": False,
                    "before": {"good": "–", "bad": "–"}, "after": {"good": "–", "bad": "–"}, "learned": learned}
        rows = P.load_rows(os.path.join(H.out, "rows.jsonl"))
        res = L.gate(rows, ctx, gold, champ, chall, int(ctx["n_eps"] * 0.6) + 1, ctx["n_eps"])
        c, n = res["champion"]["metrics"], res["challenger"]["metrics"]
        reason = res["reason"]
        if res["promote"]:
            verdict = "On episodes it never saw, it finds more good clips without adding bad ones."
        elif "known-bad" in reason:
            verdict = "It finds more good clips, but it also starts picking clips an editor marked as bad, so it isn't trusted yet."
        elif "noise" in reason:
            verdict = "It looks a little better, but the difference is too small to trust from this many choices."
        elif "validity" in reason:
            verdict = "It produced clips that break the cutting rules, so it was rejected."
        else:
            verdict = "It isn't better than the current ranker on episodes it never saw."
        count = lambda m, k: int(round((m.get(k) or 0) * 10))
        return {"promote": res["promote"], "verdict": verdict,
                "before": {"good": count(c, "precision@10"), "bad": count(c, "bad_rate@10")},
                "after": {"good": count(n, "precision@10"), "bad": count(n, "bad_rate@10")},
                "learned": learned, "saved": False}

    def eval(self):
        from .cli import main
        main(["--scripts", H.scripts, "--out", H.out, "eval"])
        return {"ok": True}


class _Server(ThreadingHTTPServer):
    """http.server sets SO_REUSEADDR, which on Windows lets a second demo bind the same port and silently share the
    traffic with an old copy. Bind exclusively instead, so a second start fails loudly."""
    allow_reuse_address = False

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(out=None, port=8765, scripts="data/scripts", host="127.0.0.1"):
    """Local by default. A hosted deploy passes --host 0.0.0.0 (and its platform's port)."""
    H.out, H.scripts = out, scripts
    try:
        srv = _Server((host, port), H)
    except OSError:
        raise SystemExit(f"Port {port} is already in use: the demo is probably already running (open "
                         f"http://127.0.0.1:{port}), or stop it first. To run a second copy, use --port.")
    print(f"Promo Finder demo on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/live")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(a.out, a.port)
