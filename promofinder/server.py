"""Demo UI: python -m promofinder serve  ->  http://127.0.0.1:8765

Flow: upload a scripts .zip (or the built-in sample) -> extracted on the server (only .txt, safe) -> parsed ->
candidates -> shortlisted clips sent to the LLM judge (progress streamed) -> ranked promos, posting plan, learning loop.

Each browser gets its own session (cookie), so several visitors can use a hosted copy without seeing each other's runs.
One pipeline runs at a time. PROMO_HOSTED=1 switches on the public-deploy limits (smaller shortlist, upload cap,
per-visitor run limit, old uploads deleted). Stdlib only. Demo clicks never save or promote a model.
"""
import argparse
import json
import os
import secrets
import shutil
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
ROOT = os.path.dirname(HERE)
SAMPLE_DIR = os.path.join(ROOT, "samples", "the-last-heir")
UPLOADS = "runs/uploads"
HOSTED = os.environ.get("PROMO_HOSTED") == "1"
MAX_UPLOAD = (20 if HOSTED else 100) * 1024 * 1024
UPLOAD_TTL = 6 * 3600                      # hosted: uploads and results are deleted after this
RUNS_PER_HOUR = int(os.environ.get("PROMO_RUNS_PER_HOUR", 6))  # hosted: LLM runs per visitor IP per hour
MAX_SESSIONS = 500
# Free tier = 50 requests/day: 10 best clips + 30 reserved episode endings + 2 stitches (~42, fewer after overlap and
# cache) + pairwise on the top 4 (12). New calls are capped at the remaining quota anyway; cached clips are free.
LLM_BUDGET = {"top_n": 10, "reserve_endings": 30, "stitch_n": 2, "pairwise_top": int(os.environ.get("PROMO_PAIRWISE_TOP", 4))}
# Hosted with a rate-limited (not day-capped) provider: every episode ending + the 10 best + pairwise on the top 4,
# about 60-70 judge calls, 1-2 minutes. The full local pool (~200 clips) takes ~5 minutes: too long for a visitor.
HOSTED_POOL = {"top_n": 10, "reserve_endings": "all", "stitch_n": 2, "pairwise_top": 4}
RUN_LOCK = threading.Lock()  # one pipeline at a time: protects the API quota and the CPU

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


class Session:
    """One visitor: their upload, their run's job state, their results folder."""

    def __init__(self, out=None, scripts="data/scripts"):
        self.out, self.scripts, self.upload, self.seen = out, scripts, None, time.time()
        self.reset()

    def reset(self, **kw):
        self.job = {"status": "idle", "step": "", "log": [], "judged": 0, "to_judge": 0, "cached": 0,
                    "pairs": 0, "pairs_total": 0, "error": None, "started": None, "finished": None, **kw}

    def log(self, msg):
        self.job["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        self.job["step"] = msg


SESSIONS, SESSIONS_LOCK = {}, threading.Lock()
RUN_LOG = {}  # ip -> [timestamps of LLM runs], hosted rate limit
DEFAULT = {"out": None, "scripts": "data/scripts"}


def get_session(sid):
    with SESSIONS_LOCK:
        s = SESSIONS.get(sid) if sid else None
        new = s is None
        if new:
            if len(SESSIONS) >= MAX_SESSIONS:  # drop the least recently seen
                del SESSIONS[min(SESSIONS, key=lambda k: SESSIONS[k].seen)]
            sid = secrets.token_urlsafe(18)
            s = SESSIONS[sid] = Session(DEFAULT["out"], DEFAULT["scripts"])
        s.seen = time.time()
        return sid, s, new


def cleanup_uploads():
    """Hosted: delete upload folders (scripts + results) older than UPLOAD_TTL."""
    if not HOSTED or not os.path.isdir(UPLOADS):
        return
    now = time.time()
    for d in os.listdir(UPLOADS):
        p = os.path.join(UPLOADS, d)
        try:
            if now - os.path.getmtime(p) > UPLOAD_TTL:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


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
    fb = L.load_feedback(fb_path(out))
    return {"manifest": d["manifest"], "promos": promos, "waves": waves,
            "feedback": {r["id"]: r["y"] for r in fb}, "model": M.load().version}


def budget():
    """Which judge is configured (re-read from .env each time) and, for OpenRouter, free requests left today."""
    load_env(override=True)
    key, base = os.environ.get("JUDGE_API_KEY"), os.environ.get("JUDGE_BASE_URL", "")
    host = urllib.parse.urlparse(base).hostname if base else None
    provider = {"integrate.api.nvidia.com": "NVIDIA", "openrouter.ai": "OpenRouter"}.get(host, host)
    info = {"model": os.environ.get("JUDGE_MODEL"), "provider": provider, "hosted": HOSTED,
            "llm_available": bool(key and os.environ.get("JUDGE_MODEL")), "sample": os.path.isdir(SAMPLE_DIR),
            "max_upload_mb": MAX_UPLOAD // 1024 // 1024}
    if key and "openrouter.ai" in base:
        try:
            req = urllib.request.Request(base.rstrip("/") + "/key", headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=10) as r:
                info["free_requests"] = json.load(r)["data"].get("free_model_daily_requests")
        except Exception as ex:
            info["free_requests_error"] = type(ex).__name__
    return info


def run_job(sess, scripts_dir, out, offline, k):
    st = sess.job
    try:
        judge = get_judge(offline=offline, reload_env=True)
        if hasattr(judge, "max_new"):
            left = (budget().get("free_requests") or {}).get("remaining")
            if left is not None:
                judge.max_new = max(0, left - 1)  # keep one request in reserve
                sess.log(f"{left} free requests left today; new judgements capped at {judge.max_new}, cached ones are free")
        model = M.load()
        prefilters = [M.load("v0000")] + ([model] if model.version != "v0000" else [])
        full = {"top_n": 200, "reserve_endings": "all", "stitch_n": 30, "pairwise_top": 0}
        if offline:
            params = full
        elif "openrouter.ai" in os.environ.get("JUDGE_BASE_URL", ""):
            params = LLM_BUDGET  # 50 requests/day: cut-down shortlist
        elif HOSTED:
            params = HOSTED_POOL
        else:
            params = dict(full, pairwise_top=LLM_BUDGET["pairwise_top"])  # local, rate-limited provider: full pool
        sess.log("judge: " + ("offline (no API)" if offline else judge.name.replace("llm:", "")))

        def on_judged(i, n, by):
            st["judged"], st["to_judge"] = i, n
            st["cached"] = getattr(judge, "stats", {}).get("cache_hits", 0)
            st["step"] = f"judging clip {i}/{n}" + (f" ({st['cached']} from cache)" if st["cached"] else "")

        rows, ctx, stats = P.featurize(scripts_dir, judge, prefilters, top_n=params["top_n"], aliases="auto",
                                       stitch_n=params["stitch_n"], reserve_endings=params["reserve_endings"],
                                       log=sess.log, progress=on_judged)
        P.save_rows(rows, os.path.join(out, "rows.jsonl"))
        promos = P.rank(rows, model, k=k)
        if params["pairwise_top"] and hasattr(judge, "compare"):
            n = min(params["pairwise_top"], len(promos))
            left = (budget().get("free_requests") or {}).get("remaining")
            if left is not None and left < n * (n - 1):
                sess.log(f"skipping pairwise re-rank: needs up to {n * (n - 1)} requests, {left} free requests left today")
                n = 0
            params = dict(params, pairwise_top=n)
        if params["pairwise_top"] and hasattr(judge, "compare"):
            n = params["pairwise_top"]
            st["pairs_total"] = n * (n - 1) // 2

            def on_pair(a, b):
                st["pairs"] += 1
                st["step"] = f"pairwise: #{a} vs #{b} ({st['pairs']}/{st['pairs_total']}, both orders)"
            promos = P.pairwise_rerank(promos, rows, judge, top=params["pairwise_top"], log=sess.log, progress=on_pair)
        stats.update(getattr(judge, "stats", {}))
        stats["tag_coverage"] = ctx.get("tag_coverage")
        man = P.manifest(model, judge, os.path.basename(os.path.dirname(scripts_dir)), {"k": k, "offline": offline, **params}, stats)
        man["show"] = ctx.get("show")
        P.write_outputs(promos, out, man, ctx["n_eps"])
        sess.log(f"done: {len(promos)} promos; {stats.get('calls', 0)} judge calls, {stats.get('cache_hits', 0)} from cache")
        sess.out, sess.scripts = out, scripts_dir
        st["status"] = "done"
    except P.JudgeUnavailable as ex:
        st["status"], st["error"], st["fix"] = "error", f"{ex} Nothing was mixed or saved.", "offline"
        sess.log(f"stopped: {ex}")
    except (ValueError, FileNotFoundError) as ex:
        st["status"], st["error"], st["fix"] = "error", str(ex), "upload"
        sess.log(f"stopped: {ex}")
    except urllib.error.URLError as ex:
        st["status"], st["error"], st["fix"] = "error", "Couldn't reach the LLM service. Try again, or run offline.", "offline"
        sess.log(f"stopped: network error ({ex})")
    except Exception as ex:  # anything unexpected: plain message in the UI, full trace in the server log
        traceback.print_exc()
        st["status"], st["error"], st["fix"] = "error", f"Something went wrong ({type(ex).__name__}). Please try again.", "retry"
        sess.log(f"failed: {type(ex).__name__}: {ex}")
    finally:
        st["finished"] = time.time()
        RUN_LOCK.release()


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if getattr(self, "_new_sid", None):
            secure = "; Secure" if HOSTED else ""
            self.send_header("Set-Cookie", f"pf_sid={self._new_sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400{secure}")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def _session(self):
        sid = None
        for part in (self.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == "pf_sid":
                sid = v
        sid, sess, new = get_session(sid)
        self._new_sid = sid if new else None
        return sess

    def _ip(self):
        fwd = self.headers.get("X-Forwarded-For")
        return fwd.split(",")[0].strip() if fwd else self.client_address[0]

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        sess = self._session()
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "demo.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/state":
            if not sess.out:
                return self._send(404, {"error": "no run yet"})
            try:
                return self._send(200, state(sess.out))
            except (FileNotFoundError, RuntimeError):
                return self._send(404, {"error": "no run yet"})
        if path == "/api/job":
            return self._send(200, sess.job)
        if path == "/api/budget":
            return self._send(200, budget())
        if path == "/healthz":
            return self._send(200, {"ok": True})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        sess = self._session()
        try:
            if self.path == "/api/upload":
                return self._send(200, self.upload(sess))
            n = int(self.headers.get("Content-Length", 0))
            if n > 64 * 1024:
                raise ValueError("Bad request.")
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                raise ValueError("Bad request.")
            if self.path == "/api/sample":
                return self._send(200, self.sample(sess))
            if self.path == "/api/start":
                return self._send(200, self.start(sess, body))
            if self.path in ("/api/feedback", "/api/retrain") and not sess.out:
                raise ValueError("Run the pipeline on a scripts zip first.")
            if self.path == "/api/feedback":
                return self._send(200, self.feedback(sess, body))
            if self.path == "/api/retrain":
                return self._send(200, self.retrain(sess))
        except zipfile.BadZipFile:
            return self._send(400, {"error": "That file isn't a valid zip."})
        except ValueError as ex:
            return self._send(400, {"error": str(ex)})
        except Exception as ex:
            traceback.print_exc()
            return self._send(500, {"error": f"Something went wrong ({type(ex).__name__}). Please try again."})
        self._send(404, {"error": "not found"})

    def _new_upload_dir(self):
        cleanup_uploads()
        base = os.path.join(UPLOADS, time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3))
        os.makedirs(base, exist_ok=True)
        return base

    def upload(self, sess):
        """Body = raw zip bytes. Saved to a fresh folder, then only .txt members are extracted (see ingest.py)."""
        if sess.job["status"] == "running":
            raise ValueError("Your run is still in progress.")
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            raise ValueError("The upload was empty.")
        if n > MAX_UPLOAD:
            raise ValueError(f"The zip must be under {MAX_UPLOAD // 1024 // 1024} MB.")
        base = self._new_upload_dir()
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
        os.remove(zpath)
        name = os.path.basename(self.headers.get("X-Filename", "upload.zip"))
        sess.upload = base
        sess.reset(status="uploaded")
        sess.log(f"received {name} ({n / 1024 / 1024:.1f} MB)")
        sess.log(f"extracted {len(written)} scripts" + (f"; skipped {len(skipped)} non-script files" if skipped else ""))
        return {"files": [os.path.basename(p) for p in written], "skipped": skipped[:20]}

    def sample(self, sess):
        """Load the built-in fictional sample show, so anyone can try the tool without their own scripts."""
        if sess.job["status"] == "running":
            raise ValueError("Your run is still in progress.")
        if not os.path.isdir(SAMPLE_DIR):
            raise ValueError("No sample show on this server.")
        base = self._new_upload_dir()
        shutil.copytree(SAMPLE_DIR, os.path.join(base, "scripts"))
        files = sorted(f for f in os.listdir(os.path.join(base, "scripts")) if f.endswith(".txt"))
        sess.upload = base
        sess.reset(status="uploaded")
        sess.log(f"loaded the sample show: {len(files)} scripts")
        return {"files": files, "skipped": [], "sample": True}

    def start(self, sess, body):
        if sess.job["status"] == "running":
            raise ValueError("Your run is still in progress.")
        if not sess.upload or not os.path.isdir(os.path.join(sess.upload, "scripts")):
            raise ValueError("Upload a scripts zip (or load the sample) first.")
        offline = bool(body.get("offline"))
        if not offline and not budget()["llm_available"]:
            raise ValueError("The AI judge isn't configured on this server. Run offline instead.")
        if not RUN_LOCK.acquire(blocking=False):
            raise ValueError("Another visitor's run is in progress. Please try again in a minute.")
        if not offline and HOSTED:  # counted only once the run can actually start
            ip, now = self._ip(), time.time()
            recent = [t for t in RUN_LOG.get(ip, []) if now - t < 3600]
            if len(recent) >= RUNS_PER_HOUR:
                RUN_LOCK.release()
                raise ValueError(f"You've used {RUNS_PER_HOUR} AI-judged runs this hour. Run offline, or try again later.")
            RUN_LOG[ip] = recent + [now]
        log_so_far = sess.job["log"]
        sess.reset(status="running", started=time.time(), log=log_so_far)
        out = os.path.join(sess.upload, "out")
        threading.Thread(target=run_job, args=(sess, os.path.join(sess.upload, "scripts"), out, offline,
                                               int(body.get("k", 25))), daemon=True).start()
        return {"ok": True}

    def feedback(self, sess, body):
        """accept / reject / clear one promo. Labels live next to the run, so every new run starts unreviewed."""
        path = fb_path(sess.out)
        promos = {p["id"]: p for p in state(sess.out)["promos"]}
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

    def retrain(self, sess):
        """Trial retrain from the editor's choices, explained in plain words. Never saved or promoted."""
        from .cli import _ctx
        champ = M.load()
        chall, note = L.retrain(champ, min_labels=4, path=fb_path(sess.out))
        if chall is None:
            return {"message": "Review at least 4 clips (accept or reject) first."}
        ctx = _ctx(sess.scripts)
        gold = [g for g in E.load_gold(os.path.join(ROOT, "gold", "gold.jsonl")) if g.get("show") == ctx["show"]]
        learned = plain_shifts(champ, chall)
        if not gold:
            return {"verdict": "There are no editor labels for this show yet, so there's no fair way to check whether the "
                               "retrained ranker is better. It is kept as a trial only.", "promote": False,
                    "before": {"good": "–", "bad": "–"}, "after": {"good": "–", "bad": "–"}, "learned": learned}
        rows = P.load_rows(os.path.join(sess.out, "rows.jsonl"))
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


class _Server(ThreadingHTTPServer):
    """http.server sets SO_REUSEADDR, which on Windows lets a second demo bind the same port and silently share the
    traffic with an old copy. Bind exclusively instead, so a second start fails loudly."""
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        import socket
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(out=None, port=8765, scripts="data/scripts", host="127.0.0.1"):
    """Local by default. A hosted deploy passes --host 0.0.0.0 (and its platform's port) with PROMO_HOSTED=1."""
    DEFAULT["out"], DEFAULT["scripts"] = out, scripts
    try:
        srv = _Server((host, port), H)
    except OSError:
        raise SystemExit(f"Port {port} is already in use: the demo is probably already running (open "
                         f"http://127.0.0.1:{port}), or stop it first. To run a second copy, use --port.")
    print(f"Promo Finder demo on http://{host}:{port}{'  [hosted mode]' if HOSTED else ''}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(a.out, a.port)
