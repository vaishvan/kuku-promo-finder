"""Local demo UI: python -m promofinder serve  ->  http://127.0.0.1:8765

Stdlib only. Binds to localhost. Feedback clicked in the UI goes to runs/feedback.demo.jsonl, never the real log,
and "retrain" in the UI evaluates a challenger without saving or promoting it.
"""
import argparse
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import evaluate as E
from . import learn as L
from . import model as M
from . import pipeline as P

HERE = os.path.dirname(__file__)
DEMO_FEEDBACK = "runs/feedback.demo.jsonl"
LOCK = threading.Lock()


def state(out, n_eps=50):
    with open(os.path.join(out, "promos.json"), encoding="utf-8") as f:
        d = json.load(f)
    rows = {r["id"]: r for r in P.load_rows(os.path.join(out, "rows.jsonl"))}
    promos = d["promos"]
    for p in promos:
        r = rows.get(p["id"], {})
        p["script"] = r.get("view", {}).get("lines", [])
        j = r.get("judge") or {}
        p["scores"] = {k: v for k, v in j.items() if k in ("hook", "cliffhanger", "standalone", "emotion")} if M.real_judge(j) else {}
    waves = [{"title": t, "why": w, "items": [p["rank"] for p in items]} for t, items, w in P.posting_plan(promos, n_eps)]
    ev = None
    if os.path.exists(os.path.join(out, "eval.json")):
        with open(os.path.join(out, "eval.json"), encoding="utf-8") as f:
            ev = json.load(f)
    fb = L.load_feedback(DEMO_FEEDBACK)
    return {"manifest": d["manifest"], "promos": promos, "waves": waves, "eval": ev,
            "feedback": {r["id"]: r["y"] for r in fb}, "out": out, "model": M.load().version}


class H(BaseHTTPRequestHandler):
    out = "runs/r1"

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
            try:
                return self._send(200, state(H.out))
            except FileNotFoundError:
                return self._send(404, {"error": f"no run in {H.out} yet: press Run"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            with LOCK:
                if self.path == "/api/feedback":
                    return self._send(200, self.feedback(body))
                if self.path == "/api/retrain":
                    return self._send(200, self.retrain())
                if self.path == "/api/run":
                    return self._send(200, self.run(body))
                if self.path == "/api/eval":
                    return self._send(200, self.eval())
        except Exception as ex:
            traceback.print_exc()
            return self._send(500, {"error": f"{type(ex).__name__}: {ex}"})
        self._send(404, {"error": "not found"})

    def feedback(self, body):
        promos = {p["id"]: p for p in state(H.out)["promos"]}
        p = promos[body["id"]]
        rec = {"id": p["id"], "episode": p["segments"][0]["episode"], "x": p["features"],
               "y": 1 if body["decision"] == "accept" else 0, "editor": "demo", "src": "ui", "schema": p.get("schema")}
        others = [r for r in L.load_feedback(DEMO_FEEDBACK) if r["id"] != p["id"]]  # latest click wins
        os.makedirs("runs", exist_ok=True)
        with open(DEMO_FEEDBACK, "w", encoding="utf-8") as f:
            for r in others + [rec]:
                f.write(json.dumps(r) + "\n")
        return {"ok": True, "labels": len(others) + 1}

    def retrain(self):
        from .cli import _ctx, _split
        champ = M.load()
        chall, note = L.retrain(champ, min_labels=4, path=DEMO_FEEDBACK)
        if chall is None:
            return {"note": note}
        ctx = _ctx("data/scripts")
        gold = E.load_gold("gold/gold.jsonl")
        rows = P.load_rows(os.path.join(H.out, "rows.jsonl"))
        res = L.gate(rows, ctx, gold, champ, chall, int(ctx["n_eps"] * 0.6) + 1, ctx["n_eps"])
        shifts = sorted(((chall.w[k] - champ.w[k], k) for k in M.FEATURES), key=lambda t: -abs(t[0]))[:5]
        return {"note": note, "promote": res["promote"], "reason": res["reason"], "ci": res["ndcg_gain_ci"],
                "champion": res["champion"]["metrics"], "challenger": res["challenger"]["metrics"],
                "shifts": [[k, round(d, 2)] for d, k in shifts], "saved": False}

    def run(self, body):
        from .cli import main
        out = body.get("out") or H.out
        args = ["--scripts", body.get("scripts") or "data/scripts", "--out", out, "run", "--k", str(body.get("k", 25))]
        if body.get("offline", True):
            args.append("--offline")
        else:
            args += ["--top-n", "25", "--stitch-n", "2", "--pairwise-top", "4"]  # free tier: 50 requests/day
        main(args)
        H.out = out
        return {"ok": True, "out": out}

    def eval(self):
        from .cli import main
        main(["--out", H.out, "eval"])
        return {"ok": True}


def serve(out="runs/r1", port=8765):
    H.out = out
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"Promo Finder demo on http://127.0.0.1:{port}  (showing {out}; Ctrl+C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/r1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    serve(a.out, a.port)
