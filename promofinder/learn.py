"""Stages 7-8: feedback log -> retrain -> champion/challenger gate."""
import csv
import json
import os

from . import candidates as C
from . import evaluate as E
from . import model as M
from .features import SCHEMA
from .pipeline import _row, rank

FEEDBACK = "runs/feedback.jsonl"  # real editor labels: append-only, never truncated
SIM_FEEDBACK = "runs/feedback.sim.jsonl"  # synthetic labels from `simulate`, kept apart (fix A2)


def log_feedback(records, path=FEEDBACK):
    """Append, skipping (id, editor) pairs already logged so re-ingesting a CSV cannot double-count."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen = {(r["id"], r.get("editor")) for r in load_feedback(path)}
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            if (r["id"], r.get("editor")) in seen:
                continue
            seen.add((r["id"], r.get("editor")))
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def load_feedback(path=FEEDBACK):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def snap(ctx, ep, a, b):
    """Move an editor's trim to the nearest legal cut points: start on a dialogue/action start, end on a beat end."""
    E_ = ctx["eps"][ep]
    beats = [e for e in E_.events if e.kind != "camera"]
    starts = sorted({e.t0 for e in beats if e.kind in ("dialogue", "action")})
    ends = sorted({e.t1 for e in beats} | {E_.end})
    return min(starts, key=lambda t: abs(t - a)), min(ends, key=lambda t: abs(t - b))


def ingest_csv(csv_path, promos_json, ctx, judge, editor="editor"):
    """Editor fills `decision` = accept|reject, optionally trim_start/trim_end (seconds, same episode).
    A trim means "this cut, not yours": trimmed window -> positive, original window -> negative (fix A1)."""
    with open(promos_json, encoding="utf-8") as f:
        promos = {p["id"]: p for p in json.load(f)["promos"]}
    recs, warnings = [], []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        d = (r.get("decision") or "").strip().lower()
        if r.get("id") not in promos:
            if d:
                warnings.append(f"unknown id {r.get('id')!r}")
            continue
        if d not in ("accept", "reject"):
            continue
        p = promos[r["id"]]
        trimmed = (r.get("trim_start") or "").strip() and (r.get("trim_end") or "").strip()
        if trimmed:
            if len(p["segments"]) != 1:
                warnings.append(f"{p['id']}: trims are only supported on single-segment promos")
            else:
                ep = p["segments"][0]["episode"]
                a, b = snap(ctx, ep, float(r["trim_start"]), float(r["trim_end"]))
                if not 20 <= b - a <= 60:
                    warnings.append(f"{p['id']}: trim {a:.1f}-{b:.1f} is {b - a:.1f}s, outside 20-60s; ignored")
                else:
                    row = _row(ctx, [(ep, a, b)])
                    recs.append({"id": C.cid([(ep, a, b)]), "episode": ep, "x": M.vec(row["feats"], judge(row["view"])),
                                 "y": 1, "editor": editor, "src": "csv-trim", "trim_of": p["id"], "schema": SCHEMA})
                    recs.append({"id": p["id"], "episode": ep, "x": p["features"], "y": 0, "editor": editor, "src": "csv-trimmed-original", "schema": p.get("schema")})
                    continue
        recs.append({"id": p["id"], "episode": p["segments"][0]["episode"], "x": p["features"],
                     "y": 1 if d == "accept" else 0, "editor": editor, "src": "csv", "schema": p.get("schema")})
    return recs, warnings


def _ep(r):
    return r["episode"] if "episode" in r else int(r["id"].split(":")[0])


def retrain(champion, min_labels=30, l2=None, path=FEEDBACK, grid=(0.1, 1, 10, 100, 1000), folds=5):
    """l2 is chosen by episode-grouped CV on the feedback labels only (held-out gold is never used to fit).
    Pairwise ranking loss when there are enough same-episode accept/reject pairs (E3), pointwise otherwise.
    Labels from another feature schema are skipped, not silently mixed (A4)."""
    allfb = [r for r in load_feedback(path) if "x" in r]
    fb = [r for r in allfb if r.get("schema") == SCHEMA]
    skipped = len(allfb) - len(fb)
    if len(fb) < min_labels:
        return None, f"only {len(fb)} labels with the current feature schema ({skipped} from other schemas skipped), need {min_labels}"
    # folds = contiguous episode blocks, so neighbouring episodes (same storyline) never straddle train/test
    uniq = sorted({_ep(r) for r in fb})
    block = {e: i * folds // len(uniq) for i, e in enumerate(uniq)}
    X, Y, G = [r["x"] for r in fb], [r["y"] for r in fb], [block[_ep(r)] for r in fb]
    EP = [_ep(r) for r in fb]
    pairwise = len(M.pairs_of(Y, EP)) >= 10

    def fit(idx, lam):
        if pairwise:
            return M.train_pairwise([X[i] for i in idx], [Y[i] for i in idx], [EP[i] for i in idx], champion, l2=lam)
        return M.train([X[i] for i in idx], [Y[i] for i in idx], champion, l2=lam)

    cv = {}
    for lam in ([l2] if l2 else grid):
        sc = []
        for f in range(folds):
            tr = [i for i in range(len(fb)) if G[i] != f]
            te = [i for i in range(len(fb)) if G[i] == f]
            a = M.auc(fit(tr, lam), [X[i] for i in te], [Y[i] for i in te])
            if a is not None:
                sc.append(a)
        cv[lam] = sum(sc) / len(sc) if sc else 0.0
    best = max(cv, key=cv.get)
    base = M.auc(champion, X, Y)
    note = (f"{len(fb)} labels ({len(M.pairs_of(Y, EP))} same-episode pairs -> {'pairwise' if pairwise else 'pointwise'} loss"
            f"{f'; {skipped} old-schema labels skipped' if skipped else ''}); cv auc by l2 "
            f"{{{', '.join(f'{k}:{v:.2f}' for k, v in cv.items())}}} -> l2={best}; champion auc on labels {base:.2f}")
    return fit(range(len(fb)), best), note


def gate(rows, ctx, gold, champion, challenger, split, n_eps, k=10, require_ci=True):
    """Promote only if, on the held-out VALIDATION gold (the frozen test split is never used here, F2):
    - every validity gate is clean,
    - the challenger adds no known-bad picks,
    - the bootstrap 10th percentile of the NDCG@10 gain is above zero (F3), not just the point estimate."""
    val = E.split_gold(gold, "val")
    res, promos = {}, {}
    for name, m in (("champion", champion), ("challenger", challenger)):
        promos[name] = rank(rows, m, k=60)
        res[name] = {"metrics": E.metrics(promos[name], val, split, n_eps), "validity": E.validity(promos[name], ctx)}
    c, n = res["champion"]["metrics"], res["challenger"]["metrics"]
    ci = E.bootstrap_diff(promos["champion"], promos["challenger"], val, split, n_eps)
    res["ndcg_gain_ci"] = ci  # (mean, p10, p90)
    clean = all(v == 0 for v in res["challenger"]["validity"].values())
    no_worse_bad = (n.get("bad_rate@10") or 0) <= (c.get("bad_rate@10") or 0)
    gain = (n.get("ndcg@10") or 0) > (c.get("ndcg@10") or 0)
    sure = ci is not None and ci[1] > 0
    res["promote"] = bool(clean and no_worse_bad and gain and (sure or not require_ci))
    if res["promote"]:
        res["reason"] = f"held-out NDCG@10 gain {ci} (mean, p10, p90) above zero, no extra bad picks, gates clean"
    elif not clean:
        res["reason"] = "validity gate failed"
    elif not no_worse_bad:
        res["reason"] = "challenger adds known-bad picks"
    elif not gain:
        res["reason"] = "challenger not better on held-out validation gold"
    else:
        res["reason"] = f"gain not distinguishable from noise: bootstrap NDCG@10 gain {ci} (mean, p10, p90) includes zero"
    return res
