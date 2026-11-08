"""Stage 6: evaluation. Gold matching, hard validity gates, ranking metrics, baselines, failure report."""
import json
import math
import random

GAIN = {"must": 3, "good": 1, "bad": 0}


def load_gold(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _segs(p):
    if "segments" in p and p["segments"] and isinstance(p["segments"][0], dict):
        return [(s["episode"], s["start"], s["end"]) for s in p["segments"]]
    return [tuple(s) for s in p["segments"]]


def iou(a, b):
    inter = sum(max(0.0, min(s[2], t[2]) - max(s[1], t[1])) for s in a for t in b if s[0] == t[0])
    da, db = sum(s[2] - s[1] for s in a), sum(s[2] - s[1] for s in b)
    return inter / (da + db - inter) if da + db - inter > 0 else 0.0


def _contain(a, b):
    """Share of the shorter clip that lies inside the other."""
    inter = sum(max(0.0, min(s[2], t[2]) - max(s[1], t[1])) for s in a for t in b if s[0] == t[0])
    return inter / min(sum(s[2] - s[1] for s in a), sum(s[2] - s[1] for s in b))


def best_match(p, gold, thr=0.5, contain=0.8):
    """Gold items are regions an editor would cut from, so a promo matches if IoU >= thr
    or it sits (>= 80%) inside the gold region. Returns (gold item, IoU)."""
    best, bi = None, -1.0
    for g in gold:
        a, b = _segs(p), _segs(g)
        i = iou(a, b)
        if (i >= thr or _contain(a, b) >= contain) and i > bi:
            best, bi = g, i
    return best, max(bi, 0.0)


def in_block(p, lo, hi):
    return all(lo <= s[0] <= hi for s in _segs(p))


# ---- hard gates ----
def validity(promos, ctx):
    """Returns {gate: n_violations}. Any non-zero count blocks promotion."""
    v = {"duration": 0, "boundary": 0, "order": 0, "overlap": 0, "safety": 0}
    for i, p in enumerate(promos):
        segs = _segs(p)
        if not 20 <= sum(s[2] - s[1] for s in segs) <= 60.0001:
            v["duration"] += 1
        for ep, a, b in segs:
            E = ctx["eps"][ep]
            beats = [e for e in E.events if e.kind != "camera"]
            ok_a = any(abs(e.t0 - a) < 0.011 and e.kind in ("dialogue", "action") for e in beats)
            ok_b = any(abs(e.t1 - b) < 0.011 for e in beats) or abs(E.end - b) < 0.011
            if not (ok_a and ok_b):
                v["boundary"] += 1
        if any(a[0] == b[0] and a[1] >= b[1] for a, b in zip(segs, segs[1:])):
            v["order"] += 1
        for q in promos[:i]:
            if any(s[0] == t[0] and s[1] < t[2] and t[1] < s[2] for s in segs for t in _segs(q)):
                v["overlap"] += 1
        if p.get("safety"):
            v["safety"] += 1
    return v


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def eps_of(p):
    return {s[0] for s in _segs(p)}


def metrics(promos, gold, lo, hi, ks=(5, 10), episodes=None):
    """promos: ranked list; gold: all gold. Only the episode block [lo,hi] (optionally a subset of its
    episodes, used by the bootstrap) is scored."""
    keep = lambda x: in_block(x, lo, hi) and (episodes is None or eps_of(x) <= episodes)
    P = [p for p in promos if keep(p)]
    G = [g for g in gold if keep(g)]
    must = [g for g in G if g["tier"] == "must"]
    pos = [g for g in G if g["tier"] in ("must", "good")]
    out = {"n_gold": len(G), "n_must": len(must), "n_returned": len(P)}
    for k in ks:
        top = P[:k]
        ms = [best_match(p, G)[0] for p in top]
        hit = sum(1 for m in ms if m and m["tier"] in ("must", "good"))
        bad = sum(1 for m in ms if m and m["tier"] == "bad")
        seen = {id(m) for m in ms if m and m["tier"] == "must"}
        gains = [GAIN[m["tier"]] if m else 0 for m in ms]
        ideal = sorted((GAIN[g["tier"]] for g in G), reverse=True)[:k]
        out[f"precision@{k}"] = round(hit / k, 3)
        out[f"bad_rate@{k}"] = round(bad / k, 3)
        out[f"must_recall@{k}"] = round(len(seen) / len(must), 3) if must else None
        out[f"ndcg@{k}"] = round(_dcg(gains) / _dcg(ideal), 3) if ideal and _dcg(ideal) else None
        out[f"distinct_eps@{k}"] = len({s[0] for p in top for s in _segs(p)})
    ious = [best_match(p, pos)[1] for p in P[:10] if best_match(p, pos)[0]]
    out["mean_iou_matched@10"] = round(sum(ious) / len(ious), 3) if ious else None
    return out


# ---- baselines: any learned system must beat these ----
def baseline_ep_end(ctx, seconds=40):
    out = []
    for ep in sorted(ctx["eps"]):
        E = ctx["eps"][ep]
        starts = [e.t0 for e in E.events if e.kind in ("dialogue", "action") and E.end - e.t0 <= seconds + 0.01 and E.end - e.t0 >= 20]
        if starts:
            out.append({"segments": [[ep, min(starts), E.end]]})
    return out


def baseline_random(rows, seed=0, n=200):
    rnd = random.Random(seed)
    pool = [r for r in rows if not r["safety"]]
    return [{"segments": r["segments"]} for r in rnd.sample(pool, min(n, len(pool)))]


def bootstrap_diff(promos_a, promos_b, gold, lo, hi, metric="ndcg@10", n=500, seed=0):
    """F3: resample held-out episodes with replacement; return (mean, p10, p90) of metric(b) - metric(a).
    With ~20 held-out episodes and a few gold items, intervals are wide: that is the point."""
    rnd = random.Random(seed)
    eps = sorted({e for g in gold if in_block(g, lo, hi) for e in eps_of(g)})
    diffs = []
    for _ in range(n):
        sample = {rnd.choice(eps) for _ in eps}
        a = metrics(promos_a, gold, lo, hi, episodes=sample).get(metric)
        b = metrics(promos_b, gold, lo, hi, episodes=sample).get(metric)
        if a is not None and b is not None:
            diffs.append(b - a)
    if not diffs:
        return None
    diffs.sort()
    q = lambda f: diffs[min(len(diffs) - 1, int(f * len(diffs)))]
    return round(sum(diffs) / len(diffs), 3), round(q(0.10), 3), round(q(0.90), 3)


def buckets(promos, gold, lo, hi, k=10):
    """F4: an unlabelled pick is unknown, not wrong. Report the three buckets instead of folding unknown into 'miss'."""
    top = [p for p in promos if in_block(p, lo, hi)][:k]
    good = bad = unl = 0
    for p in top:
        g, _ = best_match(p, [x for x in gold if in_block(x, lo, hi)])
        if g is None:
            unl += 1
        elif g["tier"] == "bad":
            bad += 1
        else:
            good += 1
    return f"{good} / {bad} / {unl}"


def unlabelled(promos, gold, lo, hi, k=20):
    """Top picks no gold item covers: the queue an editor should label next (active labelling, F4)."""
    G = [x for x in gold if in_block(x, lo, hi)]
    return [p for p in [p for p in promos if in_block(p, lo, hi)][:k] if best_match(p, G)[0] is None]


def rubric_checks(promos, lo, hi, k=10, thr=0.3):
    """F5: the checks the design promised. Rule-based unless an LLM judge produced the rows."""
    top = [p for p in promos if in_block(p, lo, hi)][:k]
    if not top:
        return {}
    share = lambda f: round(sum(1 for p in top if f(p)) / len(top), 2)
    return {"hook_in_3s": share(lambda p: p["features"].get("_hook_part", 0) >= thr),
            "ends_unresolved": share(lambda p: p["features"].get("_payoff_part", 0) >= thr),
            "standalone>=0.6": share(lambda p: p["features"].get("standalone", 0) >= 0.6),
            "policy_review": share(lambda p: p.get("policy", {}).get("label") == "review"),
            "spoiler_high": share(lambda p: p.get("spoiler") == "high"),
            "hook_types": len({p.get("hook_type") for p in top})}


def split_gold(gold, which):
    """F2: gold items tagged split='test' are a frozen test set the gate never sees; 'val' is everything else."""
    return [g for g in gold if (g.get("split") == "test") == (which == "test")]


def position(g):
    return g.get("position") or {"t": "tail", "h": "head", "r": "mid"}.get(g["id"][-1], "mid")


def recall_by_position(promos, gold, lo, hi, k=20):
    """Recall of good/must gold split by where the region sits in its episode. The original gold was 41/44
    episode tails, so a single recall number mostly measured 'can it find episode endings' (fix F1)."""
    P = [p for p in promos if in_block(p, lo, hi)][:k]
    out = {}
    for pos in ("tail", "head", "mid"):
        G = [g for g in gold if in_block(g, lo, hi) and g["tier"] in ("must", "good") and position(g) == pos]
        if G:
            hit = {best_match(p, G)[0]["id"] for p in P if best_match(p, G)[0]}
            out[pos] = f"{len(hit)}/{len(G)}"
    return out


def missed_musts(promos, gold, rows, lo, hi, k=10):
    """Must-have regions not in the top k, and whether they were ever in the judged pool (F9)."""
    P = [p for p in promos if in_block(p, lo, hi)][:k]
    out = []
    for g in [g for g in gold if in_block(g, lo, hi) and g["tier"] == "must"]:
        if any(best_match(p, [g])[0] for p in P):
            continue
        in_pool = any(best_match({"segments": r["segments"]}, [g])[0] for r in rows)
        out.append({"gold": g["id"], "note": g.get("note", ""), "why_missed": "ranked too low" if in_pool else "never reached the judged pool"})
    return out


def failure_report(promos, gold, lo, hi, k=10):
    G = [g for g in gold if in_block(g, lo, hi)]
    rows = []
    for p in [p for p in promos if in_block(p, lo, hi)][:k]:
        m, i = best_match(p, G)
        if m and m["tier"] == "bad":
            rows.append({"rank": p["rank"], "id": p["id"], "gold_note": m.get("note", ""), "hook": p["hook"][:80], "why_ranked": p["why"][:120]})
    return rows
