"""Stages 2-5 wired together. featurize() is the expensive part (judge calls) and its rows are cached;
rank() is cheap and is what retraining and evaluation call again and again."""
import csv
import hashlib
import json
import math
import os
import time

from . import candidates as C
from .features import SCHEMA, build_ctx, extract
from .judge import RUBRIC
from .model import MIN_PART, Model, real_judge, vec
from .parse import parse_dir

HANDLE = 0.4  # seconds of pre/post-roll suggested around each cut (fix C3)


def load_aliases(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def auto_aliases(episodes, path="gold/aliases.json"):
    """Use the per-show alias table only if this looks like that show (3+ of its labels appear)."""
    table = load_aliases(path)
    labels = {ev.speaker for e in episodes for ev in e.events if ev.kind == "dialogue"}
    hits = sum(1 for k in table if not k.startswith("_") and k.split("@")[0] in labels)
    return table if hits >= 3 else {}


def _row(ctx, segs):
    f, safety, view = extract(ctx, segs)
    return {"id": C.cid(segs), "segments": [list(s) for s in segs], "feats": f, "safety": safety,
            "view": view, "judge": None, "schema": SCHEMA}


def featurize(scripts_dir, judge, prefilters, top_n=200, per_ep=4, aliases=None, allow_mixed_judge=False, stitch_n=30, log=print, progress=None):
    """prefilters: list of Models. The judged pool is the UNION of each model's picks, so a challenger is not
    limited to what the champion liked (fix C5)."""
    t0 = time.time()
    episodes = parse_dir(scripts_dir)
    if not episodes:
        raise ValueError(f"no episode-NNN*.txt scripts in {scripts_dir}")
    if aliases == "auto":
        aliases = auto_aliases(episodes)
    ctx = build_ctx(episodes, aliases)
    log(f"parsed {len(episodes)} episodes" + (f" ({sum(e.dropped for e in episodes)} untimed blocks skipped)" if any(e.dropped for e in episodes) else ""))

    rows = [_row(ctx, s) for s in C.singles(episodes)]
    log(f"{len(rows)} contiguous candidates")

    # stitched promos: hook-heavy head + cliffhanger tail, same episode, best 2 of each PER EPISODE (C4)
    segs = [r for r in (_row(ctx, [s]) for s in C.short_segments(episodes)) if not r["safety"]]
    head_s = lambda r: r["feats"]["hook_arousal"] + r["feats"]["hook_dialogue"] + r["feats"]["clean_entry"]
    tail_s = lambda r: r["feats"]["cliff_content"] + r["feats"]["cliff_open_end"] + 0.3 * r["feats"]["cliff_proximity"]
    heads, tails = [], []
    for e in {r["segments"][0][0] for r in segs}:
        mine = [r for r in segs if r["segments"][0][0] == e]
        heads += [tuple(r["segments"][0]) for r in sorted(mine, key=head_s, reverse=True)[:2]]
        tails += [tuple(r["segments"][0]) for r in sorted(mine, key=tail_s, reverse=True)[:2]]
    st = [_row(ctx, s) for s in C.stitches(heads, tails)]
    # head and tail must share a character, or the "promo" splices two unrelated scenes (C4)
    st = [r for r in st if set(r["view"]["speakers"][0]) & set(r["view"]["speakers"][-1])]
    log(f"{len(st)} stitched candidates (shared character)")

    pool = {}
    for m in prefilters:
        def neutral(r, m=m):
            x = vec(r["feats"], None)
            if x["_hook_part"] < MIN_PART or x["_payoff_part"] < MIN_PART:
                return -1.0
            return m.score(x)
        kept, count = [], {}
        for r in sorted((r for r in rows if not r["safety"]), key=neutral, reverse=True):
            e = r["segments"][0][0]
            # per-episode quota of DISTINCT regions: sliding windows overlap heavily
            if count.get(e, 0) < per_ep and not any(overlaps(r["segments"], k["segments"]) for k in kept if k["segments"][0][0] == e):
                count[e] = count.get(e, 0) + 1
                kept.append(r)
        for r in kept[:top_n] + sorted((r for r in st if not r["safety"]), key=neutral, reverse=True)[:stitch_n]:
            pool[r["id"]] = r
    kept = list(pool.values())
    log(f"prefilter ({', '.join(m.version for m in prefilters)}) -> {len(kept)} candidates for the judge")

    for i, r in enumerate(kept):
        r["judge"] = judge(r["view"])
        if progress:
            progress(i + 1, len(kept), r["judge"].get("by", ""))
    stats = dict(getattr(judge, "stats", {"calls": len(kept)}))
    stats.update(judge=judge.name, seconds=round(time.time() - t0, 1), pool=len(kept))
    stats.update(episodes=len(episodes), untimed_blocks_dropped=sum(e.dropped for e in episodes),
                 tag_coverage=ctx.get("tag_coverage"), aliases_used=bool(aliases))
    fallback = sum(1 for r in kept if str(r["judge"].get("by", "")).startswith("offline-fallback"))
    stats["fallback_rows"] = fallback
    if fallback and not judge.name.startswith("offline") and fallback / len(kept) > 0.2 and not allow_mixed_judge:
        over = getattr(judge, "stats", {}).get("over_budget", 0)
        why = (f"{over} clips were not in the cache and today's free requests ran out" if over else "the API kept failing")
        raise RuntimeError(f"{fallback}/{len(kept)} clips could not be judged by the LLM ({why}); refusing to mix "
                           "judges (fix D2). Run offline, or retry after the daily reset.")
    log(f"judged by {judge.name} in {stats['seconds']}s; fallbacks {fallback}")
    return kept, ctx, stats


def overlaps(a, b):
    return any(s[0] == t[0] and s[1] < t[2] and t[1] < s[2] for s in a for t in b)


def _beat(r):
    """What story beat a clip is about: its characters and episode (for diversity, C6)."""
    chars = {c for seg in r["view"].get("speakers", []) for c in seg}
    return chars, r["segments"][0][0]


def _similar(a, b):
    """1.0 = same characters in the same or a neighbouring episode (likely the same beat, e.g. a recap)."""
    (ca, ea), (cb, eb) = a, b
    if not ca or not cb or abs(ea - eb) > 1:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def rank(rows, model: Model, k=25, per_ep_cap=2, mask_judge=False, min_part=MIN_PART, exclude_review=False,
         exclude_spoilers=False, diversity=1.0):
    """Score, drop blocked/unsafe, drop clips missing a hook or a payoff, then pick greedily with maximal marginal
    relevance: logit minus `diversity` x similarity to clips already picked (C6), no overlaps, per-episode cap
    (a stitch counts against its first episode only, I-6). Policy 'review' / spoiler 'high' are flagged, not dropped."""
    scored = []
    for r in rows:
        v = r["view"]
        pol = v.get("policy", {}).get("label", "block" if r["safety"] else "none")
        if r["safety"] or pol == "block" or (exclude_review and pol == "review") or (exclude_spoilers and v.get("spoiler") == "high"):
            continue
        x = vec(r["feats"], None if mask_judge else r["judge"])
        if x["_hook_part"] < min_part or x["_payoff_part"] < min_part:
            continue
        scored.append([model.logit(x), x, r, _beat(r)])
    out, per, chosen, beats = [], {}, [], []
    while scored and len(out) < k:
        best, best_v = None, None
        for item in scored:
            lg, x, r, bt = item
            if per.get(r["segments"][0][0], 0) >= per_ep_cap or any(overlaps(r["segments"], c) for c in chosen):
                continue
            v = lg - diversity * max((_similar(bt, o) for o in beats), default=0.0)
            if best_v is None or v > best_v:
                best, best_v = item, v
        if best is None:
            break
        scored.remove(best)
        lg, x, r, bt = best
        chosen.append(r["segments"])
        beats.append(bt)
        per[r["segments"][0][0]] = per.get(r["segments"][0][0], 0) + 1
        j = r["judge"] or {}
        v = r["view"]
        why = j.get("reason", "") if real_judge(j) else _why(x, model)
        ends = v.get("ep_ends", [None] * len(r["segments"]))
        out.append({"rank": len(out) + 1, "id": r["id"], "score": round(1 / (1 + math.exp(-lg)), 4), "logit": round(lg, 3),
                    "segments": [{"episode": g[0], "start": g[1], "end": g[2],
                                  "edit_in": round(max(0.0, g[1] - HANDLE), 2),
                                  "edit_out": round(min(e or g[2] + HANDLE, g[2] + HANDLE), 2),
                                  "timecode": f"{_tc(g[1])}-{_tc(g[2])}"} for g, e in zip(r["segments"], ends)],
                    "duration": round(v["dur"], 2), "hook": v["hook_line"], "cliffhanger": v["cliff_line"],
                    "why": why, "hook_type": v.get("hook_type"), "policy": v.get("policy", {}),
                    "spoiler": v.get("spoiler"), "judge": j.get("by"), "features": x, "schema": r.get("schema")})
    # H3: scores are uncalibrated (all near 0.99), so editors get a band by rank instead of a fake probability
    for p in out:
        p["band"] = "A" if p["rank"] <= max(1, round(0.2 * len(out))) else ("B" if p["rank"] <= round(0.6 * len(out)) else "C")
    return out


def pairwise_rerank(promos, rows, judge, top=10, log=print, progress=None):
    """D3: absolute 0-10 scores drift; re-order the top N by pairwise judgements run in BOTH orders
    (cancels position bias). Only with a judge that implements compare(); N*(N-1) calls."""
    if not hasattr(judge, "compare") or len(promos) < 2:
        return promos
    by_id = {r["id"]: r for r in rows}
    head = promos[:top]
    wins = {p["id"]: 0.0 for p in head}
    for i, a in enumerate(head):
        for b in head[i + 1:]:
            va, vb = by_id[a["id"]]["view"], by_id[b["id"]]["view"]
            p_ab = judge.compare(va, vb)          # P(a better) with a shown first
            p_ba = 1 - judge.compare(vb, va)      # same question, order swapped
            if progress:
                progress(a["rank"], b["rank"])
            p = (p_ab + p_ba) / 2
            wins[a["id"]] += p
            wins[b["id"]] += 1 - p
    head = sorted(head, key=lambda p: -wins[p["id"]])
    for p in head:
        p["pairwise_wins"] = round(wins[p["id"]], 2)
    out = head + promos[top:]
    for i, p in enumerate(out):
        p["rank"] = i + 1
    log(f"pairwise re-rank of top {len(head)} done")
    return out


def _tc(t):
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _why(x, model):
    """Explanation built only from signals that actually fired for this clip (no template claims, fix H1)."""
    parts = []
    if x.get("hook_arousal", 0) >= 0.5:
        parts.append("opens on a shock/conflict beat")
    elif x.get("hook_arousal", 0) > 0:
        parts.append("opens on a mild hook")
    if x.get("cliff_content", 0) >= 0.6:
        parts.append("ends on an open threat or demand")
    elif x.get("cliff_open_end", 0) >= 0.7:
        parts.append("ends on an unanswered line")
    if x.get("standalone", 0) >= 0.8:
        parts.append("few characters to follow")
    top = ", ".join(model.contributions(x))
    return "; ".join(parts) + (f" (top signals: {top})" if top else "") if parts else f"top signals: {top}"


def posting_plan(promos, n_eps, per_wave=5):
    """H2: the brief asks which promos to post first and why. Industry practice: lead cold audiences with the premise,
    escalate, rotate hook types (creatives fatigue in days), keep spoilers and policy-review clips back."""
    held = [p for p in promos if p.get("spoiler") == "high" or p.get("policy", {}).get("label") == "review"]
    ok = [p for p in promos if p not in held]
    premise = sorted([p for p in ok if p["segments"][0]["episode"] <= max(3, round(0.1 * n_eps))
                      and p["features"].get("standalone", 0) >= 0.6], key=lambda p: p["rank"])[:per_wave]
    rest = [p for p in ok if p not in premise]
    # round-robin across hook types so one ad set never runs five of the same opener
    by_type = {}
    for p in rest:
        by_type.setdefault(p.get("hook_type") or "other", []).append(p)
    mixed = []
    while any(by_type.values()):
        for t in list(by_type):
            if by_type[t]:
                mixed.append(by_type[t].pop(0))
    waves = [("Wave 1 - premise, for cold audiences", premise, "early-episode clips a new viewer can follow; they sell the show's setup"),
             ("Wave 2 - escalation, mixed hook types", mixed[:per_wave], "highest-ranked remaining clips, alternating hook types to slow fatigue"),
             ("Wave 3 - refresh pool", mixed[per_wave:per_wave * 3], "swap in when a wave's CTR drops >15% or CPC rises >20%"),
             ("Held back - needs a human decision", held, "spoiler risk (late reveal) or graphic violence flagged for ad-policy review")]
    return waves


def write_outputs(promos, out_dir, manifest, n_eps):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "promos.json"), "w", encoding="utf-8") as f:
        json.dump({"manifest": manifest, "promos": promos}, f, indent=1)
    with open(os.path.join(out_dir, "promos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "rank", "episodes", "segments", "duration", "hook_type", "policy", "spoiler", "hook", "cliffhanger", "why", "decision", "trim_start", "trim_end"])
        for p in promos:
            segs = "; ".join(f"Ep{s['episode']} {s['timecode']}" for s in p["segments"])
            w.writerow([p["id"], p["rank"], ",".join(str(s["episode"]) for s in p["segments"]), segs, p["duration"],
                        p["hook_type"], p["policy"].get("label"), p["spoiler"], p["hook"], p["cliffhanger"], p["why"], "", "", ""])
    lines = ["# Posting plan", "", f"Model {manifest['model']}, judge {manifest['judge']}. Order within a wave = rank.", ""]
    for title, items, why in posting_plan(promos, n_eps):
        lines += [f"## {title}", "", f"_{why}_", ""]
        for p in items:
            segs = " + ".join(f"Ep{s['episode']} {s['timecode']}" for s in p["segments"])
            flag = "; ".join(p["policy"].get("reasons", []) + ([f"spoiler {p['spoiler']}"] if p["spoiler"] == "high" else []))
            lines.append(f"- #{p['rank']} {segs} ({p['duration']}s, {p['hook_type']}) - opens \"{p['hook'].strip('"')[:60]}\", ends \"{p['cliffhanger'].strip('"')[:60]}\"" + (f" **[{flag}]**" if flag else ""))
        lines.append("")
    with open(os.path.join(out_dir, "posting_plan.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def manifest(model, judge, scripts_dir, args, stats=None):
    return {"run_id": time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}", "model": model.version, "judge": judge.name, "args": args, "scripts_dir": scripts_dir,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "feature_schema": SCHEMA,
            "rubric_hash": hashlib.sha256(RUBRIC.encode()).hexdigest()[:12], "judge_stats": stats or {}}


def save_rows(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    bad = {r.get("schema") for r in rows} - {SCHEMA}
    if bad:
        raise RuntimeError(f"{path} was built with feature schema {bad}, code is {SCHEMA}: re-run `run` (fix A4)")
    return rows
