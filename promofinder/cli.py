import argparse
import csv
import json
import os

from . import evaluate as E
from . import learn as L
from . import model as M
from . import pipeline as P
from .features import build_ctx
from .judge import get_judge
from .parse import parse_dir


ALIASES = "gold/aliases.json"


def _ctx(scripts, aliases=ALIASES):
    return build_ctx(parse_dir(scripts), P.load_aliases(aliases))


def _split(a, n_eps):
    """First held-out episode. Default: last 40% of the show is held out (any show length)."""
    return a.split or int(n_eps * 0.6) + 1


def cmd_run(a):
    judge, model = get_judge(offline=a.offline), M.load()
    prefilters = [M.load("v0000")] + ([model] if model.version != "v0000" else [])
    rows, ctx, stats = P.featurize(a.scripts, judge, prefilters, top_n=a.top_n, aliases=P.load_aliases(a.aliases),
                                   allow_mixed_judge=a.allow_mixed_judge, stitch_n=a.stitch_n,
                                   reserve_endings=a.reserve_endings if a.reserve_endings == "all" else int(a.reserve_endings))
    P.save_rows(rows, os.path.join(a.out, "rows.jsonl"))
    promos = P.rank(rows, model, k=a.k, exclude_review=a.exclude_review)
    promos = P.pairwise_rerank(promos, rows, judge, top=a.pairwise_top)
    stats.update(getattr(judge, "stats", {}))
    P.write_outputs(promos, a.out, P.manifest(model, judge, a.scripts, {k: v for k, v in vars(a).items() if k != "f"}, stats), ctx["n_eps"])
    for p in promos[:10]:
        segs = " + ".join(f"Ep{s['episode']} {s['start']:.1f}-{s['end']:.1f}" for s in p["segments"])
        print(f"#{p['rank']:>2} {p['score']:.3f} {segs}  ({p['duration']}s)\n     hook: {p['hook'][:70]}\n     ends: {p['cliffhanger'][:70]}")
    print(f"\nwrote {a.out}/promos.json, promos.csv, posting_plan.md, rows.jsonl")


def cmd_eval(a):
    ctx, gold_all = _ctx(a.scripts), E.load_gold(a.gold)
    gold_all = [g for g in gold_all if g.get("show") in (None, ctx["show"])]
    if not gold_all:
        print(f"no gold labels for this show ({ctx['show']}): ranking metrics need labels. Writing to_label.csv only.")
        promos = P.rank(P.load_rows(os.path.join(a.out, "rows.jsonl")), M.load(), k=40)
        with open(os.path.join(a.out, "to_label.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "rank", "segments", "hook", "cliffhanger", "tier (must/good/bad)"])
            for p in promos:
                w.writerow([p["id"], p["rank"], "; ".join(f"Ep{s['episode']} {s['timecode']}" for s in p["segments"]), p["hook"], p["cliffhanger"], ""])
        with open(os.path.join(a.out, "eval.json"), "w", encoding="utf-8") as f:
            json.dump({"no_gold": True, "show": ctx["show"], "to_label": len(promos)}, f)
        return
    rows = P.load_rows(os.path.join(a.out, "rows.jsonl"))
    n = ctx["n_eps"]
    SPLIT = _split(a, n)
    gold = E.split_gold(gold_all, "val")  # the frozen test split is only scored with --final (F2)
    versions = ["v0000", a.model or M.load().version]
    judge_name = (rows[0].get("judge") or {}).get("by")
    out = {"gold_val_n": len(gold), "split": SPLIT, "judge": judge_name, "systems": {}}
    systems = {}
    real = M.real_judge(rows[0].get("judge"))
    if real:  # with the offline judge the j_* features are neutral, so this row would duplicate the next one
        systems["code-only (prior weights, judge masked)"] = P.rank(rows, M.load("v0000"), k=60, mask_judge=True)
    for v in dict.fromkeys(versions):
        systems[f"weights {v}" + (" + LLM judge" if real else "")] = P.rank(rows, M.load(v), k=60)
    systems["baseline: last 40s of each episode"] = [dict(p, rank=i + 1) for i, p in enumerate(E.baseline_ep_end(ctx))]
    systems["baseline: random valid windows"] = [dict(p, rank=i + 1) for i, p in enumerate(E.baseline_random(ctx))]
    trained_on_train_block = {v for v in versions if v != "v0000"}
    for name, promos in systems.items():
        r = {"train_block": E.metrics(promos, gold, 1, SPLIT - 1), "heldout_block": E.metrics(promos, gold, SPLIT, n),
             "heldout_recall@20_by_position": E.recall_by_position(promos, gold, SPLIT, n, k=20),
             "heldout_top10_buckets": E.buckets(promos, gold, SPLIT, n)}
        if any(name.startswith(f"weights {v}") for v in trained_on_train_block):
            r["train_block"]["note"] = "in-sample: this model was trained on train-block feedback (F8)"
        if a.final:
            r["test_block"] = E.metrics(promos, E.split_gold(gold_all, "test"), SPLIT, n)
        if not name.startswith("baseline"):
            r["validity"] = E.validity(promos, ctx)
            r["rubric_checks@10"] = E.rubric_checks(promos, SPLIT, n)
        out["systems"][name] = r
    cur = P.rank(rows, M.load(versions[-1]), k=60)
    out["failures_heldout"] = E.failure_report(cur, gold, SPLIT, n)
    out["missed_musts_heldout"] = E.missed_musts(cur, gold, rows, SPLIT, n)
    unl = E.unlabelled(cur, gold, SPLIT, n)
    with open(os.path.join(a.out, "to_label.csv"), "w", newline="", encoding="utf-8") as f:  # F4: active labelling
        w = csv.writer(f)
        w.writerow(["id", "rank", "segments", "hook", "cliffhanger", "tier (must/good/bad)"])
        for p in unl:
            w.writerow([p["id"], p["rank"], "; ".join(f"Ep{s['episode']} {s['timecode']}" for s in p["segments"]), p["hook"], p["cliffhanger"], ""])
    mpath = os.path.join(a.out, "promos.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            out["cost"] = json.load(f)["manifest"].get("judge_stats", {})  # F5: cost/latency per show
    if a.final:
        out["test_usage"] = _bump_test_usage()
    with open(os.path.join(a.out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    keys = ["precision@5", "precision@10", "bad_rate@10", "must_recall@10", "ndcg@10", "distinct_eps@10"]
    blocks = ["train_block", "heldout_block"] + (["test_block"] if a.final else [])
    print(f"gold: validation split ({len(gold)} items){' + frozen test split' if a.final else ''}; held-out = episodes {SPLIT}-{n}; judge = {judge_name}")
    print(f"{'system':<42}{'block':<9}" + "".join(f"{k:>16}" for k in keys))
    for name, r in out["systems"].items():
        for blk in blocks:
            m = r[blk]
            tag = "*" if m.get("note") else ""
            print(f"{name:<42}{blk[:5] + tag:<9}" + "".join(f"{str(m.get(k)):>16}" for k in keys))
    print("  * in-sample (model trained on that block's feedback)")
    print("\nheld-out top-10 buckets (labelled good / labelled bad / unlabelled):")
    for name, r in out["systems"].items():
        print(f"  {name:<42}{r['heldout_top10_buckets']}")
    print("held-out recall@20 of good/must gold, by position in episode:")
    for name, r in out["systems"].items():
        print(f"  {name:<42}{r['heldout_recall@20_by_position']}")
    print("rubric checks on held-out top 10 (rule-based; independent only with an LLM judge):")
    for name, r in out["systems"].items():
        if "rubric_checks@10" in r:
            print(f"  {name:<42}{r['rubric_checks@10']}")
    print("\nvalidity violations:", {k: r["validity"] for k, r in out["systems"].items() if "validity" in r})
    print("cost:", out.get("cost"))
    print(f"(reports below are for {versions[-1]})")
    print("held-out failures (top-10 hitting gold 'bad'):", json.dumps(out["failures_heldout"], indent=1))
    print("held-out missed musts:", json.dumps(out["missed_musts_heldout"], indent=1))
    print(f"{len(unl)} unlabelled held-out top picks written to {a.out}/to_label.csv for review")
    if a.final:
        print(f"frozen test split has now been scored {out['test_usage']} time(s); the more it is used, the less it means")


def _bump_test_usage(path="runs/test_usage.json"):
    n = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            n = json.load(f).get("count", 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"count": n + 1}, f)
    return n + 1


def cmd_simulate(a):
    """DEMO ONLY. A synthetic editor reviews the model's own picks in the training block and accepts/rejects
    those that fall in a labelled gold region; picks outside any region stay unlabelled (a real editor would label
    them, the simulator cannot know). Gold regions are never injected as training rows (fix E1), and nothing is
    assumed about other picks in the same episode (fix E2). Same labeller as the gold set: see README.
    Writes to runs/feedback.sim.jsonl, never to the real feedback log (fix A2)."""
    gold = E.load_gold(a.gold)
    rows = P.load_rows(os.path.join(a.out, "rows.jsonl"))
    lo, hi = 1, _split(a, _ctx(a.scripts)["n_eps"]) - 1
    train_gold = [g for g in E.split_gold(gold, "val") if E.in_block(g, lo, hi)]
    recs = []
    for p in P.rank(rows, M.load(), k=400, per_ep_cap=6):
        if not E.in_block(p, lo, hi):
            continue
        g, _ = E.best_match(p, train_gold)
        if g:
            recs.append({"id": p["id"], "episode": p["segments"][0]["episode"], "x": p["features"],
                         "y": 1 if g["tier"] in ("must", "good") else 0, "editor": "simulated", "src": "sim-pick", "gold": g["id"], "schema": p.get("schema")})
    os.makedirs(os.path.dirname(L.SIM_FEEDBACK), exist_ok=True)
    open(L.SIM_FEEDBACK, "w").close()  # only the simulator's own file is reset
    n = L.log_feedback(recs, L.SIM_FEEDBACK)
    print(f"logged {n} simulated labels (positives {sum(r['y'] for r in recs)}) -> {L.SIM_FEEDBACK}")


def cmd_feedback(a):
    ctx = _ctx(a.scripts)
    recs, warnings = L.ingest_csv(a.csv, os.path.join(a.out, "promos.json"), ctx, get_judge(), a.editor)
    for w in warnings:
        print("warning:", w)
    n = L.log_feedback(recs)
    print(f"logged {n} new labels ({len(recs) - n} duplicates skipped) from {a.csv} -> {L.FEEDBACK}")


def cmd_retrain(a):
    ctx, gold = _ctx(a.scripts), E.load_gold(a.gold)
    rows = P.load_rows(os.path.join(a.out, "rows.jsonl"))
    champ = M.load()
    chall, note = L.retrain(champ, min_labels=a.min_labels, l2=a.l2, path=a.feedback)
    if chall is None:
        print("no retrain:", note)
        return
    res = L.gate(rows, ctx, gold, champ, chall, _split(a, ctx["n_eps"]), ctx["n_eps"])
    saved = M.save_new(chall, {"parent": champ.version, "note": note, "gate": res["reason"]})
    print(note, "->", saved.version)
    for k in ("champion", "challenger"):
        m = res[k]["metrics"]
        print(f"{k:<11} heldout ndcg@10={m['ndcg@10']} precision@10={m['precision@10']} bad_rate@10={m['bad_rate@10']} validity={res[k]['validity']}")
    if res["promote"] and not a.no_promote:
        M.promote(saved.version)
        print("PROMOTED", saved.version, "-", res["reason"])
    else:
        print("NOT promoted:", res["reason"])
    shifts = sorted(((chall.w[k] - champ.w[k], k) for k in M.FEATURES), key=lambda t: -abs(t[0]))[:5]
    print("largest weight shifts:", [(k, round(d, 2)) for d, k in shifts])


def cmd_promote(a):
    """Human override of the gate (e.g. after reviewing the failure report)."""
    M.promote(a.version)
    print("current model ->", a.version)


def cmd_timeline(a):
    ep = [e for e in parse_dir(a.scripts) if e.ep == a.ep][0]
    for e in ep.events:
        if e.kind in ("dialogue", "action", "sfx") and a.start <= e.t0 <= a.end:
            who = e.speaker if e.kind == "dialogue" else e.kind.upper()
            print(f"{e.t0:6.1f}-{e.t1:6.1f} {who[:14]:<14} {e.text[:95]}")
    print("episode end", ep.end)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="promofinder")
    ap.add_argument("--scripts", default="data/scripts")
    ap.add_argument("--out", default="runs/r1")
    ap.add_argument("--gold", default="gold/gold.jsonl")
    ap.add_argument("--split", type=int, default=0, help="first held-out episode (default: last 40%% of the show)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--aliases", default=ALIASES, help="per-show speaker-label -> character table (optional)")
    r = sub.add_parser("run"); r.add_argument("--k", type=int, default=25); r.add_argument("--top-n", type=int, default=200)
    r.add_argument("--pairwise-top", type=int, default=10, help="LLM judge only: re-rank this many by pairwise comparison")
    r.add_argument("--exclude-review", action="store_true", help="drop clips flagged for ad-policy review")
    r.add_argument("--stitch-n", type=int, default=30, help="stitched candidates sent to the judge")
    r.add_argument("--reserve-endings", default="all", help="episode-ending windows always shortlisted: a number, or all")
    r.add_argument("--offline", action="store_true", help="force the offline judge (no API calls)")
    r.add_argument("--allow-mixed-judge", action="store_true", help="accept a run where >20%% of rows fell back to the offline judge")
    r.set_defaults(f=cmd_run)
    e = sub.add_parser("eval"); e.add_argument("--model"); e.add_argument("--final", action="store_true", help="also score the frozen test split (counted)"); e.set_defaults(f=cmd_eval)
    s = sub.add_parser("simulate"); s.set_defaults(f=cmd_simulate)
    fb = sub.add_parser("feedback"); fb.add_argument("csv"); fb.add_argument("--editor", default="editor"); fb.set_defaults(f=cmd_feedback)
    t = sub.add_parser("retrain"); t.add_argument("--min-labels", type=int, default=30); t.add_argument("--l2", type=float, default=None); t.add_argument("--no-promote", action="store_true"); t.add_argument("--feedback", default=L.FEEDBACK, help=f"label file (simulated: {L.SIM_FEEDBACK})"); t.set_defaults(f=cmd_retrain)
    sv = sub.add_parser("serve", help="local demo UI"); sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--show", default=None, help="preload an existing run folder (default: start empty)")
    sv.set_defaults(f=lambda a: __import__("promofinder.server", fromlist=["serve"]).serve(a.show, a.port, a.scripts))
    pr = sub.add_parser("promote"); pr.add_argument("version"); pr.set_defaults(f=cmd_promote)
    tl = sub.add_parser("timeline"); tl.add_argument("ep", type=int); tl.add_argument("--start", type=float, default=0); tl.add_argument("--end", type=float, default=999); tl.set_defaults(f=cmd_timeline)
    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
