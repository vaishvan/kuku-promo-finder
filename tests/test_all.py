"""Run: python tests/test_all.py   (pytest also works). Needs data/scripts."""
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, ".")
from promofinder import candidates as C, evaluate as E, judge as J, learn as L, model as M, pipeline as P  # noqa: E402
from promofinder.features import build_ctx  # noqa: E402
from promofinder.parse import parse_dir  # noqa: E402

EPS = parse_dir("data/scripts")
CTX = build_ctx(EPS, P.load_aliases("gold/aliases.json"))
def _rows(path="runs/r1/rows.jsonl"):
    """I-3: tests build their own rows (offline judge, no network) when the cached run is missing or stale."""
    try:
        return P.load_rows(path)
    except (FileNotFoundError, RuntimeError):
        rows, _, _ = P.featurize("data/scripts", J.OfflineJudge(), [M.Model()], aliases="auto", log=lambda *_: None)
        P.save_rows(rows, path)
        return rows


ROWS = _rows()


def test_parse():
    assert len(EPS) == 50
    assert EPS[0].end == 138.0
    assert all(e.t1 >= e.t0 for ep in EPS for e in ep.events)


def test_candidates_on_boundaries_and_length():
    for w in C.singles(EPS[:5]):
        assert 20 <= w[0][2] - w[0][1] <= 60
    promos = [{"segments": s} for s in C.singles(EPS[:3])[:50]]
    v = E.validity(promos, CTX)  # sliding windows overlap each other by design; boundary/duration must be clean
    assert v["boundary"] == 0 and v["duration"] == 0


def test_validity_gate_catches_violations():
    bad = [{"segments": [[1, 11.31, 40.0]]}, {"segments": [[1, 0.0, 10.0]]}]  # off-boundary, too short
    v = E.validity(bad, CTX)
    assert v["boundary"] >= 1 and v["duration"] >= 1
    assert E.validity([{"segments": [[1, 0.0, 30.0]], "safety": True}], CTX)["safety"] == 1


def test_rank_constraints():
    out = P.rank(ROWS, M.Model(), k=30, per_ep_cap=2)
    assert len(out) >= 10
    per = {}
    for p in out:
        for e in {s["episode"] for s in p["segments"]}:  # a same-episode stitch is one promo for that episode
            per[e] = per.get(e, 0) + 1
    assert max(per.values()) <= 2
    assert sum(E.validity(out, CTX).values()) == 0
    unsafe = {r["id"] for r in ROWS if r["safety"]}
    assert not unsafe & {p["id"] for p in out}


def test_train_moves_toward_signal_and_respects_frozen():
    prior = M.Model()
    X = [{"j_hook": 1.0, "ep_pos": 0.1}] * 20 + [{"j_hook": 0.0, "ep_pos": 0.9}] * 20
    Y = [1] * 20 + [0] * 20
    m = M.train(X, Y, prior, l2=1.0)
    assert m.w["j_hook"] > prior.w["j_hook"]
    assert m.w["ep_pos"] == prior.w["ep_pos"]  # frozen
    strong = M.train(X, Y, prior, l2=1e6)
    assert abs(strong.w["j_hook"] - prior.w["j_hook"]) < 0.05  # heavy l2 = stays at prior


def test_registry_roundtrip():
    old = M.MODELS
    with tempfile.TemporaryDirectory() as d:
        M.MODELS = d
        base = M.load()
        assert base.version == "v0000"
        new = M.save_new(M.Model(bias=-1.0), {"note": "t"})
        assert new.version == "v0001" and M.load().version == "v0000"
        M.promote("v0001")
        assert M.load().version == "v0001" and M.load().bias == -1.0
    M.MODELS = old


def test_llm_judge_parse_cache_and_fallback():
    os.environ.update(JUDGE_BASE_URL="http://x", JUDGE_API_KEY="k", JUDGE_MODEL="m")
    with tempfile.TemporaryDirectory() as d:
        j = J.LLMJudge(os.path.join(d, "c.jsonl"))
        calls = []
        j._post = lambda sys_, t: calls.append(t) or 'sure {not json} then: {"hook": 9, "cliffhanger": 7, "standalone": 4, "emotion": 8, "reason": "slap {x}"}'
        view = {"lines": ["1@01.0 A: hi"], "hook_line": "hi", "cliff_line": "hi", "dur": 25}
        a, _ = j(view), j(view)
        assert a["hook"] == 9 and a["by"] == "llm:m" and len(calls) == 1  # second call cached
        j2 = J.LLMJudge(os.path.join(d, "c.jsonl"))
        assert j2(view)["hook"] == 9  # cache persists across runs
        j._post = lambda sys_, t: (_ for _ in ()).throw(RuntimeError("down"))
        f = j({"lines": ["x"], "hook_line": "x", "cliff_line": "x", "dur": 25})
        assert f["by"].startswith("offline-fallback")


def test_csv_feedback_ingest_keeps_trims_and_dedupes():
    p = [q for q in P.rank(ROWS, M.Model(), k=10) if len(q["segments"]) == 1][:4]
    seg = p[2]["segments"][0]
    with tempfile.TemporaryDirectory() as d:
        pj = os.path.join(d, "promos.json")
        with open(pj, "w") as f:
            json.dump({"promos": p}, f)
        cp = os.path.join(d, "p.csv")
        with open(cp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "decision", "trim_start", "trim_end"])
            w.writerow([p[0]["id"], "accept", "", ""])
            w.writerow([p[1]["id"], "reject", "", ""])
            w.writerow([p[2]["id"], "accept", seg["start"] + 0.4, seg["start"] + 22])  # trim, snaps to events
            w.writerow([p[3]["id"], "", "", ""])
        recs, warnings = L.ingest_csv(cp, pj, CTX, J.OfflineJudge())
        srcs = [(r["src"], r["y"]) for r in recs]
        assert ("csv", 1) in srcs and ("csv", 0) in srcs
        trim = [r for r in recs if r["src"] == "csv-trim"]
        assert len(trim) == 1 and trim[0]["y"] == 1 and "x" in trim[0], (srcs, warnings)  # fix A1: trims are kept
        assert ("csv-trimmed-original", 0) in srcs
        log = os.path.join(d, "fb.jsonl")
        assert L.log_feedback(recs, log) == len(recs)
        assert L.log_feedback(recs, log) == 0  # re-ingesting the same CSV adds nothing


def test_simulate_never_touches_real_feedback():
    real = L.FEEDBACK
    backup = open(real, encoding="utf-8").read() if os.path.exists(real) else None
    try:
        with open(real, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "sentinel", "editor": "human", "x": {}, "y": 1}) + "\n")
        from promofinder.cli import main
        main(["simulate"])
        assert any(r["id"] == "sentinel" for r in L.load_feedback(real))  # fix A2
        assert os.path.exists(L.SIM_FEEDBACK)
        assert all(r["src"] == "sim-pick" for r in L.load_feedback(L.SIM_FEEDBACK))  # fix E1: no gold rows injected
    finally:
        if backup is None:
            os.remove(real)
        else:
            with open(real, "w", encoding="utf-8") as f:
                f.write(backup)


def test_lexicons_match_whole_words_only():
    from promofinder.features import CONFLICT, HOOK_ACTION
    for w in ["staring", "roaring", "Harrington", "towering", "environment", "deadline"]:
        assert not HOOK_ACTION.search(w) and not CONFLICT.search(w), w  # fix B2
    for w in ["slapped", "struck", "gallows", "betrayed"]:
        assert HOOK_ACTION.search(w) or CONFLICT.search(w), w
    assert not CONFLICT.search("my husband the duke married me")  # fix B3: genre words carry no conflict


def test_music_cue_is_not_a_cliffhanger():
    from promofinder.features import CODE_FEATURES
    assert "cliff_marker" not in CODE_FEATURES  # fix B1: music/SFX cue feature removed
    assert "cliff_content" in CODE_FEATURES  # unresolvedness is read from story content


def test_shortlist_reserves_episode_endings():
    ends = {r["segments"][0][0] for r in ROWS if len(r["segments"]) == 1
            and abs(r["segments"][-1][2] - r["view"]["ep_ends"][-1]) < 0.01}
    # every episode ending that passes the hook-and-payoff minimum reaches the judged pool (39/50 on this show)
    assert len(ends) >= 35, len(ends)


def test_no_candidate_starts_or_ends_mid_sentence():
    cont = lambda t: t.strip().strip('"').startswith(("...", "…"))
    for (ep, a, b), in C.singles(EPS):
        E_ = CTX["eps"][ep]
        first = next(e for e in E_.events if abs(e.t0 - a) < 0.011 and e.kind in ("dialogue", "action"))
        assert not (first.kind == "dialogue" and cont(first.text)), (ep, a)
        dlg = [e for e in E_.events if e.kind == "dialogue"]
        nxt = next((e for e in dlg if e.t0 >= b - 0.01), None)
        ending = [e for e in dlg if abs(e.t1 - b) < 0.011]
        assert not (ending and nxt and cont(nxt.text) and ending[-1].text.rstrip('"').endswith(("...", "…"))), (ep, b)


def test_offline_judge_is_not_double_counted():
    x = M.vec({"hook_arousal": 0.5, "cliff_content": 0.6}, {"hook": 10, "cliffhanger": 10, "by": "offline-lexicon"})
    assert x["j_hook"] == 0.5 and x["j_cliff"] == 0.5  # fix D1
    y = M.vec({"hook_arousal": 0.5, "cliff_content": 0.6}, {"hook": 10, "cliffhanger": 10, "standalone": 5, "emotion": 5, "by": "llm:m"})
    assert y["j_hook"] == 1.0


def test_hook_and_payoff_both_required():
    no_payoff = M.vec({"hook_arousal": 1.0, "cliff_content": 0.0, "cliff_open_end": 0.0}, None)
    both = M.vec({"hook_arousal": 0.5, "cliff_content": 0.5}, None)
    assert no_payoff["hook_and_payoff"] == 0.0 and both["hook_and_payoff"] > 0.4  # fix C1
    out = P.rank(ROWS, M.Model(), k=60)
    assert all(p["features"]["_hook_part"] >= M.MIN_PART and p["features"]["_payoff_part"] >= M.MIN_PART for p in out)


def test_gate_blocks_worse_challenger():
    gold = E.load_gold("gold/gold.jsonl")
    champ = M.Model()
    worse = M.Model(w={k: -v for k, v in champ.w.items()})
    res = L.gate(ROWS, CTX, gold, champ, worse, 31, CTX["n_eps"])
    assert res["promote"] is False


def test_judge_retries_rate_limits_with_backoff():
    import urllib.error
    os.environ.update(JUDGE_BASE_URL="http://x", JUDGE_API_KEY="k", JUDGE_MODEL="m")
    with tempfile.TemporaryDirectory() as d:
        slept = []
        j = J.LLMJudge(os.path.join(d, "c.jsonl"), sleep=slept.append)
        replies = [urllib.error.HTTPError("u", 429, "rate", {}, None), urllib.error.HTTPError("u", 503, "busy", {}, None),
                   '{"hook": 6, "cliffhanger": 6, "standalone": 6, "emotion": 6, "reason": "ok"}']
        def post(sys_, t):
            r = replies.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        j._post = post
        v = j({"lines": ["1@01.0 A: hi"], "hook_line": "hi", "cliff_line": "hi", "dur": 25})
        assert v["by"] == "llm:m" and len(slept) == 2 and slept[1] > slept[0]  # fix D2: exponential backoff
        assert j.stats["retries"] == 2


def test_pairwise_compare_cancels_position_bias():
    os.environ.update(JUDGE_BASE_URL="http://x", JUDGE_API_KEY="k", JUDGE_MODEL="m")
    with tempfile.TemporaryDirectory() as d:
        j = J.LLMJudge(os.path.join(d, "c.jsonl"))
        j._post = lambda sys_, t: '{"better": "A", "confidence": 0.9}'  # a judge that always prefers the first clip
        promos = P.rank(ROWS, M.Model(), k=4)
        out = P.pairwise_rerank([dict(p) for p in promos], ROWS, j, top=4, log=lambda *_: None)
        assert {p["pairwise_wins"] for p in out[:4]} == {1.5}  # pure position bias nets out to a tie (fix D3)


def test_pairwise_training_orders_accepted_above_rejected():
    X = [{"j_hook": 0.9, "hook_arousal": 0.2}, {"j_hook": 0.1, "hook_arousal": 0.8}] * 10
    Y = [1, 0] * 10
    G = [i // 2 for i in range(20)]
    m = M.train_pairwise(X, Y, G, M.Model(), l2=0.1)
    assert m.logit(X[0]) > m.logit(X[1])  # fix E3
    assert M.pairs_of(Y, G) and all(Y[i] == 1 and Y[j] == 0 and G[i] == G[j] for i, j in M.pairs_of(Y, G))


def test_gate_uses_validation_gold_and_requires_confident_gain():
    gold = E.load_gold("gold/gold.jsonl")
    assert any(g.get("split") == "test" for g in gold)
    champ = M.Model()
    res = L.gate(ROWS, CTX, gold, champ, champ, 31, CTX["n_eps"])
    assert res["promote"] is False and res["ndcg_gain_ci"] is not None  # identical model: no gain, no promotion (F3)
    test_ids = {g["id"] for g in E.split_gold(gold, "test")}
    assert not test_ids & {g["id"] for g in E.split_gold(gold, "val")}  # F2


def test_policy_spoiler_aliases_and_schema():
    canon = CTX["canon"]
    assert canon("THE BEARDED STRANGER", 3) == "ALEXANDER" and canon("LADY JULIANA", 1) == "JULIANA"  # B6
    assert canon("BLOND NOBLEMAN", 2) == "LORD ASHCROFT" and canon("BLOND NOBLEMAN", 9) == "RUPERT"
    assert canon("ALEXANDER'S MOTHER", 35) == "ALEXANDER'S MOTHER"
    r = P._row(CTX, [(32, 119.6, 154.38)])  # branding iron scene
    assert r["view"]["policy"]["label"] == "review" and "graphic violence" in r["view"]["policy"]["reasons"][0]  # H5
    late = P._row(CTX, [(45, 0.0, 48.9)])
    assert late["view"]["spoiler"] in ("medium", "high")  # B7
    import promofinder.pipeline as PP
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rows.jsonl")
        PP.save_rows([dict(ROWS[0], schema="old")], path)
        try:
            PP.load_rows(path)
            raise AssertionError("stale rows accepted")
        except RuntimeError:
            pass  # A4


def test_outputs_have_handles_hook_types_and_a_posting_plan():
    out = P.rank(ROWS, M.Model(), k=30)
    s = out[0]["segments"][0]
    assert s["edit_in"] <= s["start"] and s["edit_out"] >= s["end"] and ":" in s["timecode"]  # C3
    waves = P.posting_plan(out, CTX["n_eps"])
    held = waves[-1][1]
    assert all(p["policy"]["label"] == "review" or p["spoiler"] == "high" for p in held)  # H2 + H5
    assert all(p not in held for _, items, _ in waves[:-1] for p in items)
    stitched = [r for r in ROWS if len(r["segments"]) > 1]
    assert all(set(r["view"]["speakers"][0]) & set(r["view"]["speakers"][-1]) for r in stitched)  # C4


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok", f.__name__)
    print(len(fns), "passed")
