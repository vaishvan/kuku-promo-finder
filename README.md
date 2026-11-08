# Promo Finder

Finds 20–60s promo clips in microdrama screenplays, learns from editor feedback, and gates every new
model on an offline evaluation. Design: [DESIGN.md](DESIGN.md). Known problems and their status:
[QUALITY_REVIEW.md](QUALITY_REVIEW.md).

Python 3.11+, standard library only. No install.

## Run it

```bash
# 1. find promos (scripts folder = episode-NNN-*.txt files)
python -m promofinder --scripts data/scripts --out runs/r1 run --k 40
#    -> promos.json (ranked; hook, cliffhanger, why, hook type, policy, spoiler, edit points)
#       promos.csv (editor sheet), posting_plan.md (what to post first), rows.jsonl (cached features)
#    flags: --exclude-review (drop ad-policy review clips), --pairwise-top N (LLM judge only), --allow-mixed-judge

# 2. evaluate (validation gold; held-out block = last 40% of episodes, --split N overrides)
python -m promofinder eval --model v0001             # also writes to_label.csv: unlabelled top picks for an editor
python -m promofinder eval --model v0001 --final     # adds the frozen test split; every use is counted

# 3. the learning loop
python -m promofinder feedback runs/r1/promos.csv      # editor fills decision=accept|reject, optional trim_start/trim_end
python -m promofinder retrain                          # CV-tuned retrain -> bootstrap gate -> promote or hold
python -m promofinder promote v0001                    # human override of the gate

# demo only: a synthetic editor labels the model's picks from validation gold (never touches the real log)
python -m promofinder simulate
python -m promofinder retrain --feedback runs/feedback.sim.jsonl --min-labels 20

python tests/test_all.py                               # 21 tests
```

Editor sheet rules: `decision` is `accept` or `reject`. A trim (`trim_start`, `trim_end` in seconds) means "cut it
like this instead": the trimmed window is snapped to legal cut points and logged as a positive, the original as a
negative. Re-ingesting the same sheet adds nothing.

Per-show speaker aliases live in `gold/aliases.json` (`"LABEL": "CHARACTER"`, `"LABEL@ep"` for one episode).
Optional: without it, honorifics are stripped and known names matched automatically.

New show at the interview: `python -m promofinder --scripts <folder> --out runs/new run`. If its files do not match
the format in `promofinder/parse.py`, only that file changes.

### Demo UI
```bash
python -m promofinder --out runs/live serve      # http://127.0.0.1:8765
```
Promos (with the script excerpt behind each cut), posting plan, evaluation, and a learning-loop tab where
Accept/Reject clicks retrain a challenger and run the promotion gate. Demo clicks go to `runs/feedback.demo.jsonl`;
the demo never saves or promotes a model. Talk track for a 3-minute recording: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

### Using a real LLM judge
Copy `.env.example` to `.env` (gitignored) and set `JUDGE_API_KEY`. Any OpenAI-compatible endpoint works; the
default is OpenRouter's free `nvidia/nemotron-3.5-super-vl:free`. Free tier = 50 requests/day, so a live run judges
a smaller pool: `run --top-n 25 --stitch-n 2 --pairwise-top 4` is about 40 calls. Every judgement is cached in
`runs/judge_cache.jsonl`, so re-runs and demos cost nothing. Swap models by changing `JUDGE_MODEL`.

## How a clip is scored

1. Code features: hook in the first 3 s (dialogue *and* camera/action text, so visual openers count), cliffhanger from
   story content in the last beats (threat, demand, arrival, unanswered question), standalone load, duration fit.
2. **Hook AND payoff:** a clip with no hook signal or no payoff signal is not ranked; the two are combined with a
   geometric mean so a strong opener cannot carry a flat ending.
3. A logistic ranker weighs the features; weights are retrained from feedback and gated.
4. Cuts never start on the second half of a split line or end right before one.

## Results (offline judge, builder-written gold v2: 64 regions; validation split 45, frozen test split 19)

Held-out block = Ep 31–50. Validation gold in that block: 2 must, 6 good, 5 bad.

| System | held-out P@10 | NDCG@10 | top-10: labelled good / bad / unlabelled | frozen test NDCG@10 (scored once) |
|---|---|---|---|---|
| random valid windows | 0.3 | 0.14 | — | 0.00 |
| last 40s of every episode | 0.3 | 0.25 | 3 / 3 / 4 | 0.39 |
| prior weights (v0000) | 0.1 | 0.14 | 1 / 0 / 9 | 0.33 |
| demo retrain (v0001, not promoted) | 0.1 | 0.14 | 1 / 0 / 9 | 0.51 |

What this says:
- **The validation gold is too thin to rank systems.** 9 of the model's held-out top 10 match no labelled region, so its precision and NDCG are mostly unknown, not low. The fix is labels, not tuning: `to_label.csv` lists the 12 unlabelled top picks for an editor.
- The model makes no known-bad picks (0/10). The episode-end baseline makes 3.
- v0001 (23 simulated labels, retrained with `--min-labels 20` as a demo) tied the champion on validation and was **not promoted**. On the frozen test split it scores higher (0.51 vs 0.33), on 6 good/must items. That is one look at a tiny set, not evidence; the gate correctly ignored it.
- The test split has now been scored once (`runs/test_usage.json`).
- Numbers are not comparable to earlier README versions: the gold was split, and features changed.
- Missed must-haves in the held-out block: Ep 41 (cathedral explodes) and Ep 45 (marriage declared void), both in the pool but ranked too low.

### Live LLM judge run (`runs/live`, nvidia/nemotron-3.5-super-vl:free via OpenRouter)

39 calls (27 rubric scores + 12 pairwise comparisons), 0 retries, 0 fallbacks, about 7.5 minutes of API time,
cost $0 (free tier). The pool was cut to 27 candidates to fit 50 requests/day.

| System (same 27-candidate pool) | train-block NDCG@10 | held-out NDCG@10 | held-out top-10 good / bad / unlabelled |
|---|---|---|---|
| code features only (judge masked) | 0.09 | 0.14 | 1 / 0 / 9 |
| code features + LLM judge | 0.25 | 0.14 | 1 / 0 / 9 |

- The judge reorders clips sensibly where gold exists (train block 0.09 → 0.25) and its reasons are specific and critical, e.g. "Opening accusation is talky, not shocking" or "'dead Duke' relationship confuses newcomers".
- Held-out it changes nothing measurable: with 27 candidates for 50 episodes, the held-out block has only a handful of judged clips and 9 of the top 10 are unlabelled.
- The random baseline here samples from the same pre-filtered pool, so it is not really random (QUALITY_REVIEW F7) and should not be read as "random does as well".
- To measure the judge properly: a paid key or a few days of the free quota to judge the full 178-candidate pool, and editor labels for `to_label.csv`.

## Known limits (read before trusting any number)
1. **Gold and simulated feedback are both written by the builder**; validation gold in the held-out block is 13 items. No inter-annotator check.
2. **Live LLM judge run once, on a 27-candidate pool** (free-tier limit); its effect on held-out quality is not measurable yet. The content-based cliffhanger feature is weak (7/26 good vs 3/15 bad episode endings fire).
3. **One show.** Generalisation untested. The alias table is per show and hand-written.
4. **Policy and spoiler labels are word lists**, not a policy-trained classifier; treat "review" as "a human looks", not as a verdict.
5. **Scores are uncalibrated** (all near 0.99); use the rank.
6. **Pairwise training** needs accept/reject pairs from the same episode; the simulated labels have none, so the demo trains pointwise.
7. **Not built:** HTTP service, scheduler, editor UI (the CSV is the interface), engagement-based feedback (hook rate, hold rate, CPI).
