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

python tests/test_all.py                               # 21 tests (build their own rows offline if needed)
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
python -m promofinder serve      # http://127.0.0.1:8765
```
1. **Upload** the scripts `.zip`. It is extracted locally into `runs/uploads/<time>/scripts`: only `.txt` members,
   member paths never used (no zip-slip), size and count limits; episode numbers come from file names, the
   `EPISODE NUMBER:` header, or file order.
2. **Run** with the LLM judge (or offline). The pipeline block streams progress: parse → candidates → shortlist →
   LLM judge (with cache hits) → pairwise re-rank → rank and posting plan. The header shows free requests left.
   Cached clips always run free; new judgements are capped at today's remaining quota, and pairwise is skipped when
   the quota is too low, so a demo cannot stall halfway.
3. **Results:** promo cards (timecodes, handles, opening/ending lines, judge scores and reason, hook type, policy and
   spoiler flags, score band, the script behind the cut), posting plan, evaluation (only when gold labels exist for
   that show), and a learning-loop tab where Accept/Reject clicks retrain a challenger and run the gate.
   Demo clicks go to `runs/feedback.demo.jsonl`; the demo never saves or promotes a model.

UI styling follows the Figma design profile from getdesign.md (black/white canvas, pill buttons, pastel color
blocks). Talk track for a 3-minute recording: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

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

## Results (2026-11-08, after all review fixes)

Gold: 64 builder-written regions; validation split 45, frozen test split 19. Held-out block = Ep 31–50
(validation gold there: 2 must, 6 good, 5 bad).

### LLM-judged run (`runs/live`, nvidia/nemotron-3.5-super-vl:free via OpenRouter, $0)

27-candidate shortlist (free tier = 50 requests/day): 9 new calls + 18 cached, 0 retries, 0 fallbacks.

| System | held-out P@10 | NDCG@10 | must-recall@10 | top-10 good / bad / unlabelled |
|---|---|---|---|---|
| random valid windows (from all ~15K candidates) | 0.0 | 0.00 | 0.0 | 0 / 1 / 9 |
| last 40s of every episode (ordered by prior score) | 0.4 | 0.41 | 1.0 | 4 / 1 / 5 |
| code features only, same shortlist | 0.1 | 0.14 | 0.0 | 1 / 0 / 9 |
| code features + LLM judge, same shortlist | 0.1 | 0.14 | 0.0 | 1 / 0 / 9 |

What this says, plainly:
- **On this labelled set, the simple baseline wins.** Once its order stopped depending on episode number, "last 40 s of every episode" finds every held-out must-have. The brief defines a promo as ending on a cliffhanger, and this show ends every episode on one, so episode endings are a strong prior.
- **The model's weak spot is the shortlist, not the judge.** The two held-out must-haves (Ep 41 cathedral explosion, Ep 45 marriage declared void) never reached the 27-clip shortlist, so the LLM never saw them. The free-tier budget forced the shortlist that small; the offline run uses 178.
- **Most of the model's picks are unlabelled** (9 of 10), mostly mid-episode moments the gold never covered. They are in `to_label.csv`. Until an editor labels them, the model's precision is unknown rather than low.
- The model makes 0 known-bad picks in its held-out top 10; the baseline makes 1.
- Next fixes, in order: guarantee each episode's ending window a place in the shortlist (cheap, and the baseline shows it matters), judge the full 178-candidate pool when quota allows, then label `to_label.csv`.

## Known limits (read before trusting any number)
1. **Gold and simulated feedback are both written by the builder**; validation gold in the held-out block is 13 items. No inter-annotator check.
2. **The LLM judge has only seen a 27-clip shortlist** (free-tier limit). On held-out labels it did not change the ranking, and the shortlist missed both held-out must-haves. The content-based cliffhanger feature is weak (7/26 good vs 3/15 bad episode endings fire).
3. **One show.** Generalisation untested. The alias table is per show and hand-written.
4. **Policy and spoiler labels are word lists**, not a policy-trained classifier; treat "review" as "a human looks", not as a verdict.
5. **Scores are uncalibrated** (all near 0.99); use the rank.
6. **Pairwise training** needs accept/reject pairs from the same episode; the simulated labels have none, so the demo trains pointwise.
7. **Not built:** HTTP service, scheduler, editor UI (the CSV is the interface), engagement-based feedback (hook rate, hold rate, CPI).
