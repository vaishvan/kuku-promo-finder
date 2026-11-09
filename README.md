# Promo Finder

**Live demo: https://promo-finder-7ln6.onrender.com** (click "Try the sample show", or upload a scripts zip; the free
server sleeps when idle, so the first visit can take about a minute).

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

python tests/test_all.py                               # 22 tests (build their own rows offline if needed)
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
python -m promofinder serve                       # http://127.0.0.1:8765, starts empty
python -m promofinder serve --show runs/live      # or preload an existing run
```
1. **Upload** the scripts `.zip`. It is extracted locally into `runs/uploads/<time>/scripts`: only `.txt` members,
   member paths never used (no zip-slip), size and count limits; episode numbers come from file names, the
   `EPISODE NUMBER:` header, or file order.
2. **Run** with the LLM judge (or offline). The pipeline block streams progress: parse → candidates → shortlist →
   LLM judge (with cache hits) → pairwise re-rank → rank and posting plan. `.env` is re-read on every run, so a new
   key or model needs no restart. If the judge can't run (rejected key, unavailable model, network, OpenRouter's daily
   quota), the run stops without mixing scores and the page explains why in plain words with a one-click fix
   ("Run offline instead", "Choose another zip", "Try again").
3. **Results:** tabs **To review**, **Accepted**, **Rejected**, **Posting plan**, **Learning loop**. Each promo card
   shows timecodes, opening and ending lines, the judge's scores and reason, hook type, ad-policy and spoiler flags,
   and the script behind the cut. Accept or Reject moves a clip into its list (Undo sends it back); those choices are
   the labels. **Learning loop → Retrain** learns from them, checks itself on held-out episodes, and explains the
   result in plain words ("cares less about a shocking opening"). Labels are stored per run
   (`<run>/feedback.demo.jsonl`), so every run starts unreviewed; the demo never saves or promotes a model.
   Evaluation numbers are deliberately not in the UI: they're in `eval` output and the Results section below.

UI styling follows the Figma design profile from getdesign.md (black/white canvas, pill buttons, pastel color
blocks). Talk track for a 3-minute recording: [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

### Public demo

**Render (free):** [Deploy to Render](https://render.com/deploy?repo=https://github.com/vaishvan/kuku-promo-finder):
sign in with GitHub, paste the NVIDIA key into the `JUDGE_API_KEY` field Render asks for, deploy. `render.yaml`
builds the `Dockerfile` (stdlib Python, no dependencies). The free plan sleeps after 15 idle minutes; the first visit
after that takes about a minute.

**Hugging Face Spaces:** `HF_TOKEN=hf_... python deploy/push_space.py` creates the Space, stores the key as a secret
and pushes. Docker Spaces now need a Hugging Face PRO plan.

GitHub Pages can't host it: Pages serves static files only, so there is no server to keep the key secret or run the
pipeline, and any key in a static site is readable by every visitor. Hosted mode
(`PROMO_HOSTED=1`, set in the image) adds what a public URL needs:
- **Per-visitor sessions** (cookie): visitors never see each other's uploads or runs; one pipeline runs at a time.
- **Try the sample show:** a short original fictional series (`samples/`, generated by `samples/make_sample.py`), so
  anyone can try it without the confidential scripts.
- **Limits:** 20 MB uploads, 6 AI-judged runs per visitor per hour, uploads and results deleted after 6 hours, and a
  smaller shortlist (every episode ending + the 10 best + pairwise on the top 4, about 1–2 minutes per 50 episodes).
- The API key lives only in the Space's secret store; the image never contains the scripts, outputs or `.env`.

### Using a real LLM judge
Copy `.env.example` to `.env` (gitignored) and set `JUDGE_API_KEY`. Any OpenAI-compatible endpoint works; the
default is the NVIDIA API catalog's free `nvidia/nemotron-3.5-super-vl-120b-a12b` with `JUDGE_NO_THINK=1` (thinking
off; ~1–2 s per call). With OpenRouter's free tier (50 requests/day) the demo automatically uses a cut-down shortlist. Every judgement is cached in
`runs/judge_cache.jsonl`, so re-runs and demos cost nothing. Swap models by changing `JUDGE_MODEL`.

## How a clip is scored

1. Code features: hook in the first 3 s (dialogue *and* camera/action text, so visual openers count), cliffhanger from
   story content in the last beats (threat, demand, arrival, unanswered question), standalone load, duration fit.
2. **Hook AND payoff:** a clip with no hook signal or no payoff signal is not ranked; the two are combined with a
   geometric mean so a strong opener cannot carry a flat ending.
3. A logistic ranker weighs the features; weights are retrained from feedback and gated.
4. Cuts never start on the second half of a split line or end right before one.

## Results (2026-11-08)

Gold: 64 builder-written regions; validation split 45, frozen test split 19. Held-out block = Ep 31–50
(validation gold there: 2 must, 6 good, 5 bad). All numbers below are on held-out validation gold.

### Shortlist fix: always judge each episode's ending

The first LLM run missed both held-out must-haves because a pure top-N shortlist never contained them. Two changes:
1. **Every episode's best ending window is reserved in the shortlist** (`--reserve-endings all`; 39 of 50 episodes
   have an ending that passes the hook-and-payoff minimum).
2. **The ending-proximity prior went from 0.3 to 2.5**, chosen from five values on the *training* episodes only
   (train NDCG@10 0.12 → 0.31), then checked on held-out once.

| System (offline judge, full 186-clip pool) | P@10 | NDCG@10 | must-recall@10 | top-10 good / bad / unlabelled |
|---|---|---|---|---|
| random valid windows | 0.0 | 0.00 | 0.0 | 0 / 1 / 9 |
| last 40 s of every episode | 0.4 | 0.41 | 1.0 | 4 / 1 / 5 |
| model before the fix | 0.1 | 0.14 | 0.0 | 1 / 0 / 9 |
| **model after the fix** | **0.3** | **0.21** | **0.5** | **3 / 0 / 7** |

- The gap to the baseline halved, with 0 known-bad picks (the baseline has 1). The baseline still leads.
- **Still missed: Ep 41's cathedral explosion.** Its ending closes on a visual beat with no dialogue ("silhouetted
  armed figures emerge through the smoke"), which the code cliffhanger signal scores 0. I did not add those words to
  the lexicon (that would be tuning on the test set); seeing visual beats is the LLM judge's job, and it now gets
  camera and action lines.
- 7 of the top 10 are unlabelled mid-episode picks (`to_label.csv`).

### LLM judge on the full fixed pool (NVIDIA API catalog, `nvidia/nemotron-3.5-super-vl-120b-a12b`, free tier, $0)

186 clips (every passing episode ending included) + pairwise re-rank of the top 10: 273 calls, 0 retries,
0 fallbacks, about 5 minutes. Thinking is switched off (`JUDGE_NO_THINK=1`): ~1–2 s per call with valid JSON.

| System (held-out, same 186-clip pool) | P@10 | NDCG@10 | must-recall@10 | top-10 good / bad / unlabelled | recall@20 of good/must endings |
|---|---|---|---|---|---|
| last 40 s of every episode | 0.4 | 0.41 | 1.0 | 4 / 1 / 5 | 6/6 |
| code features only | 0.3 | 0.21 | 0.5 | 3 / 0 / 7 | 4/6 |
| code features + LLM judge | 0.2 | 0.18 | 0.5 | 2 / 1 / 7 | 5/6 |

- **The judge fixes the failure it was meant to fix.** Ep 41's ending ("silhouetted armed figures emerge through the
  smoke") scores 0 on the code cliffhanger signal; the judge gives it cliffhanger 9 and it now reaches the top 10.
- **It also adds one known-bad pick** (Ep 37, invitations and gossip, judged "ends on a villainous wine sip and
  execution clause"). Net NDCG difference vs code-only is one pick on a 13-item label set: noise.
- Top picks are the scenes an editor would cut: Ep 32 branding iron and Ep 36 "their dead Duke is coming home" (both
  must-haves), Ep 15 garden attack, Ep 31 dungeon threat. The judge's reasons are specific and critical ("opening
  line lacks immediate physical shock"). Both top picks are flagged for ad-policy review (graphic violence), so the
  posting plan holds them back for a human.
- Judge weights (1.5 each) are still the hand-set prior; retraining on real editor labels is what would set them.
- Still missed: Ep 45 (marriage declared void), ranked too low. The episode-end baseline still leads overall.

Earlier OpenRouter run (same model, 50 requests/day): 27-clip shortlist, superseded by this one.

## Known limits (read before trusting any number)
1. **Gold and simulated feedback are both written by the builder**; validation gold in the held-out block is 13 items. No inter-annotator check.
2. **The LLM judge does not yet beat code-only on held-out labels** (0.18 vs 0.21 NDCG@10, one pick apart). It does catch visual cliffhangers the code misses. Its weights are an unvalidated prior until editor labels exist. The content-based cliffhanger feature is weak (7/26 good vs 3/15 bad episode endings fire).
3. **One show.** Generalisation untested. The alias table is per show and hand-written.
4. **Policy and spoiler labels are word lists**, not a policy-trained classifier; treat "review" as "a human looks", not as a verdict.
5. **Scores are uncalibrated** (all near 0.99); use the rank.
6. **Pairwise training** needs accept/reject pairs from the same episode; the simulated labels have none, so the demo trains pointwise.
7. **Not built:** HTTP service, scheduler, editor UI (the CSV is the interface), engagement-based feedback (hook rate, hold rate, CPI).
