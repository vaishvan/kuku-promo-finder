# Quality review: Promo Finder

Review of everything in `dev/kuku` as of 2026-11-07: framework, code, gold set, evaluation, and the
results reported in the README. Evidence comes from reading the code and from a probe script run against
the 50 episodes and the current outputs (`runs/r1`). Nothing below has been fixed yet.

## Verdict

The pipeline runs, its outputs pass the hard checks (duration, boundaries, overlap, safety lexicon), and the
retrain-and-gate loop works mechanically. The **numbers it reports cannot be trusted yet**:

1. The gold set and the simulated feedback were written by the same person (the builder), and 41 of 44 gold items are episode endings. Training and evaluation share one bias, so the "learning improved held-out NDCG" result is partly circular.
2. Several features measure the wrong thing. The worst: `cliff_marker` fires on 37 of 50 episode endings because every episode ends on a music sting, and the hook lexicon matches "ring" inside "staring", "roaring" and "whispering" (348 of 384 hits are false).
3. One documented feature is broken: editor trims are dropped and never become training labels.
4. Running the demo `simulate` command wipes any real editor feedback.

| Severity | Count | Meaning |
|---|---|---|
| Critical | 8 | Produces wrong output, loses data, or invalidates a reported result |
| High | 23 | Measurably degrades pick quality or evaluation |
| Medium | 26 | Correctness or robustness risk; matters at scale or on a new show |
| Low | 7 | Hygiene |

## Status of the critical items (updated 2026-11-07)

| # | Status | What changed | Check |
|---|---|---|---|
| A1 trims dropped | **Fixed** | Trims snap to legal cut points, get features, log as positive; the original logs as negative. Out-of-range trims are reported, not silently dropped. Re-ingesting a sheet is a no-op (A3 also fixed). | `test_csv_feedback_ingest_keeps_trims_and_dedupes` |
| A2 simulate wipes real labels | **Fixed** | Simulator writes `runs/feedback.sim.jsonl`; `retrain --feedback` picks the file; the real log is append-only. | `test_simulate_never_touches_real_feedback` |
| B1 music cue = cliffhanger | **Fixed, weak replacement** | `cliff_marker` removed; new `cliff_content` reads threat/demand/arrival/question in the last story beats; `cliff_proximity` demoted to a weak prior. The new signal fires on 7/26 good vs 3/15 bad episode endings: better than the old 37/50 blanket, but weak. Needs the LLM judge. | `test_music_cue_is_not_a_cliffhanger`; probe |
| B2 substring lexicons | **Fixed** | All lexicons anchored on word starts with explicit inflections; genre words removed from CONFLICT (B3 also fixed). Partial-word matches: 348 → 1 ("knife's"). | `test_lexicons_match_whole_words_only` |
| C1 linear sum, hook OR payoff | **Fixed** | `hook_and_payoff` = geometric mean of hook and payoff parts (prior weight 2.0); clips below 0.1 on either part are not ranked or judged. | `test_hook_and_payoff_both_required` |
| D1 live judge never run | **Fixed (2026-11-08), small pool** | Live run on OpenRouter's free nemotron: 39 calls, 0 failures, 27-candidate pool. Earlier: offline judge scores no longer feed the model (no double counting); eval labels the judge used; README states the LLM path is untested. **Still open:** no API key or local model was available, so the LLM judge has not run. | `test_offline_judge_is_not_double_counted` |
| E1 circular feedback | **Partly fixed** | Simulator no longer injects gold regions as training rows (E2 fixed too: no "one promo per episode" assumption). **Still open:** simulated labels share the gold labeller's taste; needs a second human or the live judge. | `test_simulate_never_touches_real_feedback` |
| F1 tail-biased single-labeller gold | **Partly fixed** | Gold v2: 64 regions, 23 not at an episode end, each tagged with position and labeller; eval reports recall by position and lists missed must-haves with the reason (F9 also fixed). **Still open:** one labeller, no agreement measure. | `eval` output |

Also fixed along the way: C2 (no cut starts or ends mid-sentence: 3 → 0 in the top 40), B4 ("..." continuation no
longer counts as an open ending), B5 partly (camera/action text counts toward the hook), H1 (the "why" only states
signals that fired), G1 (held-out split is 40% of any show, `--split` overrides), E4 (CV folds are contiguous
episode blocks), C4 partly (stitches limited to one episode, picked per episode). Everything else in Part 2 is open.

## Status of the high items (updated 2026-11-08)

| # | Status | What changed |
|---|---|---|
| A3 duplicate feedback | **Fixed** | Labels keyed on (id, editor); re-ingesting a sheet adds nothing. |
| A4 no feature schema | **Fixed** | A schema hash (feature list + every lexicon) is stamped on rows, promos and labels. Stale `rows.jsonl` is refused; old-schema labels are skipped and counted at retrain. |
| B3 genre words in CONFLICT | **Fixed** | Removed (with B2). |
| B4 "..." as fake cliffhanger | **Fixed** | Continuation lines do not count as open endings. |
| B5 hook ignores visuals | **Fixed** | Camera/action text in the first 3 s counts toward the hook; the judge now receives camera lines. |
| B6 standalone unreliable | **Partly fixed** | Per-show alias table (`gold/aliases.json`, with per-episode overrides such as "BLOND NOBLEMAN@2") plus automatic matching; relationship words now help. Still a heuristic: the real test ("could a new viewer say who wants what?") needs the LLM judge. |
| B7 no spoiler control | **Fixed (policy left to a human)** | Every promo carries `spoiler` low/medium/high (reveal language, late-season position); high-spoiler clips are held back in the posting plan; `rank(..., exclude_spoilers=True)` drops them. Not excluded by default because the show's title already gives away the twist. |
| C2 mid-sentence starts | **Fixed** | (critical pass) |
| C3 no handles | **Fixed** | Each segment has `edit_in`/`edit_out` with 0.4 s pre/post-roll and an mm:ss.ff timecode. |
| C4 stitches broken | **Mostly fixed** | Same episode only, best heads/tails per episode, head and tail must share a character. Not done: a judge check that the pair reads as one story. |
| C5 pool chosen by champion | **Fixed** | The judged pool is the union of the prior's and the current model's picks. A newly promoted model still needs a fresh `run`. |
| D2 no retry/backoff | **Fixed** | Exponential backoff with jitter on 408/409/425/429/5xx, timeouts and unparsable replies; per-run call/retry/failure/latency stats in the manifest; a run with >20% fallback rows is refused unless `--allow-mixed-judge`. |
| D3 absolute scoring only | **Fixed, untested live** | `LLMJudge.compare` asks which of two clips is the better promo; the top 10 are re-ranked by comparisons in both orders, so position bias cancels (test proves a judge that always says "A" nets to a tie). Not done: few-shot anchors. |
| E2 one-promo-per-episode assumption | **Fixed** | (critical pass) |
| E3 pointwise labels | **Fixed** | Pairwise ranking loss on accepted-vs-rejected pairs from the same episode, used when there are at least 10 pairs; pointwise otherwise. The simulated labels produce 0 same-episode pairs, so the demo still trains pointwise. |
| F2 test gold reused | **Fixed** | A frozen third of the gold (`split: test`) is never seen by the gate, the buckets or the to-label list; `eval --final` scores it and increments a usage counter. |
| F3 no uncertainty | **Fixed** | The gate bootstraps held-out episodes and promotes only if the 10th percentile of the NDCG@10 gain is above zero. |
| F4 unlabelled = miss | **Fixed** | Eval reports labelled-good / labelled-bad / unlabelled for each system's top 10 and writes the unlabelled picks to `to_label.csv`. |
| F5 promised checks missing | **Fixed (rule-based)** | Hook-in-3s, ends-unresolved, standalone, policy-review share, spoiler share, hook-type count, and judge cost/latency are reported. They use the same rules as the features until an LLM judge runs, and say so. |
| G1 hard-coded split | **Fixed** | (critical pass) |
| H1 template "why" | **Fixed** | (critical pass) |
| H2 no posting strategy | **Fixed** | `posting_plan.md`: wave 1 premise clips for cold audiences, wave 2 escalation with hook types round-robined, wave 3 refresh pool with the CTR/CPC swap rule, and a held-back list (spoilers, policy review). Each promo is labelled with a hook type. |
| H5 violence not flagged | **Fixed (lexicon)** | Policy label per promo: sexual content = block (never ranked), graphic violence = review (ranked, flagged, held back in the plan; `--exclude-review` drops them). 12 of the top 40 are flagged review. Not done: an LLM policy rubric. |

Medium items fixed in passing: D4 (cache key = rubric hash), D5 (JSON-mode request, first-complete-object parser,
range validation, retry on bad replies), D6 (cost/latency), F8 (in-sample rows marked), H4 (timecodes).

---

## Part 1: How the industry cuts promos, and where this tool departs from it

| # | Practice | Source | What the tool does today |
|---|---|---|---|
| I1 | Hook in the first 3 seconds. Meta measures it as **hook rate** = 3-second plays ÷ impressions (~25% average, 30%+ strong); TikTok uses 2-second views. | [AdMake](https://admakeai.com/blog/hook-rate), [AdManage](https://admanage.ai/blog/what-is-a-good-hook-rate-for-facebook-ads) | Hook scored from text lexicons only; no notion of a hook *type*. |
| I2 | The second number is **hold rate** = ThruPlays ÷ 3-second plays (40–50% average). **High hook, low hold = clickbait opener**: the hook promised something the clip did not deliver. | [AdMake](https://admakeai.com/blog/hook-rate) | Hook and payoff scored independently, then added linearly. A great opener can carry a clip with no payoff. |
| I3 | Trailer structure: cold open → build → strongest line held for the end ("put this into the end of the trailer"). Rhetorical/straight questions set up the payoff. | [Derek Lieu](https://www.derek-lieu.com/blog/2017/9/10/the-matrix-is-a-trailer-editors-dream) | Clips are contiguous windows; no check that intensity *rises* toward the end. |
| I4 | **Don't give away the ending.** The most-cited trailer mistake. | [Pexo](https://pexo.ai/tutorial/how-to-make-a-movie-trailer) | No spoiler control. Ep 42–45 (the Duke reveal) are among the top picks. |
| I5 | Microdrama ads: conflict in the first seconds, cut right before the resolution, **readable with no context and without sound**; actors overplay so the plot reads muted. | [Palai Media](https://palai.media/en/articles/iak-pratsiuie-format-short-drama) | Hook is dialogue-centric. Camera lines are dropped from hook scoring and from the judge's input, so purely visual hooks score low. |
| I6 | "Give, then take": deliver one micro-win (reveal, confrontation), then a new threat. ~8 s to establish conflict and a character to root for. | [LBB / Heckler](https://lbbonline.com/news/Micro-Dramas-4-Mistakes-Brands-Keep-Making) | No "win then threat" signal; no protagonist/rooting-interest signal. |
| I7 | Short-drama hook mix to rotate: IP/brand 25%, emotional conflict 22%, visual spectacle 20%, interactive question 18%, testimonial 15%. A 15 s clip can hold a full conflict→climax→cliffhanger arc. | [AppGrowing](https://appgrowing.net/blog/ci-games-and-short-drama-ua-strategy-the-creative-cross-over-opportunity/) | Output is not labelled by hook type, so a feed cannot be balanced across types. |
| I8 | **Creative fatigue**: AI-generated short-drama creatives fatigue in 3–5 days vs 45–100 for live action; teams need 30–50 narrative variants. Refresh when CTR drops >15% or CPC rises >20%. Meta recommends 5+ creatives per ad set. | [AppGrowing](https://appgrowing.net/blog/ci-games-and-short-drama-ua-strategy-the-creative-cross-over-opportunity/), [Meta](https://www.facebook.com/metaforbusiness/posts/your-ad-creative-is-live-but-running-on-repeat-try-5-creatives-per-ad-set-to-bea/1586667100167656/) | Returns a top-k list with only a per-episode cap. No diversity across story beats, no variant generation, no fatigue loop. |
| I9 | Ad policy: Meta bans "shocking, sensational or excessively violent content", including torture imagery and content that highlights suffering; adult nudity and sexually suggestive activity are banned; **TV/movie trailer ads need prior written permission and 18+ targeting**. TikTok bans gratuitous violence and sexually suggestive content. | [Meta Ad Standards](https://transparency.meta.com/policies/ad-standards/), [Meta sensational content](https://www.facebook.com/business/help/304110064285796), [Strike Social](https://strikesocial.com/blog/get-familiar-with-tiktok-ad-creative-policies-and-avoid-ads-rejection/) | Safety filter is a sexual-content word list only. 12 of the top 40 promos contain torture/graphic-violence language (branding iron, blood, gashes, gallows) and none are flagged. |
| I10 | Clipping tools cut on transcript/sentence boundaries; clips that start mid-thought are the most common failure; editors add a small pre-roll buffer. | [Montage](https://montage.app/blog/how-ai-video-clipping-works), [MindStudio](https://www.mindstudio.ai/blog/ai-short-form-video-creation-skill-system) | Cuts on event boundaries, but an "event" is often half a sentence (see C2). No pre-roll handle (C3). |
| I11 | Clip scores are triage, not prediction: an 84 is not better than an 81, and 20–40% of AI-picked clips are discarded by editors. | [Montage](https://montage.app/blog/how-ai-video-clipping-works), [Techpresso](https://academy.techpresso.co/reviews/is-opus-clip-worth-it) | Scores shown to 4 decimals; 35 of the top 40 are between 0.95 and 0.99. |

---

## Part 2: Problems and fixes

Evidence marked **[measured]** comes from the probe run on the 50 episodes.

### A. Data integrity and loss (fix first)

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| A1 | Critical | **Editor trims are silently dropped.** `ingest_csv` creates trim records without features (`x`), and `cmd_feedback` keeps only records with `x`. The README promises trims become positive labels. | `cli.py:96`, `learn.py:34-35` | Snap the trim to event boundaries, run `extract()` on the trimmed window, store its `x`, and log it as a positive. Log the original window as a negative, as the design doc says. Add a test. |
| A2 | Critical | **`simulate` truncates `runs/feedback.jsonl`**, deleting any real editor labels. | `cli.py:89` | Write simulated labels to a separate file (`runs/feedback.sim.jsonl`). Let `retrain` take `--feedback <file>`. Never truncate the production log. |
| A3 | High | **Duplicate feedback**: running `feedback` twice on the same CSV appends every label twice and doubles its weight. | `learn.py:13-17` | Key labels on (candidate id, editor) and keep the latest; or write an idempotency hash per CSV row. |
| A4 | High | **Feedback stores a feature snapshot with no schema version.** Any change to `features.py` makes old labels silently inconsistent with new rows. | `learn.py:33` | Store `feature_schema` (a hash of the feature list and lexicons) on every row and every label. Recompute features from segments at retrain time, or refuse to mix schemas. |
| A5 | Medium | **Non-atomic registry writes.** `current.json` is overwritten in place; a crash mid-write corrupts the pointer. Two concurrent retrains race on the next version number. | `model.py` (`save_new`, `promote`) | Write to a temp file and `os.replace()` it into place; take a lock file around `save_new` + `promote`. |
| A6 | Medium | Files opened without context managers throughout (`json.dump(..., open(...))`). On Windows this leaves locked handles. | `pipeline.py:98`, `model.py`, `cli.py:58` | Use `with open(...)`. |

### B. Features that measure the wrong thing

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| B1 | Critical | **"Ends at the end of the episode" is scored as "ends unresolved".** `cliff_marker` matches music cues ("swell", "sting", "boom", "silence"), and those are written into almost every episode ending. `cliff_proximity` rewards closeness to episode end. Quiet, resolved endings (Ep 37 gossip, Ep 38 bread) rank high. | **[measured]** 37/50 episode endings trip `cliff_marker`. These are the two held-out failures in the README. | Drop music/SFX cues from the cliffhanger signal (they describe the score, not the story). Score unresolvedness from content: an open question or threat in the last line, a reversal in the last 5 s, or an unanswered demand. Let the LLM judge's cliffhanger score carry this; keep `cliff_proximity` only as a weak prior. |
| B2 | Critical | **Lexicons match substrings, not words.** `HOOK_ACTION` "ring" matches staring, towering, roaring, whispering and *Harrington*. "iron", "brand" and "chain" pick up set dressing. | **[measured]** "ring": 384 hits, 36 are the word "ring". "dead" also hits deadly, deadline, deadbolt. | Use word boundaries (`\bring\b`), lemma lists, and ideally the LLM judge instead of lexicons for semantic signals. Add a unit test with known false positives. |
| B3 | High | **`CONFLICT` contains genre vocabulary**: "married", "wife", "husband", "duke", "never", "truth". In this show those words are in almost every scene, so `conflict_density` and `cliff_open_end` saturate. | `features.py:14` | Remove setting/genre words; keep acts (betray, threaten, accuse, arrest, slap). Better: have the judge return a conflict score. |
| B4 | High | **`cliff_open_end` rewards a line ending in "..."**, but in this script "..." usually means the sentence continues in the next event. Cutting there creates a fake cliffhanger that is really a mid-sentence cut. | **[measured]** 2 of the top 40 end mid-sentence (next line starts with "..."). | Treat "..." as open only if the next event is not a continuation by the same speaker. See C2. |
| B5 | High | **Hook ignores visuals.** Camera lines (47% of all events) are excluded from `head` and from the judge input. Practice says the first frame must work muted (I5). | `features.py:49, 59`; `features.py:108-109` | Include camera and action text in the hook view; add a `visual_hook` feature (an insert or close-up on a shocking image, a physical act). Send camera lines to the judge. |
| B6 | High | **`standalone` is unreliable.** One character appears under up to 4 speaker labels (Alexander is "THE BEARDED STRANGER", "MYSTERIOUS BEARDED MAN", "SCARRED BEARDED MAN", "ALEXANDER"), which inflates "distinct speakers". Mentioning the protagonist by name is penalised. Relationship words that *help* a cold viewer ("sister", "father", "husband") are ignored. | **[measured]** label counts per character | Normalise speaker labels to character IDs with an alias table built once per show (LLM-assisted, human-confirmed). Replace the score with a judge question: "Could a new viewer say who wants what from whom?" |
| B7 | High | **No spoiler control** (I4). Late-season reveals are preferred, not avoided. | Ep 42, 43 and 45 picks in top 40 | Add a per-show "spoiler horizon" (e.g. no promo from the last 20% of the season, or from beats tagged as reveals) and a `spoiler_risk` judge field. Make it configurable: some reveal promos are intentional (the show title already gives away "He was a Duke"). |
| B8 | Medium | `resolution` reads only emotion tags on the last 3 lines and ignores action ("walk hand-in-hand into the sunlight"). | `features.py:80-81` | Include action/camera text; use the judge's cliffhanger score as the primary resolved/unresolved signal. |
| B9 | Medium | `clean_entry` gives 0.5 to *any* clip starting on dialogue, so it barely separates clean from messy starts. | `features.py:65-66` | 1.0 at a scene start or after a pause of 1 s or more; 0 on a continuation fragment (C2). |
| B10 | Medium | `dur_fit` hard-codes a 25–55 s plateau with no source. Paid social often favours shorter cuts, and the brief allows 20–60 s. | `features.py:56` | Make the target band config per platform (e.g. TikTok 20–35 s, Reels 20–45 s) and learn its weight from feedback. |
| B11 | Medium | **Emotion depends on parenthetical tags** (99% coverage in this pack). A script without tags zeroes `emotion_intensity`, `hook_arousal` and `resolution`. | **[measured]** tag coverage 99% | Fall back to a text-based emotion score from the judge when tags are missing. Report tag coverage in the run manifest. |

### C. Candidate generation and selection

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| C1 | Critical | **Linear scoring lets one strong part carry a clip.** A promo needs hook AND payoff AND clarity; a weighted sum lets a huge hook hide a flat ending: the "clickbait" pattern of I2. | `model.py` (`logit`) | Gate on minimums (e.g. hook ≥ 5 and cliffhanger ≥ 5 from the judge) before ranking, or add interaction terms (`hook × cliff`), or rank on the geometric mean of the three rubric scores. |
| C2 | High | **Promos can start mid-sentence.** The script splits one line across events ("I said..." / "...let her go."), and windows may start on the second half. | **[measured]** 3 of the top 40 start on a "..." fragment (e.g. "...husband and wife!"); 334 candidates (2%) do. | Merge continuation events into one utterance at parse time; never start on a continuation; never end on a line whose continuation follows. |
| C3 | High | **No pre-roll/post-roll handles.** Cuts land exactly on the first syllable and last frame. | `candidates.py` (`windows`) | Output both the content boundary and a suggested edit point with 0.3–0.5 s handles, clamped to the scene. |
| C4 | High | **Stitched promos are broken in practice.** Heads and tails are picked globally by crude sums, so all stitches come from 3 episodes; there is no check that the tail pays off the head's question; none reached the top 40. | **[measured]** 30 stitched rows judged, 3 distinct episodes | Pick heads and tails per story thread (same characters, same conflict), require head ep ≤ tail ep, and ask the judge whether the pair reads as one story. Until then, disable stitching and say so. |
| C5 | High | **The candidate pool is chosen by the champion model** (prefilter uses the current weights with neutral judge scores). A challenger can only re-rank what the champion liked, so the gate is biased toward the champion and the pool never sees what a better model would find. | `pipeline.py:40-51` | Prefilter on model-agnostic features (or the union of champion and challenger top-N), and re-run featurize when a model is promoted. Record the prefilter model in the manifest. |
| C6 | Medium | **Diversity is only a per-episode cap.** Two promos of the same beat (Ep 42 and Ep 43 are both "guards refuse to arrest the Duke") can both rank; hook types are not balanced (I7, I8). | `pipeline.py:66-93` | Add maximal marginal relevance over story beats (character set + conflict), and label each promo with a hook type so the output can be balanced to a target mix. |
| C7 | Medium | Prefilter quota (4 per episode, 200 total) truncates episodes with several strong moments before the judge sees them. | `pipeline.py:44-51` | Use a score threshold plus a minimum per episode instead of a fixed quota; log what was cut. |
| C8 | Low | Window targets are a fixed list of durations; candidates between targets are skipped. | `candidates.py` | Fine as an approximation; note it, or sample ends at every beat within the band. |

### D. Judge

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| D1 | Critical | **The live LLM judge has never run.** Every reported number uses the offline judge, which reuses the same lexicons as the features (double counting) and made the prior worse (NDCG 0.45 → 0.39). | README results table | Run the LLM judge on the full pack (free tier), compare it to the gold set, and report both. Until then, label all numbers "offline judge". |
| D2 | High | **No retry or backoff** on 429s and timeouts; free tiers rate-limit hard. One failure silently falls back to the offline judge, so one run can mix two judges with different scales. | `judge.py` (`LLMJudge.__call__`) | Exponential backoff with jitter; a concurrency limit; fail the run (or mark rows unjudged) instead of mixing; record the per-row judge source counts in the manifest. |
| D3 | High | **Absolute 0–10 scoring.** LLM absolute scores drift and cluster; pairwise comparison ("which clip makes a better promo?") is more stable for ranking, and position bias must be controlled. | `judge.py` RUBRIC | Score absolutely for gating, then re-rank the top 30 with pairwise judgements in both orders (swap positions to cancel position bias). Add 3–5 few-shot anchors drawn from editor-approved clips. |
| D4 | Medium | Prompt version is a hand-edited constant; changing the rubric without bumping it serves stale cached scores. The manifest hashes the whole `judge.py` file instead of the rubric. | `judge.py` `PROMPT_VERSION`; `pipeline.py:111` | Derive the cache key and manifest field from `sha256(RUBRIC)`. |
| D5 | Medium | JSON extraction uses a greedy `\{.*\}` regex, so a reply containing two objects or any brace in the prose breaks. No schema validation of the fields. | `judge.py` | Use the provider's JSON mode or response schema; validate keys and ranges; retry once on parse failure. |
| D6 | Medium | No cost or latency logging, though the design doc lists cost per show as a metric. | `pipeline.py` | Count calls, tokens and wall time per run; write them to the manifest and `eval.json`. |

### E. Learning loop

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| E1 | Critical | **Simulated feedback and gold come from the same labeller.** `simulate` builds positives from the gold regions themselves (`sim-gold`), so the model learns the builder's tail bias and is then scored against that same bias. The "0.39 → 0.59 held-out" gain is partly circular. | `cli.py:85-88`; gold is 41/44 tails | Present the loop as a mechanics demo only. Get labels from a second person (or the live LLM judge as an independent proxy) for training, and keep editor-labelled gold strictly for evaluation. |
| E2 | High | `sim-other` assumes one promo per episode: every other pick in an episode with a positive gold item is labelled negative. Real editors cut several promos per episode. | `cli.py:83-84` | Leave non-matching picks unlabelled, or label them only where gold marks the region `bad`. |
| E3 | High | **Pointwise labels for a ranking task.** Accept/reject on single clips ignores that editors choose between alternatives. | `model.py` (`train`) | Train pairwise (accepted > rejected within the same episode or session), e.g. a logistic loss on score differences. |
| E4 | Medium | CV folds are `episode % 5`, so adjacent episodes (same storyline) land in different folds and leak into each other. The episode parser for gold ids (`"g12t"`) is a string hack. | `learn.py:39-40, 48` | Fold by contiguous episode blocks; store `episode` explicitly on every feedback record. |
| E5 | Medium | No time decay or per-editor weighting; a year-old label counts as much as today's, and one editor's taste dominates. | `learn.py` | Weight labels by recency and track agreement per editor; review editors with low agreement. |
| E6 | Medium | Prior weights (judge weights 1.5) were never validated and the offline judge hurt the prior. | `model.py` `PRIOR` | Fit the prior once on a small independent labelled set; record where each prior value came from. |

### F. Evaluation

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| F1 | Critical | **Gold is one person's labels, 44 items, 41 of them episode endings.** The metric rewards ending at episode end, which is why "last 40 s of every episode" is a strong baseline. | **[measured]** gold modes: 41 tail, 3 head | Collect 100+ regions from 2–3 editors, including mid-episode moments; report inter-annotator agreement (Cohen's kappa); stratify the gold by position in the episode. |
| F2 | High | **The held-out gold is reused for every gate decision.** Repeated promotion against one test block overfits it. | `learn.py:65-77` | Freeze a final test set that only release reviews touch; gate on a rotating validation block; log how many times each block was used. |
| F3 | High | **No uncertainty.** Each metric rests on about 10 picks; one pick moves precision by 0.1, yet the gate promotes on any increase. | README | Bootstrap confidence intervals over picks and gold; promote only if the lower bound of the improvement is above zero. |
| F4 | High | **Unlabelled picks count as misses**, so a genuinely good clip the labeller never marked lowers precision. | `evaluate.py` `metrics` | Report three buckets (matched-good, matched-bad, unlabelled); send unlabelled top picks to a reviewer and add them to gold (active labelling). |
| F5 | High | **Promised checks are missing.** The design doc lists hook-in-3s, cliffhanger, standalone-proxy, cost and latency metrics; the code has none of them. | `DESIGN.md` §5 vs `evaluate.py` | Implement them, or remove them from the doc. The hook and cliffhanger checks should come from an independent judge, not the same lexicons as the features. |
| F6 | Medium | Matching by containment (≥ 80% inside a gold region) lets a clip that starts mid-scene count as a hit; boundary quality is reported but never gated. | `evaluate.py:31-40` | Gate on start-point quality separately: the clip's start within N seconds of an editor's start, or a judge check of the first 3 s. |
| F7 | Medium | Baselines are unfair: the random baseline samples from the champion's pre-filtered pool, and the episode-end baseline is ranked by episode number. | `evaluate.py` baselines | Sample random windows from all candidates; order the episode-end baseline by a neutral score, or report order-free metrics for it. |
| F8 | Medium | **Train-block numbers for v0001 are in-sample** (it was trained on those labels) but printed next to the other systems as if comparable. | `cli.py` `cmd_eval` | Mark in-sample rows, or hide train-block metrics for models trained on that block. |
| F9 | Medium | The failure report lists only top picks that hit `bad` gold. It omits misses: must-have regions the model never ranked. | **[measured]** v0001 misses the Ep 45 must ("freed and crowned, then marriage declared void") | Add a "missed must" section and say whether the region was absent from the pool or ranked too low. |
| F10 | Medium | **No link to real performance.** Offline gold is a proxy; the industry scores promos on hook rate, hold rate, CTR and cost per install (I1, I2, I8). | — | Plan an online loop: post promos, pull hook rate, hold rate and cost per install per promo, and add them as a second label source with its own weight; flag high-hook/low-hold clips as clickbait. |

### G. Parser and portability

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| G1 | High | **Episode split is hard-coded**: `SPLIT = 31` assumes a 50-episode show. A 20-episode show at the interview would put nothing in the training block. | `cli.py:14` | Derive the split as a fraction of episodes (e.g. last 40% held out), from config. |
| G2 | Medium | Parser is tied to this pack's format (bold headers, `Start:`/`End:` lines, `episode-NNN` filenames, UTF-8). Untimed blocks are dropped silently. | `parse.py` | Validate the input and report counts of dropped or untimed blocks; keep `parse.py` as the only format-specific file and add a second adapter (e.g. SRT + scene list) to prove it. |
| G3 | Medium | Speaker labels are not normalised across episodes (see B6). | **[measured]** | Character alias table per show, as in B6. |
| G4 | Low | Paths (`models/`, `runs/`) are relative to the working directory. | `model.py`, `learn.py` | Resolve paths from a config or a `--home` flag. |

### H. Output and explanation

| # | Sev | Problem | Evidence | Fix |
|---|---|---|---|---|
| H1 | High | **The "why" for every offline-judged promo is the same template**, claiming a conflict opening and an unresolved ending whether or not that is true. | **[measured]** 1 of the top 40 claims "ends unresolved" with `cliff_open_end = 0` | Build the explanation from the actual top-contributing features and the hook/cliffhanger text, or leave it blank when no judge reason exists. |
| H2 | High | **Ranking has no posting strategy.** The brief asks which to post first and why. Practice: lead cold audiences with the premise (Ep 1 betrayal), then escalate; rotate hook types; keep spoilers out (I4, I7, I8). | `pipeline.py` `rank` | Add a posting plan on top of the ranking: premise promos first, a hook-type mix, a spoiler gate, and 5+ variants per ad set. |
| H3 | Medium | Scores look like probabilities (0.9446–0.9871, 35 of the top 40 above 0.95) and are shown to 4 decimals. | **[measured]** | Show rank and score bands (A/B/C), or calibrate scores on editor accept rates (Platt scaling). |
| H4 | Medium | Timestamps are printed as raw seconds; editors work in mm:ss:ff. | `pipeline.py:103` | Print `mm:ss.ff` alongside seconds. |
| H5 | High | **Safety covers sexual content only** (I9). Graphic violence, torture and "suffering" imagery are not flagged, though Meta bans them in ads and requires permission plus 18+ targeting for TV promos. | **[measured]** 12 of the top 40 promos contain branding-iron, blood, gash or gallows language; 0 flagged | Add a violence/torture/self-harm category; prefer the judge with a policy rubric over word lists; output a per-promo policy-risk label (none / review / block) instead of a silent drop. |

### I. Engineering and production readiness

| # | Sev | Problem | Fix |
|---|---|---|---|
| I-1 | Medium | No HTTP service, scheduler or editor UI, though "production" was the brief to the builder. | Wrap `run` / `feedback` / `retrain` in a small API (FastAPI or a stdlib server); keep the CSV as a fallback. |
| I-2 | Medium | No logging, only `print`; no run IDs linking promos, rows, model and feedback. | Use `logging`; give every run an ID and write it into every artifact. |
| I-3 | Medium | Tests depend on `runs/r1/rows.jsonl` existing and on the confidential scripts being present; they cannot run in CI. | Generate a tiny synthetic 3-episode fixture in `tests/`; build rows inside the test. |
| I-4 | Low | Constants scattered (top-n, per-episode caps, thresholds, lexicons). | One `config.toml` read at startup and written into the manifest. |
| I-5 | Low | Unused import (`sys` in `cli.py`); `features.extract` is a pass-through wrapper. | Remove. |
| I-6 | Low | Stitched promos are capped per episode against every episode they touch, which crowds out single-episode picks. | Count a stitch against its first episode only. |
| I-7 | Low | Judge calls run serially (about 200 calls per show). | Bounded thread pool once D2's backoff exists. |
| I-8 | Low | `rows.jsonl` stores no feature-schema or prefilter-model version, so stale rows are reused after code changes (see A4). | Version the rows file and refuse to load a mismatched schema. |

---

## Fix order

1. **Stop data loss and false claims:** A1, A2, H1, plus a README note that every number uses the offline judge and builder-written gold.
2. **Fix the signals that are measurably wrong:** B1, B2, B3, B4, C2, H5.
3. **Make the scoring match the brief:** C1 (hook AND payoff), B5 (visual hooks), B7 (spoilers), D1 (run the live judge).
4. **Make the evaluation honest:** F1 (more, independent, mixed-position gold), E1/E2 (separate training labels from gold), F3 (confidence intervals), F4, F9.
5. **Make the loop sound:** C5 (model-agnostic pool), E3 (pairwise), A3/A4 (dedupe, schema versions), F2 (frozen test set).
6. **Portability and production:** G1, G2, B6, I-1 to I-3, D2–D6.

## What holds up

- Code owns every timestamp; the hard checks found 0 boundary, duration or overlap violations across all systems.
- The gate refused a challenger that added known-bad picks, as designed, and a human override exists.
- Rows are cached, so retraining and evaluation re-rank without re-judging.
- Baselines and an ablation are reported next to the model, which is what exposed the weak offline judge.
