# LectureForge — Agentic vs. Non-Agentic Quiz Generation: Evaluation

## 1. Context

LectureForge generates quiz questions (MCQ, True/False, Fill-in-the-blank, Short-answer)
from a processed lecture in two different ways that already exist side by side in
`colab.py`:

- **Agentic pipeline** (`_run_quiz_agentic_loop`) — per-question Plan → Generate →
  Validate → Retry loop. Each question is generated, checked by an internal validator
  (`_validate_one_question`) against difficulty/groundedness/instruction-compliance/
  uniqueness, and retried (up to 3 attempts) if it fails.
- **Non-agentic baseline** (`generate_mcq` / `generate_tf` / `generate_fill` /
  `generate_short`) — a single LLM call that asks for all N questions at once, with no
  per-question validation or retry.

This report defines and presents the evaluation requested by the professor: an
**LLM-as-judge** assessment of question quality, and a head-to-head comparison between
the two pipelines, including a quantitative agreement score (F1) between the agentic
system's own internal validator and an independent judge.

The harness that produces these numbers is `run_quiz_evaluation()` in `colab.py` (see
the "EVALUATION HARNESS" section). **The numbers and charts below are placeholders** —
run the harness in Colab on a real lecture and replace them.

## 2. Methodology

### 2.1 Systems compared
- **Agentic**: `_run_quiz_agentic_loop`, using `QUIZ_MODEL` (`gemma3:4b`) for both
  generation and internal validation.
- **Non-agentic**: the matching legacy bulk generator, also using `QUIZ_MODEL`, in a
  single shot with no retry/validation step.

### 2.2 LLM-as-judge design
Every generated question, from both pipelines, is scored once by `judge_question()`
using an **independent, larger model** — `OLLAMA_MODEL` (`gemma3:12b`) — deliberately
different from the `gemma3:4b` model both pipelines use for generation, to reduce
same-model self-grading bias. The judge scores a 1-5 rubric:

| Dimension | Meaning |
|---|---|
| `correctness` | Factually correct and answerable from the lecture content |
| `clarity` | Unambiguous, well-formed wording |
| `difficulty_match` | Actual difficulty matches the requested target difficulty |
| `distractor_quality` | (MCQ only) wrong options are plausible but clearly incorrect |
| `overall` | Holistic quality |
| `accept` | Binary accept/reject for inclusion in a real exam |

The judge fails **closed** (`accept=False`, scores=0) on any parse error, and `accept`
is forced to `False` whenever `overall < 3`, so accept-rate and mean-score stay
internally consistent.

### 2.3 Fair-comparison controls
Both pipelines run on **identical lecture content** for each trial — the same
`board_text`/`transcript`/`board_entries`/`summary` assembled the same way the
production `/quiz/generate` route does (`_build_eval_ctx()`), at the same
`target_difficulty` ("medium" by default).

### 2.4 Metrics
For each pipeline, per quiz type: mean judge `overall`/`correctness`/`clarity`/
`difficulty_match`/`distractor_quality` (MCQ only), `judge_accept_rate`,
`duplicate_rate` (near-duplicate questions via `difflib.SequenceMatcher` ratio > 0.85,
the same threshold the production validator uses), `avg_attempts` (agentic only — always
1 for non-agentic by construction), `avg_llm_calls_per_accepted_question` (cost proxy),
and `wall_clock_sec`.

**Validator-vs-judge F1 (agentic only)** — treats the judge's `accept` as the
ground-truth label and the agentic pipeline's own internal validator `PASS`/`FAIL` verdict
(captured on the actual attempt that was kept, not the loop's always-"PASS" final status)
as the predicted label:

```
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 · precision · recall / (precision + recall)
```

This answers "how well does our agent's own validator agree with an independent LLM
judge?" — i.e., is the validator a reliable proxy for quality, or is it over/under-accepting
relative to a second opinion?

### 2.5 Sample size
Defaults: `quiz_types=(mcq, tf, fill, short)`, `num_questions=6`, `repeats=2` → 96
judged questions per full run, chosen to stay within reasonable Ollama call volume on a
free-tier Colab T4. All three are parameters to `run_quiz_evaluation()` and can be
increased for a more rigorous run.

## 3. Results (TBD — fill in after running `run_quiz_evaluation()`)

> Run in Colab: `results = run_quiz_evaluation()`. This writes `eval_results.json` and
> 4 PNG charts to `OUTPUT_DIR` (`/content/outputs`). Copy the numbers/images below.

### 3.1 Mean judge score & accept rate

| Quiz type | Agentic mean score | Non-agentic mean score | Agentic accept rate | Non-agentic accept rate |
|---|---|---|---|---|
| MCQ | TBD | TBD | TBD | TBD |
| True/False | TBD | TBD | TBD | TBD |
| Fill-blank | TBD | TBD | TBD | TBD |
| Short-answer | TBD | TBD | TBD | TBD |

![Mean judge score](outputs/eval_chart_mean_judge_score.png)
![Accept rate](outputs/eval_chart_accept_rate.png)

### 3.2 Duplicate rate & cost/latency

| Quiz type | Agentic dup. rate | Non-agentic dup. rate | Agentic avg. attempts | Agentic wall-clock (s) | Non-agentic wall-clock (s) |
|---|---|---|---|---|---|
| MCQ | TBD | TBD | TBD | TBD | TBD |
| True/False | TBD | TBD | TBD | TBD | TBD |
| Fill-blank | TBD | TBD | TBD | TBD | TBD |
| Short-answer | TBD | TBD | TBD | TBD | TBD |

![Duplicate rate](outputs/eval_chart_duplicate_rate.png)

### 3.3 Validator-vs-judge agreement (agentic only)

| Quiz type | Precision | Recall | F1 |
|---|---|---|---|
| MCQ | TBD | TBD | TBD |
| True/False | TBD | TBD | TBD |
| Fill-blank | TBD | TBD | TBD |
| Short-answer | TBD | TBD | TBD |

![Validator vs judge F1](outputs/eval_chart_validator_judge_f1.png)

## 4. Discussion (fill in once results are in)

- Did the agentic pipeline show a higher accept-rate / mean judge score than the
  non-agentic baseline, and at what cost in attempts/latency?
- Was the duplicate rate lower for the agentic pipeline, given its built-in
  existing-questions uniqueness check?
- What does the validator-vs-judge F1 say about the internal validator — does it tend to
  over-accept (low precision) or over-reject (low recall) relative to an independent
  judge?
- Which quiz type showed the biggest agentic-vs-non-agentic gap, and why might that be?

## 5. Limitations

- **The judge is an LLM, not a human rater.** Absolute scores should be read as a
  relative, comparative signal between the two pipelines, not as ground-truth quality.
- **Shared model family.** The generator (`gemma3:4b`) and judge (`gemma3:12b`) are both
  Gemma 3 models. Using a larger, separately-instantiated model as judge reduces but does
  not eliminate shared-bias risk; a fully independent model family would reduce it
  further.
- **Question-count asymmetry.** `generate_tf`/`generate_fill`/`generate_short` hardcode
  their own counts (10/10/8) and ignore `num_questions` entirely — only `generate_mcq`
  honors it. Reported metrics are means/rates specifically so they remain comparable
  despite differing raw counts; raw `n_questions` is reported alongside for transparency.
- **Small, single-lecture sample.** Default settings are bounded for a free-tier T4
  Colab session — results are illustrative. Increase `repeats` and/or run against
  multiple lectures for a stronger claim.
