# Evaluation Metrics — How Every Number Is Calculated

This document defines every metric produced by [`evaluate.py`](evaluate.py) and says
exactly how it is computed, so any number in the results can be traced back to a
formula. Nothing here is estimated or hand-tuned.

---

## 1. The workflow

```
   UI (LectureForge)                     evaluate.py
   ─────────────────                     ───────────
   Generate 200 MCQ per pipeline   →     judge each question   (quality)
   (agentic / plain / CoT / PoT)   →     re-solve each question (answer check)
   saves bulk_eval_questions.json  →     score models on gold  (calibration)
                                         → rows.csv, results.json, charts
```

**Step 1 — generate from the UI.** Run the bulk generation. It writes
`bulk_eval_questions.json`, whose `questions` map is keyed
`"<pipeline>:<difficulty>"` → list of questions.

**Step 2 — evaluate with the script.**

```bash
python evaluate.py \
  --from-bulk bulk_eval_questions.json \
  --content limits_chapter.txt \
  --gold gold_limits_100.json \
  --out eval_out
```

`--from-bulk` regroups every pipeline automatically. The difficulty stamped on each
question is the one it was **requested at** (from the key), not the model's
self-reported `difficulty` field — a model's own difficulty label is not
trustworthy, and `difficulty_match` must be scored against the *target*.

Outputs: `rows.csv` (one row per question), `results.json` (aggregates),
`gold.json`, `chart_*.png`.

---

## 2. Two things are being measured

| Axis | Question it answers |
|---|---|
| **Question quality** | Is this a good exam question? |
| **Answer correctness** | Is the option the generator *marked* actually the right one? |

A question can be beautifully written and still have the wrong answer key, so the
two are measured separately.

---

## 3. Judge metrics (LLM-as-judge, 1–5)

An **independent judge model** (`gemma3:12b` by default — deliberately *not* the
model that wrote the question) scores each question. It sees the source content,
the question, the options, and the marked answer.

Five dimensions, each an integer **1 (worst) – 5 (best)**:

| Metric | Definition given to the judge |
|---|---|
| `correctness` | factually correct, answerable from the content |
| `clarity` | unambiguous, well-formed wording |
| `difficulty_match` | actual difficulty matches the **target** difficulty |
| `distractor_quality` | the wrong options are **plausible but clearly wrong** |
| `overall` | holistic quality |

**Distractor quality** deserves a note, since it is MCQ-specific: an option set is
*bad* if the distractors are obviously silly (nobody would pick them → the question
tests nothing) **or** if a distractor is arguably also correct (ambiguous). A 5
means every wrong option is tempting to a student who has a specific
misunderstanding, yet is defensibly wrong.

The judge is told to **use the full range** and not default to 5 (5 = flawless,
4 = good, 3 = ordinary, 2 = weak, 1 = poor), because LLM judges otherwise saturate
at the top of the scale.

**Aggregation.** For a group of `N` questions (one pipeline at one difficulty), each
`mean_*` metric is the arithmetic mean:

```
mean_overall = (1/N) · Σ overall(qᵢ)
```

and likewise for correctness, clarity, difficulty_match, distractor_quality.

**Fail-closed.** If the judge call fails or returns unparseable output, the question
is scored **0 on every dimension and rejected**. A broken call can never
accidentally look like a good question.

---

## 4. Accept rate

The judge also returns a binary `accept` — "would you put this on a real exam?"

```
accept(q) = judge_says_accept(q)  AND  overall(q) ≥ 3
```

The `overall ≥ 3` clause is a consistency guard: the judge cannot accept a question
it just scored 1 or 2.

```
                 number of accepted questions
accept_rate  =  ──────────────────────────────
                          N
```
Range 0–1.

---

## 5. Duplicate rate

A per-question judge scores each question **in isolation**, so it physically cannot
see that a set asks the same thing ten times. Repetition is therefore measured
separately, without any LLM.

For each question `i`, compare its text to every **earlier** question `j < i` using
Python's `difflib.SequenceMatcher` ratio (a normalised similarity in 0–1):

```
is_duplicate(qᵢ)  =  ∃ j < i  such that  similarity(qⱼ, qᵢ) > 0.85

                    number of questions that duplicate an earlier one
duplicate_rate  =  ────────────────────────────────────────────────────
                                        N
```

The first occurrence is never counted — only the repeats. Threshold `0.85` was
chosen so that paraphrases count as duplicates but genuinely different questions on
the same topic do not.

> **Caveat.** Template-generated reference sets (`cot_limits_200.json`,
> `pot_limits_200.json`, `gold_limits_100.json`) score ~0.87 here, because
> `lim (x→2) (3x + 1)` and `lim (x→3) (2x + 4)` are string-similar. That is a
> property of those synthetic answer-key sets, **not** a defect measurement. Do not
> report their duplicate rate as if they were model-generated question sets.

---

## 6. Effective accept rate (the ranking metric)

A set of 100 questions where 90 are accepted but 70 are repeats is not a set of 90
usable questions. The accept rate alone therefore over-rewards a repetitive
generator. We discount it by the redundancy:

```
effective_accept_rate = accept_rate × (1 − duplicate_rate)
```

Interpretation: *the fraction of the set that is both exam-worthy and not a repeat.*
This is the headline number for ranking pipelines, because it is the only one that
punishes a generator for padding the set with rephrasings of one question.

---

## 7. Grounding (embedding-based, no LLM)

Is the question actually **about the source material**, or did the model invent a
topic that isn't in the chapter?

Using a sentence-embedding model (`all-MiniLM-L6-v2`, normalised vectors):

1. Split the source text into 500-character chunks `c₁ … c_m`.
2. Embed every question `qᵢ` and every chunk.
3. For each question take its **best-matching** chunk (max cosine).
4. Average over the set.

```
                 1      N
grounding  =  ─────  ·  Σ   max  cos( e(qᵢ), e(cⱼ) )
                 N     i=1   j
```

Range 0–1 (higher = better anchored in the material). A question about net present
value in a limits chapter scores low here even if it is a perfectly good question in
the abstract.

Requires `--content`; if the source text or `sentence-transformers` is unavailable,
grounding is reported as `null` rather than guessed.

---

## 8. Diversity (embedding-based, no LLM)

Duplicate rate catches *near-identical wording*. Diversity catches the subtler case:
questions that are worded differently but all probe the same idea.

For each question, find its **nearest neighbour** among the other questions (max
cosine, excluding itself), average that, and invert:

```
                       1      N
diversity  =  1  −  ─────  ·  Σ   max   cos( e(qᵢ), e(q_k) )
                       N     i=1  k ≠ i
```

Range 0–1. **Higher = more diverse.** A set where every question is a variation on
one theme has a high mean nearest-neighbour similarity → low diversity, even if no
two questions are string-duplicates.

Diversity and duplicate rate are complementary: duplicate rate is lexical and
strict; diversity is semantic and graded.

---

## 9. Answer correctness — three independent checks

Question quality says nothing about whether the **marked answer key is right**. We
check it three ways, deliberately using different evidence.

### 9.1 Judge-verify → `verified_correct_rate`

While scoring quality, the judge (which *can* see the marked answer and the source
content) is asked one extra question: *is the marked answer actually the correct
answer?* → boolean `answer_correct`.

```
verified_correct_rate = (number with answer_correct = true) / N
```

Cheap (no extra model call) but **not independent** — the judge is anchored by
seeing the answer it is asked to check.

### 9.2 Independent re-solve → `independent_match_rate`

This is the **second agent**. A separate *Answer Generator* is given the question
stem and the four options **with the marked answer removed**, and must solve it from
scratch. Its choice is then compared to what the generator marked:

```
independent_match(q) = ( solver_choice(q) == marked_answer(q) )

independent_match_rate = (number where they agree) / N
```

Because the solver never sees the marked answer, agreement is real evidence — but
it is **agreement, not truth** (see §9.3).

### 9.3 Gold calibration → `gold.json`

This is the check that makes §9.2 *interpretable*, and it is the one people forget.

If the solver agrees with the generator 90% of the time, that is only meaningful if
the solver is actually good at solving these problems. So we run both models over a
set of MCQ whose answers are **known and verified** (`gold_limits_100.json` —
answers computed with SymPy, not hand-typed):

```
                        number the model answers correctly
gold_accuracy(model) = ─────────────────────────────────────
                            number of gold questions
```
Reported overall and per difficulty, for each model.

**How to read it together:** if gold accuracy is only ~55%, then a 90%
generator↔solver agreement cannot be read as "90% of answers are correct" — two
models sharing the same blind spots will agree while both being wrong. The honest
claim is: *independent-match rate is an agreement rate, whose evidential weight is
bounded by the solver's gold accuracy.*

### 9.4 Ground-truth scoring (`--trusted`)

If a set's answers are themselves verified (our SymPy-generated sets), pass
`--trusted <name>` and the script additionally reports, against actual truth:

- `marked_vs_truth_rate` — how often the marked key is genuinely correct
- `solver_vs_truth_rate` — how often the Answer Generator is genuinely correct

---

## 10. Validator–judge agreement (F₁) — agentic pipeline only

The agentic pipeline has its **own internal validator** that accepts/rejects each
question during generation. Is that self-check any good? We test it against the
independent judge.

Treat the judge's `accept` as ground truth and the validator's `PASS` as the
prediction:

| | judge accepts | judge rejects |
|---|---|---|
| **validator PASS** | TP | FP |
| **validator FAIL** | FN | TN |

```
precision = TP / (TP + FP)      "when the validator passes a question, is it good?"
recall    = TP / (TP + FN)      "does the validator catch the good questions?"
F₁        = 2·precision·recall / (precision + recall)
```

High precision + low recall = a **conservative** validator: it never waves through a
bad question, but it throws away many good ones (wasting retries). That is the
usual failure mode and is worth reporting explicitly.

Only the agentic pipeline has an internal validator, so this metric does not exist
for plain / CoT / PoT.

---

## 11. Summary table

| Metric | Range | Source | Higher is better |
|---|---|---|---|
| `mean_correctness` | 1–5 | LLM judge | ✅ |
| `mean_clarity` | 1–5 | LLM judge | ✅ |
| `mean_difficulty_match` | 1–5 | LLM judge | ✅ |
| `mean_distractor_quality` | 1–5 | LLM judge | ✅ |
| `mean_overall` | 1–5 | LLM judge | ✅ |
| `accept_rate` | 0–1 | LLM judge | ✅ |
| `duplicate_rate` | 0–1 | difflib, no LLM | ❌ (lower better) |
| `effective_accept_rate` | 0–1 | derived | ✅ (ranking metric) |
| `grounding` | 0–1 | embeddings | ✅ |
| `diversity` | 0–1 | embeddings | ✅ |
| `verified_correct_rate` | 0–1 | LLM judge | ✅ |
| `independent_match_rate` | 0–1 | 2nd agent | ✅ (bounded by gold) |
| `gold_accuracy` | 0–1 | vs known answers | ✅ (calibrates the above) |
| `precision` / `recall` / `f1` | 0–1 | validator vs judge | ✅ (agentic only) |

---

## 12. Known limitations (state these when reporting)

1. **The judge is an LLM, not a human rater.** Absolute scores are a *relative*
   signal for comparing pipelines on identical content, not an objective quality
   measure.
2. **Judge scores saturate.** Even with the "use the full range" instruction, good
   questions cluster at 4.8–5.0, so `mean_overall` often cannot separate pipelines.
   This is exactly why `duplicate_rate` and `effective_accept_rate` exist — they are
   the metrics that actually discriminate.
3. **Generator and judge share a model family** (Gemma-3). Using a *larger*,
   separate judge mitigates self-grading bias but does not eliminate it.
4. **Answer accuracy is agreement, not truth**, unless it is measured against the
   gold set (§9.3). Always report the gold accuracy next to it.
5. **Single source chapter.** Per-cell numbers should be read with their `N`.

---

## 13. Where each metric lives in the code

All in [`evaluate.py`](evaluate.py):

| Metric | Function |
|---|---|
| judge dimensions, accept, answer_correct | `judge_question()` |
| independent answer | `solve_mcq()` |
| gold calibration | `evaluate_on_gold()` |
| duplicate rate | `duplicate_rate()` |
| grounding, diversity | `grounding_and_diversity()` |
| precision / recall / F₁ | `f1()` |
| all `mean_*`, rates, effective accept | `aggregate()` |
