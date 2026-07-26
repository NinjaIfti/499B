#!/usr/bin/env python3
"""
evaluate_requesta.py — ReQUESTA-rubric evaluation harness.

Scores generated MCQs along the EXPERT RUBRIC used in "Cognitively Diverse
Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large
Language Models" (Tian et al., arXiv:2602.03704), as a second, independent
rubric alongside evaluate_eqgbench.py.

Seven criteria, on the paper's two scales:

  BINARY (0 = Disagree, 1 = Agree)
    answer_correctness       correct answer is clearly correct to a reader who
                             has comprehended the text
    distractor_incorrectness distractors are clearly incorrect to a reader who
                             has comprehended the text

  FOUR-POINT (1 = Disagree, 2 = Neutral, 3 = Agree, 4 = Strongly Agree)
    topic_relevance          addresses a CENTRAL concept in the text, not a
                             peripheral or trivial detail
    writing_clarity          stem and options are concise and unambiguous
    distractor_linguistic    distractors match in length, clause count and
                             surface structure
    distractor_plausibility  distractors are reasonable to a reader who did NOT
                             fully comprehend the text
    distractor_uniqueness    each distractor carries a distinct meaning; none
                             are eliminable as duplicates of each other

The 4-point scale is deliberate in the paper: it affords more nuance than a
binary or 3-point scale AND has no neutral midpoint, so a rater cannot default
to the middle. That property is preserved here.

TWO THINGS THIS SCRIPT CANNOT DO — state them wherever results are published:

  1. It is an LLM judge. Tian et al. applied this rubric with TWO HUMAN EXPERTS,
     blinded to question source, calibrated over three rounds on 100 practice
     items, reporting Cohen's weighted kappa .80-1.00. Same criteria, different
     instrument. Do not describe these scores as expert ratings.
  2. It produces NO psychometrics. The paper's headline measures — item
     difficulty (p), discrimination (D) and point-biserial correlation — are
     computed from real learner responses (n=572). No judge can substitute for
     that; those cells stay empty until humans answer the questions.

FAIL-CLOSED, BUT NOT SILENTLY: a question whose judging rounds all fail is
written with rounds_ok=0 and is EXCLUDED from aggregation rather than being
folded in at the worst score. Corrupted or rate-limited rows therefore cannot
quietly drag the means down; the count of excluded rows is always reported.

USAGE (same plumbing as evaluate_eqgbench.py):
  $env:GROQ_API_KEY = "gsk_..."          # never hard-code the key
  python evaluate_requesta.py --from-bulk bulk_eval_clean.json \
      --content chapter.txt --out eval_out_requesta \
      --base-url https://api.groq.com/openai/v1 \
      --api-key-env GROQ_API_KEY --judge-model openai/gpt-oss-120b --sleep 5

Rows are checkpointed to rows.jsonl after every question, so an interrupted run
RESUMES rather than restarting.
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

import requests

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible plumbing (identical to evaluate_eqgbench.py)
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.deepseek.com"
API_KEY = ""
REQUEST_SLEEP = 0.0


def _extract_json(text):
    """First JSON object in a reply; tolerates ``` fences, prose, <think> blocks."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    start = None
    return None


def api_json(prompt, model, system="", retries=3):
    """One chat call -> parsed JSON dict, {} on failure (fail closed)."""
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": messages, "temperature": 0.6,
            "max_tokens": 4096, "stream": False}
    backoff = 2.0
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=600)
            if r.status_code == 429:
                raise requests.HTTPError("429 rate limit")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content", "")
            data = _extract_json(content)
            if isinstance(data, dict):
                if REQUEST_SLEEP:
                    time.sleep(REQUEST_SLEEP)
                return data
        except Exception as e:
            if attempt == retries:
                print(f"    [warn] judge call failed: {e}", file=sys.stderr)
                break
            time.sleep(backoff)
            backoff *= 2
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# ReQUESTA rubric
# ─────────────────────────────────────────────────────────────────────────────

BINARY_DIMS = ["answer_correctness", "distractor_incorrectness"]
SCALE_DIMS = ["topic_relevance", "writing_clarity", "distractor_linguistic",
              "distractor_plausibility", "distractor_uniqueness"]
DIMS = BINARY_DIMS + SCALE_DIMS

DIM_LONG = {
    "answer_correctness":       "Answer Correctness (0-1)",
    "distractor_incorrectness": "Distractor Incorrectness (0-1)",
    "topic_relevance":          "Topic Relevance (1-4)",
    "writing_clarity":          "Writing Clarity (1-4)",
    "distractor_linguistic":    "Distractor Linguistic Features (1-4)",
    "distractor_plausibility":  "Distractor Semantic Plausibility (1-4)",
    "distractor_uniqueness":    "Distractor Semantic Uniqueness (1-4)",
}

# Worst legal value per scale — used only to describe the scales in the prompt,
# never written into an aggregated row (failed rows are excluded instead).
DIM_RANGE = {d: (0, 1) for d in BINARY_DIMS}
DIM_RANGE.update({d: (1, 4) for d in SCALE_DIMS})


def judge_once(q, content, model):
    """One ReQUESTA-rubric judging round. Returns {dim: int} or {} on failure."""
    opts = q.get("options") or {}
    opts_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    marked = str(q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]
    topic = q.get("topic") or "(not stated)"

    prompt = f"""You are an expert rater of educational assessment items. Score this ONE
multiple-choice question against the rubric below. Be strict: reserve the top
score for genuinely excellent performance on that criterion.

Reference material (ground truth):
{content[:4500]}

Intended topic: {topic}

Question item:
Q: {q.get('question', '')}
Options:
{opts_txt}
Marked correct answer: {marked}

BINARY criteria — score 0 (Disagree) or 1 (Agree):
1. answer_correctness: the marked correct answer is clearly correct to a reader
   who HAS comprehended the reference material.
2. distractor_incorrectness: every distractor is clearly incorrect to a reader
   who HAS comprehended the reference material.

FOUR-POINT criteria — score 1 (Disagree), 2 (Neutral), 3 (Agree) or
4 (Strongly Agree). There is no neutral midpoint by design; do not default to 2.
3. topic_relevance: 1 = does not address a central concept; 2 = addresses a
   peripheral or partially relevant point; 3 = addresses a central concept in
   the material; 4 = addresses a central concept very directly and importantly.
4. writing_clarity: 1 = stem and options are excessively wordy or unclear;
   2 = wordiness or mild ambiguity that slightly impedes comprehension;
   3 = concise with minor redundancies; 4 = fully clear and concise.
5. distractor_linguistic: similarity of the distractors to the correct answer in
   LENGTH, number of clauses and surface structure. 1 = visibly different (e.g.
   much longer or shorter); 2 = noticeable differences or one major outlier;
   3 = relatively similar length and surface features; 4 = uniform and closely
   matched.
6. distractor_plausibility: 1 = distractors are implausible or obviously wrong;
   2 = some are implausible or uneven; 3 = plausible to a reader who has NOT
   comprehended the material; 4 = all highly plausible and well matched.
7. distractor_uniqueness: the extent to which each distractor conveys a DISTINCT
   meaning. 1 = several distractors are near-duplicates and easy to eliminate
   together; 4 = every distractor carries a clearly unique meaning.

Return STRICT JSON only, with exactly these keys:
{{"answer_correctness":0,"distractor_incorrectness":0,"topic_relevance":1,
"writing_clarity":1,"distractor_linguistic":1,"distractor_plausibility":1,
"distractor_uniqueness":1}}"""

    r = api_json(prompt, model,
                 system="You are a strict educational assessment rater. Return only JSON.")
    if not r:
        return {}
    out = {}
    for d in DIMS:
        lo, hi = DIM_RANGE[d]
        try:
            v = int(float(r.get(d, -1)))
        except Exception:
            return {}
        if not (lo <= v <= hi):
            return {}
        out[d] = v
    return out


def judge_voted(q, content, model, rounds):
    """Per-criterion MODE across rounds; median when rounds fully disagree.

    Returns (scores, n_votes). n_votes == 0 means every round failed; the caller
    records the row but aggregation excludes it, so a rate-limited question can
    never masquerade as a genuinely low-scoring one.
    """
    votes = []
    for _ in range(rounds):
        r = judge_once(q, content, model)
        if r:
            votes.append(r)
    if not votes:
        return {d: DIM_RANGE[d][0] for d in DIMS}, 0
    final = {}
    for d in DIMS:
        vals = [v[d] for v in votes]
        counts = Counter(vals).most_common()
        if len(counts) > 1 and counts[0][1] == counts[1][1]:
            final[d] = int(statistics.median(vals))
        else:
            final[d] = counts[0][0]
    return final, len(votes)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation / I-O
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows):
    """Mean per criterion, per (pipeline, difficulty). Rows whose judging rounds
    all failed (rounds_ok == 0) are EXCLUDED, not scored at the floor."""
    usable = [r for r in rows if r.get("rounds_ok", 0) > 0]

    def block(sub):
        n = len(sub)
        if not n:
            return None
        m = {"n": n}
        for d in DIMS:
            m[d] = round(sum(r[d] for r in sub) / n, 3)
        # Composite over the 4-point criteria only; mixing scales would be
        # meaningless, and the binary pair saturates at 1.00 in practice.
        m["composite_4pt"] = round(sum(m[d] for d in SCALE_DIMS) / len(SCALE_DIMS), 3)
        return m

    pipelines = sorted({r["pipeline"] for r in usable})
    agg = {"overall": {p: block([r for r in usable if r["pipeline"] == p])
                       for p in pipelines}}
    for d in ("easy", "medium", "hard"):
        agg[d] = {p: block([r for r in usable
                            if r["pipeline"] == p and r["difficulty"] == d])
                  for p in pipelines}
    return agg, pipelines, len(rows) - len(usable)


def load_bulk(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    groups = data.get("questions", data if isinstance(data, dict) else {})
    sets = defaultdict(list)
    for key, qs in groups.items():
        pipeline, _, diff = key.partition(":")
        for q in qs:
            if not isinstance(q, dict) or not q.get("question"):
                continue
            if diff:
                q["difficulty"] = diff
            sets[pipeline].append(q)
    return dict(sets)


def print_table(agg, pipelines, excluded):
    print("\n" + "=" * 78)
    print("ReQUESTA-RUBRIC RESULTS (overall)")
    print("=" * 78)
    print(f"{'Criterion':<42}" + "".join(f"{p:>9}" for p in pipelines))
    print("-" * 78)
    for d in DIMS:
        cells = "".join(
            f"{((agg['overall'].get(p) or {}).get(d)):>9}"
            if (agg['overall'].get(p) or {}).get(d) is not None else f"{'-':>9}"
            for p in pipelines)
        print(f"{DIM_LONG[d]:<42}{cells}")
    cells = "".join(f"{((agg['overall'].get(p) or {}).get('composite_4pt', '-')):>9}"
                    for p in pipelines)
    print("-" * 78)
    print(f"{'Composite (mean of 5 four-point items)':<42}{cells}")
    ns = "".join(f"{(agg['overall'].get(p) or {}).get('n', 0):>9}" for p in pipelines)
    print(f"{'N questions scored':<42}{ns}")
    print("=" * 78)
    if excluded:
        print(f"NOTE: {excluded} row(s) excluded from all means (rounds_ok == 0 — "
              f"every judging round failed, e.g. rate limits). Re-run to fill them.")
    print("LLM judge, NOT the paper's two blinded human experts. No item difficulty,"
          "\ndiscrimination or point-biserial here — those need real learner responses.")


def main():
    global BASE_URL, API_KEY, REQUEST_SLEEP

    ap = argparse.ArgumentParser(
        description="Score MCQs on the ReQUESTA expert rubric (arXiv:2602.03704).")
    ap.add_argument("--from-bulk", required=True, metavar="FILE",
                    help="bulk questions JSON produced by the UI")
    ap.add_argument("--content", help="Text file of the source material (ground truth)")
    ap.add_argument("--judge-model", default="deepseek-reasoner",
                    help="judge model id")
    ap.add_argument("--rounds", type=int, default=3,
                    help="independent judging rounds per question (mode voting)")
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", BASE_URL))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls, for rate-limited free tiers")
    ap.add_argument("--out", default="eval_out_requesta")
    ap.add_argument("--limit", type=int, default=0, help="Only first N per pipeline")
    ap.add_argument("--fresh", action="store_true",
                    help="delete any existing rows.jsonl and judge from scratch")
    args = ap.parse_args()

    BASE_URL = args.base_url
    API_KEY = os.environ.get(args.api_key_env, "")
    REQUEST_SLEEP = args.sleep
    if not API_KEY:
        sys.exit(f"No API key. Set it first:  $env:{args.api_key_env} = \"sk-...\"")

    os.makedirs(args.out, exist_ok=True)
    print(f"[backend] {BASE_URL}  judge={args.judge_model}  rounds={args.rounds}")

    content = ""
    if args.content:
        with open(args.content, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(f"[content] {len(content)} chars from {args.content}")
    else:
        print("[content] none given — topic_relevance judged without ground truth (weaker)")

    rows_path = os.path.join(args.out, "rows.jsonl")
    rows, done = [], set()
    if args.fresh and os.path.exists(rows_path):
        os.remove(rows_path)
        print("[fresh] removed existing rows.jsonl")
    elif os.path.exists(rows_path):
        malformed = 0
        with open(rows_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    malformed += 1
                    continue
                rows.append(r)
                # Key includes difficulty: idx restarts per difficulty group, so
                # (pipeline, idx) alone collides and would skip real questions.
                if r.get("rounds_ok", 0) > 0:
                    done.add((r["pipeline"], r.get("difficulty"), r["idx"]))
        print(f"[resume] {len(rows)} rows read, {len(done)} already judged — skipping those")
        if malformed:
            # Loud on purpose: a malformed checkpoint previously looked exactly
            # like an unfinished run, and got silently re-judged.
            print(f"[resume] WARNING: {malformed} malformed line(s) ignored. "
                  f"If this is unexpected, inspect {rows_path} before continuing — "
                  f"they will be RE-JUDGED as if never done.", file=sys.stderr)

    question_sets = load_bulk(args.from_bulk)
    print("[bulk] " + ", ".join(f"{k}={len(v)}" for k, v in question_sets.items()))

    rf = open(rows_path, "a", encoding="utf-8")
    try:
        for name, qs in question_sets.items():
            if args.limit:
                qs = qs[:args.limit]
            print(f"\n[{name}] {len(qs)} questions x {args.rounds} rounds")
            for idx, q in enumerate(qs):
                diff = (q.get("difficulty") or "medium").lower()
                if (name, diff, idx) in done:
                    continue
                scores, n_votes = judge_voted(q, content, args.judge_model, args.rounds)
                row = {
                    "pipeline": name,
                    "difficulty": diff,
                    "idx": idx,
                    "topic": q.get("topic", ""),
                    "question": (q.get("question") or "")[:300],
                    "cognitive_type": q.get("cognitive_type", ""),
                    "rounds_ok": n_votes,
                    **scores,
                }
                rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                rf.flush()
                rows.append(row)
                if (idx + 1) % 5 == 0 or idx + 1 == len(qs):
                    print(f"    {name}: {idx + 1}/{len(qs)} judged")
    finally:
        rf.close()

    if not rows:
        print("Nothing judged.")
        return

    agg, pipelines, excluded = aggregate(rows)
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": "ReQUESTA expert rubric (arXiv:2602.03704): 2 binary + 5 "
                    f"four-point criteria, mode of {args.rounds} judging rounds",
        "instrument_caveat": "Scored by an LLM judge, NOT the paper's two blinded "
                             "human expert raters (weighted kappa .80-1.00). No item "
                             "difficulty / discrimination / point-biserial: those "
                             "require real learner responses.",
        "config": {"backend": BASE_URL, "judge_model": args.judge_model,
                   "rounds": args.rounds, "pipelines": pipelines,
                   "n_rows": len(rows), "n_excluded_failed": excluded,
                   "content_chars": len(content)},
        "aggregated": agg,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cols = ["pipeline", "difficulty", "idx", "topic", "question",
            "cognitive_type", "rounds_ok"] + DIMS
    with open(os.path.join(args.out, "rows.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print_table(agg, pipelines, excluded)
    print(f"\nWrote: {args.out}/results.json, rows.csv, rows.jsonl")


if __name__ == "__main__":
    main()
