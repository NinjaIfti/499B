#!/usr/bin/env python3
"""
evaluate_eqgbench.py — EQGBench-protocol evaluation harness.

Scores generated MCQs EXACTLY along the EQGBench protocol ("From Answers to
Questions: EQGBench for Evaluating LLMs' Educational Question Generation",
arXiv:2508.10005) and nothing else:

  * Five dimensions, each scored 0 / 1 / 2  (Poor / Good / Excellent):
      KP  Knowledge-Point Alignment   — does the question target the intended
                                        knowledge point (our per-question topic)?
      QT  Question-Type Alignment     — does the item conform to the requested
                                        question type (a well-formed 4-option MCQ
                                        with exactly one correct answer)?
      QQ  Question Item Quality       — stem clarity, option quality, reasonable
                                        distractors, no ambiguity.
      SQ  Solution Explanation Quality— is the provided explanation correct,
                                        complete and pedagogically useful?
      CG  Competence-Oriented Guidance— does the item cultivate competence /
                                        higher-order thinking rather than rote
                                        recall?
  * Judge = DeepSeek-R1 (deepseek-reasoner), the judge model EQGBench uses.
  * THREE independent judging rounds per question; the per-dimension score is
    the MODE of the three rounds (median if all three disagree), replicating
    EQGBench's 3-round voting.

FAITHFULNESS NOTE: EQGBench has no public code release, so the judge prompt
below is a reconstruction from the paper's dimension definitions, not a copy of
their exact prompt. The dimensions, scale, judge model and voting rule follow
the paper; report this caveat wherever results are published.

COST WARNING: 3 rounds x 600 questions = 1800 reasoner calls. On the free tier
with --sleep for rate limits this can take many hours; rows are checkpointed to
rows.jsonl after every question, so an interrupted run RESUMES, never restarts.

USAGE (same plumbing as evaluate_deepseek.py):
  $env:DEEPSEEK_API_KEY = "sk-..."      # never hard-code the key
  python evaluate_eqgbench.py --from-bulk bulk_eval_questions.json \
      --content chapter.txt --out eval_out_eqgbench

Works with any OpenAI-compatible endpoint via --base-url/--api-key-env
(e.g. Groq's R1 distill for a faster, cheaper approximation — but then the
judge is no longer the paper's exact model; say so if you use it).
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
# OpenAI-compatible plumbing (identical to evaluate_deepseek.py)
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
# EQGBench judging
# ─────────────────────────────────────────────────────────────────────────────

DIMS = ["KP", "QT", "QQ", "SQ", "CG"]

DIM_LONG = {
    "KP": "Knowledge-Point Alignment",
    "QT": "Question-Type Alignment",
    "QQ": "Question Item Quality",
    "SQ": "Solution Explanation Quality",
    "CG": "Competence-Oriented Guidance",
}


def judge_once(q, content, model):
    """One EQGBench-style judging round. Returns {dim: 0|1|2} or {} on failure."""
    opts = q.get("options") or {}
    opts_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    marked = str(q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]
    topic = q.get("topic") or "(not stated)"
    explanation = q.get("explanation") or "(none provided)"
    diff = (q.get("difficulty") or "medium").lower()

    prompt = f"""You are an expert evaluator of educational assessment items. Evaluate this ONE
multiple-choice question along the five EQGBench dimensions. Score each
dimension 0 (Poor), 1 (Good), or 2 (Excellent). Be strict; 2 is reserved for
genuinely excellent performance on that dimension.

Reference material (ground truth):
{content[:4500]}

Intended knowledge point (topic): {topic}
Requested question type: multiple-choice question (exactly 4 options A-D, one correct)
Requested difficulty: {diff}

Question item:
Q: {q.get('question', '')}
Options:
{opts_txt}
Marked correct answer: {marked}
Provided solution explanation: {explanation}

Dimensions:
1. KP (Knowledge-Point Alignment): does the question genuinely target the
   intended knowledge point, grounded in the reference material?
2. QT (Question-Type Alignment): is it a well-formed MCQ of the requested type
   — four plausible options, exactly one defensibly correct answer?
3. QQ (Question Item Quality): is the stem clear and unambiguous, are the
   options well written, are the distractors reasonable?
4. SQ (Solution Explanation Quality): is the provided explanation correct,
   complete, and pedagogically useful for a learner?
5. CG (Competence-Oriented Guidance): does the item cultivate understanding,
   reasoning or application (higher-order competence) rather than rote recall?

Return STRICT JSON only:
{{"KP":0,"QT":0,"QQ":0,"SQ":0,"CG":0}}"""

    r = api_json(prompt, model,
                 system="You are a strict educational assessment evaluator. Return only JSON.")
    if not r:
        return {}
    out = {}
    for d in DIMS:
        try:
            v = int(float(r.get(d, -1)))
        except Exception:
            return {}
        if v not in (0, 1, 2):
            return {}
        out[d] = v
    return out


def judge_voted(q, content, model, rounds):
    """EQGBench 3-round voting: per-dimension MODE across rounds; if all rounds
    disagree on a dimension, take the median. A question where every round
    failed FAILS CLOSED: all dimensions 0."""
    votes = []
    for _ in range(rounds):
        r = judge_once(q, content, model)
        if r:
            votes.append(r)
    if not votes:
        return {d: 0 for d in DIMS}, 0
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
    """Mean per dimension (0-2) and composite mean, per (pipeline, difficulty)."""
    def block(sub):
        n = len(sub)
        if not n:
            return None
        m = {"n": n}
        for d in DIMS:
            m[d] = round(sum(r[d] for r in sub) / n, 3)
        m["composite"] = round(sum(m[d] for d in DIMS) / len(DIMS), 3)
        return m

    pipelines = sorted({r["pipeline"] for r in rows})
    agg = {"overall": {p: block([r for r in rows if r["pipeline"] == p])
                       for p in pipelines}}
    for d in ("easy", "medium", "hard"):
        agg[d] = {p: block([r for r in rows
                            if r["pipeline"] == p and r["difficulty"] == d])
                  for p in pipelines}
    return agg, pipelines


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


def print_table(agg, pipelines):
    print("\n" + "=" * 72)
    print("EQGBENCH RESULTS (overall; each dimension 0-2)")
    print("=" * 72)
    print(f"{'Dimension':<36}" + "".join(f"{p:>9}" for p in pipelines))
    print("-" * 72)
    for d in DIMS:
        cells = "".join(
            f"{((agg['overall'].get(p) or {}).get(d, None)):>9}"
            if (agg['overall'].get(p) or {}).get(d) is not None else f"{'-':>9}"
            for p in pipelines)
        print(f"{d + '  ' + DIM_LONG[d]:<36}{cells}")
    cells = "".join(f"{((agg['overall'].get(p) or {}).get('composite', '-')):>9}"
                    for p in pipelines)
    print(f"{'Composite (mean of 5)':<36}{cells}")
    ns = "".join(f"{(agg['overall'].get(p) or {}).get('n', 0):>9}" for p in pipelines)
    print("-" * 72)
    print(f"{'N questions':<36}{ns}")
    print("=" * 72)


def main():
    global BASE_URL, API_KEY, REQUEST_SLEEP
    ap = argparse.ArgumentParser(
        description="EQGBench-protocol MCQ evaluation (5 dims 0-2, DeepSeek-R1, "
                    "3-round voting).")
    ap.add_argument("--from-bulk", required=True, metavar="FILE",
                    help="bulk_eval_questions.json from the generation run")
    ap.add_argument("--content", help="Text file of the source material (ground truth)")
    ap.add_argument("--judge-model", default="deepseek-reasoner",
                    help="Judge model (default deepseek-reasoner = DeepSeek-R1, "
                         "EQGBench's judge)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="Judging rounds per question (EQGBench uses 3)")
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", BASE_URL))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="Seconds between API calls (rate limits)")
    ap.add_argument("--out", default="eval_out_eqgbench")
    ap.add_argument("--limit", type=int, default=0, help="Only first N per pipeline")
    ap.add_argument("--fresh", action="store_true")
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
        print("[content] none given — KP will be judged without ground truth (weaker)")

    rows_path = os.path.join(args.out, "rows.jsonl")
    rows, done = [], set()
    if args.fresh and os.path.exists(rows_path):
        os.remove(rows_path)
    elif os.path.exists(rows_path):
        with open(rows_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    rows.append(r)
                    done.add((r["pipeline"], r["idx"]))
                except Exception:
                    pass
        print(f"[resume] {len(rows)} rows already judged — skipping those")

    question_sets = load_bulk(args.from_bulk)
    print(f"[bulk] " + ", ".join(f"{k}={len(v)}" for k, v in question_sets.items()))

    rf = open(rows_path, "a", encoding="utf-8")
    try:
        for name, qs in question_sets.items():
            if args.limit:
                qs = qs[:args.limit]
            print(f"\n[{name}] {len(qs)} questions x {args.rounds} rounds")
            for idx, q in enumerate(qs):
                if (name, idx) in done:
                    continue
                scores, n_votes = judge_voted(q, content, args.judge_model, args.rounds)
                row = {
                    "pipeline": name,
                    "difficulty": (q.get("difficulty") or "medium").lower(),
                    "idx": idx,
                    "topic": q.get("topic", ""),
                    "question": (q.get("question") or "")[:300],
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

    agg, pipelines = aggregate(rows)
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": "EQGBench (arXiv:2508.10005): 5 dims 0-2, mode of "
                    f"{args.rounds} judging rounds",
        "config": {"backend": BASE_URL, "judge_model": args.judge_model,
                   "rounds": args.rounds, "pipelines": pipelines,
                   "n_rows": len(rows), "content_chars": len(content)},
        "aggregated": agg,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cols = ["pipeline", "difficulty", "idx", "topic", "question", "rounds_ok"] + DIMS
    with open(os.path.join(args.out, "rows.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print_table(agg, pipelines)
    print(f"\nWrote: {args.out}/results.json, rows.csv, rows.jsonl")


if __name__ == "__main__":
    main()
