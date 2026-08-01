#!/usr/bin/env python3
"""
evaluate_requesta.py — FAITHFUL ReQUESTA expert-rubric evaluation harness.

Rubric source: "Cognitively Diverse Multiple-Choice Question Generation: A
Hybrid Multi-Agent Framework with Large Language Models", Tian et al., Arizona
State University, arXiv:2602.03704v1 (local copy: 2602.03704v1.pdf).

════════════════════════════════════════════════════════════════════════════
READ THIS FIRST — ReQUESTA HAS NO JUDGE PROMPT TO COPY
════════════════════════════════════════════════════════════════════════════
EQGBench publishes an LLM judge prompt (its Figure 3). ReQUESTA does NOT, and
cannot: its rubric was applied by TWO BLINDED HUMAN EXPERT RATERS, not by a
model. Section 4.7 describes raters calibrating on 100 practice MCQs over three
rounds, scoring 200 questions with source blinded, meeting to discuss
disagreements, and consulting a third rater when needed. Reported interrater
reliability (Cohen's weighted kappa): distractor incorrectness 1.00, overall
relevance .80, writing clarity .88, distractor linguistic features .85,
distractor semantic plausibility .80, distractor semantic uniqueness .91.

So for ReQUESTA "implement their prompt" is not achievable in principle. What
IS achievable, and what this script does:
  * their RUBRIC is reproduced verbatim (Appendix A, Tables 6 and 7);
  * their DISAGREEMENT RULE is reproduced (see tie-break below);
  * the wrapper instructing an LLM to apply that rubric is OURS.
State this plainly in the paper. An LLM applying a human rubric is a different
instrument from two calibrated humans applying it, and no amount of prompt
fidelity closes that gap.

WHAT IS TAKEN VERBATIM FROM THE PAPER
─────────────────────────────────────
  * Table 6 (p.29) — both binary criteria, 0 = Disagree / 1 = Agree.
  * Table 7 (p.30) — the four-point rubric, all four anchor descriptions
    (1 Disagree / 2 Neutral / 3 Agree / 4 Strongly Agree) for six criteria.
  * Section 4.7 rationale for the 4-point scale: it "afforded more nuanced
    judgments than a binary or 3-point scale, and reduced the tendency for
    raters to default to a middle option".
  * The disagreement rule: "Remaining disagreements were resolved by adopting
    the lower score to ensure conservative quality standards" (Section 4.7).
    Hence --tie-break defaults to `lower`, NOT mode or mean. This is the
    opposite of EQGBench's rule, on purpose — the two papers differ here.

⚠ A GENUINE INCONSISTENCY INSIDE THE PAPER — handled explicitly
───────────────────────────────────────────────────────────────
The Appendix rubric (Table 7) and the reported results (Tables 3/4/5) do not
list the same criteria:
  * Table 7 defines `Overall quality` and `Coverage`, which are NEVER reported
    in any results table.
  * Tables 3/4/5 report `Distractor semantic uniqueness`, which has NO anchor
    definition anywhere in Table 7.
This script implements all nine criteria and labels each one's provenance.
`distractor_uniqueness` anchors are RECONSTRUCTED by us from the Section 4.7
prose ("the extent to which each distractor conveyed a unique meaning, as
distractors with similar meanings are easy for readers to eliminate") — flagged
with * in the output table. Only the five criteria in COMPARABLE below can be
put beside the paper's numbers.

REFERENCE NUMBERS (paper Table 4, N=200, expert-rated, for comparison)
                                    ReQUESTA   GPT-5 zero-shot
    Topic relevance                     3.36       3.12
    Writing clarity                     3.02       3.39   <- GPT-5 WINS
    Distractor linguistic features      2.28       1.92
    Distractor semantic plausibility    3.17       2.63
    Distractor semantic uniqueness      3.82       3.75   (n.s.)
  Answer correctness 1.00 both arms; distractor incorrectness .99 both arms.
  The rigour-vs-clarity trade-off is the paper's key qualitative finding:
  agentic items scored HIGHER on relevance/distractors but LOWER on clarity.

NOT PRODUCED HERE: item difficulty (p), discrimination (D) and point-biserial
correlation. Those are the paper's psychometric layer and require real learner
responses (they ran 572 Prolific participants). No LLM can substitute.

USAGE
  $env:DEEPSEEK_API_KEY = "sk-..."          # never hard-code the key
  python evaluate_requesta.py --from-bulk bulk_eval_questions.json \
      --content chapter.txt --out eval_out_requesta

Runs against EXISTING generated questions — no regeneration needed.
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

BASE_URL = "https://api.deepseek.com"
API_KEY = ""
REQUEST_SLEEP = 0.0

TEMPERATURE = 0.6
MAX_TOKENS = 4096


# ─────────────────────────────────────────────────────────────────────────────
# API plumbing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text):
    """First JSON object in a reply; tolerates ``` fences, prose, <think>."""
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
    body = {"model": model, "messages": messages, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS, "stream": False}
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
# The ReQUESTA rubric — Appendix A, Tables 6 and 7, verbatim
# ─────────────────────────────────────────────────────────────────────────────

BINARY_DIMS = ["answer_correctness", "distractor_incorrectness"]

# Order follows Table 7 (p.30), then the reported-but-undefined criterion.
SCALE_DIMS = ["overall_quality", "topic_relevance", "coverage",
              "writing_clarity", "distractor_plausibility",
              "distractor_linguistic", "distractor_uniqueness"]

DIMS = BINARY_DIMS + SCALE_DIMS

# The five criteria that can be placed beside the paper's Table 4 numbers.
# `overall_quality` and `coverage` are defined in Table 7 but never reported;
# the binary pair saturates at 1.00/0.99 in both arms.
COMPARABLE = ["topic_relevance", "writing_clarity", "distractor_linguistic",
              "distractor_plausibility", "distractor_uniqueness"]

PAPER_TABLE4 = {
    "topic_relevance":          {"ReQUESTA": 3.36, "GPT-5": 3.12},
    "writing_clarity":          {"ReQUESTA": 3.02, "GPT-5": 3.39},
    "distractor_linguistic":    {"ReQUESTA": 2.28, "GPT-5": 1.92},
    "distractor_plausibility":  {"ReQUESTA": 3.17, "GPT-5": 2.63},
    "distractor_uniqueness":    {"ReQUESTA": 3.82, "GPT-5": 3.75},
}

DIM_LONG = {
    "answer_correctness":       "Answer Correctness (0-1)",
    "distractor_incorrectness": "Distractor Incorrectness (0-1)",
    "overall_quality":          "Overall quality (1-4)",
    "topic_relevance":          "Topic relevance (1-4)",
    "coverage":                 "Coverage (1-4)",
    "writing_clarity":          "Writing clarity (1-4)",
    "distractor_plausibility":  "Distractor plausibility (1-4)",
    "distractor_linguistic":    "Linguistic features (1-4)",
    "distractor_uniqueness":    "Distractor semantic uniqueness (1-4)",
}

DIM_RANGE = {d: (0, 1) for d in BINARY_DIMS}
DIM_RANGE.update({d: (1, 4) for d in SCALE_DIMS})

# Table 6 (p.29), verbatim.
BINARY_TEXT = {
    "answer_correctness":
        "Correct answer is clearly correct to those who have comprehended the text.",
    "distractor_incorrectness":
        "Distractors are clearly incorrect to those who have comprehended the text.",
}

# Table 7 (p.30), verbatim: (1 - Disagree, 2 - Neutral, 3 - Agree, 4 - Strongly Agree).
# `distractor_uniqueness` is the sole RECONSTRUCTED entry — see the header.
SCALE_TEXT = {
    "overall_quality": (
        "The question and options are unclear, ambiguous, or verbose.",
        "Some clarity issues and/or minor ambiguity remain.",
        "Clearly worded, concise, and unambiguous.",
        "Excellent clarity: concise, unambiguous, and professional."),
    "topic_relevance": (
        "Does not address a central concept in the text.",
        "Addresses a peripheral or partially relevant point.",
        "Addresses a central concept in the text.",
        "Addresses a central concept very directly and importantly."),
    "coverage": (
        "Not covered in main points or sub-points.",
        "Partially covered in sub-points or only partially covered overall.",
        "Partially covered in a main point; fully covered in sub-points.",
        "Fully covered as a main point in the ideas sheet."),
    "writing_clarity": (
        "Stem and options are excessively wordy or unclear.",
        "Wordiness or mild ambiguity that slightly impacts comprehension.",
        "Stem and options concise with minor redundancies.",
        "Stem and options fully clear and concise."),
    "distractor_plausibility": (
        "Distractors are implausible or obviously wrong.",
        "Some distractors somewhat implausible or uneven.",
        "Distractors plausible to those who have not comprehended the text.",
        "All distractors are highly plausible and well matched."),
    "distractor_linguistic": (
        "Distractors visibly different from the correct answer "
        "(e.g., much longer or shorter).",
        "Some distractors have noticeable differences or one major difference.",
        "Distractors have relatively similar length and surface features.",
        "All distractors are uniform in linguistic features and closely matched."),
    # RECONSTRUCTED from Section 4.7 prose. No anchor table exists.
    "distractor_uniqueness": (
        "Several distractors convey near-identical meanings and can be "
        "eliminated together.",
        "Some distractors overlap in meaning, or one pair is close to redundant.",
        "Distractors mostly convey distinct meanings, with minor overlap.",
        "Every distractor conveys a clearly unique meaning."),
}

RECONSTRUCTED = {"distractor_uniqueness"}

# Ours, not the paper's — ReQUESTA used human raters (see header).
JUDGE_SYSTEM = ("You are one of two trained expert raters scoring "
                "multiple-choice questions against a fixed rubric. Apply the "
                "rubric exactly as written. Return only JSON.")


def build_prompt(q, content, dims):
    opts = q.get("options") or {}
    opts_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    marked = str(q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]
    topic = q.get("topic") or "(not stated)"

    lines = [
        "Score this ONE multiple-choice question against the rubric below.",
        "",
        "Source text the question was written from:",
        content[:4500] if content else "(not supplied)",
        "",
        f"Intended topic: {topic}",
        "",
        "Question item:",
        f"Q: {q.get('question', '')}",
        "Options:",
        opts_txt,
        f"Marked correct answer: {marked}",
        "",
    ]

    bin_dims = [d for d in dims if d in BINARY_DIMS]
    if bin_dims:
        lines.append("BINARY CRITERIA — score 0 (Disagree) or 1 (Agree):")
        for d in bin_dims:
            lines.append(f"- {d}: {BINARY_TEXT[d]}")
        lines.append("")

    sc_dims = [d for d in dims if d in SCALE_DIMS]
    if sc_dims:
        lines.append(
            "FOUR-POINT CRITERIA — score 1 (Disagree), 2 (Neutral), "
            "3 (Agree) or 4 (Strongly Agree).")
        lines.append(
            "The 4-point scale is deliberate: it affords more nuanced "
            "judgments than a binary or 3-point scale and removes the middle "
            "option raters default to. Do not default to 2.")
        lines.append("")
        for d in sc_dims:
            a = SCALE_TEXT[d]
            lines.append(f"- {d}:")
            for i, anchor in enumerate(a, start=1):
                lines.append(f"    {i} = {anchor}")
        lines.append("")

    keys = ", ".join(f'"{d}": {{"justification": "<one sentence>", '
                     f'"score": <int>}}' for d in dims)
    lines.append("Give a one-sentence justification BEFORE the score for each "
                 "criterion. Return STRICT JSON only, exactly these keys:")
    lines.append("{" + keys + "}")
    return "\n".join(lines)


def _coerce(val):
    """Accept {'score': n, ...} or a bare number."""
    if isinstance(val, dict):
        val = val.get("score")
    try:
        return int(float(val))
    except Exception:
        return None


def judge_once(q, content, model, dims):
    """One rubric pass. Returns ({dim: int}, {dim: str}) or ({}, {}) on failure."""
    r = api_json(build_prompt(q, content, dims), model, system=JUDGE_SYSTEM)
    if not r:
        return {}, {}
    scores, justs = {}, {}
    for d in dims:
        lo, hi = DIM_RANGE[d]
        v = _coerce(r.get(d))
        if v is None or not (lo <= v <= hi):
            return {}, {}          # partial parses are failures, never floors
        scores[d] = v
        raw = r.get(d)
        if isinstance(raw, dict):
            justs[d] = str(raw.get("justification", ""))[:300]
    return scores, justs


def judge_voted(q, content, model, rounds, dims, tie_break):
    """N independent passes reduced to one score per criterion.

    tie_break='lower' reproduces Section 4.7: "Remaining disagreements were
    resolved by adopting the lower score to ensure conservative quality
    standards." Note this applies to ANY disagreement across rounds, not only
    to a mode tie — the paper's raters resolved every unresolved disagreement
    downward. 'mode' and 'mean' are offered for sensitivity checks only.

    n_votes == 0 means every round failed; aggregation excludes such rows, so a
    rate-limited question can never masquerade as a genuinely low-scoring one.
    """
    votes, justs = [], {}
    for _ in range(rounds):
        s, j = judge_once(q, content, model, dims)
        if s:
            votes.append(s)
            for k, v in j.items():
                justs.setdefault(k, v)
    if not votes:
        return {}, {}, 0
    final = {}
    for d in dims:
        vals = [v[d] for v in votes]
        if tie_break == "lower":
            final[d] = min(vals)
        elif tie_break == "mean":
            final[d] = round(sum(vals) / len(vals), 4)
        else:
            counts = Counter(vals).most_common()
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                final[d] = int(statistics.median(vals))
            else:
                final[d] = counts[0][0]
    return final, justs, len(votes)


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation / I-O
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows, dims):
    """Mean per criterion, per (pipeline, difficulty). Rows whose judging rounds
    all failed (rounds_ok == 0) are EXCLUDED, not scored at the floor."""
    usable = [r for r in rows if r.get("rounds_ok", 0) > 0]

    def block(sub):
        if not sub:
            return None
        m = {"n": len(sub)}
        for d in dims:
            vals = [r[d] for r in sub if r.get(d) is not None]
            m[d] = round(sum(vals) / len(vals), 3) if vals else None
        comp = [m[d] for d in COMPARABLE if d in dims and m.get(d) is not None]
        # Composite over the COMPARABLE five only. Mixing the 0-1 binary pair
        # into a 1-4 mean would be meaningless, and overall_quality/coverage
        # have no paper counterpart to compare against.
        m["composite_comparable"] = round(sum(comp) / len(comp), 3) if comp else None
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


def print_table(agg, pipelines, excluded, dims):
    w = 78 + 22
    print("\n" + "=" * w)
    print("ReQUESTA EXPERT-RUBRIC RESULTS  (Appendix A, Tables 6-7)")
    print("=" * w)
    head = f"{'Criterion':<36}" + "".join(f"{p:>11}" for p in pipelines)
    head += f"{'| paper ReQ':>13}{'paper GPT-5':>13}"
    print(head)
    print("-" * w)
    for d in dims:
        cells = ""
        for p in pipelines:
            v = (agg["overall"].get(p) or {}).get(d)
            cells += f"{v:>11}" if v is not None else f"{'-':>11}"
        ref = PAPER_TABLE4.get(d)
        if ref:
            cells += f"{'| ' + str(ref['ReQUESTA']):>13}{ref['GPT-5']:>13}"
        else:
            cells += f"{'| -':>13}{'-':>13}"
        flag = " *" if d in RECONSTRUCTED else ""
        print(f"{DIM_LONG[d] + flag:<36}{cells}")
    print("-" * w)
    for label, key in (("Composite (mean of comparable 5)", "composite_comparable"),
                       ("N questions scored", "n")):
        cells = ""
        for p in pipelines:
            v = (agg["overall"].get(p) or {}).get(key)
            cells += f"{v:>11}" if v is not None else f"{'-':>11}"
        print(f"{label:<36}{cells}")
    print("=" * w)
    if excluded:
        print(f"NOTE: {excluded} row(s) excluded from all means (rounds_ok == 0 — "
              f"every judging round failed, e.g. rate limits). Re-run to fill them.")
    print("* anchors RECONSTRUCTED — no anchor table exists in the paper for "
          "this criterion.")
    print("`Overall quality` and `Coverage` are defined in Table 7 but never "
          "reported in the\npaper's results, so they have no reference column.")
    print("\n⚠ INSTRUMENT MISMATCH: these are LLM scores. The paper's numbers "
          "come from TWO\nBLINDED HUMAN EXPERTS (weighted kappa .80-1.00) who "
          "calibrated on 100 practice items.\nThe columns sit side by side for "
          "orientation, NOT as a like-for-like comparison.")
    print("No item difficulty / discrimination / point-biserial here — those "
          "need real\nlearner responses (the paper ran 572 participants).")


def main():
    global BASE_URL, API_KEY, REQUEST_SLEEP

    ap = argparse.ArgumentParser(
        description="Score MCQs on the ReQUESTA expert rubric "
                    "(arXiv:2602.03704, Appendix A Tables 6-7).")
    ap.add_argument("--from-bulk", required=True, metavar="FILE",
                    help="bulk questions JSON produced by the UI")
    ap.add_argument("--content", help="Text file of the source material")
    ap.add_argument("--judge-model", default="deepseek-reasoner",
                    help="judge model id")
    ap.add_argument("--rounds", type=int, default=3,
                    help="independent rubric passes per question")
    ap.add_argument("--tie-break", choices=["lower", "mode", "mean"],
                    default="lower",
                    help="how to reduce disagreeing rounds. 'lower' reproduces "
                         "the paper's conservative rule (Section 4.7); the "
                         "others are for sensitivity checks only")
    ap.add_argument("--dims", default=",".join(DIMS),
                    help="comma-separated subset of criteria to score")
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

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    bad = [d for d in dims if d not in DIMS]
    if bad:
        sys.exit(f"Unknown criterion/criteria: {bad}. Valid: {DIMS}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[backend] {BASE_URL}  judge={args.judge_model}  rounds={args.rounds}")
    print(f"[tie-break] {args.tie_break}"
          + ("  (paper's conservative rule)" if args.tie_break == "lower" else
             "  (NOT the paper's rule — sensitivity check only)"))

    content = ""
    if args.content:
        with open(args.content, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(f"[content] {len(content)} chars from {args.content}")
    else:
        print("[content] none given — topic_relevance and coverage are judged "
              "without ground truth (much weaker)")

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
        print(f"[resume] {len(rows)} rows read, {len(done)} already judged — "
              f"skipping those")
        if malformed:
            # Loud on purpose: a malformed checkpoint previously looked exactly
            # like an unfinished run, and got silently re-judged forever.
            print(f"[resume] WARNING: {malformed} malformed line(s) ignored. "
                  f"If this is unexpected, inspect {rows_path} before "
                  f"continuing — they will be RE-JUDGED as if never done.",
                  file=sys.stderr)

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
                scores, justs, n_votes = judge_voted(
                    q, content, args.judge_model, args.rounds, dims,
                    args.tie_break)
                row = {
                    "pipeline": name,
                    "difficulty": diff,
                    "idx": idx,
                    "topic": q.get("topic", ""),
                    "question": (q.get("question") or "")[:300],
                    "cognitive_type": q.get("cognitive_type", ""),
                    "rounds_ok": n_votes,
                    **scores,
                    **{f"{k}_justification": v for k, v in justs.items()},
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

    agg, pipelines, excluded = aggregate(rows, dims)
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": "ReQUESTA expert rubric (arXiv:2602.03704). Binary criteria "
                    "verbatim from Appendix A Table 6; four-point anchors "
                    "verbatim from Table 7. Disagreements across rounds "
                    f"resolved by '{args.tie_break}' over {args.rounds} passes.",
        "instrument_caveat": "ReQUESTA publishes NO judge prompt — its rubric "
                             "was applied by two blinded HUMAN expert raters "
                             "(Cohen's weighted kappa .80-1.00) who calibrated "
                             "on 100 practice MCQs. The rubric here is theirs "
                             "verbatim; the instruction wrapper around it is "
                             "ours. An LLM applying a human rubric is a "
                             "different instrument and must be reported as such.",
        "deviations_from_paper": [
            "distractor_uniqueness anchors are RECONSTRUCTED from Section 4.7 "
            "prose: it is reported in Tables 3/4/5 but has no anchor row in "
            "Appendix Table 7.",
            "overall_quality and coverage are defined in Table 7 but never "
            "reported in the paper's results, so no reference values exist.",
            "Item difficulty, discrimination and point-biserial are NOT "
            "computed — they require real learner responses (572 participants "
            "in the paper).",
            "The paper compared ReQUESTA against a GPT-5 zero-shot baseline on "
            "OpenStax expository passages; our corpus and generator differ.",
        ],
        "paper_reference_scores": {"table_4": PAPER_TABLE4,
                                   "answer_correctness": 1.00,
                                   "distractor_incorrectness": 0.99},
        "config": {"backend": BASE_URL, "judge_model": args.judge_model,
                   "rounds": args.rounds, "tie_break": args.tie_break,
                   "dims": dims, "pipelines": pipelines,
                   "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                   "n_rows": len(rows), "n_excluded_failed": excluded,
                   "content_chars": len(content)},
        "aggregated": agg,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cols = ["pipeline", "difficulty", "idx", "topic", "question",
            "cognitive_type", "rounds_ok"] + dims
    with open(os.path.join(args.out, "rows.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print_table(agg, pipelines, excluded, dims)
    print(f"\nWrote: {args.out}/results.json, rows.csv, rows.jsonl")


if __name__ == "__main__":
    main()
