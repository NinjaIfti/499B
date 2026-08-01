#!/usr/bin/env python3
"""
evaluate_eqgbench.py — FAITHFUL EQGBench evaluation harness.

Protocol source: "From Answers to Questions: EQGBench for Evaluating LLMs'
Educational Question Generation", Zhou et al., Beijing Normal University,
arXiv:2508.10005v1 (local copy: ref.pdf).

WHAT IS TAKEN VERBATIM FROM THE PAPER
─────────────────────────────────────
  * The judge SYSTEM prompt and the Query scaffold — transcribed from Figure 3
    (p.7), "Evaluation Prompt Design".
  * The KP scoring criteria, BOTH worked examples (0-point and 2-point) and the
    full Note block — transcribed verbatim from Figure 3.
  * The QT / QQ / SQ / CG scoring criteria — transcribed verbatim from the
    Excellent / Good / Poor anchors in Section 4 (pp.5-6).
  * The output format:  [Scoring Justification]: ...<eoa>  /  [Score]: ...<eoa>
  * ONE DIMENSION PER CALL. Figure 3 shows a prompt scoped to a single
    dimension ("Dimension: Knowledge Point Alignment"), so each dimension is a
    separate judging call. This is 5x the calls of a single combined prompt.
  * Judge = DeepSeek-R1 (deepseek-reasoner); temperature 0.6; max output 4096
    tokens (Section 5.1/5.2).
  * THREE independent rounds per (question, dimension); final score = MODE of
    the rounds; "In case no mode existed, the arithmetic mean was used as the
    final score" (Section 5.2) — so the no-mode tie-break is MEAN, not median.

WHAT IS *NOT* FROM THE PAPER (disclose wherever these results are published)
───────────────────────────────────────────────────────────────────────────
  1. Figure 3 prints the worked examples for the KP dimension ONLY. QT/QQ/SQ/CG
     use the same verbatim scaffold and their verbatim Section 4 anchors, but
     carry NO few-shot examples, because the paper does not supply any.
  2. The CG dimension in Section 4 defines only TWO anchors — Excellent (2) and
     Poor (0). No "Good" (1) anchor exists in the paper. The 1-point wording
     used here is OURS. CG is the dimension the paper reports as collapsing
     for every model (DeepSeek-R1 math CG = 0.17), so this gap matters: treat
     our CG numbers as the least comparable of the five.
  3. EQGBench evaluates Chinese middle-school math/physics/chemistry against a
     user instruction. We evaluate English MCQs against source material, so the
     "User Input" is reconstructed from each question's topic/difficulty.
  4. EQGBench has no public code release. Everything above is transcribed from
     the PDF, not cloned from an implementation.

REFERENCE NUMBERS (paper Table 2, DeepSeek-R1 judged, for sanity-checking a run)
    DeepSeek-R1 (math):  KP 1.98  QT 1.95  QQ 1.95  SQ 1.96  CG 0.17
    GPT-4o      (math):  KP 1.96  QT 1.82  QQ 1.76  SQ 1.72  CG 0.21
  KP/QT/QQ/SQ saturate near 1.9-2.0 for every strong model; CG collapses.
  If our KP/QT/QQ/SQ land far below ~1.8, suspect the harness, not the pipeline.

  Human validation (Table 3): Score Consistency KP .965 QT .973 QQ .898
  SQ .885 CG .920; Ranking Consistency 1.00 for all dimensions EXCEPT SQ (.50).

COST: 5 dims x 3 rounds x N questions. For N=600 that is 9,000 reasoner calls.
Rows are checkpointed after every (question, dimension), so a run RESUMES.
Smoke-test with --limit 5 before committing to a full run.

USAGE
  $env:DEEPSEEK_API_KEY = "sk-..."          # never hard-code the key
  python evaluate_eqgbench.py --from-bulk bulk_eval_questions.json \
      --content chapter.txt --out eval_out_eqgbench

Any OpenAI-compatible endpoint works via --base-url/--api-key-env, but then the
judge is no longer the paper's model — say so if you use one.
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import requests

BASE_URL = "https://api.deepseek.com"
API_KEY = ""
REQUEST_SLEEP = 0.0

# Paper settings, Section 5.1: "The temperature parameter was set to 0.6 ...
# The maximum output length was capped at 4096 tokens".
TEMPERATURE = 0.6
MAX_TOKENS = 4096


# ─────────────────────────────────────────────────────────────────────────────
# API plumbing
# ─────────────────────────────────────────────────────────────────────────────

def api_text(prompt, model, system="", retries=3):
    """One chat call -> raw assistant text, "" on failure (fail closed)."""
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
            content = r.json()["choices"][0]["message"].get("content", "") or ""
            if content.strip():
                if REQUEST_SLEEP:
                    time.sleep(REQUEST_SLEEP)
                return content
        except Exception as e:
            if attempt == retries:
                print(f"    [warn] judge call failed: {e}", file=sys.stderr)
                break
            time.sleep(backoff)
            backoff *= 2
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# The EQGBench rubric — Figure 3 (p.7) and Section 4 (pp.5-6), verbatim
# ─────────────────────────────────────────────────────────────────────────────

DIMS = ["KP", "QT", "QQ", "SQ", "CG"]

DIM_LONG = {
    "KP": "Knowledge Point Alignment",
    "QT": "Question Type Alignment",
    "QQ": "Question Item Quality",
    "SQ": "Solution Explanation Quality",
    "CG": "Competence-Oriented Guidance",
}

# Verbatim from Figure 3. This is the paper's own system prompt.
JUDGE_SYSTEM = (
    "You are an experienced middle school exam question designer with 20 years "
    "of expertise. Based on the following evaluation dimension, please strictly "
    "score the given question according to the scoring criteria, in combination "
    "with the user input and the generated question."
)

# Per-dimension blocks.
#   "dimension" + "criteria"  -> verbatim from the paper.
#   "examples" / "note"       -> verbatim ONLY for KP (Figure 3); the paper
#                                supplies none for the other four.
#   "reconstructed"           -> True where we had to author an anchor.
DIM_SPEC = {
    "KP": {
        "dimension": "Knowledge Point Alignment. This measures whether the "
                     "generated question accurately matches and adequately "
                     "covers the specified knowledge point.",
        "criteria": """0 points: The question item fails to correctly reflect the knowledge point. There is a significant mismatch between the knowledge point used in the question and the one specified by the user, or it is from a different subject.
1 point: The question item is generally relevant in topic but does not directly address or include the user-specified knowledge point.
2 points: The question item basically covers the user-specified knowledge point, with no significant omissions.""",
        # Both worked examples, transcribed verbatim from Figure 3.
        "examples": """Example for 0 points:
Question Item: A city surveyed 1,000 residents to determine awareness of garbage sorting. Of those, 920 were aware. Based on this data, answer the following: 3. Estimate the margin of error for the city's garbage-sorting awareness rate at a 95% confidence level, rounded to two decimal places. Assume the population variance is unknown and estimate using sample variance.
Scoring Justification: "Confidence level" is not part of the middle school curriculum.
Example for 2 points:
User Request: Please create a multiple-choice question on addition and subtraction of polynomials at the middle school level.
Question Item: We define a linear equation in one variable $ax - b$ to be a "difference-solution equation" if its solution is $b - a$. For example, the solution of $2x - 4$ is 2, and $2 - 4 = 2$, so $2x - 4$ is a difference-solution equation.
Scoring Justification: Even though the concept is newly defined, the solution process involves addition and subtraction of polynomials, thus the specified knowledge point is covered.""",
        # Verbatim Note block from Figure 3.
        "note": """Note: As long as the question includes the specified knowledge point in any part, it is considered "basically covered" and earns 2 points. For example, if only one sub-question among several involves the knowledge point, or if only one option in a multiple-choice question does, it still counts as basic coverage.
Do not apply a lowest-score-first principle unless there are multiple sub-questions—in that case, the final score should be the lowest score among all the sub-questions.
Important: Only evaluate the content within the <question item> tags; ignore <solution explanation> and <answer>.
If the question is incomplete and cannot stand alone, the score is automatically 0 for this dimension.""",
        "reconstructed": False,
    },
    "QT": {
        "dimension": "Question Type Alignment. This dimension evaluates whether "
                     "the type of the generated question (e.g., choice, "
                     "fill-in-the-blank, problem) matches the user's selection "
                     "and adheres to the standard formatting requirements for "
                     "the type. For example, a single-choice question should "
                     "include four options; a fill-in-the-blank question should "
                     "provide an underline, parentheses, or another clear "
                     "indicator for the answer; a problem may be presented as a "
                     "comprehensive problem that integrates various formats "
                     "like selection or calculation.",
        "criteria": """0 points (Poor): The question's type is completely inconsistent with the user's specification, or the format is too disorganized to be identified.
1 point (Good): The question's type is generally consistent with the user's specification, but there are minor errors in detail or formatting.
2 points (Excellent): The question's type is identical to the user's specification and adheres to the standard format for that type.""",
        "examples": "",
        "note": "",
        "reconstructed": False,
    },
    "QQ": {
        "dimension": "Question Item Quality. This dimension assesses whether the "
                     "generated question is expressed clearly, has an "
                     "unambiguous objective, uses standardized terminology, and "
                     "is solvable with a unique or definitive answer. This "
                     "ensures that students can accurately understand the "
                     "question's intent and complete the task.",
        "criteria": """0 points (Poor): The language is confusing or unclear, with significant issues such as redundancy, logical fallacies, or typos.
1 point (Good): The language of the question is ambiguous, or technical terms are used incorrectly.
2 points (Excellent): The question is clear, concise, and easy for students to understand.""",
        "examples": "",
        "note": "",
        "reconstructed": False,
    },
    "SQ": {
        "dimension": "Solution Explanation Quality. This dimension evaluates the "
                     "correctness, rigor, and completeness of the explanation "
                     "provided for the generated question. It also requires that "
                     "the knowledge involved in the explanation is appropriate "
                     "for the cognitive level and curriculum requirements of the "
                     "target academic stage, and that the correct answer can be "
                     "derived from the explanation.",
        "criteria": """0 points (Poor): The explanation process is flawed and cannot lead to the correct answer, or no explanation is provided at all (only the final answer is given).
1 point (Good): The explanation contains logical leaps, lacks clarity, or has issues like repetition.
2 points (Excellent): The explanation is correct, logically sound, and meets all requirements of the question.""",
        "examples": "",
        "note": "",
        "reconstructed": False,
    },
    "CG": {
        "dimension": "Competence-Oriented Guidance. This dimension evaluates "
                     "whether the generated question integrates or simulates a "
                     "realistic scenario, including but not limited to cultural "
                     "contexts, practical subject applications, or real-life "
                     "situations. It measures the question's value in guiding "
                     "students to apply knowledge and develop higher-order "
                     "competencies.",
        # The paper (Section 4, p.6) defines ONLY Excellent and Poor for CG.
        # The 1-point line below is OURS — see the disclosure at the top.
        "criteria": """0 points (Poor): The question is a purely abstract application of knowledge points, lacking any contextual design.
1 point (Good): [NOT DEFINED IN THE PAPER — reconstructed] The question gestures at a context or application but the scenario is token, generic, or not genuinely required to solve the problem.
2 points (Excellent): The question incorporates a rich, contextual scenario that is directly relevant to solving the problem.""",
        "examples": "",
        "note": "",
        "reconstructed": True,
    },
}


def format_generated_question(q):
    """Render one MCQ in EQGBench's tagged output format (Figure 2).

    The KP Note says "Only evaluate the content within the <question item>
    tags; ignore <solution explanation> and <answer>" — that instruction is
    meaningless unless the item is actually presented with those tags.
    """
    opts = q.get("options") or {}
    opts_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    marked = str(q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]
    explanation = (q.get("explanation") or q.get("rationale")
                   or q.get("solution") or "").strip()
    stem = (q.get("question") or "").strip()
    return (
        f"<question item>{stem}\n{opts_txt}</question item>\n"
        f"<solution explanation>{explanation or '(none provided)'}"
        f"</solution explanation>\n"
        f"<answer>{marked or '(none provided)'}</answer>"
    )


def format_user_input(q):
    """Reconstruct an EQGBench-style user instruction from question metadata.

    EQGBench ships a real user query per sample (Table 1). We do not have one,
    so we synthesise the equivalent instruction from topic + difficulty. This
    is a documented deviation, not paper text.
    """
    topic = q.get("topic") or "(not stated)"
    diff = (q.get("difficulty") or "medium").lower()
    return (f"Please create a single-choice question on the topic "
            f"\"{topic}\", with a difficulty level of {diff}.")


def build_prompt(dim, q, content):
    """Assemble the Figure 3 prompt for ONE dimension."""
    spec = DIM_SPEC[dim]
    parts = [
        "Query:",
        f"Dimension: {spec['dimension']}",
        "Scoring Criteria:",
        spec["criteria"],
    ]
    if spec["examples"]:
        parts.append(spec["examples"])
    if content:
        # Not in Figure 3: EQGBench judges against a user instruction, we judge
        # against source material. Kept clearly separate from the paper's text.
        parts.append("Reference material the question was generated from "
                     "(supplied by this harness, not part of the original "
                     "EQGBench prompt):\n" + content[:4500])
    parts.append(f"User Input: {format_user_input(q)}")
    parts.append(f"Generated Question: {format_generated_question(q)}")
    if spec["note"]:
        parts.append(spec["note"])
    parts.append("Please output the response in the following format:\n"
                 "[Scoring Justification]: ...<eoa>\n"
                 "[Score]: ...<eoa>")
    return "\n\n".join(parts)


_SCORE_RE = re.compile(r"\[?Score\]?\s*[:：]\s*\**\s*([0-2])", re.IGNORECASE)


def parse_score(text):
    """Pull the integer out of '[Score]: N<eoa>'. Returns None if unparseable.

    Deliberately strict: an unreadable reply is a FAILED round, not a 0. A
    parser that silently returns 0 would make rate limits look like bad
    questions — the exact failure mode that corrupted the earlier run.
    """
    if not text:
        return None
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _SCORE_RE.search(body)
    if m:
        return int(m.group(1))
    # Fallback: a bare digit on the last non-empty line.
    for line in reversed([l.strip() for l in body.splitlines() if l.strip()]):
        m2 = re.search(r"\b([0-2])\b", line)
        if m2:
            return int(m2.group(1))
    return None


def parse_justification(text):
    if not text:
        return ""
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\[?Scoring Justification\]?\s*[:：]\s*(.*?)(?:<eoa>|\[Score\])",
                  body, re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else "")[:400]


def judge_dimension(q, content, dim, model, rounds):
    """Score ONE dimension over N rounds. Returns (score, n_ok, justification).

    Voting rule, Section 5.2: mode of the rounds; if no mode exists, the
    ARITHMETIC MEAN. The mean is kept unrounded so a 0/1/2 split scores 1.0
    rather than being silently floored.
    """
    prompt = build_prompt(dim, q, content)
    votes, first_just = [], ""
    for _ in range(rounds):
        text = api_text(prompt, model, system=JUDGE_SYSTEM)
        s = parse_score(text)
        if s is not None:
            votes.append(s)
            if not first_just:
                first_just = parse_justification(text)
    if not votes:
        return None, 0, ""
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        score = sum(votes) / len(votes)          # paper: arithmetic mean
    else:
        score = float(counts[0][0])
    return round(score, 4), len(votes), first_just


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation / I-O
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(rows):
    """Mean per dimension, per (pipeline, difficulty).

    A dimension is averaged over the questions where it actually scored. Rows
    with rounds_ok == 0 for that dimension are EXCLUDED, never counted as 0 —
    otherwise rate-limit failures would masquerade as genuine low scores.
    """
    def block(sub):
        if not sub:
            return None
        m = {"n": len(sub)}
        present = []
        for d in DIMS:
            vals = [r[d] for r in sub if r.get(d) is not None
                    and r.get(f"{d}_rounds_ok", 0) > 0]
            m[d] = round(sum(vals) / len(vals), 3) if vals else None
            m[f"{d}_n"] = len(vals)
            if m[d] is not None:
                present.append(m[d])
        # EQGBench's per-subject "total score" is the sum of the five dims
        # (Section 5.3.2 quotes totals like 9.02 out of 10).
        m["total"] = round(sum(present), 3) if len(present) == len(DIMS) else None
        m["composite"] = round(sum(present) / len(present), 3) if present else None
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
    print("\n" + "=" * 78)
    print("EQGBENCH RESULTS  (each dimension 0-2; paper Table 2 for comparison)")
    print("=" * 78)
    print(f"{'Dimension':<38}" + "".join(f"{p:>13}" for p in pipelines))
    print("-" * 78)
    for d in DIMS:
        cells = ""
        for p in pipelines:
            v = (agg["overall"].get(p) or {}).get(d)
            cells += f"{v:>13}" if v is not None else f"{'-':>13}"
        flag = " *" if DIM_SPEC[d]["reconstructed"] else ""
        print(f"{d + '  ' + DIM_LONG[d] + flag:<38}{cells}")
    print("-" * 78)
    for label, key in (("Total (sum of 5 dims, max 10)", "total"),
                       ("Composite (mean of 5 dims, max 2)", "composite"),
                       ("N questions", "n")):
        cells = ""
        for p in pipelines:
            v = (agg["overall"].get(p) or {}).get(key)
            cells += f"{v:>13}" if v is not None else f"{'-':>13}"
        print(f"{label:<38}{cells}")
    print("=" * 78)
    print("* CG's 1-point anchor is NOT in the paper (only Excellent/Poor are\n"
          "  defined). CG is the least comparable dimension — disclose this.")
    print("Paper Table 2 reference, DeepSeek-R1 (math): KP 1.98 QT 1.95 QQ 1.95\n"
          "SQ 1.96 CG 0.17. KP/QT/QQ/SQ saturate near ceiling for strong models;\n"
          "CG collapses for ALL of them. Scores far below ~1.8 on KP/QT/QQ/SQ\n"
          "suggest a harness problem, not a pipeline problem.")


def main():
    global BASE_URL, API_KEY, REQUEST_SLEEP

    ap = argparse.ArgumentParser(
        description="Faithful EQGBench evaluation (arXiv:2508.10005): 5 dims "
                    "scored 0-2, one dimension per call, DeepSeek-R1, 3-round "
                    "mode voting with arithmetic-mean tie-break.")
    ap.add_argument("--from-bulk", required=True, metavar="FILE",
                    help="bulk questions JSON produced by the UI")
    ap.add_argument("--content", help="Text file of the source material")
    ap.add_argument("--judge-model", default="deepseek-reasoner",
                    help="judge model id (paper uses DeepSeek-R1)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="independent rounds per (question, dimension); paper uses 3")
    ap.add_argument("--dims", default=",".join(DIMS),
                    help="comma-separated subset of dimensions to judge")
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", BASE_URL))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between calls, for rate-limited free tiers")
    ap.add_argument("--out", default="eval_out_eqgbench")
    ap.add_argument("--limit", type=int, default=0, help="Only first N per pipeline")
    ap.add_argument("--fresh", action="store_true",
                    help="delete any existing rows.jsonl and judge from scratch")
    args = ap.parse_args()

    BASE_URL = args.base_url
    API_KEY = os.environ.get(args.api_key_env, "")
    REQUEST_SLEEP = args.sleep
    if not API_KEY:
        sys.exit(f"No API key. Set it first:  $env:{args.api_key_env} = \"sk-...\"")

    dims = [d.strip().upper() for d in args.dims.split(",") if d.strip()]
    bad = [d for d in dims if d not in DIMS]
    if bad:
        sys.exit(f"Unknown dimension(s): {bad}. Valid: {DIMS}")

    os.makedirs(args.out, exist_ok=True)
    print(f"[backend] {BASE_URL}  judge={args.judge_model}  rounds={args.rounds}")
    print(f"[dims]    {dims}   (one API call per question per dimension)")

    content = ""
    if args.content:
        with open(args.content, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(f"[content] {len(content)} chars from {args.content}")
    else:
        print("[content] none given — KP is judged against the reconstructed "
              "user instruction only (weaker)")

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
                # Key MUST include difficulty: idx restarts per difficulty
                # group, so (pipeline, idx) alone collides and silently skips
                # real questions. A dimension counts as done only if at least
                # one of its rounds parsed.
                for d in DIMS:
                    if r.get(f"{d}_rounds_ok", 0) > 0:
                        done.add((r["pipeline"], r.get("difficulty"), r["idx"], d))
        print(f"[resume] {len(rows)} rows read, {len(done)} "
              f"(question, dimension) pairs already judged — skipping those")
        if malformed:
            print(f"[resume] WARNING: {malformed} malformed line(s) ignored. "
                  f"Inspect {rows_path} before continuing — those questions "
                  f"will be RE-JUDGED as if never done.", file=sys.stderr)

    question_sets = load_bulk(args.from_bulk)
    print("[bulk] " + ", ".join(f"{k}={len(v)}" for k, v in question_sets.items()))

    total_calls = sum(min(len(v), args.limit or len(v))
                      for v in question_sets.values()) * len(dims) * args.rounds
    print(f"[cost] up to {total_calls} judge calls for this run")

    by_key = {(r["pipeline"], r.get("difficulty"), r["idx"]): r for r in rows}
    rf = open(rows_path, "a", encoding="utf-8")
    try:
        for name, qs in question_sets.items():
            if args.limit:
                qs = qs[:args.limit]
            print(f"\n[{name}] {len(qs)} questions x {len(dims)} dims x "
                  f"{args.rounds} rounds")
            for idx, q in enumerate(qs):
                diff = (q.get("difficulty") or "medium").lower()
                key = (name, diff, idx)
                row = by_key.get(key)
                if row is None:
                    row = {
                        "pipeline": name, "difficulty": diff, "idx": idx,
                        "topic": q.get("topic", ""),
                        "question": (q.get("question") or "")[:300],
                        "cognitive_type": q.get("cognitive_type", ""),
                    }
                    by_key[key] = row
                    rows.append(row)
                changed = False
                for d in dims:
                    if (name, diff, idx, d) in done:
                        continue
                    score, n_ok, just = judge_dimension(
                        q, content, d, args.judge_model, args.rounds)
                    row[d] = score
                    row[f"{d}_rounds_ok"] = n_ok
                    row[f"{d}_justification"] = just
                    changed = True
                if changed:
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rf.flush()
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
        "protocol": "EQGBench (arXiv:2508.10005). Judge system prompt, Query "
                    "scaffold, KP criteria + both worked examples + Note block "
                    "transcribed verbatim from Figure 3 (p.7); QT/QQ/SQ/CG "
                    "anchors verbatim from Section 4 (pp.5-6). One dimension "
                    f"per call, {args.rounds} rounds, mode with arithmetic-mean "
                    "tie-break (Section 5.2). temperature=0.6, max_tokens=4096.",
        "deviations_from_paper": [
            "Figure 3 supplies worked examples for KP only; QT/QQ/SQ/CG are "
            "zero-shot because the paper prints no examples for them.",
            "The CG 1-point ('Good') anchor does not exist in the paper — "
            "Section 4 defines only Excellent (2) and Poor (0). Our 1-point "
            "wording is authored by this harness. CG is therefore the least "
            "comparable of the five dimensions.",
            "EQGBench is Chinese middle-school math/physics/chemistry judged "
            "against a real user query; we judge English MCQs and reconstruct "
            "the 'User Input' from each question's topic and difficulty.",
            "Reference source material is appended to the prompt; the original "
            "EQGBench prompt has no such block.",
            "No public code release exists — everything is transcribed from "
            "the PDF, not cloned.",
        ],
        "paper_reference_scores": {
            "DeepSeek-R1_math": {"KP": 1.98, "QT": 1.95, "QQ": 1.95,
                                 "SQ": 1.96, "CG": 0.17},
            "GPT-4o_math": {"KP": 1.96, "QT": 1.82, "QQ": 1.76,
                            "SQ": 1.72, "CG": 0.21},
            "note": "Table 2. KP/QT/QQ/SQ saturate ~1.9-2.0; CG collapses.",
        },
        "config": {"backend": BASE_URL, "judge_model": args.judge_model,
                   "rounds": args.rounds, "dims": dims, "pipelines": pipelines,
                   "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
                   "n_rows": len(rows), "content_chars": len(content)},
        "aggregated": agg,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    cols = (["pipeline", "difficulty", "idx", "topic", "question",
             "cognitive_type"]
            + DIMS + [f"{d}_rounds_ok" for d in DIMS])
    with open(os.path.join(args.out, "rows.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print_table(agg, pipelines)
    print(f"\nWrote: {args.out}/results.json, rows.csv, rows.jsonl")


if __name__ == "__main__":
    main()
