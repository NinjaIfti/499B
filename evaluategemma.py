import argparse
import csv
import difflib
import json
import os
import re
import sys
import time
from collections import defaultdict

import requests


# Ollama

OLLAMA = "http://localhost:11434"


def _extract_json(text):
    """Pull the first JSON object."""
    if not text:
        return None
    text = text.strip()
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


def ollama_json(prompt, model, system="", num_ctx=8192, retries=2):
    """Call Ollama"""
    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }
    if system:
        body["system"] = system
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=300)
            r.raise_for_status()
            data = _extract_json(r.json().get("response", ""))
            if isinstance(data, dict):
                return data
        except Exception as e:
            if attempt == retries:
                print(f"    [warn] ollama call failed: {e}", file=sys.stderr)
        time.sleep(1)
    return {}



# A. Question quality — LLM as judge


DIFF_DESC = {
    "easy":   "tests basic recall, definitions, single-concept understanding",
    "medium": "requires understanding + application, may combine 2 concepts",
    "hard":   "requires deep analysis, multi-step reasoning, edge cases",
}


def judge_question(q, content, model):
    """Score ONE question 1-5 on four dimensions, decide accept/reject, and
    verify whether the MARKED answer is actually correct."""
    diff = (q.get("difficulty") or "medium").lower()
    opts = q.get("options") or {}
    opts_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    marked = str(q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]

    prompt = f"""You are an impartial, strict but fair exam-quality judge. Score this ONE
MCQ on a 1-5 scale for each dimension.

Target difficulty: {diff} - {DIFF_DESC.get(diff, '')}
Content snippet (ground truth):
{content[:4500]}

Question:
Q: {q.get('question','')}
Options:
{opts_txt}
Marked correct answer: {marked}

Score 1 (worst) to 5 (best) on:
1. correctness        - factually correct, answerable from the content
2. clarity            - unambiguous, well-formed wording
3. difficulty_match   - actual difficulty matches the target
4. distractor_quality - wrong options plausible but clearly wrong
5. overall            - holistic quality

BE DISCRIMINATING - use the FULL range, do not default to 5:
  5 = flawless; 4 = good; 3 = ordinary; 2 = weak; 1 = poor.
Penalise generic phrasing, near-duplicate framing, weak distractors, shallow recall.

Then decide accept (true/false), and separately decide answer_correct:
is the MARKED answer actually the correct answer to this question?

Return STRICT JSON:
{{"correctness":1,"clarity":1,"difficulty_match":1,"distractor_quality":1,
  "overall":1,"accept":true,"answer_correct":true,"reason":""}}"""

    r = ollama_json(prompt, model,
                    system="You are a strict but fair quiz judge. Return only JSON.")
    if not r:
        # fail closed: a broken judge call must not look like a good question
        return {"correctness": 0, "clarity": 0, "difficulty_match": 0,
                "distractor_quality": 0, "overall": 0,
                "accept": False, "answer_correct": False, "reason": "judge failed"}

    def num(k):
        try:
            return max(1, min(5, int(float(r.get(k, 0)))))
        except Exception:
            return 0

    out = {k: num(k) for k in
           ("correctness", "clarity", "difficulty_match", "distractor_quality", "overall")}
    out["accept"] = bool(r.get("accept", False)) and out["overall"] >= 3
    out["answer_correct"] = bool(r.get("answer_correct", False))
    out["reason"] = str(r.get("reason", ""))[:200]
    return out



# B. Answer correctness 

def solve_mcq(q, content, model):
    """Solve an MCQ from scratch WITHOUT being shown the marked answer."""
    opts = q.get("options") or {}
    opts_txt = "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))
    ctx = f"Reference material:\n{content[:3000]}\n\n" if content else ""
    prompt = (f"{ctx}Answer this multiple-choice question. Work out the answer step by "
              f"step, then give the single best option.\n\n"
              f"Q: {q.get('question','')}\n{opts_txt}\n\n"
              f'Return STRICT JSON: {{"answer":"A"}}')
    r = ollama_json(prompt, model,
                    system="You are an expert solver. Return only JSON.")
    choice = str(r.get("answer", "")).strip().upper()[:1]
    return choice if choice in ("A", "B", "C", "D") else ""


    """Gold calibration: how accurately can each model solve KNOWN-answer MCQ?
    This is what makes the independent-match rate interpretable — if the solver
    is only 55% accurate on questions whose answers we know, then 'the solver
    agrees with the generator' is weak evidence of correctness."""
    print(f"\n[GOLD] calibrating {len(models)} model(s) on {len(gold)} known MCQ")
    per_model = {}
    for m in models:
        hits = 0
        by_diff = defaultdict(lambda: [0, 0])
        by_type = defaultdict(lambda: [0, 0])   # computational vs conceptual
        for i, g in enumerate(gold, 1):
            truth = str(g.get("answer") or g.get("correct_answer") or "").strip().upper()[:1]
            got = solve_mcq(g, content, m)
            ok = (got == truth)
            hits += ok
            d = (g.get("difficulty") or "n/a").lower()
            by_diff[d][0] += ok
            by_diff[d][1] += 1
            t = (g.get("type") or "untyped").lower()
            by_type[t][0] += ok
            by_type[t][1] += 1
            if i % 10 == 0 or i == len(gold):
                print(f"    {m}: {hits}/{i}")
        per_model[m] = {
            "accuracy": round(hits / len(gold), 3) if gold else None,
            "correct": hits, "n": len(gold),
            "by_difficulty": {d: round(c / t, 3) for d, (c, t) in by_diff.items() if t},
            # Per-stratum accuracy. The generated questions are a MIX of computational
            # and conceptual, so a single overall gold number under- or over-states how
            # much to trust the solver — read the stratum matching the question type.
            "by_type": {k: {"accuracy": round(c / t, 3), "correct": c, "n": t}
                        for k, (c, t) in by_type.items() if t},
        }
    result = {"n": len(gold), "per_model": per_model}
    _write_json(os.path.join(out_dir, "gold.json"), result)
    for m, v in per_model.items():
        print(f"  [GOLD] {m}: {v['correct']}/{v['n']} = {v['accuracy']:.1%}")
        for k, s in v["by_type"].items():
            print(f"           {k:<14} {s['accuracy']:.1%}  ({s['correct']}/{s['n']})")
    return result



# Set-level metrics


def duplicate_rate(questions, threshold=0.85):
    """Fraction of questions that near-duplicate an earlier one."""
    texts = [(q.get("question") or "").strip().lower() for q in questions]
    if not texts:
        return 0.0
    dups = 0
    for i, t in enumerate(texts):
        if t and any(difflib.SequenceMatcher(None, p, t).ratio() > threshold
                     for p in texts[:i] if p):
            dups += 1
    return dups / len(texts)


_EMBEDDER = None


def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDER = False   # unavailable
    return _EMBEDDER or None


def grounding_and_diversity(questions, content):
    """grounding  = mean cosine of each question to its best-matching chunk of
                    the source text (is the question actually about the material?)
       diversity  = 1 - mean nearest-neighbour cosine between questions
                    (are the questions different from each other?)"""
    model = _embedder()
    if not model or not questions:
        return {"grounding": None, "diversity": None}
    import numpy as np
    qs = [(q.get("question") or "") for q in questions if q.get("question")]
    if not qs:
        return {"grounding": None, "diversity": None}
    qe = model.encode(qs, normalize_embeddings=True)

    grounding = None
    if content:
        chunks = [content[i:i + 500] for i in range(0, min(len(content), 40000), 500)]
        if chunks:
            ce = model.encode(chunks, normalize_embeddings=True)
            grounding = float(np.mean(np.max(qe @ ce.T, axis=1)))

    diversity = None
    if len(qs) > 1:
        sim = qe @ qe.T
        np.fill_diagonal(sim, -1.0)
        diversity = float(1.0 - np.mean(np.max(sim, axis=1)))

    return {"grounding": round(grounding, 3) if grounding is not None else None,
            "diversity": round(diversity, 3) if diversity is not None else None}


def f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if (p + r) else 0.0}


# Aggregation


def aggregate(rows, content):
    """Roll per-question rows up into per-(pipeline, difficulty) and per-pipeline
    metric blocks."""
    def block(sub):
        n = len(sub)
        if not n:
            return None

        def mean(k):
            vals = [r[k] for r in sub if isinstance(r.get(k), (int, float))]
            return round(sum(vals) / len(vals), 3) if vals else None

        qs = [{"question": r["question"]} for r in sub]
        dup = round(duplicate_rate(qs), 3)
        acc = round(sum(1 for r in sub if r["judge_accept"]) / n, 3)
        m = {
            "n": n,
            "mean_overall":            mean("judge_overall"),
            "mean_correctness":        mean("judge_correctness"),
            "mean_clarity":            mean("judge_clarity"),
            "mean_difficulty_match":   mean("judge_difficulty_match"),
            "mean_distractor_quality": mean("judge_distractor_quality"),
            "accept_rate":             acc,
            "duplicate_rate":        dup,
            "effective_accept_rate": round(acc * (1 - dup), 3),
            # answer correctness, the two independent checks
            "verified_correct_rate":  round(sum(1 for r in sub if r["answer_correct"]) / n, 3),
            "independent_match_rate": round(sum(1 for r in sub if r["independent_match"]) / n, 3),
        }
        m.update(grounding_and_diversity(qs, content))

        # if the set carries TRUSTED ground-truth answers, score against them too
        truth = [r for r in sub if r.get("true_answer")]
        if truth:
            m["marked_vs_truth_rate"] = round(
                sum(1 for r in truth if r["marked_answer"] == r["true_answer"]) / len(truth), 3)
            m["solver_vs_truth_rate"] = round(
                sum(1 for r in truth if r["independent_choice"] == r["true_answer"]) / len(truth), 3)
        return m

    pipelines = sorted({r["pipeline"] for r in rows})
    diffs = ["easy", "medium", "hard"]
    agg = {"overall": {p: block([r for r in rows if r["pipeline"] == p]) for p in pipelines}}
    for d in diffs:
        agg[d] = {p: block([r for r in rows if r["pipeline"] == p and r["difficulty"] == d])
                  for p in pipelines}
    return agg, pipelines



# I/O


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("questions", [])
    return [q for q in data if isinstance(q, dict) and q.get("question")]


def load_bulk(path):
    """Read the UI's bulk_eval_questions.json"""
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
                q["difficulty"] = diff          # target difficulty, not self-reported
            sets[pipeline].append(q)
    return dict(sets)


def write_csv(rows, path):
    cols = ["pipeline", "difficulty", "idx", "topic", "question", "marked_answer",
            "true_answer", "judge_overall", "judge_correctness", "judge_clarity",
            "judge_difficulty_match", "judge_distractor_quality", "judge_accept",
            "answer_correct", "independent_choice", "independent_match", "judge_reason"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def render_charts(agg, pipelines, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [charts] matplotlib unavailable — skipped")
        return
    diffs = ["easy", "medium", "hard"]

    def chart(metric, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        width = 0.8 / max(1, len(pipelines))
        for pi, p in enumerate(pipelines):
            vals = [((agg.get(d, {}).get(p) or {}).get(metric) or 0) for d in diffs]
            off = (pi - (len(pipelines) - 1) / 2) * width
            ax.bar([i + off for i in range(len(diffs))], vals, width, label=p)
        ax.set_xticks(range(len(diffs)))
        ax.set_xticklabels([d.upper() for d in diffs])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
        plt.close(fig)

    chart("independent_match_rate", "Answer accuracy (independent re-solve)", "rate", "chart_answer_accuracy.png")
    chart("verified_correct_rate",  "Answer correct (judge-verified)",        "rate", "chart_answer_verified.png")
    chart("accept_rate",            "Judge accept rate",                      "rate", "chart_accept_rate.png")
    chart("effective_accept_rate",  "Effective accept (duplicate-penalised)", "rate", "chart_effective_accept.png")
    chart("duplicate_rate",         "Duplicate rate",                         "rate", "chart_duplicate_rate.png")
    chart("mean_overall",           "Mean judge score",                       "1-5",  "chart_mean_overall.png")
    print(f"  [charts] written to {out_dir}")


def print_table(agg, pipelines):
    metrics = [
        ("mean_overall",           "Judge overall (1-5)"),
        ("accept_rate",            "Accept rate"),
        ("duplicate_rate",         "Duplicate rate"),
        ("effective_accept_rate",  "Effective accept"),
        ("grounding",              "Grounding"),
        ("independent_match_rate", "Answer acc. (re-solve)"),
        ("verified_correct_rate",  "Answer correct (judge)"),
    ]
    print("\n" + "=" * 78)
    print("RESULTS (overall)")
    print("=" * 78)
    print(f"{'Metric':<26}" + "".join(f"{p:>13}" for p in pipelines))
    print("-" * 78)
    for key, label in metrics:
        cells = ""
        for p in pipelines:
            v = (agg["overall"].get(p) or {}).get(key)
            cells += f"{'-' if v is None else f'{v:.3f}':>13}"
        print(f"{label:<26}{cells}")
    print("-" * 78)
    ns = "".join(f"{(agg['overall'].get(p) or {}).get('n', 0):>13}" for p in pipelines)
    print(f"{'N questions':<26}{ns}")
    print("=" * 78)



# Main


def main():
    global OLLAMA
    ap = argparse.ArgumentParser(
        description="Standalone MCQ evaluation harness (LLM-as-judge + answer correctness).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=FILE",
                    help="Question set to evaluate, e.g. --set cot=cot_limits_200.json "
                         "(repeatable, one per pipeline)")
    ap.add_argument("--from-bulk", metavar="FILE",
                    help="Read bulk_eval_questions.json produced by the UI's bulk "
                         "generation and evaluate every pipeline in it "
                         "(agentic/nonagentic/cot/pot) — no --set needed")
    ap.add_argument("--gold", help="JSON of KNOWN-answer MCQ for solver calibration")
    ap.add_argument("--content", help="Text file of the source material (for grounding + judging)")
    ap.add_argument("--judge-model", default="gemma3:12b")
    ap.add_argument("--solver-model", default=None,
                    help="Model for the independent Answer Generator (default: judge model)")
    ap.add_argument("--gold-models", default=None,
                    help="Comma-separated models to calibrate on gold (default: judge + gemma3:4b)")
    ap.add_argument("--ollama", default=OLLAMA)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--limit", type=int, default=0, help="Only evaluate the first N per set")
    ap.add_argument("--trusted", action="append", default=[],
                    help="Set names whose 'correct_answer' is VERIFIED ground truth "
                         "(e.g. --trusted cot). Adds accuracy-vs-truth columns.")
    ap.add_argument("--fresh", action="store_true", help="Ignore any previous rows and restart")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    OLLAMA = args.ollama
    solver_model = args.solver_model or args.judge_model
    os.makedirs(args.out, exist_ok=True)

    content = ""
    if args.content:
        with open(args.content, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        print(f"[content] {len(content)} chars from {args.content}")
    else:
        print("[content] none given — grounding disabled, judging without source material")

    # resume 
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

    #  collect the question sets to evaluate 
    question_sets = {}          
    if args.from_bulk:
        question_sets.update(load_bulk(args.from_bulk))
        print(f"[bulk] {args.from_bulk}: "
              + ", ".join(f"{k}={len(v)}" for k, v in question_sets.items()))
    for spec in args.set:
        if "=" not in spec:
            sys.exit(f"--set expects NAME=FILE, got: {spec}")
        name, path = spec.split("=", 1)
        question_sets[name] = load_questions(path)

    # judge + solve every question in every set 
    if question_sets:
        rf = open(rows_path, "a", encoding="utf-8")
        try:
            for name, qs in question_sets.items():
                if args.limit:
                    qs = qs[:args.limit]
                trusted = name in args.trusted
                print(f"\n[{name}] {len(qs)} questions"
                      + ("  (answers trusted as ground truth)" if trusted else ""))

                for idx, q in enumerate(qs):
                    if (name, idx) in done:
                        continue
                    marked = str(q.get("correct_answer") or q.get("answer") or "").strip().upper()[:1]

                    jr = judge_question(q, content, args.judge_model)      
                    choice = solve_mcq(q, content, solver_model)           

                    row = {
                        "pipeline": name,
                        "difficulty": (q.get("difficulty") or "medium").lower(),
                        "idx": idx,
                        "topic": q.get("topic", ""),
                        "question": (q.get("question") or "")[:300],
                        "marked_answer": marked,
                      
                        "true_answer": marked if trusted else "",
                        "judge_overall":            jr["overall"],
                        "judge_correctness":        jr["correctness"],
                        "judge_clarity":            jr["clarity"],
                        "judge_difficulty_match":   jr["difficulty_match"],
                        "judge_distractor_quality": jr["distractor_quality"],
                        "judge_accept":             jr["accept"],
                        "judge_reason":             jr["reason"],
                        "answer_correct":           jr["answer_correct"],
                        "independent_choice":       choice,
                        "independent_match":        bool(choice) and choice == marked,
                    }
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rf.flush()
                    rows.append(row)
                    if (idx + 1) % 5 == 0 or idx + 1 == len(qs):
                        print(f"    {name}: {idx + 1}/{len(qs)} judged")
        finally:
            rf.close()



    # aggregate + report 
    if not rows:
        print("\nNo question sets evaluated (use --set NAME=FILE or --from-bulk FILE). Done.")
        return

    agg, pipelines = aggregate(rows, content)
    results = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {"judge_model": args.judge_model, "solver_model": solver_model,
                   "pipelines": pipelines, "n_rows": len(rows),
                   "content_chars": len(content)},
        "aggregated": agg,
        "gold_calibration": gold_result,
    }
    _write_json(os.path.join(args.out, "results.json"), results)
    write_csv(rows, os.path.join(args.out, "rows.csv"))
    if not args.no_charts:
        render_charts(agg, pipelines, args.out)

    print_table(agg, pipelines)
    if gold_result:
        print("\nGOLD CALIBRATION (accuracy on known-answer MCQ — how much to trust the solver)")
        for m, v in gold_result["per_model"].items():
            print(f"  {m:<16} {v['accuracy']:.1%}  ({v['correct']}/{v['n']})")
    print(f"\nWrote: {args.out}/results.json, rows.csv, rows.jsonl"
          + (", gold.json" if gold_result else ""))


if __name__ == "__main__":
    main()
