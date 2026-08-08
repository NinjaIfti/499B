#!/usr/bin/env python3
"""
merged_agents.py — agent-MERGING ablation runner for LectureAssist.

STANDALONE. This file never imports colab.py and never modifies it. colab.py
carries `!pip` magics, so it is not importable as a plain module anyway; the
pieces it shares with the production pipeline (LLM wrapper, JSON repair chain,
prompts, schemas, cognitive router) are therefore PORTED here verbatim so the
two merged pipelines stay byte-comparable with the existing `agentic` control
run. If you change a prompt in colab.py, mirror it here or the comparison dies.

------------------------------------------------------------------------------
WHAT THIS RUNS
------------------------------------------------------------------------------
Two pipelines, each merging exactly ONE pair of agents relative to the full
agentic control. Everything else is held identical, so any difference is
attributable to the merge and nothing else.

  merge_vr : Validator + Refiner merged into ONE call.
             Control does: validate (call 1) -> if FAIL, regenerate (call 2).
             Here:         one call both judges the draft AND, when it fails
                           with action=regenerate, returns the rewritten
                           question in the same reply.
             Planner, retrieval tool and cognitive routing are UNCHANGED.

  merge_pg : Planner + Generator merged into ONE call.
             Control does: plan all topics up front (call 1), then generate
                           each question against its assigned topic.
             Here:         no standalone planner. Each question call picks its
                           own topic from the summary AND drafts the question,
                           given the topics already used so far.
             Validator and Refiner stay SEPARATE (as in control), retrieval
             and cognitive routing UNCHANGED.

------------------------------------------------------------------------------
WHY COST IS RECORDED
------------------------------------------------------------------------------
The point of merging is to cut LLM calls. Quality alone cannot show that, so
every question records `llm_calls`, and the artifact carries calls-per-accepted
-question per group. Report that column beside quality or the result is
unreadable: "merging barely hurt quality" only means something next to "and it
halved the calls".

------------------------------------------------------------------------------
OUTPUT
------------------------------------------------------------------------------
Writes the SAME artifact shape the existing harnesses already read:

    {"questions": {"<pipeline>:<difficulty>": [ {question, options, ...}, ... ]},
     "meta": {...}}

so evaluate_eqgbench.py and evaluate_requesta.py score it unchanged --
they split the group key on ":" into (pipeline, difficulty).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    python merged_agents.py --content chapter.txt --n 100 --pipeline both
    python merged_agents.py --content chapter.txt --n 100 --pipeline merge_vr
    python merged_agents.py --content chapter.txt --n 100 --resume   # continue

Then judge the output exactly as before:
    python evaluate_eqgbench.py --questions merged_eval_questions.json
    python evaluate_requesta.py --questions merged_eval_questions.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time

import requests as http_requests

# Windows consoles default to cp1252 and raise UnicodeEncodeError on the
# box-drawing/arrow characters in this file's progress output, killing a run
# mid-generation. Colab is already UTF-8; force it everywhere else rather than
# degrading the output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — mirrors colab.py. Keep these in sync or runs are not comparable.
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_BASE   = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_MODEL  = "gemma3:12b"     # primary (unused on the generation path here)
QUIZ_MODEL    = "gemma3:4b"      # generator/validator — PINNED, never escalates
QUIZ_CTX      = 16384
PIN_QUIZ_MODEL = True

# Cognitive routing is ON in the control run, so it is ON here too.
COGNITIVE_ROUTING = True

EMBED_MODEL   = "all-MiniLM-L6-v2"
RAG_TOP_K     = 5
RAG_CHUNK     = 900              # chars per chunk
RAG_OVERLAP   = 150

DIFFICULTIES  = ("easy", "medium", "hard")


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALL COUNTER — the whole point of the merge experiment
# ─────────────────────────────────────────────────────────────────────────────
class _Calls:
    def __init__(self):
        self.total = 0
        self._mark = 0

    def bump(self):
        self.total += 1

    def mark(self):
        self._mark = self.total

    def since_mark(self):
        return self.total - self._mark


CALLS = _Calls()


# ─────────────────────────────────────────────────────────────────────────────
# LLM WRAPPER + JSON REPAIR CHAIN — ported verbatim from colab.py
# Do not "clean this up": every branch here exists because gemma3:4b produced
# the exact malformation it repairs.
# ─────────────────────────────────────────────────────────────────────────────
def call_ollama(prompt, system="You are an expert educational AI assistant.",
                json_mode=False, max_retries=3, timeout=300,
                model=None, num_ctx=None):
    """Send a prompt to the local Ollama instance and return the response text."""
    payload = {
        "model":  model or OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        # num_predict is an OUTPUT-token ceiling. Without it Ollama caps
        # generation at ~128-256 tokens, which truncated MCQ JSON mid-object
        # and made the pinned 4B "return no questions".
        "options": {"num_ctx": num_ctx or 16384, "num_predict": 8192,
                    "temperature": 0.2},
    }
    if json_mode:
        # Deliberately NOT sending Ollama's format="json" grammar — under the
        # grammar, gemma's typographic quotes stall generation and the reply
        # arrives truncated and unrepairable. Without it the reply arrives
        # COMPLETE and the repair chain below fixes the quoting afterwards.
        payload["prompt"] = re.sub(r"[‘’‚‛]", "'", re.sub(r'[“”„‟]', "'", prompt))
        payload["system"] = system + (' In JSON output use the plain ASCII double-'
                                      'quote character (") for JSON syntax only. '
                                      'Never use curly/typographic quotes, and '
                                      'never put a double quote inside a string '
                                      "value — quote words with single quotes ' "
                                      'instead.')

    for attempt in range(1, max_retries + 1):
        try:
            CALLS.bump()
            resp = http_requests.post(f"{OLLAMA_BASE}/api/generate",
                                      json=payload, timeout=timeout)
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if json_mode:
                text = re.sub(r'^```(?:json)?\s*', '', text)
                text = re.sub(r'\s*```$', '', text)
            return text
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"  LLM attempt {attempt} failed: {e} — retrying…")
            time.sleep(2)


def _fix_bad_escapes(raw):
    """Escape backslash sequences that are not valid JSON escapes (LaTeX-ish
    \\lim, \\epsilon … whose lone backslash makes json.loads reject the reply)."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)


def _fix_smart_quotes(raw):
    """Convert curly quotes to ASCII ONLY where they act as JSON delimiters."""
    fixed = re.sub(r'(?<=[{\[,:])(\s*)[“”]', r'\1"', raw)
    return re.sub(r'[“”](?=\s*[}\],:])', '"', fixed)


def _escape_inner_quotes(raw):
    """Escape ASCII double quotes that are CONTENT, not JSON delimiters."""
    out, n = [], len(raw)
    for i, ch in enumerate(raw):
        if ch == '"':
            j = i - 1
            while j >= 0 and raw[j] in ' \t\r\n':
                j -= 1
            prev = raw[j] if j >= 0 else ''
            k = i + 1
            while k < n and raw[k] in ' \t\r\n':
                k += 1
            nxt = raw[k] if k < n else ''
            is_open  = prev in '{[,:' or prev == ''
            is_close = nxt in ':,}]' or nxt == ''
            if not is_open and not is_close and prev != '\\':
                out.append('\\"')
                continue
        out.append(ch)
    return ''.join(out)


try:
    from json_repair import repair_json as _json_repair
except Exception:
    _json_repair = None


def _drop_stray_eol_quotes(raw):
    """Remove a lone quote dangling after a comma at end-of-line."""
    return re.sub(r',[ \t]*"[ \t]*(?=\r?\n)', ',', raw)


def _close_options_object(raw):
    """Insert the missing `}` that closes "options" before "correct_answer"."""
    return re.sub(r'("options"\s*:\s*\{[^{}]*?)(,\s*"correct_answer")',
                  r'\1}\2', raw)


def _slice_json_block(raw):
    """Cut leading/trailing prose around the outermost JSON value."""
    starts = [i for i in (raw.find("{"), raw.find("[")) if i != -1]
    start  = min(starts) if starts else -1
    end    = max(raw.rfind("}"), raw.rfind("]"))
    if start == -1 or end <= start:
        return raw
    return raw[start:end + 1]


def call_ollama_json(prompt, system="You are an expert educational AI assistant.",
                     fallback=None, model=None, num_ctx=None):
    """call_ollama with JSON mode. Returns parsed dict/list, or *fallback*."""
    last_err, raw = None, ""
    for attempt in (1, 2):
        try:
            raw = call_ollama(prompt, system=system, json_mode=True,
                              model=model, num_ctx=num_ctx)
        except Exception as e:
            print(f"  LLM call failed: {e}")
            break
        base = _slice_json_block(raw)
        sq   = _fix_smart_quotes(base)
        lq   = _drop_stray_eol_quotes(sq)
        co   = _close_options_object(lq)
        iq   = _escape_inner_quotes(co)
        cands = []
        for c in (raw, base, sq, lq, co, iq):
            for v in (c, _fix_bad_escapes(c)):
                if v not in cands:
                    cands.append(v)
        for candidate in cands:
            try:
                return json.loads(candidate, strict=False)
            except Exception as e:
                last_err = e
        if _json_repair is not None:
            try:
                obj = _json_repair(base, return_objects=True)
                if isinstance(obj, (dict, list)) and obj:
                    print("  ⚠ JSON recovered by json_repair fallback")
                    return obj
            except Exception as e:
                last_err = e
        pos     = getattr(last_err, "pos", 0) or 0
        snippet = raw[max(0, pos - 60):pos + 60].replace("\n", " ")
        print(f"  JSON parse error (attempt {attempt}/2): {last_err} — near: …{snippet}…")
    return fallback if fallback is not None else {}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PROMPT MATERIAL — ported verbatim from colab.py
# ─────────────────────────────────────────────────────────────────────────────
MCQ_SCHEMA = ('{"questions":[{"question":"","options":{"A":"","B":"","C":"","D":""},'
              '"correct_answer":"A","explanation":"","topic":"","difficulty":"",'
              '"bloom_level":"","source_timestamp":""}]}')

DIFF_DESCRIPTIONS = {
    "easy":   "tests basic recall, definitions, single-concept understanding",
    "medium": "requires understanding + application, may combine 2 concepts",
    "hard":   "requires deep analysis, multi-step reasoning, edge cases, synthesis",
}

COGNITIVE_TRACKS = {
    "text_based": (
        "TEXT-BASED (recall) — target a fact, definition or relationship stated "
        "explicitly in the content. The answer must be locatable in the text."
    ),
    "inferential": (
        "INFERENTIAL — the answer must NOT appear verbatim. It must require "
        "combining two or more separate points from the content, or reasoning to "
        "a consequence the material implies but never states outright."
    ),
    "main_idea": (
        "MAIN IDEA (synthesis) — target the overarching argument or central theme "
        "of the material as a whole, not any single local detail."
    ),
}
_COGNITIVE_ORDER = ("text_based", "inferential", "main_idea")

MCQ_STRATEGIES = [
    "Concept check style: test one core concept precisely with plausible distractors.",
    "Scenario style: short practical situation and ask the best next reasoning choice.",
    "Compare-and-justify style: ask which option is best and why others are weaker.",
]


def _cognitive_type_for_slot(idx):
    """Round-robin the cognitive types across slots — deterministic, no LLM call."""
    return _COGNITIVE_ORDER[idx % len(_COGNITIVE_ORDER)]


def _topic_coverage(topic, content_snip):
    """Fraction of a topic's significant words present in the content."""
    words = [w for w in re.split(r'\W+', (topic or '').lower()) if len(w) > 3]
    if not words:
        return 1.0
    low = content_snip.lower()
    return sum(1 for w in words if w in low) / len(words)


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL — same embedder and index type as the control run
# ─────────────────────────────────────────────────────────────────────────────
class LectureRAG:
    """SentenceTransformer + FAISS inner-product index over L2-normalised
    vectors, so inner product == cosine similarity. Same configuration as the
    control run; if it differs, the retrieval tool is not held constant and the
    merge comparison is confounded."""

    def __init__(self):
        self.embedder = None
        self.index    = None
        self.chunks   = []

    def build(self, text):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except Exception as e:
            raise RuntimeError(
                "Retrieval needs sentence-transformers + faiss:\n"
                "    pip install sentence-transformers faiss-cpu\n"
                "Or pass --no-retrieval to run without the tool — but note that "
                "disables an agent the control run HAD, so the result is then an "
                "ablation, not a clean merge comparison.\n"
                f"(import failed: {e})"
            )
        self.embedder = SentenceTransformer(EMBED_MODEL)
        step   = max(1, RAG_CHUNK - RAG_OVERLAP)
        self.chunks = [text[i:i + RAG_CHUNK]
                       for i in range(0, len(text), step)
                       if text[i:i + RAG_CHUNK].strip()]
        vecs = self.embedder.encode(self.chunks, batch_size=64,
                                    show_progress_bar=False).astype('float32')
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        print(f"  RAG index built — {len(self.chunks)} chunks.")

    def query(self, question, top_k=RAG_TOP_K):
        if self.index is None:
            return ""
        import faiss
        vec = self.embedder.encode([question]).astype('float32')
        faiss.normalize_L2(vec)
        _, idxs = self.index.search(vec, top_k)
        return "\n\n".join(self.chunks[i] for i in idxs[0] if i < len(self.chunks))


rag = LectureRAG()


def _fetch_more_context(topic, question_text, top_k=RAG_TOP_K):
    """Remediation tool: pull extra lecture context for an under-grounded topic."""
    if rag.index is None:
        return ""
    q = f"{topic} {question_text}".strip()
    try:
        return rag.query(q, top_k=top_k) or ""
    except Exception:
        return ""


def _gather_grounding(topic, content_snip, summary, max_tool_calls=2):
    """Model-directed tool use, run BEFORE drafting: the generator decides for
    itself whether it holds enough material on the topic and, if not, issues its
    own search_lecture(query). UNCHANGED from control in both pipelines."""
    if rag.index is None:
        return content_snip
    if _topic_coverage(topic, content_snip) >= 0.6:
        return content_snip

    enriched = content_snip
    for _ in range(max(1, max_tool_calls)):
        prompt = (
            f"You will write a MCQ question on the topic '{topic}'.\n"
            "Tool available — search_lecture(query): returns more text from THIS lecture.\n"
            "Does the CONTENT below already contain enough specific material on that "
            "topic to write a well-grounded question?\n"
            '- If yes: return {"action":"ready"}\n'
            '- If not: return {"action":"search","query":"<short search query>"}\n\n'
            f"CONTENT:\n{enriched[:2800]}\n\nSUMMARY:\n{summary[:500]}\n\n"
            'Return STRICT JSON: {"action":"ready|search","query":"..."}'
        )
        data   = call_ollama_json(prompt, fallback={"action": "ready"},
                                  model=QUIZ_MODEL, num_ctx=QUIZ_CTX)
        action = str((data or {}).get("action", "ready")).strip().lower()
        query  = str((data or {}).get("query", "")).strip()
        if action != "search" or not query:
            break
        extra = _fetch_more_context(query, "")
        if not extra or extra[:160] in enriched:
            break
        enriched = extra + "\n\n" + enriched
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC PRE-CHECKS — free, no LLM call, identical in both pipelines
# ─────────────────────────────────────────────────────────────────────────────
def _structural_reject(q):
    """Reject malformed MCQs before spending a validation call on them."""
    qtxt = (q.get("question") or "").strip()
    opts = q.get("options") or {}
    if not qtxt:
        return "empty question text"
    if not isinstance(opts, dict) or len(opts) < 4:
        return "fewer than 4 options"
    if any(not str(v).strip() for v in opts.values()):
        return "an option is empty"
    if str(q.get("correct_answer", "")).strip().upper() not in ("A", "B", "C", "D"):
        return "correct_answer is not one of A-D"
    return ""


def _duplicate_reject(q, existing):
    """Cheap deterministic uniqueness gate, identical to the control run."""
    qtxt_low = (q.get("question") or "").strip().lower()
    if not qtxt_low:
        return ""
    for ex in existing:
        ex_text = (ex.get("question") or "").strip().lower()
        if not ex_text:
            continue
        if ex_text == qtxt_low:
            return "duplicate of an existing question"
        ratio = difflib.SequenceMatcher(None, ex_text, qtxt_low).ratio()
        if ratio > 0.85:
            return f"too similar ({int(ratio * 100)}%) to an existing question"
    return ""


def _resolve_action(checks, raw_action):
    """Same deterministic action guards as the control validator."""
    action = str(raw_action or "").strip().lower()
    if action not in ("regenerate", "fetch_context", "replan_topic"):
        action = ""
    if checks.get("grounded") == "FAIL":
        action = action or "fetch_context"
    elif checks.get("difficulty") == "FAIL":
        action = "regenerate"
    return action or "regenerate"


def _apply_verdict_policy(checks, reason, user_instr):
    """The control run's blocking policy, reproduced exactly:
      - grounded must pass (hard)
      - instruction must pass only when the user actually gave instructions
      - difficulty mismatch is NON-blocking (the 4B validator is noisy on it)
    Returns (passed, reason)."""
    hard_fail = []
    if checks.get("grounded") == "FAIL":
        hard_fail.append("grounded")
    if user_instr and checks.get("instruction") == "FAIL":
        hard_fail.append("instruction")
    if hard_fail:
        return False, reason or f"failed checks: {', '.join(hard_fail)}"
    if checks.get("difficulty") == "FAIL":
        return True, "difficulty borderline; accepted (grounded + instruction-safe)"
    return True, reason


# ─────────────────────────────────────────────────────────────────────────────
# THE GENERATOR — shared by both pipelines when the planner is separate
# ─────────────────────────────────────────────────────────────────────────────
def _generate_draft(topic, difficulty, content, summary, user_instr,
                    existing, strategy, cognitive_type, feedback=""):
    """One drafting call. Prompt is the control run's, unchanged."""
    diff_desc = DIFF_DESCRIPTIONS.get(difficulty, "")
    cog_desc  = COGNITIVE_TRACKS.get(cognitive_type or "", "")
    cog_block = f"Cognitive type (mandatory): {cog_desc}\n" if cog_desc else ""

    feedback_block = ""
    if feedback:
        feedback_block = (f"\nPREVIOUS ATTEMPT FAILED: {feedback}\n"
                          "You MUST fix this exact issue in the new question.\n")

    existing_summary = ""
    if existing:
        existing_summary = ("\nDO NOT duplicate or paraphrase these accepted questions:\n- "
                            + "\n- ".join((q.get("question") or "")[:90] for q in existing)
                            + "\n")

    instr_line = f"\nStudent instructions: {user_instr}\n" if user_instr else ""

    prompt = (
        f"Generate exactly ONE MCQ question wrapped in a JSON array.\n"
        f"Topic: '{topic}'\n"
        f"Difficulty: {difficulty.upper()} — {diff_desc}\n"
        f"Question strategy for this attempt: {strategy}\n"
        f"{cog_block}"
        "GROUNDING RULE: the question and its answer MUST be answerable using ONLY "
        "the CONTENT below. Do NOT introduce formulas, methods, numbers, or concepts "
        "that are not present in it — not even to make it harder. Make it harder by "
        "deeper reasoning over the SAME material, never by adding outside topics.\n"
        f"{feedback_block}{existing_summary}{instr_line}"
        f"Output STRICT JSON in this exact shape (a 'questions' array containing exactly 1 object):\n"
        f"{MCQ_SCHEMA}\n\n"
        f"CONTENT:\n{content[:5500]}\n\n"
        f"SUMMARY:\n{summary[:1200]}"
    )
    data = call_ollama_json(prompt, fallback={"questions": []},
                            model=QUIZ_MODEL, num_ctx=QUIZ_CTX)
    qs = data.get("questions", []) if isinstance(data, dict) else []
    return qs[0] if qs else None


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE 1 — merge_vr : VALIDATOR + REFINER IN ONE CALL
# ─────────────────────────────────────────────────────────────────────────────
def _validate_and_refine(question, content_snip, difficulty, user_instr):
    """THE MERGE. One call does what control does in two.

    Control: validator returns PASS/FAIL + action; on action=regenerate a
    SECOND call redrafts the question from the original prompt plus the failure
    reason. Here the validator is told to repair what it rejects, and returns
    the corrected question inline.

    Note the asymmetry that makes this a fair merge rather than a shortcut:
    only 'regenerate' can be absorbed into this call. 'fetch_context' and
    'replan_topic' need work OUTSIDE the model (a RAG query, a topic swap), so
    they are still returned as actions for the loop to act on — exactly as in
    control. Merging cannot remove a step that was never an LLM call.

    Returns (passed, reason, checks, refined_question_or_None)."""
    qtxt = (question.get("question") or "")[:300]
    ans  = str(question.get("correct_answer") or "")[:120]
    opts = question.get("options") or {}
    diff_desc  = DIFF_DESCRIPTIONS.get(difficulty, "")
    instr_note = (f"\nMust comply with student instructions: {user_instr}"
                  if user_instr else "")

    prompt = (
        "You are a strict but fair quiz quality validator AND rewriter. "
        "Apply 3 checks to ONE question:\n"
        f"1. DIFFICULTY: Is it genuinely {difficulty.upper()}? ({diff_desc})\n"
        "2. GROUNDED: Is it answerable from the content snippet below?\n"
        f"3. INSTRUCTION: Does it follow the student instructions?{instr_note}\n\n"
        f"Content snippet:\n{content_snip[:2500]}\n\n"
        "Question to validate:\n"
        f"Q: {qtxt}\n"
        f"Options: {json.dumps(opts, ensure_ascii=False)[:500]}\n"
        f"A: {ans}\n\n"
        "If it FAILS, also choose the single best remediation 'action':\n"
        "- 'fetch_context': it fails because the snippet lacks the needed facts "
        "(GROUNDED fail) — more lecture context would let a good question be written.\n"
        "- 'replan_topic': the topic itself does not appear in the lecture at all — "
        "no amount of rewriting or context will help; a different topic is needed.\n"
        "- 'regenerate': wording/clarity/difficulty issues a plain rewrite can fix.\n\n"
        "CRITICAL — you are also the rewriter. If and ONLY IF action is "
        "'regenerate', you MUST return the corrected question in the 'refined' "
        "field, fully rewritten to fix the exact problem you identified, still "
        "answerable using ONLY the content snippet above, with 4 options and the "
        "correct one marked. For 'fetch_context' or 'replan_topic', or when "
        "overall is PASS, set \"refined\":null — do not invent material you do "
        "not have.\n\n"
        "Return STRICT JSON:\n"
        '{"difficulty":"PASS|FAIL","grounded":"PASS|FAIL",'
        '"instruction":"PASS|FAIL","overall":"PASS|FAIL",'
        '"action":"regenerate|fetch_context|replan_topic",'
        '"reason":"1-line concrete fix hint if FAIL",'
        '"refined":{"question":"","options":{"A":"","B":"","C":"","D":""},'
        '"correct_answer":"A","explanation":"","topic":"","difficulty":"",'
        '"bloom_level":"","source_timestamp":""}}'
    )

    result = call_ollama_json(
        prompt,
        system="You are a strict but fair quiz validator and rewriter. Return only JSON.",
        fallback={"overall": "PASS", "refined": None},
        model=QUIZ_MODEL, num_ctx=QUIZ_CTX,
    )
    if not isinstance(result, dict):
        return True, "", {"difficulty": "PASS", "grounded": "PASS",
                          "instruction": "PASS", "action": "regenerate"}, None

    checks = {
        "difficulty":  str(result.get("difficulty",  "PASS")).upper(),
        "grounded":    str(result.get("grounded",    "PASS")).upper(),
        "instruction": str(result.get("instruction", "PASS")).upper(),
    }
    checks["action"] = _resolve_action(checks, result.get("action"))
    passed, reason   = _apply_verdict_policy(checks, result.get("reason", ""),
                                             user_instr)

    refined = result.get("refined")
    if not isinstance(refined, dict) or _structural_reject(refined):
        refined = None      # a malformed rewrite is no rewrite
    else:
        # The rewriter routinely drops metadata fields it was not asked to
        # think about. Carry them over from the question it rewrote, or the
        # artifact loses `topic` — which EQGBench needs to reconstruct the
        # user instruction it scores against.
        for field in ("topic", "difficulty", "bloom_level", "source_timestamp"):
            if not str(refined.get(field, "")).strip():
                refined[field] = question.get(field, "")
    return passed, reason, checks, refined


def _one_question_merge_vr(topic, difficulty, content_snip, summary, user_instr,
                            existing, idx, max_attempts=3):
    """Per-question loop with validate+refine merged.

    The saving: on a 'regenerate' failure the control run spends validate(1) +
    redraft(1) = 2 calls per retry; here the single merged call returns both the
    verdict and the rewrite, so the retry costs 1."""
    qid       = f"q{idx + 1}"
    cog_type  = _cognitive_type_for_slot(idx) if COGNITIVE_ROUTING else None
    working   = _gather_grounding(topic, content_snip, summary)
    cur_topic = topic
    fetched   = replanned = False
    last_fail = ""
    best      = None
    CALLS.mark()

    draft = None
    for attempt in range(1, max_attempts + 1):
        strategy = MCQ_STRATEGIES[(attempt - 1) % len(MCQ_STRATEGIES)]

        # First attempt drafts; later attempts may already hold a refined draft
        # handed back by the merged call, in which case no drafting call is made.
        if draft is None:
            draft = _generate_draft(cur_topic, difficulty, working, summary,
                                    user_instr, existing, strategy, cog_type,
                                    feedback=last_fail if attempt > 1 else "")
        if draft is None:
            last_fail = "model returned no question"
            continue

        bad = _structural_reject(draft) or _duplicate_reject(draft, existing)
        if bad:
            last_fail, draft = bad, None
            continue

        passed, reason, checks, refined = _validate_and_refine(
            draft, working, difficulty, user_instr)

        if passed:
            draft["_verdict"]     = "PASS"
            draft["_attempts"]    = attempt
            draft["_llm_calls"]   = CALLS.since_mark()
            draft["_fail_reason"] = reason
            return draft

        best      = best or draft
        last_fail = reason
        action    = checks["action"]

        # Same escalation ladder as control: don't repeat a remediation.
        if action == "fetch_context" and fetched:
            action = "replan_topic"
        if action == "replan_topic" and replanned:
            action = "regenerate"

        if action == "fetch_context":
            extra = _fetch_more_context(cur_topic, draft.get("question", ""))
            if extra:
                working = extra + "\n\n" + working
                fetched = True
            draft = None                      # must redraft with new context
        elif action == "replan_topic":
            new_topic = _replan_topic(cur_topic, summary, existing)
            if new_topic:
                cur_topic = new_topic
                replanned = True
            draft = None                      # must redraft on the new topic
        else:
            # action == regenerate: THE MERGE PAYS OFF. The rewrite already
            # came back in the validation reply, so the next iteration
            # validates it directly instead of spending a drafting call.
            draft = refined                   # None -> falls back to redrafting

    # Retry budget exhausted: keep the best attempt, explicitly labelled.
    # Never silently recorded as a clean PASS — otherwise the pipeline's own
    # accept rate would be a fiction.
    if best is not None:
        best["_verdict"]     = "OVERRIDE"
        best["_attempts"]    = max_attempts
        best["_llm_calls"]   = CALLS.since_mark()
        best["_fail_reason"] = last_fail
    return best


def _replan_topic(current_topic, summary, existing):
    """Remediation tool shared by both pipelines: ask for a different topic that
    IS covered in the lecture. Same as control."""
    used = ", ".join(sorted({(q.get("topic") or "") for q in existing if q.get("topic")}))
    prompt = (
        f"The topic '{current_topic}' does not appear in this lecture, so no "
        "question can be grounded in it. Choose ONE different topic that IS "
        "covered in the summary below and is not already heavily used.\n\n"
        f"Already used topics: {used or 'none'}\n\n"
        f"SUMMARY:\n{summary[:1800]}\n\n"
        'Return STRICT JSON: {"topic":"<the new topic>"}'
    )
    data = call_ollama_json(prompt, fallback={}, model=QUIZ_MODEL, num_ctx=QUIZ_CTX)
    return str((data or {}).get("topic", "")).strip()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE 2 — merge_pg : PLANNER + GENERATOR IN ONE CALL
# ─────────────────────────────────────────────────────────────────────────────
def _plan_and_generate(difficulty, content, summary, user_instr, existing,
                       strategy, cognitive_type, feedback=""):
    """THE MERGE. One call selects the topic AND writes the question.

    Control spends one planner call up front for the whole batch, then one
    drafting call per question against an assigned topic. Here there is no
    planner: each drafting call is shown the topics already used and must pick
    an unused one that the summary actually covers, then write the question for
    it in the same reply.

    This trades a global view for a local one — the model can no longer balance
    topic counts across the whole quiz, only avoid what it has already seen.
    Whether that costs diversity is precisely the thing being measured."""
    diff_desc = DIFF_DESCRIPTIONS.get(difficulty, "")
    cog_desc  = COGNITIVE_TRACKS.get(cognitive_type or "", "")
    cog_block = f"Cognitive type (mandatory): {cog_desc}\n" if cog_desc else ""

    used_topics = sorted({(q.get("topic") or "").strip()
                          for q in existing if (q.get("topic") or "").strip()})
    used_block = (f"\nTopics already used (choose a DIFFERENT one): "
                  f"{', '.join(used_topics)}\n" if used_topics else "")

    existing_summary = ""
    if existing:
        existing_summary = ("\nDO NOT duplicate or paraphrase these accepted questions:\n- "
                            + "\n- ".join((q.get("question") or "")[:90] for q in existing)
                            + "\n")

    feedback_block = ""
    if feedback:
        feedback_block = (f"\nPREVIOUS ATTEMPT FAILED: {feedback}\n"
                          "You MUST fix this exact issue in the new question.\n")

    instr_line = f"\nStudent instructions: {user_instr}\n" if user_instr else ""

    prompt = (
        "You are a combined quiz PLANNER and GENERATOR. In one step:\n"
        "STEP 1 (plan): read the SUMMARY and CONTENT and choose ONE specific "
        "topic that the lecture genuinely covers and that is not already used.\n"
        "STEP 2 (generate): write exactly ONE MCQ question on that topic.\n\n"
        f"Difficulty: {difficulty.upper()} — {diff_desc}\n"
        f"Question strategy for this attempt: {strategy}\n"
        f"{cog_block}{used_block}"
        "GROUNDING RULE: the question and its answer MUST be answerable using ONLY "
        "the CONTENT below. Do NOT introduce formulas, methods, numbers, or concepts "
        "that are not present in it — not even to make it harder. Make it harder by "
        "deeper reasoning over the SAME material, never by adding outside topics.\n"
        "The 'topic' field MUST hold the topic you chose in STEP 1.\n"
        f"{feedback_block}{existing_summary}{instr_line}"
        f"Output STRICT JSON in this exact shape (a 'questions' array containing exactly 1 object):\n"
        f"{MCQ_SCHEMA}\n\n"
        f"CONTENT:\n{content[:5500]}\n\n"
        f"SUMMARY:\n{summary[:1200]}"
    )
    data = call_ollama_json(prompt, fallback={"questions": []},
                            model=QUIZ_MODEL, num_ctx=QUIZ_CTX)
    qs = data.get("questions", []) if isinstance(data, dict) else []
    return qs[0] if qs else None


def _validate_only(question, content_snip, difficulty, user_instr):
    """The control run's SEPARATE validator, unchanged. Used by merge_pg so that
    the only thing differing from control is the plan/generate merge."""
    qtxt = (question.get("question") or "")[:300]
    ans  = str(question.get("correct_answer") or "")[:120]
    diff_desc  = DIFF_DESCRIPTIONS.get(difficulty, "")
    instr_note = (f"\nMust comply with student instructions: {user_instr}"
                  if user_instr else "")

    prompt = (
        "You are a strict but fair quiz quality validator. Apply 3 checks to ONE question:\n"
        f"1. DIFFICULTY: Is it genuinely {difficulty.upper()}? ({diff_desc})\n"
        "2. GROUNDED: Is it answerable from the content snippet below?\n"
        f"3. INSTRUCTION: Does it follow the student instructions?{instr_note}\n\n"
        f"Content snippet:\n{content_snip[:2500]}\n\n"
        "Question to validate:\n"
        f"Q: {qtxt}\n"
        f"A: {ans}\n\n"
        "If it FAILS, also choose the single best remediation 'action':\n"
        "- 'fetch_context': it fails because the snippet lacks the needed facts "
        "(GROUNDED fail) — more lecture context would let a good question be written.\n"
        "- 'replan_topic': the topic itself does not appear in the lecture at all — "
        "no amount of rewriting or context will help; a different topic is needed.\n"
        "- 'regenerate': wording/clarity/difficulty issues a plain rewrite can fix.\n\n"
        "Return STRICT JSON:\n"
        '{"difficulty":"PASS|FAIL","grounded":"PASS|FAIL",'
        '"instruction":"PASS|FAIL","overall":"PASS|FAIL",'
        '"action":"regenerate|fetch_context|replan_topic",'
        '"reason":"1-line concrete fix hint if FAIL"}'
    )
    result = call_ollama_json(
        prompt,
        system="You are a strict but fair quiz validator. Return only JSON.",
        fallback={"overall": "PASS"}, model=QUIZ_MODEL, num_ctx=QUIZ_CTX,
    )
    if not isinstance(result, dict):
        return True, "", {"difficulty": "PASS", "grounded": "PASS",
                          "instruction": "PASS", "action": "regenerate"}
    checks = {
        "difficulty":  str(result.get("difficulty",  "PASS")).upper(),
        "grounded":    str(result.get("grounded",    "PASS")).upper(),
        "instruction": str(result.get("instruction", "PASS")).upper(),
    }
    checks["action"] = _resolve_action(checks, result.get("action"))
    passed, reason   = _apply_verdict_policy(checks, result.get("reason", ""),
                                             user_instr)
    return passed, reason, checks


def _one_question_merge_pg(difficulty, content_snip, summary, user_instr,
                           existing, idx, max_attempts=3):
    """Per-question loop with plan+generate merged, validator/refiner separate."""
    cog_type = _cognitive_type_for_slot(idx) if COGNITIVE_ROUTING else None
    working  = content_snip
    fetched  = False
    last_fail = ""
    best      = None
    CALLS.mark()

    for attempt in range(1, max_attempts + 1):
        strategy = MCQ_STRATEGIES[(attempt - 1) % len(MCQ_STRATEGIES)]
        draft = _plan_and_generate(difficulty, working, summary, user_instr,
                                   existing, strategy, cog_type,
                                   feedback=last_fail if attempt > 1 else "")
        if draft is None:
            last_fail = "model returned no question"
            continue

        # The merged call chose its own topic, so grounding is gathered AFTER
        # the topic is known rather than before it — the tool is still available
        # on the retry path, which is where control uses it too.
        bad = _structural_reject(draft) or _duplicate_reject(draft, existing)
        if bad:
            last_fail = bad
            continue

        # The merged call is responsible for choosing the topic; if it omitted
        # one, the used-topics block on later questions silently degrades, so
        # make the omission visible rather than letting it read as "no topic".
        if not str(draft.get("topic", "")).strip():
            draft["topic"] = "Unspecified"

        passed, reason, checks = _validate_only(draft, working, difficulty,
                                                user_instr)
        if passed:
            draft["_verdict"]     = "PASS"
            draft["_attempts"]    = attempt
            draft["_llm_calls"]   = CALLS.since_mark()
            draft["_fail_reason"] = reason
            return draft

        best      = best or draft
        last_fail = reason
        action    = checks["action"]

        # replan_topic is a no-op here by construction: the generator replans
        # itself on every call, which is the whole point of the merge. Treat it
        # as a plain regenerate rather than spending a call on a separate
        # replanner that this pipeline does not have.
        if action == "fetch_context" and not fetched:
            extra = _fetch_more_context(draft.get("topic") or "",
                                        draft.get("question", ""))
            if extra:
                working = extra + "\n\n" + working
                fetched = True

    if best is not None:
        best["_verdict"]     = "OVERRIDE"
        best["_attempts"]    = max_attempts
        best["_llm_calls"]   = CALLS.since_mark()
        best["_fail_reason"] = last_fail
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER (separate) — used by merge_vr only, unchanged from control
# ─────────────────────────────────────────────────────────────────────────────
def plan_quiz(num, difficulty, summary, user_instr=""):
    """The control run's QuizPlannerAgent prompt, reduced to manual-difficulty
    mode (the bulk eval runs a fixed target difficulty per group)."""
    prompt = (
        f"You are a quiz planning agent. Decide how to distribute {num} "
        f"MCQ questions across topics for a student.\n\n"
        f"Quiz type: MCQ\n"
        f"Target difficulty: {difficulty.upper()} (uniform, manual mode)\n"
        f"Total questions: {num}\n"
        f"Student instructions: {user_instr or 'None'}\n\n\n"
        f"Lecture summary (excerpt):\n{summary[:2000]}\n\n"
        "Return STRICT JSON (no extra text):\n"
        '{"topics":["topic1","topic2"],'
        '"per_topic_count":{"topic1":5,"topic2":5},'
        '"per_topic_difficulty":{"topic1":"easy","topic2":"medium"},'
        '"bloom_targets":["Remember","Apply"],'
        '"focus_notes":"what to emphasize or avoid based on instructions"}\n\n'
        "Rules:\n"
        f"- Sum of per_topic_count MUST equal {num}\n"
        f"- MANUAL: all topics use {difficulty} difficulty\n"
        "- Use 2-4 topics maximum"
    )
    plan = call_ollama_json(
        prompt, system="You are an expert quiz planning agent. Return only valid JSON.",
        fallback=None, model=QUIZ_MODEL, num_ctx=QUIZ_CTX,
    )
    if not isinstance(plan, dict) or not plan.get("topics"):
        topics = ["Core Concepts", "Key Principles", "Applications"]
        base, rem = num // len(topics), num % len(topics)
        counts = {t: base for t in topics}
        counts[topics[0]] += rem
        plan = {"topics": topics, "per_topic_count": counts,
                "per_topic_difficulty": {t: difficulty for t in topics},
                "focus_notes": "Fallback plan — planner LLM returned invalid JSON"}

    counts = plan.get("per_topic_count", {})
    if counts:
        total = sum(counts.values())
        if total != num:
            tlist = list(counts.keys())
            counts[tlist[-1]] = max(1, counts[tlist[-1]] + (num - total))
            plan["per_topic_count"] = counts

    slots = []
    for topic in plan.get("topics", []):
        for _ in range(counts.get(topic, 0)):
            slots.append((topic, plan.get("per_topic_difficulty", {})
                          .get(topic, difficulty)))
    if not slots:
        slots = [("General", difficulty)] * num
    slots = slots[:num]
    while len(slots) < num:
        slots.append(slots[-1] if slots else ("General", difficulty))
    return slots


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY — map-reduce, so the planner can propose topics from anywhere in the
# document rather than only from its first few pages
# ─────────────────────────────────────────────────────────────────────────────
def build_summary(text, chunk_chars=9000, max_chunks=14):
    print("  Building map-reduce summary…")
    chunks = [text[i:i + chunk_chars]
              for i in range(0, len(text), chunk_chars)][:max_chunks]
    partials = []
    for i, ch in enumerate(chunks, 1):
        out = call_ollama(
            "Summarise the key topics, definitions and results in this lecture "
            "excerpt as concise bullet points. No preamble.\n\n" + ch,
            system="You are a precise academic summariser.",
            model=QUIZ_MODEL, num_ctx=QUIZ_CTX,
        )
        partials.append((out or "").strip())
        print(f"    chunk {i}/{len(chunks)}")
    merged = "\n".join(partials)[:20000]
    final = call_ollama(
        "Merge these partial summaries into one structured summary of the whole "
        "chapter: list the main topics and the key definitions/results under "
        "each. No preamble.\n\n" + merged,
        system="You are a precise academic summariser.",
        model=QUIZ_MODEL, num_ctx=QUIZ_CTX,
    )
    return (final or merged).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(name, n_per_diff, content, summary, difficulties,
                 user_instr="", max_attempts=3, existing_groups=None):
    """Generate n_per_diff questions per difficulty for one merged pipeline."""
    groups = dict(existing_groups or {})
    for diff in difficulties:
        key  = f"{name}:{diff}"
        have = list(groups.get(key, []))
        if len(have) >= n_per_diff:
            print(f"  [{key}] already has {len(have)} — skipping (resume)")
            continue

        print(f"\n  ── {key} — generating {n_per_diff - len(have)} more ──")
        slots = None
        if name == "merge_vr":
            # merge_vr keeps the SEPARATE planner (only validate+refine merged)
            slots = plan_quiz(n_per_diff, diff, summary, user_instr)
            print(f"     plan: {sorted({t for t, _ in slots})}")

        # A question whose every attempt is rejected by the deterministic
        # structural/duplicate gate yields nothing at all. A plain
        # `for idx in range(n)` would then leave the group short of n and the
        # groups unbalanced, which quietly biases any per-group rate. Instead
        # keep filling until n is reached, with a hard slot budget so a
        # pathological topic cannot loop forever. Any remaining shortfall is
        # reported rather than hidden.
        slot_budget = (n_per_diff - len(have)) + max(5, n_per_diff // 4)
        slots_used  = 0

        while len(have) < n_per_diff and slots_used < slot_budget:
            idx = len(have)          # slot index == accepted count, so the
            slots_used += 1          # cognitive round-robin stays aligned
            t0 = time.time()
            if name == "merge_vr":
                topic, _ = slots[idx] if idx < len(slots) else ("General", diff)
                q = _one_question_merge_vr(topic, diff, content, summary,
                                           user_instr, have, idx, max_attempts)
            elif name == "merge_pg":
                q = _one_question_merge_pg(diff, content, summary, user_instr,
                                           have, idx, max_attempts)
            else:
                raise ValueError(f"unknown pipeline {name}")

            if q is None:
                print(f"     q{idx+1}: FAILED (no question produced)")
                continue

            q["difficulty"] = diff.upper()
            q["_pipeline"]  = name
            q["_secs"]      = round(time.time() - t0, 1)
            if COGNITIVE_ROUTING:
                q["cognitive_type"] = _cognitive_type_for_slot(idx)
            have.append(q)
            groups[key] = have
            print(f"     q{idx+1}: {q.get('_verdict')} "
                  f"({q.get('_llm_calls')} calls, {q['_secs']}s) "
                  f"topic={str(q.get('topic'))[:40]!r}")
            yield groups        # let the caller checkpoint after every question

        if len(have) < n_per_diff:
            print(f"  ⚠ [{key}] SHORT: {len(have)}/{n_per_diff} after "
                  f"{slots_used} slots (budget {slot_budget}). Groups are "
                  f"unbalanced — say so when reporting per-group rates.")
    return groups


def summarise_cost(groups):
    """Calls per accepted question, per group — the column that makes a merge
    result interpretable."""
    out = {}
    for key, qs in groups.items():
        if not qs:
            continue
        calls  = [q.get("_llm_calls", 0) for q in qs]
        passed = [q for q in qs if q.get("_verdict") == "PASS"]
        out[key] = {
            "n":                    len(qs),
            "pass":                 len(passed),
            "override":             len(qs) - len(passed),
            "total_llm_calls":      sum(calls),
            "calls_per_question":   round(sum(calls) / len(qs), 2),
            "calls_per_accepted":   (round(sum(calls) / len(passed), 2)
                                     if passed else None),
            "mean_attempts":        round(sum(q.get("_attempts", 1) for q in qs)
                                          / len(qs), 2),
            "mean_secs":            round(sum(q.get("_secs", 0) for q in qs)
                                          / len(qs), 1),
        }
    return out


def write_artifact(path, groups, meta):
    payload = {
        "questions": {k: v for k, v in groups.items() if v},
        "meta": {**meta, "cost": summarise_cost(groups)},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--content", default="chapter.txt",
                    help="lecture/chapter text file (default: chapter.txt)")
    ap.add_argument("--summary", default="",
                    help="pre-built summary file; built map-reduce if omitted")
    ap.add_argument("--out", default="merged_eval_questions.json")
    ap.add_argument("--pipeline", default="both",
                    choices=["merge_vr", "merge_pg", "both"])
    ap.add_argument("--n", type=int, default=100,
                    help="questions PER DIFFICULTY per pipeline (default 100)")
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--instructions", default="")
    ap.add_argument("--no-retrieval", action="store_true",
                    help="run without the search_lecture tool (changes the "
                         "experiment: this is then an ablation, not a merge)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing --out file")
    args = ap.parse_args()

    if not os.path.exists(args.content):
        sys.exit(f"content file not found: {args.content}")
    content = open(args.content, encoding="utf-8", errors="ignore").read()
    print(f"Loaded {len(content):,} chars from {args.content}")

    # fail fast if Ollama is not actually up — better than 300s timeouts later
    try:
        http_requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10).raise_for_status()
    except Exception as e:
        sys.exit(f"Ollama not reachable at {OLLAMA_BASE}: {e}")

    if args.summary and os.path.exists(args.summary):
        summary = open(args.summary, encoding="utf-8").read()
        print(f"Loaded summary from {args.summary} ({len(summary):,} chars)")
    else:
        summary = build_summary(content)
        with open("merged_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary built ({len(summary):,} chars) → merged_summary.txt")

    if not args.no_retrieval:
        rag.build(content)
    else:
        print("  ⚠ retrieval DISABLED (--no-retrieval): not a clean merge comparison")

    groups, meta_prev = {}, {}
    if args.resume and os.path.exists(args.out):
        prev = json.load(open(args.out, encoding="utf-8"))
        groups    = prev.get("questions", {})
        meta_prev = prev.get("meta", {})
        print(f"Resuming — existing groups: "
              f"{ {k: len(v) for k, v in groups.items()} }")

    names = ["merge_vr", "merge_pg"] if args.pipeline == "both" else [args.pipeline]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]

    meta = {
        "generator_model": QUIZ_MODEL,
        "pinned":          PIN_QUIZ_MODEL,
        "cognitive_routing": COGNITIVE_ROUTING,
        "retrieval":       (not args.no_retrieval),
        "content_file":    args.content,
        "content_chars":   len(content),
        "n_per_difficulty": args.n,
        "max_attempts":    args.max_attempts,
        "started":         meta_prev.get("started") or time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("merge_vr = validator+refiner in one call; "
                 "merge_pg = planner+generator in one call. "
                 "Compare against the `agentic` control run."),
    }

    t0 = time.time()
    for name in names:
        print(f"\n{'='*66}\nPIPELINE: {name}\n{'='*66}")
        # run_pipeline yields after every question so a crash costs one question
        for snapshot in run_pipeline(name, args.n, content, summary, diffs,
                                     args.instructions, args.max_attempts,
                                     existing_groups=groups):
            groups = snapshot
            write_artifact(args.out, groups, meta)

    write_artifact(args.out, groups, meta)
    mins = (time.time() - t0) / 60

    print(f"\n{'='*66}\nDONE in {mins:.1f} min — wrote {args.out}")
    print(f"Total LLM calls this run: {CALLS.total}")
    print(f"{'group':<22}{'n':>5}{'pass':>6}{'calls/q':>9}{'calls/acc':>11}{'secs/q':>8}")
    for key, c in sorted(summarise_cost(groups).items()):
        print(f"{key:<22}{c['n']:>5}{c['pass']:>6}{c['calls_per_question']:>9}"
              f"{str(c['calls_per_accepted']):>11}{c['mean_secs']:>8}")
    print("\nNow judge it with the existing harnesses:")
    print(f"  python evaluate_eqgbench.py --questions {args.out}")
    print(f"  python evaluate_requesta.py --questions {args.out}")


if __name__ == "__main__":
    main()
