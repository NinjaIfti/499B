"""Test for the summary reviewer-mode guard in colab.py.

Background
----------
`summarize_document()` summarises long documents map-reduce style. The final
reduce call is the only summarisation call whose payload is already-polished
summary prose rather than raw source text. Without a delimiter and an explicit
output constraint, gemma3 reads the message as "here is a document produced in
response to that instruction -- comment on it" and returns praise plus
improvement suggestions instead of the merged summary. That commentary then
passes SummaryAgent.validate (it is long, and contains none of the three
refusal phrases that check looks for) and is shown to the user as the summary.

colab.py cannot be imported outside Google Colab -- it contains `!pip` magics
and Colab-only runtime state -- so this test extracts the guard function's
source with ast and execs just that function in isolation.

Run from the repo root:  py -3 tools/test_summary_guard.py
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLAB = os.path.join(ROOT, "colab.py")

FUNC_NAME = "_looks_like_review_not_summary"


def load_guard():
    """Pull FUNC_NAME out of colab.py and exec it standalone."""
    with open(COLAB, encoding="utf-8") as f:
        src = f.read()
    cleaned = "\n".join(
        "" if re.match(r"^\s*[!%]", line) else line for line in src.split("\n")
    )
    tree = ast.parse(cleaned)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == FUNC_NAME:
            ns = {}
            exec(compile(ast.Module([node], []), COLAB, "exec"), ns)
            return ns[FUNC_NAME]
    raise AssertionError(
        "%s() not found at module level in colab.py" % FUNC_NAME
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# VERBATIM reviewer-mode output reported by the user. This is the bug.
REVIEW_MODE = """Okay, that's a comprehensive and well-structured summary! You've successfully combined the partial summaries into one detailed document covering all topics, definitions, theorems, and examples. The organization is excellent, with clear headings and subheadings for each section. The inclusion of worked example numbers helps to cross-reference back to the original material.

Here are a few minor suggestions that could further enhance it (though these are truly optional):

Consistency in Formatting: While you've done a great job, there are slight variations in how definitions and theorems are presented (e.g., some have more context than others). Striving for even greater consistency would make the summary feel even more polished.
Brief Explanations of "Why": For some theorems or concepts, adding a very brief explanation of why they're important or how they connect to other ideas could be helpful (especially for someone using this as a study guide).
Linking Between Parts: Consider adding brief transition sentences between sections to highlight how concepts build upon each other.

Overall, this is an outstanding summary! It effectively captures the essence of the document and presents it in a clear, organized, and accessible manner. The level of detail is excellent, making it valuable for review or study purposes."""

# A genuine merged summary of the same source material. Must NOT be flagged.
REAL_SUMMARY = """# Limits -- Chapter Summary

## 1. The Idea of a Limit
The limit describes the value a function approaches as the input approaches a
point, whether or not the function is defined there. Notation: lim_{x->a} f(x) = L.

## 2. Finding Limits
- **Direct substitution** applies when f is continuous at a.
- **Factoring and cancelling** resolves 0/0 indeterminate forms.
- **Conjugate multiplication** handles expressions containing radicals.

## 3. The Squeeze Theorem
If g(x) <= f(x) <= h(x) near a, and lim g(x) = lim h(x) = L, then lim f(x) = L.
Worked example 2.14 uses this to establish lim_{x->0} x sin(1/x) = 0.

## 4. Continuity
A function f is continuous at a when f(a) is defined, the limit exists, and the
two are equal. The Intermediate Value Theorem follows from continuity on a
closed interval.

## 5. Asymptotes
A vertical asymptote occurs at x = a when the one-sided limits diverge. A
horizontal asymptote y = L occurs when lim_{x->inf} f(x) = L."""

# A study-guide-flavoured summary that addresses the reader directly. This is
# the most likely false positive, so it is an explicit test case.
STUDY_GUIDE_TONE = """# Limits -- Study Notes

Remember that a limit tells you what a function approaches, not necessarily what
it equals. You can evaluate most limits by direct substitution when the function
is continuous.

When you hit a 0/0 form, you should factor and cancel first. The Squeeze Theorem
is worth knowing because it lets you find limits that direct calculation cannot
reach. Consider reviewing worked examples 2.12 through 2.15 before the exam."""

# Concatenated partials -- the fallback value. Must NOT be flagged.
PARTIALS = """--- Part 1 ---
Key topics: intuitive definition of a limit, one-sided limits, notation.
Definitions: limit of a function, one-sided limit.
Worked examples: 2.1, 2.2, 2.3.

--- Part 2 ---
Key topics: limit laws, the Squeeze Theorem, trigonometric limits.
Theorems: sum/product/quotient laws, Squeeze Theorem.
Worked examples: 2.11, 2.14."""


CASES = [
    ("user-reported reviewer-mode output", REVIEW_MODE, True),
    ("genuine merged summary", REAL_SUMMARY, False),
    ("study-guide tone, addresses reader", STUDY_GUIDE_TONE, False),
    ("concatenated partials (the fallback)", PARTIALS, False),
    ("empty string", "", False),
]


def main():
    try:
        guard = load_guard()
    except AssertionError as exc:
        print("FAIL %s" % exc)
        return 1

    failures = 0
    for name, text, expected in CASES:
        got = bool(guard(text))
        status = "ok  " if got == expected else "FAIL"
        if got != expected:
            failures += 1
        print("%s %-38s expected=%-5s got=%-5s" % (status, name, expected, got))

    if failures:
        print("\n%d/%d case(s) failed" % (failures, len(CASES)))
        return 1
    print("\nall %d cases passed" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
