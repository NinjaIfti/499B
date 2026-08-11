"""Static checks for the LectureAssist single-page frontend and Colab backend.

Run from the repo root:  py -3 tools/check_ui.py

There is no test framework in this project and colab.py cannot be imported
outside Colab, so these four checks are the automated safety net:
  1. the extracted <script> block parses as JavaScript
  2. <div> tags balance in the HTML
  3. colab.py parses as Python once Colab magics are stripped
  4. no evaluation UI has crept back in

They are static only. They cannot tell you the UI looks correct -- always
also run the browser checks listed in the implementation plan.
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "LectureAssis.html")
PY = os.path.join(ROOT, "colab.py")

failures = []
notes = []


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def check_js(html):
    """Extract the final <script> block and run `node --check` on it."""
    start = html.rindex("<script>") + len("<script>")
    end = html.rindex("</script>")
    js = html[start:end]

    if shutil.which("node") is None:
        notes.append("node not found on PATH - JS syntax NOT checked")
        return

    tmp = os.path.join(tempfile.gettempdir(), "lectureassist_check.js")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(js)
    proc = subprocess.run(
        ["node", "--check", tmp], capture_output=True, text=True
    )
    os.remove(tmp)
    if proc.returncode != 0:
        failures.append("JS syntax error:\n" + (proc.stderr or proc.stdout).strip())


def check_divs(html):
    opens = len(re.findall(r"<div\b", html))
    closes = len(re.findall(r"</div>", html))
    if opens != closes:
        failures.append(
            "div tags unbalanced: %d opened, %d closed" % (opens, closes)
        )


def check_python(src):
    """colab.py contains `!pip` / `%cd` Colab magics; blank them before parsing."""
    cleaned = "\n".join(
        "" if re.match(r"^\s*[!%]", line) else line for line in src.split("\n")
    )
    try:
        ast.parse(cleaned)
    except SyntaxError as exc:
        failures.append(
            "colab.py syntax error at line %s: %s" % (exc.lineno, exc.msg)
        )


# Targeted patterns, not a bare /eval|bulk/ search: the substring "eval"
# occurs innocently inside "retrieval", which is a natural word to use when
# describing the Plan -> Retrieve -> Generate -> Validate loop. A bare search
# would fail the build on correct UI copy.
EVAL_PATTERNS = [
    r"/eval/",                       # the removed Flask routes
    r"eval_results|bulk_eval|bulk_questions|bulk_csv",   # removed downloads
    r"runEvaluationBenchmark|runBulkEvaluation|runQuizCompare",
    r"downloadEvalResults|pollEvalStatus|renderBulkResult|updateBulkHint",
    r"evalStatus|evalFeed|evalBar|evalPhase|evalLive",
    r"bulkTotal|bulkFeed|bulkBar|bulkPhase|bulkLive|bulkStatus|bulkGold",
    r"Run Evaluation|Bulk evaluation|Run Benchmark|Download Eval",
]


def check_no_eval(html):
    hits = []
    for i, line in enumerate(html.split("\n"), 1):
        for pattern in EVAL_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                hits.append("  line %d: %s" % (i, line.strip()[:100]))
                break
    if hits:
        failures.append(
            "evaluation UI found in LectureAssis.html:\n" + "\n".join(hits)
        )


def main():
    html = read(HTML)
    check_js(html)
    check_divs(html)
    check_no_eval(html)
    check_python(read(PY))

    for note in notes:
        print("WARN " + note)
    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("OK  js syntax, div balance, colab.py syntax, no-eval")
    return 0


if __name__ == "__main__":
    sys.exit(main())
