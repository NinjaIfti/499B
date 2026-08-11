# Demo Cockpit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live LectureAssist capstone demo read unmistakably as an agentic multi-agent system, and remove the screens that look empty or unfinished.

**Architecture:** Six tasks against a self-contained single-page frontend (`LectureAssis.html`) plus one line in the Colab Flask backend (`colab.py`). Every change is additive — no existing feature is rewired. Task 1 unlocks live agent data; Tasks 2–3 consume it on the Upload page; Tasks 4–6 are independent frontend work.

**Tech Stack:** Vanilla ES5-style JS (no build step, no framework, no bundler), hand-written CSS using custom-property design tokens, Flask + ngrok backend running in Google Colab.

## Global Constraints

Copied from `docs/superpowers/specs/2026-08-11-demo-ui-cockpit-design.md`:

- **No evaluation surfaces.** Do not reintroduce any `eval` or `bulk` UI, endpoint, route, or download. `LectureAssis.html` must contain zero case-insensitive matches for `eval` or `bulk` when the work is done.
- **No fabricated numbers.** Every stat displayed must trace to a real backend value. If a source is missing or empty, display `—`. Never a placeholder, estimate, or fallback value. In particular do **not** source topic counts from `/quiz/topics`, which invents four generic topics (`Core Concepts`, `Key Principles`, `Applications`, `Problem Solving`) when nothing is detected — see `colab.py:5596-5598`.
- **No new backend endpoints.** The only backend change in this plan is the one-line alias in Task 1.
- **ES5-compatible JS only.** The file uses `var`, `function(){}`, and string concatenation throughout. Match it. No arrow functions, no template literals, no `const`/`let`, no optional chaining.
- **Theme via tokens only.** Dark mode works by redefining CSS custom properties under `[data-theme="dark"]` (line 979). Any new CSS must use existing tokens (`var(--surface)`, `var(--ink)`, `var(--ink2)`, `var(--muted)`, `var(--border)`, `var(--border2)`, `var(--bg)`, `var(--g1)`, `var(--green)`, `var(--amber)`, `var(--red)`, `var(--cyan)`, and the `-s` softened variants). Never hard-code a hex colour that needs to differ between themes.
- **No new dependencies.** No CDN links, no npm, no build step.
- **Fonts in use:** `'Cabinet Grotesk'` for headings, `'DM Sans'` for body, `'DM Mono'` for monospace.
- **Mobile breakpoint is `@media(max-width:860px)`** at line 1058. Sticky positioning must be disabled below it.
- **E is the designated cut.** If time runs out, Task 6 is dropped. Tasks 1–5 must be complete and working first.

---

## Testing Reality (read before Task 1)

**This repository has no test framework, and this plan does not introduce one.** There is no pytest suite, no jest, no test runner. `colab.py` cannot even be imported outside Colab — it contains `!pip` shell magics and depends on Colab-only runtime state, so `ast.parse` with magics stripped is the only static check available for it.

Inventing a test suite here would be busywork that does not make the demo safer. Instead, **Task 0 builds a real verification script** that every subsequent task runs as its gate. It performs four checks that genuinely catch the failure modes of a single-file frontend:

1. JavaScript syntax (`node --check` on the extracted `<script>` block)
2. HTML tag balance (`<div>` open/close counts)
3. Python syntax of `colab.py` (with Colab magics stripped)
4. The no-evaluation constraint

**These are static checks. They cannot tell you the UI looks right.** Every task therefore also carries a **Browser check** — an explicit list of what to click and what to see. Those must actually be performed against a running Colab backend before the demo. A task is not done because the script passed.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `tools/check_ui.py` | **Create.** Static verification gate for all later tasks. | 0 |
| `colab.py` | Flask backend. One line changed so `/agent_log` streams live. | 1 |
| `LectureAssis.html` | The entire frontend — CSS in `<style>` (lines 9–1069), markup (1071–1639), JS in `<script>` (1640–3418). All UI work lands here. | 2–6 |

`LectureAssis.html` is a 3.4k-line single file. That is the established pattern of this project and this plan does **not** split it — restructuring the frontend days before a demo would be reckless. New CSS is appended to the section it belongs to; new JS is appended near the functions it collaborates with.

---

## Task 0: Verification harness

**Files:**
- Create: `tools/check_ui.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `py -3 tools/check_ui.py` — exits `0` when all checks pass, exits `1` and prints the failures otherwise. Every later task runs this as its final gate.

- [ ] **Step 1: Create the check script**

```python
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


def check_no_eval(html):
    hits = []
    for i, line in enumerate(html.split("\n"), 1):
        if re.search(r"eval|bulk", line, re.IGNORECASE):
            hits.append("  line %d: %s" % (i, line.strip()[:100]))
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
```

- [ ] **Step 2: Run it against the current tree**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval` and exit code 0. The evaluation removal already landed, so this passes today. If it fails, stop — something is wrong before any of this plan's work has started.

- [ ] **Step 3: Verify the script actually catches a break**

A gate that cannot fail is not a gate. Temporarily corrupt the file, confirm the failure, then restore:

```bash
cp LectureAssis.html /tmp/la_backup.html
printf '\n<script>function broken( {</script>\n' >> LectureAssis.html
py -3 tools/check_ui.py; echo "exit=$?"
cp /tmp/la_backup.html LectureAssis.html
rm /tmp/la_backup.html
```

Expected: the middle command prints a `FAIL` and `exit=1`; after restoring, `py -3 tools/check_ui.py` prints `OK` again.

- [ ] **Step 4: Commit**

```bash
git add tools/check_ui.py
git commit -m "Add static check script for frontend and backend"
```

---

## Task 1: Make `/agent_log` stream during a run

**Files:**
- Modify: `colab.py:4854`

**Interfaces:**
- Consumes: nothing.
- Produces: `GET /agent_log` returns `{log: [...], total: N}` where `log` grows **during** a pipeline run instead of staying empty until it ends. Each entry is `{agent: str, level: str, time: ISO8601 str, message: str, duration_sec?: float}`. Task 2 and Task 3 both depend on this.

**Why this is needed:** `OrchestratorAgent.run()` currently assigns `state.agent_log = []` — a brand-new list — at pipeline start, while every `_log()` call (`colab.py:1965` and `colab.py:4912`) appends to `ctx.agent_log`, a different list. `state.agent_log` is what `/agent_log` serves, so it stays empty for the whole run and is only populated by the assignment at `colab.py:4896` after everything finishes. Aliasing the two makes the endpoint live. `state.agent_log` is written at exactly these two sites and read only by `/agent_log`, so nothing else is affected.

- [ ] **Step 1: Confirm the current behaviour before changing it**

Run: `py -3 -c "import re; s=open('colab.py',encoding='utf-8').read().split('\n'); print(repr(s[4853])); print(repr(s[4895]))"`

Expected output:
```
'        state.agent_log  = []'
'        state.agent_log = ctx.agent_log'
```

If those two lines are not what you see, the file has shifted — find them with `grep -n "state.agent_log" colab.py` and adjust, but do not proceed blindly.

- [ ] **Step 2: Change the reset into an alias**

In `colab.py`, inside `OrchestratorAgent.run()`, replace:

```python
        state.agent_log  = []
```

with:

```python
        # Alias, not a fresh list: every agent appends to ctx.agent_log, so
        # sharing the object is what makes GET /agent_log stream live during
        # the run instead of staying empty until the pipeline finishes.
        state.agent_log  = ctx.agent_log
```

Leave the assignment at the end of `run()` (`state.agent_log = ctx.agent_log`) exactly as it is. It becomes a harmless no-op and documents the same intent.

- [ ] **Step 3: Verify Python still parses**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval`

- [ ] **Step 4: Browser check**

This one needs the real backend and is worth doing now, because Tasks 2 and 3 are built on it.

1. Restart the Colab notebook so the edited `colab.py` is loaded.
2. Connect the UI to the printed ngrok URL.
3. Upload a file and press **Process Content**.
4. While the pipeline is still running (progress bar below 100%), open `<ngrok-url>/agent_log` in a second browser tab and refresh it a few times.

Expected: `total` is greater than 0 and **increases** between refreshes, while processing is still in progress. Before this change it stayed at `0` until the very end. If it stays 0, the alias did not take effect — most likely the notebook was not restarted.

- [ ] **Step 5: Commit**

```bash
git add colab.py
git commit -m "Stream agent log live during pipeline runs"
```

---

## Task 2: Live agent ticker and step timers on the Upload page

**Files:**
- Modify: `LectureAssis.html` — CSS after line 337, markup inside `progCard`, JS near `updSteps`

**Interfaces:**
- Consumes: `GET /agent_log` from Task 1. Existing helpers `apiFetch(path, opts)`, `esc(s)`, and the `pollStatus()` loop running on a 2000 ms interval (`LectureAssis.html:1844`).
- Produces:
  - `renderAgentTicker(log)` — appends any entries not yet drawn, autoscrolls, updates the count chip.
  - `pollAgentTicker()` — fetches `/agent_log` and calls `renderAgentTicker`.
  - `paintStepTimers()` — writes elapsed seconds into the four step pills.
  - `resetRunUi()` — clears ticker and timer state at the start of a run. **Task 3 calls this too.**
  - Module-level state `_tickerCount` (number), `_stepStart` (object), `_stepFrozen` (object).
  - DOM ids `agentTicker`, `atBody`, `atCount` — Task 3 does not use these, but Task 5's disconnected banner must not collide with them.

- [ ] **Step 1: Add the CSS**

Insert immediately after line 337 (`.step-p.done{...}`), before the `.quiz-live-notice` block:

```css

/* ── Live agent ticker (Upload page) ── */
.agent-ticker{
  margin-top:14px;border:1px solid var(--border);border-radius:10px;
  background:var(--bg);overflow:hidden;display:none;position:relative;z-index:1;
}
.at-head{
  display:flex;align-items:center;gap:8px;padding:8px 12px;
  border-bottom:1px solid var(--border);
  font-size:11px;font-weight:700;color:var(--muted);
  text-transform:uppercase;letter-spacing:.06em;
}
.at-dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulseDot 1.6s infinite}
.at-count{margin-left:auto;font-weight:600;letter-spacing:0;text-transform:none}
.at-body{max-height:168px;overflow-y:auto;padding:6px 0}
.at-body::-webkit-scrollbar{width:4px}
.at-body::-webkit-scrollbar-thumb{background:var(--border2);border-radius:10px}
.at-row{
  display:flex;align-items:baseline;gap:8px;padding:4px 12px;
  font-size:12px;line-height:1.5;animation:atIn .25s ease;
}
@keyframes atIn{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.at-agent{
  flex-shrink:0;font-family:'DM Mono',monospace;font-size:10.5px;font-weight:600;
  padding:1px 7px;border-radius:100px;
  background:var(--pri-s);color:var(--g1);border:1px solid rgba(99,102,241,0.18);
}
.at-msg{flex:1;min-width:0;color:var(--ink2)}
.at-dur{flex-shrink:0;font-size:10.5px;color:var(--muted);font-family:'DM Mono',monospace}
.at-row[data-level="PLAN"] .at-agent{background:var(--cyan-s);color:var(--cyan);border-color:rgba(6,182,212,0.25)}
.at-row[data-level="DONE"] .at-agent{background:var(--green-s);color:var(--green);border-color:rgba(16,185,129,0.25)}
.at-row[data-level="RETRY"] .at-agent{background:var(--amber-s);color:var(--amber);border-color:rgba(245,158,11,0.25)}
.at-row[data-level="ERROR"] .at-agent,
.at-row[data-level="FAIL"] .at-agent{background:var(--red-s);color:var(--red);border-color:rgba(239,68,68,0.25)}
.sp-t{margin-left:6px;opacity:.75;font-family:'DM Mono',monospace;font-size:10px}
.step-p.active{animation:stepPulse 1.4s infinite}
@keyframes stepPulse{
  0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,0)}
  50%{box-shadow:0 0 0 4px rgba(99,102,241,0.12)}
}
```

Note `position:relative;z-index:1` on `.agent-ticker`: `.prog-card` has a `::after` shimmer overlay spanning `inset:0` (line 309–313). Without the stacking context the shimmer paints over the ticker rows.

- [ ] **Step 2: Add the markup**

In `page-upload`, inside `<div class="prog-card" id="progCard">`, immediately after the closing `</div>` of `<div class="step-row" id="stepRow">` and before `<div class="quiz-live-notice" id="quizLiveNotice" ...>`:

```html

        <div class="agent-ticker" id="agentTicker">
          <div class="at-head">
            <span class="at-dot"></span> Agent activity
            <span class="at-count" id="atCount">0 decisions</span>
          </div>
          <div class="at-body" id="atBody"></div>
        </div>
```

- [ ] **Step 3: Replace `updSteps` and add the ticker functions**

Replace the whole existing `updSteps` function with the block below. The keyword and percentage-marker logic is unchanged — only timer bookkeeping is added.

```js
// ── Live agent ticker + per-step timers (Upload page) ──
// /agent_log streams during a run (colab.py aliases state.agent_log to the
// orchestrator's live list), so named agents can be shown deciding things
// while OCR/Whisper/Summary/RAG are still working. Without this the Upload
// page shows a bare progress bar for several minutes.
var _tickerCount = 0;    // how many log entries have already been drawn
var _stepStart   = {};   // step key -> ms timestamp when it first went active
var _stepFrozen  = {};   // step key -> final seconds, once the step is done

function resetRunUi() {
  _tickerCount = 0;
  _stepStart   = {};
  _stepFrozen  = {};
  var body = document.getElementById('atBody');
  if (body) body.innerHTML = '';
  var wrap = document.getElementById('agentTicker');
  if (wrap) wrap.style.display = 'none';
  var c = document.getElementById('atCount');
  if (c) c.textContent = '0 decisions';
  document.querySelectorAll('.step-p .sp-t').forEach(function(el){ el.remove(); });
  document.querySelectorAll('.step-p').forEach(function(el){
    el.classList.remove('active','done');
  });
}

function renderAgentTicker(log) {
  var body = document.getElementById('atBody');
  var wrap = document.getElementById('agentTicker');
  if (!body || !wrap || !log.length) return;
  wrap.style.display = 'block';
  // Append only what is new, so existing rows do not re-run their animation
  // on every poll.
  for (var i = _tickerCount; i < log.length; i++) {
    var e   = log[i] || {};
    var dur = (typeof e.duration_sec === 'number') ? (e.duration_sec + 's') : '';
    var row = document.createElement('div');
    row.className = 'at-row';
    row.setAttribute('data-level', e.level || 'INFO');
    row.innerHTML = '<span class="at-agent">' + esc(e.agent || 'agent') + '</span>' +
                    '<span class="at-msg">'   + esc(e.message || '') + '</span>' +
                    '<span class="at-dur">'   + esc(dur) + '</span>';
    body.appendChild(row);
  }
  if (log.length > _tickerCount) {
    _tickerCount = log.length;
    body.scrollTop = body.scrollHeight;
    var c = document.getElementById('atCount');
    if (c) c.textContent = _tickerCount + ' decision' + (_tickerCount === 1 ? '' : 's');
  }
}

function pollAgentTicker() {
  if (!API_BASE) return;
  apiFetch('/agent_log').then(function(r){ return r.json(); }).then(function(d){
    var log = d.log || [];
    // A fresh run replaces the backend list, so a shorter list means restart.
    if (log.length < _tickerCount) {
      _tickerCount = 0;
      var body = document.getElementById('atBody');
      if (body) body.innerHTML = '';
    }
    renderAgentTicker(log);
  }).catch(function(){ /* transient poll failures are ignored */ });
}

function paintStepTimers() {
  ['ocr','whisper','summary','rag'].forEach(function(k){
    var p = document.querySelector('[data-k="' + k + '"]');
    if (!p) return;
    var secs = null;
    if (_stepFrozen[k] != null)  secs = _stepFrozen[k];
    else if (_stepStart[k])      secs = Math.round((Date.now() - _stepStart[k]) / 1000);
    var t = p.querySelector('.sp-t');
    if (secs == null) { if (t) t.remove(); return; }
    if (!t) { t = document.createElement('span'); t.className = 'sp-t'; p.appendChild(t); }
    t.textContent = secs + 's';
  });
}

function updSteps(stage, pct) {
  var kw={ocr:['OCR','board','EasyOCR'],whisper:['Whisper','speech','Transcrib'],
    summary:['summary','Summary'],rag:['RAG','FAISS']};
  var sl=stage.toLowerCase();
  Object.keys(kw).forEach(function(k){
    var p=document.querySelector('[data-k="'+k+'"]');if(!p)return;
    if(kw[k].some(function(w){return sl.includes(w.toLowerCase());})){
      p.classList.add('active');p.classList.remove('done');
      if(!_stepStart[k]) _stepStart[k]=Date.now();
    }
  });
  var marks={ocr:35,whisper:55,summary:78,rag:100};
  Object.keys(marks).forEach(function(k){
    if(pct>=marks[k]){
      var p=document.querySelector('[data-k="'+k+'"]');
      if(p){
        p.classList.add('done');p.classList.remove('active');
        if(_stepStart[k] && _stepFrozen[k]==null){
          _stepFrozen[k]=Math.round((Date.now()-_stepStart[k])/1000);
        }
      }
    }
  });
  paintStepTimers();
}
```

- [ ] **Step 4: Hook the ticker into the existing poll**

In `pollStatus()`, immediately after the line `updSteps(d.stage||'',d.pct||0);`, add:

```js
    pollAgentTicker();
```

This satisfies the spec's request that the ticker stop polling once the run is done, without any extra guard: `pollStatus` already calls `clearInterval(pollInt)` when `d.state==='done'`, so the ticker fetch dies with the poll loop that drives it. The rendered rows persist, leaving the finished trail on screen.

- [ ] **Step 5: Reset state when a run starts and when the form is cleared**

In `startProc()`, immediately after the line `if(!API_BASE){...return;}`, add:

```js
  resetRunUi();
```

In `clearAll()`, immediately before the final `if(pollInt)clearInterval(pollInt);`, add:

```js
  resetRunUi();
```

- [ ] **Step 6: Run the checks**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval`

- [ ] **Step 7: Browser check**

1. Connect to the backend, upload a file, press **Process Content**.
2. Within a few seconds of the progress bar starting, the **Agent activity** panel appears inside the progress card.
3. Rows accumulate *while* the run is in progress, each showing an agent-name chip and a message; the panel autoscrolls to the newest.
4. Rows with `level` of `DONE` are green, `RETRY` amber, `ERROR`/`FAIL` red, `PLAN` cyan.
5. The step pills show a live-counting `Ns` suffix; the active one pulses; when a step completes its number freezes.
6. Press **Clear**, then start a second run — the ticker empties and starts from zero rather than continuing the old list.
7. Toggle dark mode with the moon button while the ticker has rows: text stays legible, no hard-coded light-theme colours.

- [ ] **Step 8: Commit**

```bash
git add LectureAssis.html
git commit -m "Add live agent ticker and step timers to Upload page"
```

---

## Task 3: Hero strip with honest run stats

**Files:**
- Modify: `LectureAssis.html` — CSS after the `.page-sub` rule (line 202), markup at the top of `page-upload`, JS in `onDone` and `updCounts`

**Interfaces:**
- Consumes: `resetRunUi()` from Task 2 (extended here, not redefined). `GET /agent_log` from Task 1. Existing `onDone(d)`, `updCounts()`, `apiFetch`, `esc`.
- Produces:
  - `setHeroStat(id, value)` — writes a value, or `—` when the value is null/undefined/empty.
  - `fillHeroStats(d)` — populates all four pills from a `/status` done-payload plus two follow-up fetches.
  - DOM ids `hsWords`, `hsTopics`, `hsQs`, `hsAgents`.

**Honesty rule (Global Constraint):** every pill either shows a number traceable to a real backend value, or `—`. `setHeroStat` enforces this; do not bypass it.

- [ ] **Step 1: Add the CSS**

Insert immediately after line 202 (`.page-sub{...}`):

```css

/* ── Upload hero ── */
.hero{
  display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:22px 24px;margin-bottom:16px;
  box-shadow:var(--shadow);
  background-image:
    radial-gradient(120% 140% at 0% 0%, rgba(99,102,241,0.10) 0%, transparent 55%),
    radial-gradient(100% 140% at 100% 0%, rgba(6,182,212,0.08) 0%, transparent 55%);
}
.hero-main{flex:1;min-width:260px}
.hero-kicker{
  display:inline-block;font-size:10.5px;font-weight:800;letter-spacing:.07em;
  text-transform:uppercase;color:var(--g1);
  background:var(--pri-s);border:1px solid rgba(99,102,241,0.18);
  border-radius:100px;padding:3px 10px;margin-bottom:10px;
}
.hero-title{
  font-family:'Cabinet Grotesk',sans-serif;font-size:1.35rem;font-weight:800;
  color:var(--ink);letter-spacing:-0.02em;margin-bottom:6px;
}
.hero-sub{font-size:13px;color:var(--ink2);line-height:1.65;max-width:60ch}
.hero-stats{display:flex;gap:10px;flex-wrap:wrap}
.hs{
  min-width:92px;padding:10px 14px;border-radius:12px;
  background:var(--bg);border:1px solid var(--border);text-align:center;
}
.hs-v{
  font-family:'Cabinet Grotesk',sans-serif;font-size:1.15rem;font-weight:800;
  color:var(--g1);line-height:1.2;
}
.hs-l{font-size:10.5px;color:var(--muted);font-weight:600;margin-top:2px}
@media(max-width:860px){
  .hero-stats{width:100%}
  .hs{flex:1}
}
```

The `@media` block here is a second, local one — that is fine and keeps the rule next to what it modifies. Do not move it into the block at line 1058.

- [ ] **Step 2: Add the markup**

In `page-upload`, immediately after the closing `</div>` of `<div class="page-head">` and before the first `<div class="card">`:

```html

      <div class="hero">
        <div class="hero-main">
          <div class="hero-kicker">Agentic · Multimodal · Runs fully local</div>
          <div class="hero-title">Turn a lecture into a study kit</div>
          <div class="hero-sub">Upload a lecture video or a document. A pipeline of specialist agents transcribes the audio, reads the board, summarises the material, and builds quizzes — every question planned, generated, and validated before it reaches you.</div>
        </div>
        <div class="hero-stats">
          <div class="hs"><div class="hs-v" id="hsWords">—</div><div class="hs-l">transcript</div></div>
          <div class="hs"><div class="hs-v" id="hsTopics">—</div><div class="hs-l">topics found</div></div>
          <div class="hs"><div class="hs-v" id="hsQs">—</div><div class="hs-l">questions</div></div>
          <div class="hs"><div class="hs-v" id="hsAgents">—</div><div class="hs-l">agent decisions</div></div>
        </div>
      </div>
```

- [ ] **Step 3: Add the hero stat functions**

Add immediately after the `paintStepTimers` function from Task 2:

```js
// ── Upload hero stats ──
// Every pill is either a real backend number or an em dash. Never a
// placeholder, an estimate, or a fallback -- these numbers get shown to an
// examining panel.
function setHeroStat(id, value) {
  var el = document.getElementById(id);
  if (!el) return;
  var missing = (value === null || value === undefined || value === '' ||
                 (typeof value === 'number' && isNaN(value)));
  el.textContent = missing ? '—' : value;
}

function fillHeroStats(d) {
  d = d || {};

  // transcript size: words for video, extracted characters for documents
  if (d.transcript && d.transcript.trim()) {
    var words = d.transcript.trim().split(/\s+/).length;
    setHeroStat('hsWords', words.toLocaleString() + ' w');
  } else if (d.doc_chars) {
    setHeroStat('hsWords', Math.round(d.doc_chars / 1000) + 'k ch');
  } else {
    setHeroStat('hsWords', null);
  }

  // topics: ONLY the AI-detected list. /quiz/topics is deliberately not used
  // here -- it invents four generic topics when nothing is detected
  // (colab.py:5596-5598), which would put a fabricated number on screen.
  apiFetch('/exam_hints').then(function(r){ return r.json(); }).then(function(h){
    var ai = (h && h.ai_analysis) ? h.ai_analysis : {};
    var list = ai.ai_important_topics;
    setHeroStat('hsTopics', (list && list.length) ? list.length : null);
  }).catch(function(){ setHeroStat('hsTopics', null); });

  // agent decisions
  apiFetch('/agent_log').then(function(r){ return r.json(); }).then(function(a){
    setHeroStat('hsAgents', a.total ? a.total : null);
  }).catch(function(){ setHeroStat('hsAgents', null); });

  // questions is filled by updCounts(), which already fetches all four types
}
```

- [ ] **Step 4: Total the question counts in `updCounts`**

Replace the existing `updCounts` function with:

```js
function updCounts() {
  var total = 0, seen = 0;
  [{k:'mcq',ep:'/quiz/mcq',id:'cnt-mcq'},{k:'tf',ep:'/quiz/true_false',id:'cnt-tf'},
   {k:'fill',ep:'/quiz/fill_blank',id:'cnt-fill'},{k:'short',ep:'/quiz/short_answer',id:'cnt-short'}]
  .forEach(function(t){
    apiFetch(t.ep).then(function(r){return r.json();}).then(function(d){
      var el=document.getElementById(t.id);if(el)el.textContent=(d.total||0)+' questions';
      total += (d.total||0);
    }).catch(function(){}).then(function(){
      seen++;
      // once all four have settled, publish the sum to the hero pill
      if (seen === 4) setHeroStat('hsQs', total > 0 ? total : null);
    });
  });
}
```

The trailing `.then()` after `.catch()` runs on both success and failure, so `seen` reaches 4 even when an endpoint errors.

- [ ] **Step 5: Call `fillHeroStats` on completion**

In `onDone(d)`, immediately after the line `updCounts();`, add:

```js
  fillHeroStats(d);
```

- [ ] **Step 6: Reset the pills when a new run starts**

In `resetRunUi()` (added in Task 2), append before its closing brace:

```js
  ['hsWords','hsTopics','hsQs','hsAgents'].forEach(function(id){ setHeroStat(id, null); });
```

- [ ] **Step 7: Run the checks**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval`

- [ ] **Step 8: Browser check**

1. Load the page fresh — the hero is visible above the upload cards and all four pills read `—`.
2. Process a file. On completion, `transcript`, `questions`, and `agent decisions` show real numbers.
3. `topics found` shows a number **only if** exam hints were detected. Cross-check by opening `<ngrok-url>/exam_hints` — if `ai_analysis.ai_important_topics` is absent or empty, the pill must still read `—`. **A number here with an empty list is a bug, not a cosmetic issue.**
4. Cross-check `agent decisions` against `total` at `<ngrok-url>/agent_log` — they must match exactly.
5. Press **Clear** — all four pills return to `—`.
6. Narrow the window below 860px — the stat pills wrap and stretch instead of overflowing.

- [ ] **Step 9: Commit**

```bash
git add LectureAssis.html
git commit -m "Add Upload hero strip with run stats"
```

---

## Task 4: Agent Workspace promotion

**Files:**
- Modify: `LectureAssis.html` — `.main` rule (line 197), `.agent-workspace` rule (line 484), new CSS after the `.phase-step` rules, JS near `setPhase`, sidebar markup for the live chip

**Interfaces:**
- Consumes: existing `awState` (`{total, rendered, rows, phase, done, passCount, overrideCount, session}`), `setPhase(phase)`, `setLiveStatus(text, progFraction)`, and the `quiz_thinking_active` flag already read in `pollStatus`.
- Produces:
  - `openAgentWorkspace()` — reveals the workspace, marks it live, scrolls it into view (once per run).
  - `updateAwCounters()` — writes the derived counter line.
  - `setQuizLiveChip(active)` — toggles the sidebar pulse.
  - DOM ids `awCounters`, `navQuizLive`.

**Two CSS blockers must be fixed or the sticky stepper silently does nothing.** Both are pre-existing:

1. `.agent-workspace` has `overflow:hidden` (line 486). A `position:sticky` child's scroll container becomes that box — which never scrolls — so the child never sticks.
2. `.main` has `overflow-x:hidden` (line 197). Per the CSS overflow spec, when one axis is not `visible` the other computes to `auto`, so `.main` is also a scroll container that never scrolls. `overflow-x:clip` clips identically **without** creating a scroll container, which is the correct fix.

`.sidebar` already uses `position:sticky;top:60px` successfully because it is a grid child of `.layout`, outside `.main`. The topbar is 60px tall and itself sticky, so 60px is the correct offset.

- [ ] **Step 1: Fix the two overflow blockers**

Change line 197 from:

```css
.main{padding:28px 32px 60px;overflow-x:hidden;min-height:calc(100vh - 60px)}
```

to:

```css
/* overflow-x:clip, not hidden -- `hidden` would make .main a scroll container
   (the other axis computes to auto), which breaks position:sticky for the
   phase stepper inside the Agent Workspace. `clip` clips without scrolling. */
.main{padding:28px 32px 60px;overflow-x:clip;min-height:calc(100vh - 60px)}
```

Then add immediately after the `.agent-workspace{...}` rule (after line 488):

```css
/* While a run is live the stepper sticks, so overflow must not clip it.
   The head keeps the top corners rounded on its own. */
.agent-workspace.live{overflow:visible}
.agent-workspace.live .aw-head{border-radius:var(--r) var(--r) 0 0}
```

- [ ] **Step 2: Add sticky stepper, counter, and sidebar chip CSS**

Insert after the `.phase-step .ps-label` rule (line 550):

```css
.agent-workspace.live .phase-stepper{
  position:sticky;top:60px;z-index:20;
  background:var(--surface);
  backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);
  box-shadow:0 4px 16px rgba(99,102,180,0.08);
}
.aw-counters{
  display:flex;gap:8px;flex-wrap:wrap;
  padding:10px 20px;border-bottom:1px solid var(--border);
}
.awc{
  font-size:11px;font-weight:700;padding:2px 10px;border-radius:100px;
  background:var(--bg);border:1px solid var(--border);color:var(--ink2);
}
.awc b{color:var(--g1);font-weight:800}
.awc.ok b{color:var(--green)}
.awc.warn b{color:var(--amber)}
.nav-live{
  margin-left:auto;flex-shrink:0;display:none;align-items:center;gap:5px;
  font-size:9.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;
  color:var(--green);
}
.nav-live.on{display:inline-flex}
.nav-live i{
  width:6px;height:6px;border-radius:50%;background:var(--green);
  animation:pulseDot 1.6s infinite;font-style:normal;
}
@media(max-width:860px){
  .agent-workspace.live .phase-stepper{position:static}
}
```

The final `@media` disables sticky below the mobile breakpoint, as required by the Global Constraints.

- [ ] **Step 3: Add the counter row markup**

In the Agent Workspace block, immediately after the closing `</div>` of `<div class="phase-stepper" id="phaseStepper">` and before `<div class="live-status">`:

```html

        <div class="aw-counters" id="awCounters"></div>
```

- [ ] **Step 4: Add the sidebar live chip**

Replace the Quiz Engine nav button with:

```html
    <button class="nav-btn" onclick="goPage('quiz')" id="nav-quiz">
      <div class="nav-icon">📝</div> Quiz Engine
      <span class="nav-live" id="navQuizLive"><i></i>live</span>
      <span class="nav-ready" id="dot-quiz"></span>
    </button>
```

`.nav-ready` uses `margin-left:auto` (line 191) to push itself right; `.nav-live` also uses `margin-left:auto`, and since it comes first it takes the space, leaving the ready dot immediately after it. That is the intended layout.

- [ ] **Step 5: Add the three functions**

Add immediately after the existing `setPhase` function:

```js
// ── Agent Workspace promotion ──
// The Plan -> Generate -> Validate -> Retry loop is the actual contribution
// being demonstrated, so when a run starts we surface it rather than waiting
// for someone to scroll and find it.
var _awScrolled = false;

function openAgentWorkspace() {
  var ws = document.getElementById('agentWorkspace');
  if (!ws) return;
  ws.style.display = 'block';
  ws.classList.add('live');
  if (!_awScrolled) {
    _awScrolled = true;
    // rAF so the scroll happens after the element is actually laid out
    requestAnimationFrame(function(){
      ws.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
}

function closeAgentWorkspaceLive() {
  var ws = document.getElementById('agentWorkspace');
  if (ws) ws.classList.remove('live');
  _awScrolled = false;
}

function updateAwCounters() {
  var el = document.getElementById('awCounters');
  if (!el) return;
  var rows     = awState.rows || {};
  var keys     = Object.keys(rows);
  var retries  = 0;
  var agents   = {};
  keys.forEach(function(k){
    var r = rows[k] || {};
    if (r.attempt && r.attempt > 1) retries += (r.attempt - 1);
  });
  (awState.agentNames || []).forEach(function(n){ agents[n] = 1; });
  var done = (awState.passCount || 0) + (awState.overrideCount || 0);

  var parts = [];
  parts.push('<span class="awc">questions <b>' + done + '/' + (awState.total || 0) + '</b></span>');
  parts.push('<span class="awc ok">validated <b>' + (awState.passCount || 0) + '</b></span>');
  if (awState.overrideCount) {
    parts.push('<span class="awc warn">override <b>' + awState.overrideCount + '</b></span>');
  }
  if (retries) {
    parts.push('<span class="awc warn">retries <b>' + retries + '</b></span>');
  }
  var nAgents = Object.keys(agents).length;
  if (nAgents) {
    parts.push('<span class="awc">agents <b>' + nAgents + '</b></span>');
  }
  el.innerHTML = parts.join('');
}

function setQuizLiveChip(active) {
  var chip = document.getElementById('navQuizLive');
  if (!chip) return;
  if (active) chip.classList.add('on');
  else        chip.classList.remove('on');
}
```

- [ ] **Step 6: Track agent names so the `agents` counter is real**

`awState` has no agent-name set today, so `updateAwCounters` would always report 0 agents.

**There are TWO `awState` object literals and the field must be added to both.** `resetAgentWorkspace(total)` does not mutate `awState` — it *reassigns* it to a brand-new literal. A field added only to the initial declaration silently disappears the first time a run resets, and the agents counter would read 0 for the rest of the session.

First literal — the `var awState = {` declaration. Insert after the `session:` line, keeping the trailing-comma style:

```js
  agentNames:   [],      // distinct agent names seen in this run
```

Second literal — inside `resetAgentWorkspace`, change:

```js
  awState = {
    total: total || 0, rendered: 0, rows: {},
    phase: null, done: false, passCount: 0, overrideCount: 0,
    session: prevSession,
  };
```

to:

```js
  awState = {
    total: total || 0, rendered: 0, rows: {},
    phase: null, done: false, passCount: 0, overrideCount: 0,
    session: prevSession, agentNames: [],
  };
```

Then in `processNewEvents`, inside the `for` loop, immediately after the line `var e = entries[i];` and before `appendRawLog(e);`:

```js
    if (e && e.agent && awState.agentNames.indexOf(e.agent) === -1) {
      awState.agentNames.push(e.agent);
    }
```

Do not add a reset anywhere else — `resetAgentWorkspace` is the single reset point, and it is now covered.

- [ ] **Step 6b: Verify both literals were changed**

Run: `py -3 -c "s=open('LectureAssis.html',encoding='utf-8').read(); print('agentNames occurrences:', s.count('agentNames'))"`

Expected: `5` — one per literal definition (2), the `indexOf` and `push` calls in `processNewEvents` (2), and the read in `updateAwCounters` (1). A count of 4 means one of the two literals was missed, which is the failure mode this step exists to catch.

- [ ] **Step 7: Wire the calls**

At the end of `processNewEvents`, immediately before its closing brace, add:

```js
  updateAwCounters();
```

In `pollStatus()`, replace the existing line:

```js
    showQuizLiveNotice(d.quiz_thinking_active === true);
```

with:

```js
    showQuizLiveNotice(d.quiz_thinking_active === true);
    setQuizLiveChip(d.quiz_thinking_active === true);
    if (d.quiz_thinking_active === true) openAgentWorkspace();
    else if (_prevQuizThinkingActive) closeAgentWorkspaceLive();
```

This sits before the existing `_prevQuizThinkingActive` assignment further down, so it still reads the previous value. Do not move that assignment.

- [ ] **Step 8: Run the checks**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval`

- [ ] **Step 9: Browser check**

1. Process a file (or press **Generate Quiz** on the Quiz Engine page) and go to Quiz Engine.
2. The Agent Workspace appears and the page scrolls to it automatically — once, not repeatedly on every poll.
3. **Scroll down while the run is live**: the Plan/Generate/Validate/Done stepper stays pinned below the topbar while pipeline rows scroll under it. *If it does not stick, Step 1 was not applied correctly — re-check both the `.main` and `.agent-workspace.live` rules.*
4. The counter row shows `questions n/N`, `validated n`, and `agents n` with a non-zero agent count; retries/override chips appear only when those actually occur.
5. The sidebar Quiz Engine row shows a pulsing green `live` chip while generating, and it disappears when the run ends.
6. When the run ends, the stepper stops being sticky.
7. Check the mobile layout below 860px — the stepper is not sticky and nothing overlaps.
8. Toggle dark mode while the workspace is live — the sticky stepper background is opaque, not transparent, so rows do not show through it.

- [ ] **Step 10: Commit**

```bash
git add LectureAssis.html
git commit -m "Promote Agent Workspace during live runs"
```

---

## Task 5: Guided empty states and disconnected banner

**Files:**
- Modify: `LectureAssis.html` — CSS after line 907, the five static `.empty` blocks, new JS near `goPage`, banner markup at the top of `<main>`

**Interfaces:**
- Consumes: existing `goPage(p)`, `API_BASE`.
- Produces:
  - `refreshConnBanner()` — shows/hides the not-connected banner.
  - DOM id `connBanner`.

The locked states are written as **static markup**, not generated by a helper. They must render correctly before any JS executes, and each one is written once and never re-rendered, so a `lockedState()` builder would be dead code on its first day.

- [ ] **Step 1: Add the CSS**

Insert immediately after line 907 (`.empty-sub{...}`):

```css
.empty-cta{
  margin-top:16px;display:inline-flex;align-items:center;gap:7px;
  font-family:'DM Sans',sans-serif;font-size:12px;font-weight:700;
  padding:7px 16px;border-radius:8px;cursor:pointer;
  background:var(--pri-s);color:var(--g1);
  border:1.5px solid rgba(99,102,241,0.3);
  transition:all var(--tr);
}
.empty-cta:hover{background:var(--g1);color:#fff;border-color:var(--g1)}

/* ── Not-connected banner ── */
.conn-banner{
  display:none;align-items:center;gap:10px;
  background:var(--amber-s);border:1px solid rgba(245,158,11,0.3);
  border-radius:10px;padding:10px 14px;margin-bottom:16px;
  font-size:13px;color:var(--ink2);
}
.conn-banner.on{display:flex}
.conn-banner b{color:var(--amber)}
.conn-banner button{
  margin-left:auto;font-family:'DM Sans',sans-serif;font-size:12px;font-weight:700;
  padding:5px 13px;border-radius:8px;cursor:pointer;
  background:var(--amber);color:#fff;border:none;
}
```

- [ ] **Step 2: Add the banner markup**

Immediately after `<main class="main">`, before the first page div:

```html

    <div class="conn-banner" id="connBanner">
      <span>🔌</span>
      <span><b>Backend not connected.</b> Paste the ngrok URL printed by your Colab notebook to start.</span>
      <button onclick="document.getElementById('setupModal').style.display='flex'">Connect</button>
    </div>
```

- [ ] **Step 3: Add the banner function**

Add immediately before the `goPage` function:

```js
// A grey box reads as unimplemented. Every locked page instead states what it
// will show and offers the step that unlocks it (static markup, see below).
function refreshConnBanner() {
  var b = document.getElementById('connBanner');
  if (!b) return;
  if (API_BASE) b.classList.remove('on');
  else          b.classList.add('on');
}
```

- [ ] **Step 4: Replace the five static empty states**

Each of these is currently a one-line `.empty` block in the markup. Replace the inner content of each container with the version below.

In `page-flashcards`, replace the contents of `<div id="flashArea">`:

```html
        <div class="empty">
          <div class="empty-icon">🃏</div>
          <div class="empty-title">Flashcards unlock after a lecture is processed</div>
          <div class="empty-sub">Key terms and definitions are pulled from your material, then turned into a flip-card deck.</div>
          <button class="empty-cta" onclick="goPage('upload')">Upload a lecture →</button>
        </div>
```

In `page-hints`, replace the contents of `<div id="hintsContent">`:

```html
        <div class="empty">
          <div class="empty-icon">🔥</div>
          <div class="empty-title">Exam Focus unlocks after a lecture is processed</div>
          <div class="empty-sub">Detects the moments your lecturer emphasised and ranks the topics most likely to be examined.</div>
          <button class="empty-cta" onclick="goPage('upload')">Upload a lecture →</button>
        </div>
```

In `page-performance`, replace the contents of `<div id="perfContent">`:

```html
        <div class="empty">
          <div class="empty-icon">📊</div>
          <div class="empty-title">Performance tracking starts with your first graded quiz</div>
          <div class="empty-sub">Scores are broken down per topic, and weak areas are fed back into the difficulty adapter.</div>
          <button class="empty-cta" onclick="goPage('quiz')">Go to Quiz Engine →</button>
        </div>
```

In `page-quiz-past`, replace the contents of `<div id="generatedRunsArea">`:

```html
          <div class="empty">
            <div class="empty-icon">📂</div>
            <div class="empty-title">No saved sets yet</div>
            <div class="empty-sub">Every agentic generation is snapshotted here so you can reload it as the active quiz later.</div>
            <button class="empty-cta" onclick="goPage('quiz')">Generate a quiz →</button>
          </div>
```

In `page-agentlog`, replace the contents of `<div class="card" id="agentLogContent">`:

```html
        <div class="empty">
          <div class="empty-icon">🤖</div>
          <div class="empty-title">No agent log yet</div>
          <div class="empty-sub">The full decision trail of every agent in the last pipeline run appears here, with per-agent timing.</div>
          <button class="empty-cta" onclick="goPage('upload')">Upload a lecture →</button>
        </div>
```

- [ ] **Step 5: Call `refreshConnBanner` at the three points connection state changes**

At the end of `goPage(p)`, before its closing brace:

```js
  refreshConnBanner();
```

In `connectBackend()`, inside the success handler `.then(function(d) {` — immediately after the existing line `updCounts();` (which follows `sessionStorage.setItem('qf_api_base', raw);`):

```js
      refreshConnBanner();
```

That is the first point where the connection is confirmed rather than merely attempted. `API_BASE` is assigned optimistically earlier at `var prev = API_BASE; API_BASE = raw;` so the banner must **not** be refreshed there — a failed connection restores `prev` and the banner would already have been wrongly hidden.

And on initial load, alongside the other startup calls at the bottom of the script:

```js
refreshConnBanner();
```

- [ ] **Step 6: Run the checks**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval`

The div-balance check matters most in this task — five markup blocks were edited.

- [ ] **Step 7: Browser check**

1. Load the page **without** connecting — the amber not-connected banner is visible; its **Connect** button opens the setup modal.
2. Connect — the banner disappears, and stays gone as you navigate between pages.
3. Before processing anything, visit Flashcards, Exam Focus, Performance, Past quiz sets, and Agent Log. Each shows an explanatory card with a working button, not a grey box.
4. Click each button and confirm it lands on the page it names.
5. Process a lecture, then revisit those pages — real content replaces the locked states.
6. Toggle dark mode on a locked state and on the banner — both remain legible.

- [ ] **Step 8: Commit**

```bash
git add LectureAssis.html
git commit -m "Add guided empty states and not-connected banner"
```

---

## Task 6: Quiz mode bar — Simple / Advanced

> **This is the designated cut.** If time is short before the demo, skip this task entirely. Tasks 1–5 stand on their own.

**Files:**
- Modify: `LectureAssis.html` — CSS after line 456, `qmode-bar` markup, JS near `onQModeChange`

**Interfaces:**
- Consumes: existing `onQModeChange()`, `onPlanModeChange()`, `generateQuiz()`.
- Produces: `toggleQmodeAdvanced()`; DOM ids `qmodeAdvanced`, `qmodeAdvBtn`.

**Every control keeps its existing `id`.** `generateQuiz()`, `onQModeChange()` and `onPlanModeChange()` read those ids and are **not** modified by this task. The only change is which controls are visible by default.

`display:contents` is the key technique: the wrapper must not become a flex item itself, or the five wrapped controls would be laid out in a nested row instead of participating in `.qmode-bar`'s flex layout.

- [ ] **Step 1: Add the CSS**

Insert immediately after line 456 (`.qmode-label{...}`):

```css
/* display:contents so the wrapper does not become a flex item -- the wrapped
   .qmode-group children must keep participating in .qmode-bar's own layout. */
.qmode-adv{display:contents}
.qmode-adv[hidden]{display:none}
.qmode-advbtn{
  align-self:flex-end;font-family:'DM Sans',sans-serif;
  font-size:11.5px;font-weight:700;padding:7px 13px;border-radius:8px;
  cursor:pointer;background:transparent;color:var(--ink2);
  border:1.5px solid var(--border2);transition:all var(--tr);
  white-space:nowrap;
}
.qmode-advbtn:hover{background:var(--bg);color:var(--g1);border-color:rgba(99,102,241,0.3)}
.qmode-advbtn .cav{display:inline-block;transition:transform var(--tr)}
.qmode-advbtn.open .cav{transform:rotate(180deg)}
```

- [ ] **Step 2: Wrap the advanced controls**

In the `qmode-bar`, the current child order is: Quiz Mode, Plan Mode (`planGroup`), Manual Topics (`manualTopicsGroup`), Difficulty (`diffGroup`), Questions (`countGroup`), Action (`actionGroup`), Instructions, Generate button.

Reorder so the three simple controls come first, then the wrapper, then the buttons. Open the wrapper immediately after the Quiz Mode group's closing `</div>`:

```html
        <div class="qmode-adv" id="qmodeAdvanced" hidden>
```

Move the `countGroup` block so it sits **before** that opening tag (directly after the Quiz Mode group), since Questions is a simple control.

Then close the wrapper after the Instructions group's closing `</div>`, immediately before the Generate button:

```html
        </div>
        <button class="qmode-advbtn" id="qmodeAdvBtn" onclick="toggleQmodeAdvanced()">Advanced <span class="cav">▾</span></button>
```

Resulting order: Quiz Mode → Questions → `<div class="qmode-adv" hidden>` [ Plan Mode, Manual Topics, Difficulty, Action, Instructions ] `</div>` → Advanced button → Generate Quiz button.

- [ ] **Step 3: Add the toggle function**

Add immediately before `onQModeChange`:

```js
// Simple by default: Quiz Mode, Questions, Generate. Everything else lives
// behind Advanced. All ids are unchanged, so generateQuiz() and the existing
// onQModeChange()/onPlanModeChange() visibility rules keep working as-is.
function toggleQmodeAdvanced() {
  var wrap = document.getElementById('qmodeAdvanced');
  var btn  = document.getElementById('qmodeAdvBtn');
  if (!wrap || !btn) return;
  var opening = wrap.hasAttribute('hidden');
  if (opening) wrap.removeAttribute('hidden');
  else         wrap.setAttribute('hidden', '');
  btn.classList.toggle('open', opening);
  // re-apply the mode-dependent visibility rules for the controls we just showed
  if (opening) onQModeChange();
}
```

- [ ] **Step 4: Run the checks**

Run: `py -3 tools/check_ui.py`
Expected: `OK  js syntax, div balance, colab.py syntax, no-eval`

- [ ] **Step 5: Browser check**

1. On Quiz Engine, the bar shows exactly three controls: **Quiz Mode**, **Questions**, **Generate Quiz**, plus the **Advanced ▾** button.
2. Click **Advanced** — Plan Mode, Difficulty, Action and Instructions appear inline in the same row (not in a nested sub-row, and not stacked); the chevron rotates.
3. Switch Quiz Mode to **Chain-of-Thought (MCQ)** with Advanced open — Plan Mode hides and Difficulty shows, exactly as before this change.
4. Switch Plan Mode to **Manual Topics** — the topic multi-select appears and is populated.
5. Generate a quiz in **Adaptive** mode with Advanced closed — generation works normally.
6. Generate again in **Manual** mode with a chosen difficulty and an instruction string — the settings are respected.
7. Collapse Advanced and generate once more — still works; hidden controls keep their values.
8. Below 860px the bar stacks vertically and the toggle still works.

- [ ] **Step 6: Commit**

```bash
git add LectureAssis.html
git commit -m "Collapse quiz mode bar into simple and advanced sections"
```

---

## Final verification

Run after all tasks, against a live backend. These are the spec's acceptance criteria.

- [ ] `py -3 tools/check_ui.py` prints `OK` and exits 0.
- [ ] `git status` is clean; every task committed separately.
- [ ] Upload → Process shows named agent entries in the ticker **during** OCR/Whisper/Summary/RAG, not only at the end.
- [ ] Step pills show elapsed time; the active step pulses.
- [ ] Each hero pill shows a number traceable to its documented source, or `—`. Cross-check `agent decisions` against `/agent_log` `total`, and `topics found` against `/exam_hints`.
- [ ] Starting a generation auto-opens and scrolls to the Agent Workspace; the stepper stays pinned while rows scroll; the sidebar shows the live chip.
- [ ] Flashcards / Exam Focus / Performance / Past quiz sets / Agent Log show guided cards with working buttons before processing.
- [ ] The quiz mode bar shows three controls; Advanced reveals five more; generation works in Adaptive, Manual, CoT and PoT.
- [ ] Dark mode is legible on every new surface.
- [ ] The layout holds below 860px.
- [ ] No `eval` or `bulk` match anywhere in `LectureAssis.html` (the check script enforces this).

---

## Rollback

Every task is a separate commit, so any single change can be reverted independently:

```bash
git log --oneline -8          # find the commit
git revert <sha>              # undo just that task
```

The two riskiest changes, if something looks wrong under demo pressure:

- **Task 4's `.main{overflow-x:clip}`** — if any page develops a horizontal scrollbar, revert that one declaration to `hidden`. The only consequence is that the phase stepper stops sticking; nothing else breaks.
- **Task 1's backend alias** — if `/agent_log` misbehaves, restore `state.agent_log = []`. The ticker and the `agent decisions` pill go quiet (the pill correctly falls back to `—`), but processing itself is completely unaffected.
