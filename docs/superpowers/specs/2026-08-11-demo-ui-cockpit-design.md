# Demo Cockpit — UI design for the LectureAssist capstone demo

**Date:** 2026-08-11
**Scope:** `LectureAssis.html` (frontend) + one line in `colab.py` (backend)
**Goal:** Make the live capstone demo read unmistakably as an *agentic multi-agent system*, not a chat wrapper, and remove the moments where the UI looks empty or unfinished.

---

## 1. Context

LectureAssist is a single-page frontend (`LectureAssis.html`, ~3.4k lines, self-contained CSS + vanilla JS) talking to a Flask backend running in Google Colab behind ngrok (`colab.py`).

The UI is already mature: CSS custom-property design tokens, light/dark themes, a gradient system, drag-and-drop upload zones, a sidebar with per-page "ready" dots, and an Agent Workspace that renders a Plan→Generate→Validate→Done stepper with per-question pipeline rows.

Evaluation surfaces were removed from both frontend and backend immediately before this work (707 lines deleted). Nothing in this design reintroduces them.

### Demo format

The presentation is a **full live run**: a lecture video or document is uploaded on stage, `Process Content` is pressed, and the panel watches the pipeline execute end to end before any quiz is generated.

This is the single most important constraint in this document. It means there are several minutes during which the audience is looking at a progress bar, and what fills that time determines whether the system reads as agentic.

### The four problems being solved

Identified by the project owner:

1. The agentic story is not obvious enough — the Agent Workspace is `display:none` until a quiz run starts, and sits below the fold on a different page from where processing happens.
2. Weak first impression — the Upload page is the landing screen and carries no statement of what the system does or what it just produced.
3. Empty and disconnected states — before processing, most pages show grey placeholder boxes.
4. The Quiz Engine mode bar is cluttered — seven controls in one row.

---

## 2. Key finding: the agent log is not live

This constrains change A and is the reason a backend edit is required at all.

In `colab.py`, `OrchestratorAgent.run()`:

```python
ctx = AgentContext(content_type, file_path)
state.session_id = state._make_session_id(ctx.source_name)
state.agent_log  = []          # <- reset to a NEW empty list
```

and only at the very end of the pipeline:

```python
state.agent_log = ctx.agent_log   # <- populated here, after everything finishes
```

Both `BaseAgent._log()` and `OrchestratorAgent._log()` append exclusively to `ctx.agent_log`. `state.agent_log` is what `/agent_log` serves. Therefore **`/agent_log` returns an empty list for the entire duration of a run** and only fills in once the run is already complete.

Consequence: during OCR, Whisper, Summary and RAG — the long stages — there is currently no live agent activity available to the frontend at all. `/status` carries `quiz_thinking[]`, but that is only populated during quiz generation, which happens at the end.

**Fix:** alias instead of replace, so the list `/agent_log` serves is the same object the agents append to:

```python
state.agent_log = ctx.agent_log   # alias: live for the whole run
```

The existing end-of-pipeline assignment then becomes a harmless no-op and is left in place. This is a one-line change with no behavioural risk: `state.agent_log` is only read by `/agent_log` and written at these two sites.

---

## 3. Design

Five changes, labelled A–E. All are additive — no existing feature is rewired, and no working code path is replaced. A requires the backend line above; B–E are frontend-only.

### A. Live agent ticker on the Upload page

The `progCard` element gains an **Agent activity** feed directly below the existing four step pills.

- **Data source:** `/agent_log`, polled on the existing `pollStatus` cadence. Each entry already carries `{agent, level, time, message, duration_sec?}`.
- **Rendering:** the most recent ~6 entries, newest at the bottom, auto-scrolled. Each row is an agent-name chip, the message, and the duration when present. Colour is keyed off `level` reusing the existing token palette — `PLAN` cyan, `DONE` green, `RETRY` amber, `ERROR`/`FAIL` red, everything else muted.
- **Only rendered while a run is in flight**, and it keeps its final state after completion so the finished trail stays on screen.
- The four step pills (`OCR`, `Whisper`, `Summary`, `RAG`) gain a per-step elapsed timer and a pulse on the currently active step. Today they only toggle `active`/`done` with no sense of time passing.

*Rationale:* this is the change that directly addresses the demo format. It converts several minutes of dead air into a continuously updating trace of named agents making decisions, on the screen the presenter is already standing on.

### B. Hero strip and run payoff on the Upload page

A compact hero above the two upload cards:

- Product name, a one-line statement of what the system does.
- Four stat pills reading `—` before any run, filled in on completion:

  | Pill | Source | Note |
  |---|---|---|
  | Transcript length (words) | `d.transcript` in `onDone(d)`; `d.doc_chars` when a document was uploaded instead | Already consumed by `onDone` |
  | Topics detected | `/exam_hints` → `ai_analysis.ai_important_topics` length | **Not** `/quiz/topics.total` — that endpoint falls back to four invented generic topics (`Core Concepts`, `Key Principles`, …) when nothing is detected, which would put a fabricated number on stage. If `ai_important_topics` is absent or empty, the pill stays `—`. |
  | Questions generated | Sum of the four `total` values already fetched by `updCounts()` | No new request |
  | Agent decisions logged | `/agent_log` → `total` | Same fetch the ticker already makes |

  Any pill whose source is missing stays `—`. No pill is ever populated with a placeholder or estimated value.

*Rationale:* fixes the landing impression, and gives the presenter a closing beat when processing finishes — a concrete "here is what it just did" instead of a toast that disappears.

### C. Agent Workspace promotion

- When a generation starts, the workspace auto-opens and scrolls into view rather than waiting to be found.
- The phase stepper becomes sticky within the workspace while a run is live, so Plan→Generate→Validate→Done stays visible while pipeline rows scroll underneath.
- A live counter line summarising the run: agents involved, questions in flight, retries, replans. `awState` already tracks `total`, `passCount`, `overrideCount` and per-row `attempt`; the counter derives from those and needs no new backend data.
- A pulsing "live" chip appears in the sidebar next to Quiz Engine while `quiz_thinking_active` is true, reusing the existing `.nav-ready` dot slot.

*Rationale:* makes the Plan→Generate→Validate→Retry loop — the actual research contribution — the thing on screen during generation.

### D. Guided empty and disconnected states

- A single reusable "locked" empty-state card replacing the current grey `.empty` boxes: icon, a sentence describing what the page will show, and a button that navigates to the step which unlocks it (usually Upload).
- A dismissible banner when `API_BASE` is unset, stating the backend is not connected and offering the setup modal.

*Rationale:* if a panel member clicks Flashcards or Exam Focus before processing, they should see stated intent rather than an empty container that reads as unimplemented.

### E. Quiz mode bar: Simple / Advanced

The seven controls collapse to three by default — Quiz Mode, Questions, Generate Quiz — with an `Advanced ▾` toggle revealing Plan Mode, Manual Topics, Difficulty, Action and Instructions.

- Implemented as a show/hide wrapper around the existing `.qmode-group` elements. Every control keeps its current `id`, so `generateQuiz()`, `onQModeChange()` and `onPlanModeChange()` are untouched.
- The existing conditional visibility logic in `onQModeChange()` (which already hides `diffGroup` and `planGroup` depending on mode) continues to apply *within* the advanced section.

*Rationale:* the bar currently reads as a debugging console. Three controls reads as a product.

**This is the designated cut.** If time runs short before the demo, E is dropped first — it is the only change that touches a screen the presenter must interact with live, and therefore carries the most rehearsal risk.

---

## 4. Non-goals

- No visual reskin. The token system, type scale and dark mode are already coherent; changing them would consume the remaining time without improving the demo.
- No restructuring of the Quiz Engine page beyond E's show/hide wrapper.
- No reintroduction of any evaluation UI, endpoint, or download.
- No new backend endpoints. The only backend change is the one-line alias in section 2.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| The backend alias changes `/agent_log` behaviour mid-run | The endpoint is read-only. Its only existing consumer is `loadAgentLog()`, which already renders whatever list it receives and is triggered manually via Refresh; a shorter list mid-run is a valid state for it. The ticker becomes the second consumer. Verified by syntax check plus a live run before the demo. |
| Polling `/agent_log` alongside `/status` doubles request volume | Both are cheap in-memory reads; the poll interval is unchanged. The ticker fetch is skipped entirely once the run reports `done`. |
| Sticky stepper interacts badly with the mobile layout | Sticky positioning is scoped to the workspace container and applied only above the existing mobile breakpoint. |
| Frontend edits break the single-file page | `node --check` on the extracted `<script>` block and a tag-balance check after each stage, as was done for the evaluation removal. |

---

## 6. Verification

The work is complete when, on a real run against the Colab backend:

1. Uploading a file and pressing Process shows named agent entries appearing in the Upload-page ticker *during* OCR/Whisper/Summary/RAG — not only at the end.
2. Step pills show elapsed time and the active step pulses.
3. On completion, each hero stat pill either carries a number traceable to its documented source or remains `—`. No pill shows a fallback or estimated value.
4. Starting a quiz generation auto-opens and scrolls to the Agent Workspace, the stepper stays visible while rows scroll, and the sidebar shows a live chip.
5. Visiting Flashcards / Exam Focus / Performance before processing shows guided cards with a working navigation button, not grey boxes.
6. The Quiz mode bar shows three controls, and Advanced reveals the other five with generation still working in every mode.
7. `node --check` passes on the extracted script; `ast.parse` passes on `colab.py` with Colab magics stripped.
8. No occurrence of `eval` or `bulk` remains in `LectureAssis.html`.
