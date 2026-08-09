/* Builds the two ILLUSTRATED figures (robot/avatar style) as native .drawio:
 *   lectureassist-architecture-illustrated.drawio
 *   pipeline-merge-illustrated.drawio
 * Mascots come from mascots.js, embedded as SVG data URIs.
 *
 * Icon captions sit BELOW each node and are outside its geometry, so any edge
 * attached to a node's bottom is drawn straight through its own caption. Every
 * vertical / feedback connector is therefore a FREE edge with explicit
 * endpoints routed under the caption band. Do not "simplify" these back into
 * attached edges.
 *
 * This file is UTF-8 and contains em-dashes, arrows and middle dots. Do NOT
 * round-trip it through PowerShell Get-Content/Set-Content: that re-encodes it
 * as cp1252 and destroys those characters irreversibly.
 *
 *   node build_illustrated.js <outdir>
 *   drawio -x -f pdf -e --crop -b 12 -o <name>.pdf <name>.drawio
 */
const fs = require("fs");
const path = require("path");
const M = require("./mascots");

const OUT = process.argv[2] || ".";
const INK = "#1F3864", EDGE = "#34495E", SUB = "#5A5A5A";
const GRN = "#1E8449", RED = "#C0392B", TEAL = "#16A085", PUR = "#7D3C98";

const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                          .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const iconStyle = u =>
  `shape=image;html=1;imageAspect=0;aspect=fixed;image=${u};`
  + `verticalLabelPosition=bottom;verticalAlign=top;labelBackgroundColor=none;`
  + `fontSize=11;fontStyle=1;fontColor=${INK};fontFamily=Calibri;`;

const two = (a, b) => `&lt;b&gt;${esc(a)}&lt;/b&gt;`
  + (b ? `&lt;br/&gt;&lt;font color=&quot;${SUB}&quot; style=&quot;font-size:9px&quot;&gt;${esc(b)}&lt;/font&gt;` : "");

const icon = (id, u, label, x, y, w = 64, h = 68) =>
  `<mxCell id="${id}" value="${label}" style="${iconStyle(u)}" vertex="1" parent="1">`
  + `<mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry"/></mxCell>`;

const box = (id, label, x, y, w, h, dashed, color) =>
  `<mxCell id="${id}" value="${esc(label)}" style="rounded=1;arcSize=4;html=1;fillColor=none;`
  + `strokeColor=${color || EDGE};strokeWidth=1.6;${dashed ? "dashed=1;dashPattern=8 5;" : ""}`
  + `verticalAlign=top;align=left;spacingLeft=10;spacingTop=6;fontSize=12;fontStyle=1;`
  + `fontColor=${color || INK};fontFamily=Calibri;" vertex="1" parent="1">`
  + `<mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry"/></mxCell>`;

// html=1 means the value is parsed as HTML, so a literal \n collapses to a
// space -- it has to be an explicit <br/> or every callout renders on one line.
const note = (id, label, x, y, w, h, color) =>
  `<mxCell id="${id}" value="${esc(label).replace(/\n/g, "&lt;br/&gt;")}" `
  + `style="rounded=1;arcSize=14;html=1;whiteSpace=wrap;fillColor=#FFFFFF;`
  + `strokeColor=${color};strokeWidth=1.6;fontSize=11;fontColor=${color};`
  + `fontFamily=Calibri;align=center;" vertex="1" parent="1">`
  + `<mxGeometry x="${x}" y="${y}" width="${w}" height="${h}" as="geometry"/></mxCell>`;

const text = (id, label, x, y, w, o = {}) =>
  `<mxCell id="${id}" value="${esc(label)}" style="text;html=1;align=${o.align || "left"};`
  + `verticalAlign=middle;fontSize=${o.fs || 10};fontColor=${o.color || SUB};`
  + `${o.bold ? "fontStyle=1;" : ""}fontFamily=Calibri;" vertex="1" parent="1">`
  + `<mxGeometry x="${x}" y="${y}" width="${w}" height="16" as="geometry"/></mxCell>`;

/* chunky outlined block arrow, as in the reference figure - straight runs only */
const flex = c => `edgeStyle=none;shape=flexArrow;html=1;fillColor=#FFFFFF;`
  + `strokeColor=${c || EDGE};strokeWidth=1.4;width=9;endWidth=17;endSize=5;startSize=0;`
  + `fontSize=10;fontStyle=1;fontColor=${c || INK};fontFamily=Calibri;labelBackgroundColor=#FFFFFF;`;

/* thin orthogonal connector for feedback / L-shaped routes */
const wire = (c, dashed) => `edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;`
  + `strokeColor=${c || EDGE};strokeWidth=1.8;${dashed ? "dashed=1;dashPattern=7 5;" : ""}`
  + `endArrow=blockThin;endFill=1;endSize=6;fontSize=10;fontStyle=1;`
  + `fontColor=${c || INK};fontFamily=Calibri;labelBackgroundColor=#FFFFFF;`;

/* free edge: explicit endpoints, so nothing is anchored to a node whose caption
 * would be crossed. pts = [[x,y] start, ...waypoints, [x,y] end] */
function fedge(id, pts, style, label) {
  const s = pts[0], t = pts[pts.length - 1], mid = pts.slice(1, -1);
  return `<mxCell id="${id}" value="${label ? esc(label) : ""}" style="${style}" edge="1" parent="1">`
    + `<mxGeometry relative="1" as="geometry">`
    + `<mxPoint x="${s[0]}" y="${s[1]}" as="sourcePoint"/>`
    + `<mxPoint x="${t[0]}" y="${t[1]}" as="targetPoint"/>`
    + (mid.length ? `<Array as="points">${mid.map(p => `<mxPoint x="${p[0]}" y="${p[1]}"/>`).join("")}</Array>` : "")
    + `</mxGeometry></mxCell>`;
}

const doc = (cells, w, h) =>
  `<mxGraphModel dx="1400" dy="900" grid="0" page="1" pageWidth="${w}" pageHeight="${h}" `
  + `adaptiveColors="auto"><root><mxCell id="0"/><mxCell id="1" parent="0"/>`
  + cells.join("") + `</root></mxGraphModel>`;

/* ===================== FIGURE 1 - system architecture ===================== */
function architecture() {
  const c = [];
  const U = {
    student: M.uri(M.student()), docu: M.uri(M.doc()),
    prep: M.uri(M.gear("#2E86C1")), plan: M.uri(M.robot("#F4C542", "bulb")),
    ctrl: M.uri(M.gear(PUR)), gen: M.uri(M.robot("#E67E22", "q")),
    val: M.uri(M.robot(GRN, "check")), fmt: M.uri(M.gear("#2E86C1")),
    lens: M.uri(M.lens()), qw: M.uri(M.robot("#2E86C1", "lens")),
    cm: M.uri(M.robot(PUR, "doc")), gate: M.uri(M.gear(TEAL)),
    ok: M.uri(M.badge("check")), no: M.uri(M.badge("cross")),
  };

  c.push(box("clus", "Generator Cluster — one question at a time", 432, 44, 344, 356));
  c.push(box("rag", "Grounding & Retrieval Module — search_lecture tool", 812, 44, 380, 300, true, TEAL));

  c.push(icon("student", U.student, two("Learner"), 46, 62, 64, 64));
  c.push(icon("docu", U.docu, two("Lecture video / PDF"), 176, 62, 64, 64));
  c.push(icon("prep", U.prep, two("Preprocessor", "OCR + Whisper → fused timeline"), 316, 62, 64, 64));
  c.push(icon("plan", U.plan, two("Quiz Planner", "topic × difficulty slots"), 316, 206, 64, 68));
  c.push(icon("ctrl", U.ctrl, two("Controller", "cognitive routing"), 316, 350, 64, 64));

  c.push(icon("g1", U.gen, two("Text-based Generator"), 466, 78, 64, 68));
  c.push(icon("g2", U.gen, two("Inferential Generator"), 466, 190, 64, 68));
  c.push(icon("g3", U.gen, two("Main-idea Generator"), 466, 300, 64, 68));

  c.push(icon("r1", U.gate, two("Coverage Gate"), 846, 84, 64, 64));
  c.push(icon("r3", U.lens, two("FAISS Top-k"), 1046, 84, 64, 64));
  c.push(icon("r4", U.cm, two("Context Merger"), 1046, 210, 64, 68));
  c.push(icon("r2", U.qw, two("Query Writer"), 846, 210, 64, 68));

  c.push(icon("val", U.val, two("Validator", "grounding · difficulty · compliance"), 566, 486, 64, 68));
  c.push(icon("ok", U.ok, "", 452, 498, 44, 44));
  c.push(icon("no", U.no, "", 700, 498, 44, 44));
  c.push(icon("fmt", U.fmt, two("Formatter"), 316, 486, 64, 64));
  c.push(note("ref", "Refiner\nregenerate · fetch context · replan topic", 800, 486, 250, 66, RED));

  c.push(fedge("e1", [[116, 94], [170, 94]], flex()));
  c.push(fedge("e2", [[246, 94], [310, 94]], flex()));
  c.push(fedge("e3", [[348, 164], [348, 200]], flex()));
  c.push(fedge("e4", [[348, 312], [348, 344]], flex()));
  c.push(fedge("e5", [[386, 382], [428, 382]], flex()));

  c.push(fedge("e6", [[536, 112], [842, 112]], flex(TEAL)));
  c.push(fedge("e7", [[916, 116], [1042, 116]], flex(TEAL)));
  c.push(fedge("e8", [[1078, 176], [1078, 206]], flex(TEAL)));
  c.push(fedge("e9", [[1042, 244], [916, 244]], flex(TEAL)));
  c.push(fedge("e10", [[842, 244], [792, 244], [792, 224], [536, 224]],
               wire(TEAL, true), "retrieved context"));

  c.push(fedge("e11", [[498, 396], [498, 432], [598, 432], [598, 482]], wire()));
  c.push(fedge("e12", [[562, 520], [500, 520]], flex(GRN), "PASS"));
  c.push(fedge("e13", [[448, 520], [384, 520]], flex(GRN)));
  c.push(fedge("e14", [[634, 520], [696, 520]], flex(RED), "FAIL"));
  c.push(fedge("e15", [[748, 520], [796, 520]], flex(RED)));
  c.push(fedge("e16", [[925, 482], [925, 424], [780, 424], [780, 390]],
               wire(RED, true), "retry within budget"));
  c.push(fedge("e17", [[312, 518], [28, 518], [28, 94], [42, 94]],
               wire(), "formatted quiz"));

  c.push(text("foot", "All agents run gemma3:4b served locally by Ollama; the independent judge is "
    + "gemma3:12b. The Controller assigns each slot exactly one cognitive track, so the spread "
    + "is enforced rather than requested.", 36, 616, 1150, { fs: 10 }));

  return doc(c, 1220, 648);
}

/* ==================== FIGURE 2 - merge comparison ==================== */
function merges() {
  const c = [];
  const U = {
    plan: M.uri(M.robot("#F4C542", "bulb")), gen: M.uri(M.robot("#E67E22", "q")),
    val: M.uri(M.robot(GRN, "check")), ref: M.uri(M.robot(RED, "merge")),
    pair: M.uri(M.robotPair(PUR)), ok: M.uri(M.badge("check")),
  };
  const LY = [58, 258, 458], NY = l => LY[l] + 40, RAIL = l => LY[l] + 166;
  const CX = [70, 250, 430, 610, 830];
  const lane = [
    ["CONTROL — full agentic pipeline", "as implemented in colab.py", "#1F4E79",
     "4 LLM calls\nper retried question"],
    ["merge_vr — Validator + Refiner merged", "one call judges AND rewrites", PUR,
     "3 LLM calls\nper retried question"],
    ["merge_pg — Planner + Generator merged", "no separate planner call", PUR,
     "4 LLM calls\nbatch planner call removed"],
  ];

  lane.forEach((L, i) => {
    c.push(`<mxCell id="ln${i}" value="${esc(L[0])}" style="rounded=1;arcSize=4;html=1;`
      + `fillColor=${i ? "#FBF7FD" : "#F5F8FA"};strokeColor=none;verticalAlign=top;align=left;`
      + `spacingLeft=12;spacingTop=6;fontSize=12;fontStyle=1;fontColor=${L[2]};fontFamily=Calibri;" `
      + `vertex="1" parent="1"><mxGeometry x="20" y="${LY[i]}" width="1150" height="192" as="geometry"/></mxCell>`);
    c.push(text(`ls${i}`, L[1], 34, LY[i] + 26, 420));
    c.push(note(`lc${i}`, L[3], 960, NY(i) + 6, 190, 56, L[2]));
  });

  const step = (id, u, a, l, col) => c.push(icon(id, u, two(a), CX[col], NY(l), 64, 68));

  step("a1", U.plan, "Quiz Planner", 0, 0);
  step("a2", U.gen, "Generator", 0, 1);
  step("a3", U.val, "Validator", 0, 2);
  step("a4", U.ref, "Refiner", 0, 3);
  step("a5", U.ok, "Accepted MCQ", 0, 4);

  step("b1", U.plan, "Quiz Planner", 1, 0);
  step("b2", U.gen, "Generator", 1, 1);
  c.push(icon("b3", U.pair, two("Validator + Refiner", "merged into one call"), 500, NY(1), 104, 68));
  step("b5", U.ok, "Accepted MCQ", 1, 4);

  c.push(icon("c1", U.pair, two("Planner + Generator", "merged into one call"), 140, NY(2), 104, 68));
  step("c3", U.val, "Validator", 2, 2);
  step("c4", U.ref, "Refiner", 2, 3);
  step("c5", U.ok, "Accepted MCQ", 2, 4);

  const mid = l => NY(l) + 34;
  const arr = (id, x1, x2, l) => c.push(fedge(id, [[x1, mid(l)], [x2, mid(l)]], flex()));

  arr("m1", 140, 244, 0); arr("m2", 320, 424, 0); arr("m3", 500, 604, 0); arr("m4", 680, 824, 0);
  arr("m5", 140, 244, 1); arr("m6", 320, 494, 1); arr("m7", 610, 824, 1);
  arr("m8", 250, 424, 2); arr("m9", 500, 604, 2); arr("m10", 680, 824, 2);

  /* retry rails live BELOW the caption band so they never cross a label */
  const retry = (id, sx, tx, l) => c.push(fedge(id,
    [[sx, NY(l) + 108], [sx, RAIL(l)], [tx, RAIL(l)], [tx, NY(l) + 110]],
    wire(RED, true), "FAIL — retry"));
  retry("rt0", 642, 282, 0);
  retry("rt1", 552, 282, 1);
  retry("rt2", 642, 192, 2);

  c.push(text("mf", "Call counts are structural — counted from the implementation, not measured "
    + "runtimes. Retrieval, cognitive routing and the pinned gemma3:4b generator are identical "
    + "across all three.", 22, 668, 1140, { fs: 10 }));

  return doc(c, 1190, 696);
}

fs.writeFileSync(path.join(OUT, "lectureassist-architecture-illustrated.drawio"), architecture(), "utf8");
/* ============ FIGURE 3 - condensed variant for the synopsis page ============
 * The full architecture is far too dense for a one-page synopsis: at 164 mm
 * text width its 11px captions render at about 4 pt. This variant is designed
 * on a WIDE, SHORT canvas with deliberately oversized type, so that after
 * \includegraphics scales it to \linewidth the captions land near 8 pt.
 * Scale factor = 164.3mm / (1300px * 25.4/96) = 0.478, so 22px -> ~7.9pt. */
function synopsisFigure() {
  const c = [];
  const F = 22, FS = 18;                       // oversized on purpose - see above
  const big = (a, b) => `&lt;b&gt;&lt;font style=&quot;font-size:${F}px&quot;&gt;${esc(a)}&lt;/font&gt;&lt;/b&gt;`
    + (b ? `&lt;br/&gt;&lt;font color=&quot;${SUB}&quot; style=&quot;font-size:${FS}px&quot;&gt;${esc(b)}&lt;/font&gt;` : "");

  const U = {
    student: M.uri(M.student()), docu: M.uri(M.doc()),
    prep: M.uri(M.gear("#2E86C1")), idx: M.uri(M.lens()),
    plan: M.uri(M.robot("#F4C542", "bulb")), gen: M.uri(M.robot("#E67E22", "q")),
    val: M.uri(M.robot(GRN, "check")), fmt: M.uri(M.gear("#2E86C1")),
    ok: M.uri(M.badge("check")), no: M.uri(M.badge("cross")),
  };

  const CX = [90, 400, 710, 1020], IW = 84, IH = 88;
  const R1 = 40, R2 = 218, CAP1 = R1 + IH, MIDY = 186;
  const cx = i => CX[i] + IW / 2;

  const put = (id, u, a, b, col, y) =>
    c.push(icon(id, u, big(a, b), CX[col], y, IW, IH));

  put("s1", U.student, "Learner", "", 0, R1);
  put("s2", U.docu, "Lecture video / PDF", "", 1, R1);
  put("s3", U.prep, "Preprocessor", "", 2, R1);
  put("s4", U.idx, "Summary + FAISS index", "", 3, R1);

  put("t1", U.plan, "Quiz Planner", "topic × difficulty", 0, R2);
  put("t2", U.gen, "Generator", "3 cognitive tracks", 1, R2);
  put("t3", U.val, "Validator", "grounding · difficulty", 2, R2);
  put("t4", U.fmt, "Formatter", "quiz to the learner", 3, R2);

  const st = flex(); st.replace("width=9", "width=14");
  const big2 = c2 => flex(c2).replace("width=9;endWidth=17;endSize=5",
                                      "width=14;endWidth=26;endSize=8");

  c.push(fedge("q1", [[CX[0] + IW, R1 + 44], [CX[1] - 6, R1 + 44]], big2()));
  c.push(fedge("q2", [[CX[1] + IW, R1 + 44], [CX[2] - 6, R1 + 44]], big2()));
  c.push(fedge("q3", [[CX[2] + IW, R1 + 44], [CX[3] - 6, R1 + 44]], big2()));

  /* wrap: right side down, across, into the second row */
  c.push(fedge("qw", [[cx(3), CAP1 + 34], [cx(3), MIDY], [cx(0), MIDY], [cx(0), R2 - 6]],
               wire(), "grounded content"));

  c.push(fedge("q4", [[CX[0] + IW, R2 + 44], [CX[1] - 6, R2 + 44]], big2()));
  c.push(fedge("q5", [[CX[1] + IW, R2 + 44], [CX[2] - 6, R2 + 44]], big2()));
  c.push(fedge("q6", [[CX[2] + IW, R2 + 44], [CX[3] - 6, R2 + 44]], big2(GRN)));
  c.push(icon("qok", U.ok, "", (CX[2] + IW + CX[3]) / 2 - 20, R2 + 24, 40, 40));

  /* retry loop, below the caption band so it crosses nothing */
  c.push(fedge("qr", [[cx(2), R2 + IH + 46], [cx(2), R2 + IH + 76],
                      [cx(1), R2 + IH + 76], [cx(1), R2 + IH + 48]],
               wire(RED, true), "FAIL — regenerate / fetch context / replan"));
  /* no cross badge here: at this size it lands on the Validator's caption, and
   * the retry edge is already red and labelled FAIL, so it earns nothing. */

  return doc(c, 1300, 404);
}

fs.writeFileSync(path.join(OUT, "pipeline-merge-illustrated.drawio"), merges(), "utf8");
fs.writeFileSync(path.join(OUT, "synopsis-architecture.drawio"), synopsisFigure(), "utf8");
console.log("wrote illustrated .drawio files to", OUT);
