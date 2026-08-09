/* Builds the merge-comparison figure as a native .drawio file, with the agent
 * icons embedded as SVG data URIs.
 *
 * This script is the SOURCE OF TRUTH. Edit it, re-run it, then re-export the
 * SVG. Editing the .drawio in the draw.io GUI instead will be silently
 * overwritten the next time this runs.
 *
 *   node pipeline-merge-comparison.build.js .
 *   "C:\Program Files\draw.io\draw.io.exe" -x -f svg -b 12 \
 *       -o pipeline-merge-comparison.svg pipeline-merge-comparison.drawio
 */
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");

const OUT_DIR = process.argv[2] || ".";
const BASE = "pipeline-merge-comparison";

const C = {
  a:      "#1B4F72",   // stage A / planner-side
  b:      "#7E5109",   // stage B / generation-side
  merged: "#6C3483",   // a merged agent — the thing the figure is about
  fail:   "#922B21",   // failure + retry path only
  ink:    "#333333",
  sub:    "#5A5A5A",
  line:   "#8A8A8A",
};

/* ---------- icons: 24x24, stroke-based so they stay crisp when scaled ---- */
const ICONS = {
  planner: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 3h8v3H8z" fill="${c}" stroke="none"/>
<path d="M8 11h3M8 15h3"/><path d="M14 10.6l1.3 1.3 2.4-2.6M14 14.6l1.3 1.3 2.4-2.6"/></g>`,

  generator: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M6 3h8l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M14 3v5h5"/>
<path d="M9.5 14.2l.9-2 .9 2 2 .9-2 .9-.9 2-.9-2-2-.9z" fill="${c}" stroke="none"/></g>`,

  validator: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M12 2.6l7 2.6v6c0 4.6-3 8.3-7 10.2-4-1.9-7-5.6-7-10.2v-6z"/><path d="M8.6 11.8l2.4 2.4 4.4-4.8"/></g>`,

  refiner: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M20 11a8 8 0 1 0-.7 4.4"/><path d="M20 4.5V11h-6.2"/></g>`,

  merge: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<path d="M3 5h4.2c1.6 0 3 .9 3.8 2.3l2 3.4c.8 1.4 2.2 2.3 3.8 2.3H21"/>
<path d="M3 19h4.2c1.6 0 3-.9 3.8-2.3l.7-1.2"/><path d="M17.4 9.6L21 13l-3.6 3.4"/></g>`,

  accepted: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<circle cx="12" cy="12" r="9"/><path d="M7.8 12.3l2.8 2.8 5.6-6.2"/></g>`,

  data: c => `<g fill="none" stroke="${c}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
<ellipse cx="12" cy="5.8" rx="7.4" ry="2.8"/><path d="M4.6 5.8v12.4c0 1.5 3.3 2.8 7.4 2.8s7.4-1.3 7.4-2.8V5.8"/>
<path d="M4.6 12c0 1.5 3.3 2.8 7.4 2.8s7.4-1.3 7.4-2.8"/></g>`,
};

const svgIcon = (name, color) =>
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24">${ICONS[name](color)}</svg>`;

const iconDataUri = (name, color) =>
  "data:image/svg+xml," + Buffer.from(svgIcon(name, color), "utf8").toString("base64");

/* ---------- layout ------------------------------------------------------- */
const NW = 178, NH = 54, GAP = 22;
const COL = i => 44 + i * (NW + GAP);          // 44, 244, 444, 644, 844
const LANE_X = 20, LANE_W = 1216;
const LANE_H = 168;
const LANE_Y = [128, 316, 504];
const nodeY = l => LANE_Y[l] + 42;
const retryY = l => LANE_Y[l] + 132;

const lanes = [
  { t: "CONTROL — full agentic pipeline", s: "as implemented in colab.py", c: C.a,
    calls: "4 LLM calls", callsub: "per retried question" },
  { t: "merge_vr — Validator + Refiner merged", s: "one call judges AND rewrites", c: C.merged,
    calls: "3 LLM calls", callsub: "per retried question" },
  { t: "merge_pg — Planner + Generator merged", s: "no separate planner call", c: C.merged,
    calls: "4 LLM calls", callsub: "batch planner call removed" },
];

const N = [];
const add = o => (N.push(o), o);

// lane 0 — control
add({ id:"a1", l:0, c:0, w:NW, t:"Quiz Planner",  s:"topic × difficulty slots", ic:"planner",   col:C.a });
add({ id:"a2", l:0, c:1, w:NW, t:"Generator",     s:"drafts one MCQ",           ic:"generator", col:C.b });
add({ id:"a3", l:0, c:2, w:NW, t:"Validator",     s:"PASS / FAIL + action",     ic:"validator", col:C.b });
add({ id:"a4", l:0, c:3, w:NW, t:"Refiner",       s:"redrafts on FAIL",         ic:"refiner",   col:C.b });
add({ id:"a5", l:0, c:4, w:NW, t:"Accepted MCQ",  s:"or OVERRIDE",              ic:"accepted",  col:C.a });

// lane 1 — merge_vr
add({ id:"b1", l:1, c:0, w:NW, t:"Quiz Planner",  s:"unchanged",                ic:"planner",   col:C.a });
add({ id:"b2", l:1, c:1, w:NW, t:"Generator",     s:"drafts one MCQ",           ic:"generator", col:C.b });
add({ id:"b3", l:1, c:2, w:2*NW+GAP, t:"Validator + Refiner  (merged)",
      s:"one call returns the verdict and the rewritten question", ic:"merge", col:C.merged, m:true });
add({ id:"b4", l:1, c:4, w:NW, t:"Accepted MCQ",  s:"or OVERRIDE",              ic:"accepted",  col:C.a });

// lane 2 — merge_pg
add({ id:"c1", l:2, c:0, w:2*NW+GAP, t:"Planner + Generator  (merged)",
      s:"one call picks the topic and drafts the MCQ", ic:"merge", col:C.merged, m:true });
add({ id:"c2", l:2, c:2, w:NW, t:"Validator",     s:"PASS / FAIL + action",     ic:"validator", col:C.b });
add({ id:"c3", l:2, c:3, w:NW, t:"Refiner",       s:"redrafts on FAIL",         ic:"refiner",   col:C.b });
add({ id:"c4", l:2, c:4, w:NW, t:"Accepted MCQ",  s:"or OVERRIDE",              ic:"accepted",  col:C.a });

const byId = Object.fromEntries(N.map(n => [n.id, n]));
const X = n => COL(n.c), Y = n => nodeY(n.l);

const FLOWS = [["a1","a2"],["a2","a3"],["a3","a4"],["a4","a5"],
               ["b1","b2"],["b2","b3"],["b3","b4"],
               ["c1","c2"],["c2","c3"],["c3","c4"]];
// retry edges: [from, to, lane]
const RETRIES = [["a4","a2",0],["b3","b2",1],["c3","c1",2]];

const SRC = { x: 20, y: 50, w: 1216, h: 48,
              t:"Grounded content + FAISS retrieval index",
              s:"identical input to all three configurations below — extraction, fusion and indexing are unchanged",
              ic:"data", col:C.a };

const FOOT = "Call counts are structural — counted from the implementation, "
           + "not measured runtimes. Retrieval, cognitive routing and the pinned "
           + "gemma3:4b generator are identical across all three configurations.";

const W = 1256, H = LANE_Y[2] + LANE_H + 46;

/* ---------- draw.io XML -------------------------------------------------- */
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
                          .replace(/>/g,"&gt;").replace(/"/g,"&quot;");

function boxStyle(n) {
  const fill = n.m ? "#F4ECF7" : "#FFFFFF";
  const sw   = n.m ? 2 : 1;
  return `shape=label;html=1;whiteSpace=wrap;rounded=1;arcSize=12;`
       + `fillColor=${fill};strokeColor=${n.col};strokeWidth=${sw};`
       + `image=${iconDataUri(n.ic, n.col)};imageWidth=22;imageHeight=22;`
       + `imageAlign=left;imageVerticalAlign=middle;spacingLeft=34;align=left;`
       + `verticalAlign=middle;fontSize=11;fontColor=${C.ink};fontFamily=Calibri;`;
}

function laneStyle(i) {
  return `rounded=1;arcSize=6;html=1;fillColor=${i===0?"#F7F9FA":"#FBF7FD"};`
       + `strokeColor=none;verticalAlign=top;align=left;`
       + `spacingLeft=12;spacingTop=8;fontSize=12;fontStyle=1;`
       + `fontColor=${lanes[i].c};fontFamily=Calibri;`;
}

let x = [];
x.push(`<mxGraphModel dx="1400" dy="800" grid="0" page="1" pageWidth="${W}" pageHeight="${H}" adaptiveColors="auto">`);
x.push(`<root><mxCell id="0"/><mxCell id="1" parent="0"/>`);

lanes.forEach((L, i) => {
  x.push(`<mxCell id="lane${i}" value="${esc(L.t.toUpperCase())}" style="${laneStyle(i)}" vertex="1" parent="1">`
       + `<mxGeometry x="${LANE_X}" y="${LANE_Y[i]}" width="${LANE_W}" height="${LANE_H}" as="geometry"/></mxCell>`);
  x.push(`<mxCell id="lanesub${i}" value="${esc(L.s)}" style="text;html=1;align=left;verticalAlign=middle;`
       + `fontSize=10;fontColor=${C.sub};fontFamily=Calibri;" vertex="1" parent="1">`
       + `<mxGeometry x="${LANE_X+12}" y="${LANE_Y[i]+27}" width="480" height="16" as="geometry"/></mxCell>`);
  x.push(`<mxCell id="calls${i}" value="${esc(L.calls)}&#10;${esc(L.callsub)}" `
       + `style="rounded=1;arcSize=30;html=1;whiteSpace=wrap;fillColor=#FFFFFF;strokeColor=${L.c};`
       + `dashed=1;fontSize=10;fontStyle=1;fontColor=${L.c};fontFamily=Calibri;" vertex="1" parent="1">`
       + `<mxGeometry x="${COL(5)}" y="${nodeY(i)+6}" width="150" height="42" as="geometry"/></mxCell>`);
});

const htmlLabel = n =>
  `&lt;b&gt;${esc(n.t)}&lt;/b&gt;&lt;br/&gt;`
  + `&lt;font color=&quot;${C.sub}&quot; style=&quot;font-size:9px&quot;&gt;${esc(n.s)}&lt;/font&gt;`;

x.push(`<mxCell id="src" value="${htmlLabel(SRC)}" style="${boxStyle(SRC)}" vertex="1" parent="1">`
     + `<mxGeometry x="${SRC.x}" y="${SRC.y}" width="${SRC.w}" height="${SRC.h}" as="geometry"/></mxCell>`);

N.forEach(n => {
  x.push(`<mxCell id="${n.id}" value="${htmlLabel(n)}" `
       + `style="${boxStyle(n)}" vertex="1" parent="1">`
       + `<mxGeometry x="${X(n)}" y="${Y(n)}" width="${n.w}" height="${NH}" as="geometry"/></mxCell>`);
});

const edgeStyle = (c, dashed) =>
  `edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=${c};strokeWidth=1.4;`
  + (dashed ? `dashed=1;dashPattern=6 4;` : ``)
  + `endArrow=blockThin;endFill=1;endSize=5;fontSize=9;fontColor=${c};fontFamily=Calibri;`
  + `labelBackgroundColor=#FFFFFF;`;

FLOWS.forEach(([s, t], i) => {
  x.push(`<mxCell id="f${i}" style="${edgeStyle(byId[t].col, false)}" edge="1" parent="1" `
       + `source="${s}" target="${t}"><mxGeometry relative="1" as="geometry"/></mxCell>`);
});

RETRIES.forEach(([s, t, l], i) => {
  x.push(`<mxCell id="r${i}" value="FAIL — retry" style="${edgeStyle(C.fail, true)}exitX=0.5;exitY=1;`
       + `entryX=0.5;entryY=1;" edge="1" parent="1" source="${s}" target="${t}">`
       + `<mxGeometry relative="1" as="geometry"><Array as="points">`
       + `<mxPoint x="${X(byId[s])+byId[s].w/2}" y="${retryY(l)}"/>`
       + `<mxPoint x="${X(byId[t])+byId[t].w/2}" y="${retryY(l)}"/>`
       + `</Array></mxGeometry></mxCell>`);
});

x.push(`<mxCell id="foot" value="${esc(FOOT)}" style="text;html=1;align=left;`
     + `verticalAlign=middle;fontSize=9;fontColor=${C.sub};fontFamily=Calibri;" vertex="1" parent="1">`
     + `<mxGeometry x="${LANE_X+2}" y="${H-28}" width="1180" height="16" as="geometry"/></mxCell>`);

x.push(`</root></mxGraphModel>`);
const xml = x.join("\n");
fs.writeFileSync(path.join(OUT_DIR, BASE + ".drawio"), xml, "utf8");

/* ---------- browser URL -------------------------------------------------- */
const compressed = zlib.deflateRawSync(encodeURIComponent(xml)).toString("base64");
const payload = encodeURIComponent(JSON.stringify({ type:"xml", compressed:true, data:compressed }));
const url = "https://app.diagrams.net/?grid=0&pv=0&border=10&edit=_blank#create=" + payload;
fs.writeFileSync(path.join(OUT_DIR, BASE + ".url.txt"), url, "utf8");

console.log("drawio :", path.join(OUT_DIR, BASE + ".drawio"), fs.statSync(path.join(OUT_DIR, BASE+".drawio")).size, "bytes");
console.log("nodes  :", N.length, " icons:", Object.keys(ICONS).length, " canvas:", W+"x"+H);
console.log("urllen :", url.length);
