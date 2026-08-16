/* Hand-authored flat-vector mascot set for the LectureAssist figures.
 * Every asset is a plain SVG string so it can be embedded in .drawio as a
 * base64 data URI. No external assets, no network, no licensing questions.
 */

const NAVY = "#1F4E79";
const DARK = "#143A5C";
const SCRN = "#EAF2FB";
const SKIN = "#F2C9A0";

/* A robot mascot. `accent` colours the antenna bulb and chest light, so the
 * same body reads as a different agent without redrawing it. `badge` hangs a
 * small role glyph off the lower-right (bulb = planning, lens = retrieval,
 * q = question writing, doc = formatting, check = validation). */
function robot(accent = "#2E86C1", badge = "") {
  const badges = {
    bulb: `<g transform="translate(44,44)">
<circle cx="9" cy="9" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.4"/>
<path d="M9 3.6a3.6 3.6 0 0 0-2.1 6.5v1.5h4.2V10.1A3.6 3.6 0 0 0 9 3.6z" fill="#F4C542" stroke="${DARK}" stroke-width="1"/>
<path d="M7.4 13.2h3.2M7.8 14.9h2.4" stroke="${DARK}" stroke-width="1.1" stroke-linecap="round"/></g>`,
    lens: `<g transform="translate(44,44)">
<circle cx="9" cy="9" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.4"/>
<circle cx="8" cy="8" r="4" fill="none" stroke="${DARK}" stroke-width="1.6"/>
<path d="M11.2 11.2l3.1 3.1" stroke="${DARK}" stroke-width="1.8" stroke-linecap="round"/></g>`,
    q: `<g transform="translate(44,44)">
<circle cx="9" cy="9" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.4"/>
<path d="M6.6 6.6a2.5 2.5 0 1 1 3.1 3.4c-.6.4-.9.9-.9 1.6" fill="none" stroke="${DARK}" stroke-width="1.6" stroke-linecap="round"/>
<circle cx="8.8" cy="14" r="1.1" fill="${DARK}"/></g>`,
    doc: `<g transform="translate(44,44)">
<circle cx="9" cy="9" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.4"/>
<path d="M6 4.6h4.2L12.4 7v6.8a.6.6 0 0 1-.6.6H6a.6.6 0 0 1-.6-.6V5.2a.6.6 0 0 1 .6-.6z" fill="none" stroke="${DARK}" stroke-width="1.3"/>
<path d="M7.2 9.4h3.6M7.2 11.4h3.6" stroke="${DARK}" stroke-width="1.1" stroke-linecap="round"/></g>`,
    check: `<g transform="translate(44,44)">
<circle cx="9" cy="9" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.4"/>
<path d="M5.2 9.2l2.6 2.6 5-5.4" fill="none" stroke="#1E8449" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></g>`,
    merge: `<g transform="translate(44,44)">
<circle cx="9" cy="9" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.4"/>
<path d="M4 5h2.2c1 0 1.9.6 2.4 1.5l1 1.7c.5.9 1.4 1.5 2.4 1.5H14" fill="none" stroke="${DARK}" stroke-width="1.5" stroke-linecap="round"/>
<path d="M4 13h2.2c1 0 1.9-.6 2.4-1.5" fill="none" stroke="${DARK}" stroke-width="1.5" stroke-linecap="round"/>
<path d="M11.8 7.8L14 9.7l-2.2 1.9" fill="none" stroke="${DARK}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></g>`,
  };
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 68" width="64" height="68">
<path d="M32 7v8" stroke="${DARK}" stroke-width="2" stroke-linecap="round"/>
<circle cx="32" cy="5" r="3.6" fill="${accent}" stroke="${DARK}" stroke-width="1.4"/>
<rect x="5" y="26" width="6" height="12" rx="3" fill="${NAVY}" stroke="${DARK}" stroke-width="1.4"/>
<rect x="53" y="26" width="6" height="12" rx="3" fill="${NAVY}" stroke="${DARK}" stroke-width="1.4"/>
<rect x="11" y="14" width="42" height="30" rx="9" fill="${NAVY}" stroke="${DARK}" stroke-width="1.6"/>
<rect x="16" y="19" width="32" height="20" rx="6" fill="${SCRN}" stroke="${DARK}" stroke-width="1.3"/>
<circle cx="25.5" cy="28" r="3.4" fill="${NAVY}"/><circle cx="38.5" cy="28" r="3.4" fill="${NAVY}"/>
<circle cx="26.7" cy="26.8" r="1.1" fill="#FFFFFF"/><circle cx="39.7" cy="26.8" r="1.1" fill="#FFFFFF"/>
<path d="M27 33.5q5 3.4 10 0" fill="none" stroke="${NAVY}" stroke-width="1.6" stroke-linecap="round"/>
<rect x="18" y="47" width="28" height="17" rx="6" fill="${NAVY}" stroke="${DARK}" stroke-width="1.6"/>
<rect x="26" y="51" width="12" height="9" rx="3" fill="${accent}" stroke="${DARK}" stroke-width="1.2"/>
${badges[badge] || ""}</svg>`;
}

/* Two robots fused — used for the merged agents in the comparison figure. */
function robotPair(accent = "#7D3C98") {
  const one = (x, s) => `<g transform="translate(${x},6) scale(${s})">
<path d="M32 7v8" stroke="${DARK}" stroke-width="2.4" stroke-linecap="round"/>
<circle cx="32" cy="5" r="3.6" fill="${accent}" stroke="${DARK}" stroke-width="1.6"/>
<rect x="11" y="14" width="42" height="30" rx="9" fill="${NAVY}" stroke="${DARK}" stroke-width="1.8"/>
<rect x="16" y="19" width="32" height="20" rx="6" fill="${SCRN}" stroke="${DARK}" stroke-width="1.4"/>
<circle cx="25.5" cy="28" r="3.4" fill="${NAVY}"/><circle cx="38.5" cy="28" r="3.4" fill="${NAVY}"/>
<circle cx="26.7" cy="26.8" r="1.1" fill="#FFFFFF"/><circle cx="39.7" cy="26.8" r="1.1" fill="#FFFFFF"/>
<path d="M27 33.5q5 3.4 10 0" fill="none" stroke="${NAVY}" stroke-width="1.7" stroke-linecap="round"/>
<rect x="18" y="47" width="28" height="15" rx="6" fill="${NAVY}" stroke="${DARK}" stroke-width="1.8"/>
<rect x="26" y="50" width="12" height="8" rx="3" fill="${accent}" stroke="${DARK}" stroke-width="1.3"/></g>`;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 104 68" width="104" height="68">
${one(0, 0.82)}${one(40, 0.82)}
<circle cx="52" cy="60" r="10" fill="#FFFFFF" stroke="${accent}" stroke-width="2"/>
<path d="M47 56.5h4c1 0 1.6.5 2 1.2M47 63.5h4c1 0 1.6-.5 2-1.2M53 60h4" fill="none" stroke="${accent}" stroke-width="1.7" stroke-linecap="round"/>
<path d="M55.4 57.7L57.8 60l-2.4 2.3" fill="none" stroke="${accent}" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
}

/* The learner. Graduation cap so the role reads without a caption. */
function student() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
<circle cx="32" cy="32" r="30" fill="#DCE9F7" stroke="${NAVY}" stroke-width="2"/>
<path d="M12 58a20 16 0 0 1 40 0z" fill="${NAVY}"/>
<path d="M22 52c0-4 4.4-7 10-7s10 3 10 7" fill="#2E86C1"/>
<circle cx="32" cy="31" r="11" fill="${SKIN}" stroke="${DARK}" stroke-width="1.2"/>
<path d="M21 26c2-6 7-9 11-9s9 3 11 9z" fill="${DARK}"/>
<path d="M32 10L50 17 32 24 14 17z" fill="${NAVY}" stroke="${DARK}" stroke-width="1.2" stroke-linejoin="round"/>
<path d="M46 19.2V26" stroke="${DARK}" stroke-width="1.4" stroke-linecap="round"/>
<circle cx="46" cy="27.4" r="1.8" fill="#F4C542" stroke="${DARK}" stroke-width="1"/></svg>`;
}

function gear(color = "#7D3C98") {
  const teeth = Array.from({ length: 8 }, (_, i) =>
    `<rect x="29" y="1.5" width="6" height="11" rx="2" fill="${color}" stroke="${DARK}" stroke-width="1.1" transform="rotate(${i * 45} 32 32)"/>`
  ).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
${teeth}<circle cx="32" cy="32" r="21" fill="${color}" stroke="${DARK}" stroke-width="1.6"/>
<circle cx="32" cy="32" r="9" fill="#FFFFFF" stroke="${DARK}" stroke-width="1.6"/></svg>`;
}

function doc() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
<path d="M14 6h26l12 12v40a2 2 0 0 1-2 2H14a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z" fill="#FFFFFF" stroke="${NAVY}" stroke-width="2"/>
<path d="M40 6v12h12" fill="#DCE9F7" stroke="${NAVY}" stroke-width="2" stroke-linejoin="round"/>
<path d="M19 28h26M19 36h26M19 44h17" stroke="#2E86C1" stroke-width="2.6" stroke-linecap="round"/></svg>`;
}

function lens() {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">
<circle cx="27" cy="27" r="18" fill="#DCE9F7" stroke="${NAVY}" stroke-width="3"/>
<circle cx="27" cy="27" r="18" fill="none" stroke="#FFFFFF" stroke-width="1"/>
<path d="M40 40l16 16" stroke="${NAVY}" stroke-width="6" stroke-linecap="round"/>
<path d="M20 22a10 10 0 0 1 8-5" fill="none" stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round"/></svg>`;
}

function badge(kind) {
  const ok = kind === "check";
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="48" height="48">
<circle cx="24" cy="24" r="21" fill="${ok ? "#1E8449" : "#C0392B"}" stroke="#FFFFFF" stroke-width="3"/>
${ok
  ? `<path d="M14 24.6l6.4 6.4L34 16.8" fill="none" stroke="#FFFFFF" stroke-width="4.6" stroke-linecap="round" stroke-linejoin="round"/>`
  : `<path d="M16 16l16 16M32 16L16 32" fill="none" stroke="#FFFFFF" stroke-width="4.6" stroke-linecap="round"/>`}</svg>`;
}

const uri = svg => "data:image/svg+xml," + Buffer.from(svg, "utf8").toString("base64");

module.exports = { robot, robotPair, student, gear, doc, lens, badge, uri,
                   NAVY, DARK, SCRN };
