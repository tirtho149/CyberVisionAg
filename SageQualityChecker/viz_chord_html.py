"""Generate an interactive D3.js chord diagram from disease_label_correct_only.json.

Every VALID crop is an arc around the circle. Ribbons connect crop pairs that
share at least MIN_SHARED CORRECT disease labels; ribbon thickness scales with
shared count. Hovering a crop arc isolates its chords and surfaces a tooltip
listing the shared diseases.

Output: viz_chord.html (single self-contained file).
"""

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_JSON = SCRIPT_DIR / "disease_label_correct_only.json"
OUT_HTML   = SCRIPT_DIR / "viz_chord.html"

# Only draw a chord if two crops share at least this many CORRECT diseases.
MIN_SHARED = 1


def build_data(entries, min_shared=MIN_SHARED):
    # Order crops by # of CORRECT diseases (descending) — larger arcs at top.
    entries = sorted(entries, key=lambda e: -len(e.get("diseases", [])))
    crops = [e["crop"] for e in entries]
    sets  = [set(e.get("diseases", [])) for e in entries]

    # Sparse adjacency: only crop pairs with >= min_shared overlap
    edges = []
    for i in range(len(crops)):
        for j in range(i + 1, len(crops)):
            shared = sets[i] & sets[j]
            if len(shared) >= min_shared:
                edges.append({
                    "source": i,
                    "target": j,
                    "value": len(shared),
                    "shared": sorted(shared),
                })

    nodes = [
        {"name": crops[i], "diseases": sorted(sets[i]), "count": len(sets[i])}
        for i in range(len(crops))
    ]
    return nodes, edges


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Chord — Cross-crop Diseases</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  :root {
    --bg: #FFFDF7;
    --ink: #111111;
    --muted: #555555;
  }
  html, body {
    margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: "Georgia", "Times New Roman", serif;
  }
  header {
    padding: 18px 28px 6px;
  }
  header h1 {
    margin: 0; font-size: 22px; font-weight: bold; letter-spacing: 0.2px;
  }
  header p {
    margin: 4px 0 0; color: var(--muted); font-size: 13px;
  }
  #controls {
    padding: 8px 28px;
    display: flex; gap: 22px; align-items: center;
    flex-wrap: wrap;
    color: var(--muted); font-size: 13px;
  }
  #controls label { display: flex; align-items: center; gap: 8px; }
  #controls input[type=range] { width: 180px; }
  #controls #info { margin-left: auto; font-size: 12px; color: #777; }

  #chart { width: 100vw; height: calc(100vh - 100px); }
  svg { width: 100%; height: 100%; display: block; }

  .arc { cursor: pointer; }
  .arc-label { font-size: 7px; fill: #222; pointer-events: none;
               font-family: "Georgia", serif; }
  .arc-label.highlight { font-weight: bold; font-size: 9px; fill: #000; }
  .ribbon { fill-opacity: 0.55; mix-blend-mode: multiply; }
  .ribbon.faded { fill-opacity: 0.04; }
  .ribbon.active { fill-opacity: 0.8; }

  .tooltip {
    position: fixed; pointer-events: auto; background: white;
    border: 1px solid #ccc; padding: 12px 14px; border-radius: 8px;
    font-size: 12px; line-height: 1.45; max-width: 520px; max-height: 70vh;
    overflow-y: auto; z-index: 99;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12); font-family: Georgia, serif;
    opacity: 0; transition: opacity 0.1s;
  }
  .tooltip.pinned { box-shadow: 0 8px 28px rgba(0,0,0,0.2); border-color: #888; }
  .tooltip h4 { margin: 0 0 4px; font-size: 14px; }
  .tooltip .meta { color: #777; font-size: 11px; margin-bottom: 8px; }
  .tooltip .hint { color: #999; font-size: 10px; margin-top: 8px; font-style: italic; }
  .tooltip .section-head {
    font-weight: bold; font-size: 12px; margin: 8px 0 4px;
    color: #333; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .tooltip .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 12px; }
  .tooltip .cols div {
    font-size: 11px; padding: 1px 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .tooltip ul { margin: 4px 0 0 16px; padding: 0; }
  .tooltip li { margin: 0; font-size: 11px; }
  .tooltip .partner {
    display: flex; justify-content: space-between; padding: 1px 0;
    font-size: 11px;
  }
  .tooltip .partner b { color: #4D96FF; margin-left: 8px; }
  .tooltip .close {
    float: right; cursor: pointer; color: #aaa; font-size: 16px; line-height: 1;
    padding: 0 4px; margin: -4px -6px 0 0;
  }
  .tooltip .close:hover { color: #333; }
</style>
</head>
<body>
<header>
  <h1>Chord diagram — cross-crop disease labels</h1>
  <p id="subtitle"></p>
</header>
<div id="controls">
  <label>Min shared diseases per chord:
    <input type="range" id="minShared" min="1" max="12" step="1" value="__MIN_SHARED__" />
    <span id="minSharedVal">__MIN_SHARED__</span>
  </label>
  <label><input type="checkbox" id="showLabels" checked /> Show labels</label>
  <span id="info"></span>
</div>
<div id="chart"></div>
<div id="tooltip" class="tooltip"></div>

<script>
const NODES = __NODES_JSON__;
const EDGES = __EDGES_JSON__;
const N = NODES.length;

const container = document.getElementById('chart');
const tooltip = d3.select('#tooltip');
const subtitle = document.getElementById('subtitle');
const info = document.getElementById('info');

subtitle.textContent =
  `${N} crops · ${EDGES.length} chord candidates (min shared = 1) · ribbon thickness = # shared diseases`;

// Colour scale — categorical, repeated across 228 with perceptual spacing
function nodeColor(i) {
  // evenly spaced hues
  const hue = (i * 137.508) % 360;       // golden angle → good separation
  const sat = 65 + (i % 3) * 8;
  const lgt = 55 + (i % 2) * 6;
  return `hsl(${hue}, ${sat}%, ${lgt}%)`;
}

function layout(minShared) {
  // Build an N×N matrix from the filtered edges.
  const mat = Array.from({length: N}, () => new Float32Array(N));
  let kept = 0;
  for (const e of EDGES) {
    if (e.value < minShared) continue;
    mat[e.source][e.target] += e.value;
    mat[e.target][e.source] += e.value;
    kept++;
  }
  return { mat, kept };
}

let minShared = +document.getElementById('minShared').value;
let showLabels = true;

function draw() {
  container.innerHTML = '';
  const w = container.clientWidth;
  const h = container.clientHeight;
  const outerR = Math.min(w, h) / 2 - 60;
  const innerR = outerR - 14;

  const svg = d3.select(container).append('svg')
      .attr('viewBox', `${-w/2} ${-h/2} ${w} ${h}`);

  const { mat, kept } = layout(minShared);
  info.textContent = `Rendering ${kept.toLocaleString()} chords`;
  document.getElementById('minSharedVal').textContent = minShared;

  const chord = d3.chord().padAngle(0.003).sortSubgroups(d3.descending)(mat);
  const arc = d3.arc().innerRadius(innerR).outerRadius(outerR);
  const ribbon = d3.ribbon().radius(innerR - 1);

  // Ribbons
  const ribbons = svg.append('g').attr('class', 'ribbons')
    .selectAll('path').data(chord).enter()
    .append('path')
      .attr('class', 'ribbon')
      .attr('d', ribbon)
      .attr('fill', d => nodeColor(d.source.index))
      .on('mouseenter', ribbonHover)
      .on('mouseleave', ribbonLeave);

  // Arcs
  const arcs = svg.append('g').attr('class', 'arcs')
    .selectAll('path').data(chord.groups).enter()
    .append('path')
      .attr('class', 'arc')
      .attr('d', arc)
      .attr('fill', d => nodeColor(d.index))
      .attr('stroke', '#fff')
      .attr('stroke-width', 0.6)
      .on('mouseenter', arcHover)
      .on('mouseleave', arcLeave);

  // Labels
  const labels = svg.append('g').attr('class', 'labels')
    .selectAll('text').data(chord.groups).enter()
    .append('text')
      .attr('class', 'arc-label')
      .attr('dy', '0.35em')
      .attr('transform', d => {
        const angle = (d.startAngle + d.endAngle) / 2;
        const rot = (angle * 180 / Math.PI - 90);
        const x = outerR + 4;
        return `rotate(${rot}) translate(${x},0)${angle > Math.PI ? ' rotate(180)' : ''}`;
      })
      .attr('text-anchor', d => {
        const angle = (d.startAngle + d.endAngle) / 2;
        return angle > Math.PI ? 'end' : 'start';
      })
      .style('display', showLabels ? null : 'none')
      .text(d => NODES[d.index].name);

  // ── Interaction helpers ─────────────────────────────────────────────────
  let pinned = false;

  function arcContent(idx) {
    const n = NODES[idx];
    const partners = chord
      .filter(c => (c.source.index === idx || c.target.index === idx) && c.source.value > 0)
      .map(c => {
        const other = c.source.index === idx ? c.target.index : c.source.index;
        // Compute shared diseases explicitly between the two crops
        const setOther = new Set(NODES[other].diseases);
        const shared = n.diseases.filter(d => setOther.has(d));
        return { name: NODES[other].name, count: shared.length, shared };
      })
      .sort((a,b) => b.count - a.count);

    const diseaseCols = n.diseases
      .map(d => `<div>• ${escapeHtml(d)}</div>`).join('');

    const partnerRows = partners.length
      ? partners.map(p =>
          `<div class="partner"><span>${escapeHtml(p.name)}</span>` +
          `<b>${p.count} shared</b></div>`).join('')
      : '<div style="color:#888; font-size:11px; padding:2px 0;">No shared diseases with any other crop</div>';

    return `
      <span class="close" onclick="hideTooltipForce()">×</span>
      <h4>${escapeHtml(n.name)}</h4>
      <div class="meta">${n.count} diseases · ${partners.length} chord connections</div>
      <div class="section-head">All diseases (${n.diseases.length})</div>
      <div class="cols">${diseaseCols}</div>
      <div class="section-head">Connected crops (${partners.length})</div>
      ${partnerRows}
      <div class="hint">Click arc to pin · click × or outside to close</div>
    `;
  }

  function ribbonContent(c) {
    const a = NODES[c.source.index], b = NODES[c.target.index];
    const setB = new Set(b.diseases);
    const shared = a.diseases.filter(d => setB.has(d));
    return `
      <span class="close" onclick="hideTooltipForce()">×</span>
      <h4>${escapeHtml(a.name)} ↔ ${escapeHtml(b.name)}</h4>
      <div class="meta">${shared.length} shared disease${shared.length!==1?'s':''}</div>
      <ul>${shared.map(d => `<li>${escapeHtml(d)}</li>`).join('')}</ul>
    `;
  }

  function arcHover(evt, d) {
    if (pinned) return;
    const idx = d.index;
    highlightArc(idx);
    showTooltip(evt, arcContent(idx));
  }
  function arcLeave() {
    if (pinned) return;
    clearHighlight();
    hideTooltip();
  }
  function arcClick(evt, d) {
    const idx = d.index;
    if (pinned && tooltip.classed('pinned') && tooltip.attr('data-idx') == String(idx)) {
      unpin();
    } else {
      pinned = true;
      highlightArc(idx);
      tooltip.classed('pinned', true).attr('data-idx', idx);
      showTooltip(evt, arcContent(idx));
    }
    evt.stopPropagation();
  }

  function ribbonHover(evt, c) {
    if (pinned) return;
    ribbons.classed('faded', r => r !== c);
    ribbons.filter(r => r === c).classed('active', true);
    showTooltip(evt, ribbonContent(c));
  }
  function ribbonLeave() {
    if (pinned) return;
    ribbons.classed('faded', false).classed('active', false);
    hideTooltip();
  }

  function highlightArc(idx) {
    ribbons.classed('faded', r => r.source.index !== idx && r.target.index !== idx);
    ribbons.classed('active', r => r.source.index === idx || r.target.index === idx);
    labels.classed('highlight', l => l.index === idx);
  }
  function clearHighlight() {
    ribbons.classed('faded', false).classed('active', false);
    labels.classed('highlight', false);
  }
  function unpin() {
    pinned = false;
    tooltip.classed('pinned', false).attr('data-idx', null);
    clearHighlight();
    hideTooltip();
  }
  window.hideTooltipForce = unpin;

  arcs.on('click', arcClick);
  d3.select('svg').on('click', () => { if (pinned) unpin(); });

  function showTooltip(evt, html) {
    tooltip.html(html).style('opacity', 1);
    moveTooltip(evt);
  }
  function hideTooltip() {
    if (!pinned) tooltip.style('opacity', 0);
  }
  function moveTooltip(evt) {
    if (pinned) return;
    const pad = 14;
    const x = Math.min(evt.clientX + pad, window.innerWidth  - tooltip.node().offsetWidth  - 10);
    const y = Math.min(evt.clientY + pad, window.innerHeight - tooltip.node().offsetHeight - 10);
    tooltip.style('left', x + 'px').style('top', y + 'px');
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
  svg.on('mousemove', moveTooltip);
}

draw();

// Controls
document.getElementById('minShared').addEventListener('input', function() {
  minShared = +this.value;
  draw();
});
document.getElementById('showLabels').addEventListener('change', function() {
  showLabels = this.checked;
  draw();
});
window.addEventListener('resize', draw);
</script>
</body>
</html>
"""


def main():
    entries = json.load(INPUT_JSON.open())
    print(f"Read {len(entries)} VALID crops from {INPUT_JSON.name}")
    nodes, edges = build_data(entries, min_shared=MIN_SHARED)
    print(f"Computed {len(edges):,} crop-pair chords (min shared = {MIN_SHARED})")

    html = (HTML_TEMPLATE
            .replace("__NODES_JSON__", json.dumps(nodes, ensure_ascii=False))
            .replace("__EDGES_JSON__", json.dumps(edges, ensure_ascii=False))
            .replace("__MIN_SHARED__", str(MIN_SHARED)))

    OUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUT_HTML.stat().st_size // 1024
    print(f"[✓] HTML → {OUT_HTML} ({size_kb} KB)")


if __name__ == "__main__":
    main()
