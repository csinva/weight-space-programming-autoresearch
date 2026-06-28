// Agents panel — the headline of the redesign. A vertical stack of proposer cards
// (p00..p09), each making the MODEL + a SEPARATE EFFORT meter the visual focal point.
import { html, Badge, modelStyle, effortStyle, agentColor, fmtNum } from "../lib/ui.js";

// Pretty status: ACTIVE / IDLE / STOPPED with a colored dot.
function StatusTag({ status }) {
  const s = (status || "").toLowerCase();
  const cls = s === "active" ? "dot-active" : s === "stopped" ? "dot-stopped" : "dot-idle";
  return html`<span class="sx-ag-status">
    <span class="dot ${cls}"></span>${s || "idle"}
  </span>`;
}

// MODEL chip — mono name, color-keyed by the model family, with a leading dot.
function ModelChip({ model }) {
  const { color, family } = modelStyle(model);
  return html`<span class="sx-ag-model" style=${`--mc:${color}`} title=${family}>
    <span class="sx-ag-model-dot"></span>
    <span class="sx-ag-model-name">${model || "—"}</span>
  </span>`;
}

// EFFORT meter — a SEPARATE 5-segment level indicator + the effort word.
function EffortMeter({ effort }) {
  const { rank, max } = effortStyle(effort);
  const segs = [];
  for (let i = 0; i < max; i++) {
    segs.push(html`<span class="sx-ag-seg ${i < rank ? "is-on" : ""}" style=${`--i:${i}`}></span>`);
  }
  const word = effort && rank ? effort.toLowerCase() : "—";
  const lvl = rank <= 1 ? "lo" : rank <= 3 ? "mid" : "hi";
  return html`<span class="sx-ag-effort" data-lvl=${lvl} title=${`effort: ${word}`}>
    <span class="sx-ag-effort-bars">${segs}</span>
    <span class="sx-ag-effort-word">${word}</span>
  </span>`;
}

function AgentCard({ a }) {
  const accent = agentColor(a.id);
  return html`<article class="sx-ag-card" style=${`--accent:${accent}`}>
    <span class="sx-ag-bar"></span>
    <div class="sx-ag-cbody">
      <header class="sx-ag-top">
        <span class="sx-ag-id">${a.id}</span>
        <div class="sx-ag-meta">
          <${ModelChip} model=${a.model} />
          <span class="sx-ag-vsep"></span>
          <${EffortMeter} effort=${a.effort} />
          <${StatusTag} status=${a.status} />
        </div>
      </header>

      <div class="sx-ag-stats">
        <span class="sx-ag-stat"><b>${a.models_count}</b> models</span>
        <span class="sx-ag-dotsep">·</span>
        <span class="sx-ag-stat">best <b class="num">${fmtNum(a.best_score, 4)}</b></span>
      </div>

      <p class="sx-ag-dir"><span class="sx-ag-arrow">→</span>${a.direction || "—"}</p>
      ${a.prev_direction ? html`<p class="sx-ag-prev">
        <span class="sx-ag-prevtag">prev</span>${a.prev_direction}
      </p>` : null}
    </div>
  </article>`;
}

export default function AgentsPanel({ agents }) {
  const list = Array.isArray(agents) ? agents : [];
  if (!list.length) {
    return html`<div class="sx-ag">
      <div class="sx-ag-head"><span class="label">Agents</span></div>
      <div class="empty">No proposers yet.</div>
    </div>`;
  }
  return html`<div class="sx-ag">
    <div class="sx-ag-head">
      <span class="label">Agents</span>
      <span class="sx-ag-count num">${list.length}</span>
    </div>
    <div class="sx-ag-stack">
      ${list.map((a) => html`<${AgentCard} key=${a.id} a=${a} />`)}
    </div>
  </div>`;
}
