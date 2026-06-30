// Agents panel — a vertical stack of proposer cards (p00..p09). Styled to match the
// original dashboard's restrained, monochrome roster: the proposer id, one muted mono
// meta line (model · effort · count · best), the agent's self-chosen direction, and a
// status badge — no colour-coded chips or meters.
import { html, Badge, fmtNum } from "../lib/ui.js";

// Status pill: ACTIVE is a solid (dark) pill, everything else an outline/soft pill —
// the same vocabulary the attempts table uses.
function StatusTag({ status }) {
  const s = (status || "idle").toLowerCase();
  if (s === "active") return html`<${Badge} kind="dark">active<//>`;
  if (s === "stopped") return html`<${Badge} kind="soft">stopped<//>`;
  return html`<${Badge} kind="outline">${s}<//>`;
}

function AgentCard({ a }) {
  const meta = [a.model || "—", a.effort].filter(Boolean).join(" · ");
  return html`<article class="sx-ag-card">
    <header class="sx-ag-top">
      <span class="sx-ag-id">${a.id}</span>
      <${StatusTag} status=${a.status} />
    </header>

    <div class="sx-ag-meta">
      ${meta ? html`<span class="sx-ag-model">${meta}</span>
        <span class="sx-ag-dotsep">·</span>` : null}
      <span><b>${a.models_count}</b> models</span>
      <span class="sx-ag-dotsep">·</span>
      <span>best <b class="num">${fmtNum(a.best_score, 4)}</b></span>
    </div>

    <p class="sx-ag-dir"><span class="sx-ag-arrow">→</span>${a.direction || "—"}</p>
    ${a.prev_direction ? html`<p class="sx-ag-prev">
      <span class="sx-ag-prevtag">prev</span>${a.prev_direction}
    </p>` : null}
  </article>`;
}

export default function AgentsPanel({ agents }) {
  const list = Array.isArray(agents) ? agents : [];
  if (!list.length) {
    return html`<div class="sx-ag">
      <div class="empty">No proposers yet.</div>
    </div>`;
  }
  return html`<div class="sx-ag">
    <div class="sx-ag-stack">
      ${list.map((a) => html`<${AgentCard} key=${a.id} a=${a} />`)}
    </div>
  </div>`;
}
