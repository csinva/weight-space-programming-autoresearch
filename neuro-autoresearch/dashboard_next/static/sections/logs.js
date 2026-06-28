import {
  html, useState, useEffect,
  modelStyle, relTime, fmtNum,
} from "../lib/ui.js";
import { api } from "../lib/api.js";

// Escape raw text before injecting via innerHTML (tool bodies, prose).
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function AgentCard({ agent, selected, onSelect }) {
  const color = modelStyle(agent.model).color;
  const status = (agent.status || "").toLowerCase();
  return html`
    <button
      class="sx-lg-card ${selected ? "is-selected" : ""}"
      onClick=${() => onSelect(agent.id)}>
      <div class="sx-lg-card-top">
        <span class="sx-lg-pid serif">${agent.id}</span>
        <span class="sx-lg-badge" style=${`--m:${color}`}>${agent.model || "model"}</span>
        <span class="sx-lg-status">${status || "—"}</span>
      </div>
      <div class="sx-lg-card-bot">
        <span class="sx-lg-stats num">
          ${agent.models_count ?? 0} models · best ${fmtNum(agent.best_score, 4)}
        </span>
        <span class="sx-lg-time num">${relTime(agent.last_activity)}</span>
      </div>
    </button>`;
}

function Entry({ entry }) {
  const type = entry.type;
  if (type === "tool_call") {
    const body = entry.details && entry.details.body;
    return html`
      <div class="sx-lg-tool">
        <div class="sx-lg-chip mono">${entry.content}</div>
        ${body ? html`<div class="sx-lg-toolbody mono"
            dangerouslySetInnerHTML=${{ __html: escapeHtml(body) }}></div>` : null}
      </div>`;
  }
  if (type === "system") {
    return html`<div class="sx-lg-system mono">${entry.content}</div>`;
  }
  // text (and any fallback)
  return html`<p class="sx-lg-text">${entry.content}</p>`;
}

export default function Logs({ agents }) {
  const list = agents || [];
  const [selected, setSelected] = useState(null);
  const [log, setLog] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Default-select the first agent once the roster is available.
  useEffect(() => {
    if (selected == null && list.length) setSelected(list[0].id);
  }, [list, selected]);

  // Fetch the transcript whenever the selection changes.
  useEffect(() => {
    if (!selected) return;
    let alive = true;
    setLoading(true);
    setError(null);
    api.log(selected)
      .then((d) => { if (alive) { setLog(d); setLoading(false); } })
      .catch((e) => { if (alive) { setError(String(e)); setLog(null); setLoading(false); } });
    return () => { alive = false; };
  }, [selected]);

  const entries = [];
  if (log && Array.isArray(log.turns)) {
    for (const turn of log.turns) {
      for (const e of (turn.entries || [])) entries.push(e);
    }
  }

  return html`
    <div class="sx-lg">
      <aside class="sx-lg-left">
        <div class="label sx-lg-heading">Agents</div>
        <div class="sx-lg-cards">
          ${list.length
            ? list.map((a) => html`
                <${AgentCard} key=${a.id} agent=${a}
                  selected=${a.id === selected}
                  onSelect=${setSelected} />`)
            : html`<div class="empty">no agents</div>`}
        </div>
      </aside>
      <section class="sx-lg-right">
        ${!selected
          ? html`<div class="empty">select an agent to view its transcript</div>`
          : loading
          ? html`<div class="empty">loading transcript…</div>`
          : error
          ? html`<div class="banner-error">failed to load log: ${error}</div>`
          : entries.length
          ? html`<div class="sx-lg-feed">
              ${entries.map((e, i) => html`<${Entry} key=${i} entry=${e} />`)}
            </div>`
          : html`<div class="empty">no transcript entries</div>`}
      </section>
    </div>`;
}
