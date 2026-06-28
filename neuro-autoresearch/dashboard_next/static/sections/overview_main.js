// Overview — LEFT column. Three stacked blocks: a best-score headline, the score
// chart (delegated to ./chart.js), and a sortable attempts table. The chart is owned
// by a separate module and must NOT be implemented here.
import {
  html, useState, useMemo,
  StatusBadge, fmtNum, fmtTime, agentColor,
} from "../lib/ui.js";
import Chart from "./chart.js";

// How many rows to show collapsed, and the hard cap when expanded (perf with ~2k rows).
const PREVIEW_ROWS = 8;
const MAX_ROWS = 300;

// ---- counts pill (mono "N scored / M total") ----------------------------
function Counts({ scored, total }) {
  return html`<span class="sx-ov-counts num">
    ${scored} <span class="muted">scored</span> / ${total} <span class="muted">total</span>
  </span>`;
}

// ---- 1) best score headline ---------------------------------------------
function BestScore({ best_score, best_title, scored, total }) {
  return html`<div class="card sx-ov-best">
    <div class="sx-ov-best-main">
      <div class="label">Best Score</div>
      <div class="sx-ov-best-num serif">${fmtNum(best_score, 5)}</div>
      <div class="sx-ov-best-sub muted">${best_title || "—"}</div>
    </div>
    <div class="sx-ov-best-counts num">
      <div><span class="sx-ov-cnum">${scored}</span> <span class="muted">scored</span></div>
      <div><span class="sx-ov-cnum">${total}</span> <span class="muted">total</span></div>
    </div>
  </div>`;
}

// ---- 3) attempts table ---------------------------------------------------
function AttemptsTable({ scored, total, attempts }) {
  const [dir, setDir] = useState("desc");
  const [expanded, setExpanded] = useState(false);

  const sorted = useMemo(() => {
    const arr = attempts.slice();
    arr.sort((a, b) => {
      const av = a.score == null ? -Infinity : a.score;
      const bv = b.score == null ? -Infinity : b.score;
      return dir === "desc" ? bv - av : av - bv;
    });
    return arr;
  }, [attempts, dir]);

  const capped = expanded && sorted.length > MAX_ROWS;
  const rows = expanded ? sorted.slice(0, MAX_ROWS) : sorted.slice(0, PREVIEW_ROWS);

  return html`<div class="card sx-ov-tablecard">
    <div class="sx-ov-cardhead">
      <span class="label">Attempts</span>
      <${Counts} scored=${scored} total=${total} />
    </div>

    ${sorted.length === 0
      ? html`<div class="empty">No attempts yet.</div>`
      : html`<table class="sx-ov-table">
          <thead>
            <tr>
              <th class="sx-ov-th-rank">#</th>
              <th class="sx-ov-th-score" onClick=${() => setDir(dir === "desc" ? "asc" : "desc")}>
                Score <span class="sx-ov-sort">${dir === "desc" ? "↓" : "↑"}</span>
              </th>
              <th>Agent</th>
              <th>Status</th>
              <th class="sx-ov-th-time">Time</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map(
              (a, i) => html`<tr key=${a.hash}>
                <td class="num sx-ov-rank">${i + 1}</td>
                <td class="num sx-ov-score">${a.score == null ? "—" : fmtNum(a.score, 5)}</td>
                <td class="mono sx-ov-agent">
                  <span class="sx-ov-adot-i" style=${`background:${agentColor(a.agent)}`}></span>${a.agent}
                </td>
                <td><${StatusBadge} status=${a.status} /></td>
                <td class="sx-ov-time">${fmtTime(a.time)}</td>
              </tr>`
            )}
          </tbody>
        </table>`}

    ${capped
      ? html`<div class="sx-ov-cap">showing first ${MAX_ROWS} of ${sorted.length}</div>`
      : null}

    ${sorted.length > PREVIEW_ROWS
      ? html`<button class="sx-ov-showall" onClick=${() => setExpanded(!expanded)}>
          ${expanded ? "SHOW LESS" : `SHOW ALL ${sorted.length} ATTEMPTS`}
        </button>`
      : null}
  </div>`;
}

// ---- main ---------------------------------------------------------------
export default function OverviewMain({ meta, attempts }) {
  const list = attempts || [];
  const m = meta || {};

  const scoredCount = m.scored ?? list.filter((a) => a.score != null).length;
  const total = m.total ?? list.length;

  return html`<div class="sx-ov">
    <${BestScore}
      best_score=${m.best_score}
      best_title=${m.best_title}
      scored=${scoredCount}
      total=${total} />

    <${Chart} attempts=${list} />

    <${AttemptsTable} scored=${scoredCount} total=${total} attempts=${list} />
  </div>`;
}
