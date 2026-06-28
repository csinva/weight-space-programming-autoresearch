// Knowledge — per-proposer "best model" gallery. Full-width vertical list; each entry
// is a collapsible two-column row: source code (left) + a rich stats panel (right).
import {
  html, useState,
  Badge, fmtNum, fmtParams, fmtSeconds, fmtTime,
  modelStyle, effortStyle, agentColor,
} from "../lib/ui.js";

// ---- safe syntax tinting --------------------------------------------------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

const PY_KW = new Set((
  "False None True and as assert async await break class continue def del elif " +
  "else except finally for from global if import in is lambda nonlocal not or " +
  "pass raise return try while with yield self cls"
).split(" "));

// One-pass tokenizer over the WHOLE source (so triple-quoted strings spanning
// many lines are handled correctly). Every emitted chunk is HTML-escaped first.
const TOKEN_RE =
  /("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\b\d+\.?\d*(?:e[-+]?\d+)?\b|\b[A-Za-z_]\w*\b)/g;

function highlightPython(source) {
  const src = String(source);
  let out = "";
  let last = 0;
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(src)) !== null) {
    if (m.index > last) out += escapeHtml(src.slice(last, m.index));
    const tok = m[0];
    const safe = escapeHtml(tok);
    const c0 = tok[0];
    if (c0 === "#") {
      out += `<span class="sx-kn-c">${safe}</span>`;
    } else if (c0 === '"' || c0 === "'") {
      out += `<span class="sx-kn-s">${safe}</span>`;
    } else if (/^\d/.test(tok)) {
      out += `<span class="sx-kn-n">${safe}</span>`;
    } else if (PY_KW.has(tok)) {
      out += `<span class="sx-kn-k">${safe}</span>`;
    } else {
      out += safe;
    }
    last = m.index + tok.length;
  }
  if (last < src.length) out += escapeHtml(src.slice(last));
  return out;
}

// ---- small pieces ---------------------------------------------------------
function EffortMeter({ effort }) {
  const { rank, max } = effortStyle(effort);
  const pips = [];
  for (let i = 0; i < max; i++) {
    pips.push(html`<span class=${"sx-kn-pip" + (i < rank ? " on" : "")}></span>`);
  }
  return html`
    <span class="sx-kn-effort" title=${"effort: " + (effort || "unknown")}>
      <span class="sx-kn-pips">${pips}</span>
      <span class="sx-kn-effort-lab">${effort || "—"}</span>
    </span>`;
}

function CodeBlock({ source }) {
  const [copied, setCopied] = useState(false);
  const lineCount = String(source).split("\n").length;
  let gutter = "";
  for (let i = 1; i <= lineCount; i++) gutter += (i > 1 ? "\n" : "") + i;
  const code = highlightPython(source);

  function copy() {
    try {
      navigator.clipboard.writeText(source);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch (e) { /* clipboard unavailable */ }
  }

  return html`
    <div class="sx-kn-code">
      <button class=${"sx-kn-copy" + (copied ? " is-done" : "")} onClick=${copy}>
        ${copied ? "copied ✓" : "copy"}
      </button>
      <div class="sx-kn-code-scroll">
        <div class="sx-kn-code-inner">
          <pre class="sx-kn-gutter">${gutter}</pre>
          <pre class="sx-kn-pre"><code dangerouslySetInnerHTML=${{ __html: code }}></code></pre>
        </div>
      </div>
    </div>`;
}

function StatTile({ big, small, label }) {
  return html`
    <div class="sx-kn-tile">
      <div class="sx-kn-tile-big num">${big}</div>
      ${small ? html`<div class="sx-kn-tile-small num">${small}</div>` : null}
      <div class="sx-kn-tile-lab">${label}</div>
    </div>`;
}

function Entry({ item, defaultOpen }) {
  const [open, setOpen] = useState(!!defaultOpen);
  const ms = modelStyle(item.model);
  const acolor = agentColor(item.agent);

  const header = html`
    <button class="sx-kn-head" onClick=${() => setOpen(!open)} aria-expanded=${open}>
      <span class="sx-kn-caret">${open ? "▾" : "▸"}</span>
      <span class="sx-kn-agent" style=${`--ac:${acolor}`}>${item.agent}</span>
      <span class="sx-kn-title serif">${item.title || "untitled model"}</span>
      <span class="sx-kn-head-spacer"></span>
      <span class="sx-kn-mbadge" style=${`--mc:${ms.color}`}>
        ${ms.family}${item.effort ? html` · <span class="sx-kn-mbadge-eff">${item.effort}</span>` : null}
      </span>
      <span class="sx-kn-file mono">models/${item.model_id}.py</span>
      <span class="sx-kn-score-chip num">${fmtNum(item.test_corr, 4)}</span>
    </button>`;

  if (!open) return html`<div class="sx-kn-entry">${header}</div>`;

  return html`
    <div class="sx-kn-entry is-open">
      ${header}
      <div class="sx-kn-body">
        <div class="sx-kn-left">
          <${CodeBlock} source=${item.source || ""} />
        </div>
        <div class="sx-kn-right">
          <div class="sx-kn-scores">
            <div class="sx-kn-score-main">
              <div class="sx-kn-score-val num">${fmtNum(item.test_corr, 5)}</div>
              <div class="sx-kn-score-lab">test corr</div>
            </div>
            ${item.roi_corr !== null && item.roi_corr !== undefined ? html`
              <div class="sx-kn-score-sub">
                <div class="sx-kn-score-sub-val num">${fmtNum(item.roi_corr, 5)}</div>
                <div class="sx-kn-score-lab">roi corr</div>
              </div>` : null}
          </div>

          <div class="sx-kn-tiles">
            <${StatTile} big=${fmtParams(item.n_params)}
              small=${item.n_params ? Number(item.n_params).toLocaleString() : "—"}
              label="parameters" />
            <${StatTile} big=${fmtSeconds(item.seconds)} label="runtime" />
          </div>

          ${item.insight ? html`
            <div class="sx-kn-callout">
              <div class="sx-kn-callout-lab">Insight</div>
              <div class="sx-kn-callout-txt">${item.insight}</div>
            </div>` : null}

          ${item.description ? html`
            <div class="sx-kn-notes">
              <div class="sx-kn-notes-lab">Experiment notes</div>
              <div class="sx-kn-notes-txt">${item.description}</div>
            </div>` : null}

          <div class="sx-kn-author">
            <div class="sx-kn-author-row">
              <span class="sx-kn-author-lab">Authored by</span>
              <span class="sx-kn-mbadge" style=${`--mc:${ms.color}`}>${item.model || "model"}</span>
            </div>
            <div class="sx-kn-author-row">
              <${EffortMeter} effort=${item.effort} />
              <span class="sx-kn-time mono">${fmtTime(item.created_at)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>`;
}

export default function Knowledge({ items }) {
  const list = Array.isArray(items) ? items : [];
  if (!list.length) {
    return html`<div class="sx-kn"><div class="empty">No scored models yet — the knowledge gallery fills in as proposers land their first scored model.</div></div>`;
  }
  return html`
    <div class="sx-kn">
      <div class="sx-kn-intro">
        <h2 class="sx-kn-h2 serif">Knowledge</h2>
        <p class="sx-kn-sub">Each proposer's best scored model — its idea, code, and what it learned. Ranked by test correlation.</p>
      </div>
      <div class="sx-kn-list">
        ${list.map((it, i) => html`<${Entry} key=${it.model_id || i} item=${it} defaultOpen=${i === 0} />`)}
      </div>
    </div>`;
}
