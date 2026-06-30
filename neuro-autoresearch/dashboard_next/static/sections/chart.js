// Chart card — the score/best evolution chart with a "Replay evolution" animation.
// Extracted & upgraded from overview_main.js. Self-contained SVG (no chart library).
import {
  html, useState, useEffect, useRef, useMemo,
  agentColor, fmtNum, fmtTime, StatusBadge,
} from "../lib/ui.js";

// ---- constants & local helpers -----------------------------------------
const VIEW_N = { all: Infinity, "50": 50, "20": 20 };
const REPLAY_MS = 12000;       // target full-replay duration, regardless of N

function Segmented({ value, options, onChange }) {
  return html`<div class="sx-ch-seg">
    ${options.map(
      (o) => html`<button
        class=${"sx-ch-seg-btn" + (value === o.value ? " is-on" : "")}
        onClick=${() => onChange(o.value)}>${o.label}</button>`
    )}
  </div>`;
}

// evenly-spaced tick values across [lo, hi]
function ticksFor(lo, hi, count = 5) {
  if (!(hi > lo)) hi = lo + 1;
  const out = [];
  for (let i = 0; i < count; i++) out.push(lo + ((hi - lo) * i) / (count - 1));
  return out;
}

// ---- inline icons -------------------------------------------------------
const IcoPlay = () => html`<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
  <path d="M4 2.8 12.5 8 4 13.2Z" fill="currentColor" />
</svg>`;
const IcoPause = () => html`<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
  <rect x="3.5" y="2.8" width="3" height="10.4" rx="1" fill="currentColor" />
  <rect x="9.5" y="2.8" width="3" height="10.4" rx="1" fill="currentColor" />
</svg>`;
const IcoExpand = () => html`<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"
    fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
  <path d="M9.5 2.5H13.5V6.5M6.5 13.5H2.5V9.5M13.5 2.5 9 7M2.5 13.5 7 9" />
</svg>`;

// ---- hover/pin detail card ---------------------------------------------
function PointTip({ mk, pinned, W, H }) {
  const d = mk.d || {};
  const leftPct = (mk.x / W) * 100;
  const below = mk.y / H < 0.34;            // near the top edge -> drop below the point
  const style =
    `left:${leftPct}%; top:${(mk.y / H) * 100}%;` +
    `transform:translate(-50%, ${below ? "16px" : "calc(-100% - 14px)"});`;
  return html`<div class=${"sx-ch-tip" + (below ? " is-below" : "") + (pinned ? " is-pinned" : "")}
      style=${style}>
    <div class="sx-ch-tip-head">
      <span class="sx-ch-tip-dot" style=${`background:${mk.color}`}></span>
      <span class="sx-ch-tip-agent mono">${d.agent}</span>
      <span class="sx-ch-tip-idx mono">#${d.index}</span>
      <${StatusBadge} status=${d.status} />
      ${pinned ? html`<span class="sx-ch-tip-pin mono">pinned</span>` : null}
    </div>
    <div class="sx-ch-tip-score num">${fmtNum(d.score, 4)}
      <span class="sx-ch-tip-best mono">best ${fmtNum(d.best, 4)}</span>
    </div>
    ${d.title ? html`<div class="sx-ch-tip-title">${d.title}</div>` : null}
    <div class="sx-ch-tip-time mono">${fmtTime(d.time)}</div>
  </div>`;
}

// ---- main ---------------------------------------------------------------
export default function Chart({ attempts }) {
  const [scale, setScale] = useState("linear");
  const [xaxis, setXaxis] = useState("index");
  const [view, setView] = useState("all");
  const [expanded, setExpanded] = useState(false);

  // replay state: `replaying` drives the loop; `cursor` = how many attempts
  // (oldest->k) are revealed. cursor survives a pause so play resumes.
  const [replaying, setReplaying] = useState(false);
  const [cursor, setCursor] = useState(null);
  const rafRef = useRef(0);

  const ordered = useMemo(
    () => (attempts || []).slice().sort((a, b) => a.index - b.index),
    [attempts]
  );
  const scored = useMemo(() => ordered.filter((a) => a.score != null), [ordered]);
  const N = ordered.length;

  // ---- the replay animation loop ----
  useEffect(() => {
    if (!replaying) return;
    if (N < 2) { setReplaying(false); return; }
    const startK = cursor != null && cursor < N ? cursor : 0;
    const t0 = performance.now();
    const tick = (now) => {
      const k = startK + Math.ceil(((now - t0) / REPLAY_MS) * N);
      if (k >= N) {
        setCursor(null);          // restore the user's VIEW window
        setReplaying(false);
        return;
      }
      setCursor(Math.max(1, k));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line
  }, [replaying]);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  const toggleReplay = () => {
    if (replaying) { setReplaying(false); return; }      // pause -> VIEW restored
    if (cursor != null && cursor >= N) setCursor(null);  // finished -> restart
    setReplaying(true);
  };

  // during replay we OVERRIDE the VIEW window with attempts[1..cursor]
  const overriding = replaying && cursor != null;

  // window of attempts (scored + unscored) that defines range + present agents
  const windowAttempts = useMemo(() => {
    if (overriding) return ordered.slice(0, cursor);
    const n = VIEW_N[view];
    if (n === Infinity) return scored;
    return scored.slice(Math.max(0, scored.length - n));
  }, [ordered, scored, view, overriding, cursor]);

  const pts = useMemo(() => windowAttempts.filter((a) => a.score != null), [windowAttempts]);

  const agentsPresent = useMemo(
    () => Array.from(new Set(windowAttempts.map((a) => a.agent))).sort(),
    [windowAttempts]
  );

  // agent filter: null = ALL on; otherwise a Set of explicitly-on agents
  const [onAgents, setOnAgents] = useState(null);
  const isOn = (id) => onAgents === null || onAgents.has(id);
  const allOn = onAgents === null;
  const toggleAgent = (id) => {
    const base = onAgents === null ? new Set(agentsPresent) : new Set(onAgents);
    if (base.has(id)) base.delete(id); else base.add(id);
    if (base.size === 0 || base.size === agentsPresent.length) setOnAgents(null);
    else setOnAgents(base);
  };

  const rangeLabel = windowAttempts.length
    ? `${windowAttempts[0].index}-${windowAttempts[windowAttempts.length - 1].index}`
    : "—";

  // ---- geometry ----
  const W = 720;
  const H = expanded ? 440 : 300;
  const padL = 44, padR = 16, padT = 16, padB = 28;

  const geom = useMemo(() => {
    if (!pts.length) return null;

    let run = -Infinity;
    const e = pts.map((p) => {
      run = Math.max(run, p.score);
      return { ...p, best: run };
    });

    const tx = (v) => (scale === "log" ? Math.log10(Math.max(v, 1e-4)) : v);
    const xv = (p) => (xaxis === "time" ? new Date(p.time).getTime() : p.index);

    const xs = e.map(xv);
    let xmin = Math.min(...xs), xmax = Math.max(...xs);
    if (xmin === xmax) { xmin -= 1; xmax += 1; }

    const yvals = e.flatMap((p) => [tx(p.score), tx(p.best)]);
    let ymin = Math.min(...yvals), ymax = Math.max(...yvals);
    const pad = (ymax - ymin) * 0.14 || Math.abs(ymax) * 0.14 || 0.05;
    ymin -= pad; ymax += pad;
    if (scale === "linear" && ymin < 0) ymin = 0;

    const px = (x) => padL + ((x - xmin) / (xmax - xmin)) * (W - padL - padR);
    const py = (yt) => padT + (1 - (yt - ymin) / (ymax - ymin)) * (H - padT - padB);

    const xpix = e.map((p) => px(xv(p)));
    const scoreY = e.map((p) => py(tx(p.score)));
    const bestY = e.map((p) => py(tx(p.best)));

    let bestPath = `M ${xpix[0].toFixed(1)} ${bestY[0].toFixed(1)}`;
    for (let i = 1; i < e.length; i++) {
      bestPath += ` L ${xpix[i].toFixed(1)} ${bestY[i - 1].toFixed(1)}`;
      bestPath += ` L ${xpix[i].toFixed(1)} ${bestY[i].toFixed(1)}`;
    }

    const yticks = ticksFor(ymin, ymax, 5).map((yt) => ({
      y: py(yt),
      label: fmtNum(scale === "log" ? Math.pow(10, yt) : yt, scale === "log" ? 4 : 3),
    }));

    const step = Math.max(1, Math.ceil(e.length / 9));
    const xticks = [];
    for (let i = 0; i < e.length; i += step) {
      const p = e[i];
      xticks.push({
        x: xpix[i],
        label: xaxis === "time"
          ? new Date(p.time).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })
          : "#" + p.index,
      });
    }

    // Every scored attempt is a dot, coloured by its proposer (like the original).
    // Improved-best points are drawn last (on top) with a ring so they read clearly.
    const markers = e.map((p, i) => ({
      x: xpix[i], y: scoreY[i],
      improved: p.status === "improved",
      on: isOn(p.agent),
      color: agentColor(p.agent),
      d: { index: p.index, agent: p.agent, title: p.title,
           score: p.score, best: p.best, time: p.time, status: p.status },
    }));
    markers.sort((a, b) => (a.improved === b.improved ? 0 : a.improved ? 1 : -1));

    return { bestPath, yticks, xticks, markers };
    // eslint-disable-next-line
  }, [pts, scale, xaxis, expanded, onAgents, H]);

  const canReplay = N >= 2;

  // ---- point inspection: nearest-marker hover + click-to-pin ----
  const svgRef = useRef(null);
  const [hoverIdx, setHoverIdx] = useState(-1);
  const [pinnedIdx, setPinnedIdx] = useState(-1);

  // any change to the plotted set invalidates a held index
  useEffect(() => { setHoverIdx(-1); setPinnedIdx(-1); }, [pts, scale, xaxis, onAgents]);

  const nearestMarker = (ev) => {
    const svg = svgRef.current, mk = geom && geom.markers;
    if (!svg || !mk) return -1;
    const rect = svg.getBoundingClientRect();
    if (!rect.width || !rect.height) return -1;
    const sx = rect.width / W, sy = rect.height / H;
    const vx = (ev.clientX - rect.left) / sx;
    const vy = (ev.clientY - rect.top) / sy;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < mk.length; i++) {
      if (!mk[i].on) continue;
      const dx = (mk[i].x - vx) * sx, dy = (mk[i].y - vy) * sy;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    return bestD <= 20 * 20 ? best : -1;   // within ~20px
  };

  const onMove = (ev) => setHoverIdx(nearestMarker(ev));
  const onLeave = () => setHoverIdx(-1);
  const onClick = (ev) => {
    const i = nearestMarker(ev);
    setPinnedIdx((cur) => (i >= 0 && i === cur ? -1 : i));
  };

  const activeIdx = pinnedIdx >= 0 ? pinnedIdx : hoverIdx;
  const active = geom && activeIdx >= 0 ? geom.markers[activeIdx] : null;

  return html`<div class=${"card sx-ch" + (expanded ? " is-expanded" : "")}>
    <!-- row 1: scale / x-axis / view  +  replay / expand -->
    <div class="sx-ch-controls">
      <div class="sx-ch-ctrl">
        <span class="label">Scale</span>
        <${Segmented} value=${scale} onChange=${setScale}
          options=${[{ value: "linear", label: "LINEAR" }, { value: "log", label: "LOG" }]} />
      </div>
      <div class="sx-ch-ctrl">
        <span class="label">X-Axis</span>
        <${Segmented} value=${xaxis} onChange=${setXaxis}
          options=${[{ value: "index", label: "INDEX" }, { value: "time", label: "TIME" }]} />
      </div>
      <div class="sx-ch-ctrl">
        <span class="label">View</span>
        <${Segmented} value=${view} onChange=${setView}
          options=${[
            { value: "all", label: "ALL" },
            { value: "50", label: "LAST 50" },
            { value: "20", label: "LAST 20" },
          ]} />
      </div>
      <div class="sx-ch-icons">
        <button
          class=${"sx-ch-ico" + (replaying ? " is-on" : "")}
          disabled=${!canReplay}
          title=${replaying ? "Pause replay" : "Replay evolution"}
          aria-label=${replaying ? "Pause replay" : "Replay evolution"}
          onClick=${toggleReplay}>
          ${replaying ? html`<${IcoPause} />` : html`<${IcoPlay} />`}
        </button>
        <button
          class=${"sx-ch-ico" + (expanded ? " is-on" : "")}
          title="Expand chart"
          aria-label="Expand chart"
          onClick=${() => setExpanded((v) => !v)}>
          <${IcoExpand} />
        </button>
      </div>
    </div>

    <!-- row 2: agent filter + range pill -->
    <div class="sx-ch-agents">
      <span class="label">Agents</span>
      <button class=${"sx-ch-chip" + (allOn ? " is-on" : "")}
        onClick=${() => setOnAgents(null)}>ALL</button>
      ${agentsPresent.map(
        (id) => html`<button
          class=${"sx-ch-adot" + (isOn(id) ? " is-on" : "")}
          title=${id}
          onClick=${() => toggleAgent(id)}>
          <span class="sx-ch-adot-i" style=${`background:${agentColor(id)}`}></span>
          <span class="sx-ch-adot-n">${id.replace(/^p0?/, "")}</span>
        </button>`
      )}
      <span class="sx-ch-rangewrap">
        <span class="label">Range</span>
        <span class=${"sx-ch-range" + (overriding ? " is-live" : "")}>${rangeLabel}</span>
      </span>
    </div>

    <div class="sx-ch-legend">
      <span><span class="sx-ch-lg-dot"></span>Score</span>
      <span><span class="sx-ch-lg sx-ch-lg-best"></span>Best</span>
    </div>

    ${geom
      ? html`<div class="sx-ch-plot">
          <div class="sx-ch-svgwrap">
            <svg ref=${svgRef} viewBox=${`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img"
              aria-label="Score evolution chart"
              onMouseMove=${onMove} onMouseLeave=${onLeave} onClick=${onClick}>
              ${geom.yticks.map(
                (t) => html`<g>
                  <line class="sx-ch-grid" x1=${padL} x2=${W - padR} y1=${t.y} y2=${t.y} />
                  <text class="sx-ch-ytxt" x=${padL - 7} y=${t.y + 3} text-anchor="end">${t.label}</text>
                </g>`
              )}
              ${geom.xticks.map(
                (t) => html`<text class="sx-ch-xtxt" x=${t.x} y=${H - 9} text-anchor="middle">${t.label}</text>`
              )}
              <path class="sx-ch-best" d=${geom.bestPath} />
              ${geom.markers.map(
                (mk) => html`<circle
                  class=${"sx-ch-mk" + (mk.on ? "" : " is-off") + (mk.improved ? " is-imp" : "")}
                  cx=${mk.x} cy=${mk.y} r=${mk.improved ? 3.1 : 2.2}
                  style=${`fill:${mk.color}`} />`
              )}
              ${active ? html`
                <line class="sx-ch-cross" x1=${active.x} x2=${active.x} y1=${padT} y2=${H - padB} />
                <circle class="sx-ch-mk-hi" cx=${active.x} cy=${active.y} r="6"
                  style=${`stroke:${active.color}`} />` : null}
            </svg>
            ${active ? html`<${PointTip} mk=${active} pinned=${pinnedIdx >= 0} W=${W} H=${H} />` : null}
          </div>
          <div class="sx-ch-foot">
            <span class="label">${overriding ? "Replaying" : "Showing"}</span>
            <span class="num">${pts.length}</span>
            <span class="muted">scored</span>
            ${overriding
              ? html`<span class="muted">· ${cursor} / ${N} attempts</span>`
              : active ? null
              : html`<span class="sx-ch-hint muted">· hover a point to inspect · click to pin</span>`}
          </div>
        </div>`
      : html`<div class="empty">No scored attempts in view yet.</div>`}
  </div>`;
}
