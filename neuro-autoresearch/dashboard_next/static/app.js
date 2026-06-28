// dashboard_next shell: header (logo + run selector + nav + live status chip), data
// fetching, SSE-driven refresh, and page composition. Section components are drop-in
// modules (see CONTRACT.md) — the shell owns layout; sections own their content.

import {
  html, render, useState, useEffect, useCallback,
  Label, fmtNum,
} from "./lib/ui.js";
import { api, subscribe } from "./lib/api.js";

import OverviewMain from "./sections/overview_main.js";
import AgentsPanel from "./sections/agents.js";
import Knowledge from "./sections/knowledge.js";
import Logs from "./sections/logs.js";

const NAV = [
  { id: "overview", label: "Overview" },
  { id: "knowledge", label: "Knowledge" },
  { id: "logs", label: "Logs" },
];

function RunSelector({ meta, onSwitch }) {
  const [open, setOpen] = useState(false);
  const runs = meta?.runs;
  return html`
    <div class="runsel" onMouseLeave=${() => setOpen(false)}>
      <button class="runsel-btn" onClick=${() => setOpen((o) => !o)}>
        <span class="muted mono">${meta?.task || "—"}</span>
        <span class="runsel-run mono">${meta?.run || ""}</span>
        <span class="runsel-caret">▾</span>
      </button>
      ${open && runs && html`
        <div class="runsel-menu card">
          ${runs.tasks.map((t) => html`
            <div class="runsel-group" key=${t.slug}>
              <div class="label runsel-task">${t.slug}</div>
              ${t.runs.map((r) => html`
                <button class="runsel-item ${r.run === meta.run && t.slug === meta.task ? "is-cur" : ""}"
                        key=${r.run}
                        onClick=${() => { setOpen(false); onSwitch(t.slug, r.run); }}>
                  <span class="mono">${r.run}</span>
                  <span class="runsel-meta">
                    <span class="dot dot-${r.status === "running" ? "active" : "stopped"}"></span>
                    <span class="num faint">${r.attempts}</span>
                  </span>
                </button>`)}
            </div>`)}
        </div>`}
    </div>`;
}

function StatusChip({ meta }) {
  if (!meta) return null;
  const s = meta.status || "stopped";
  return html`
    <div class="chip">
      <span class="dot dot-${s}"></span>
      <span class="chip-status">${s}</span>
      <span class="chip-sep">·</span>
      <span class="num">${meta.total ?? 0}</span><span class="faint"> att</span>
      <span class="chip-sep">·</span>
      <span class="faint">best </span><span class="num">${meta.best_score != null ? fmtNum(meta.best_score, 4) : "—"}</span>
      ${meta.best_index ? html`<span class="chip-idx num faint">#${meta.best_index}</span>` : null}
    </div>`;
}

function Header({ meta, page, setPage, onSwitch }) {
  return html`
    <header class="topbar">
      <div class="topbar-left">
        <span class="brand serif">neuro·autoresearch</span>
        <span class="brand-slash">/</span>
        <${RunSelector} meta=${meta} onSwitch=${onSwitch} />
      </div>
      <nav class="nav">
        ${NAV.map((n) => html`
          <button key=${n.id} class="nav-item ${page === n.id ? "is-active" : ""}"
                  onClick=${() => setPage(n.id)}>${n.label}</button>`)}
      </nav>
      <div class="topbar-right"><${StatusChip} meta=${meta} /></div>
    </header>`;
}

function App() {
  const [page, setPage] = useState("overview");
  const [meta, setMeta] = useState(null);
  const [attempts, setAttempts] = useState([]);
  const [agents, setAgents] = useState([]);
  const [knowledge, setKnowledge] = useState([]);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [m, a, ag, kn] = await Promise.all([
        api.meta(), api.attempts(), api.agents(), api.knowledge(),
      ]);
      setMeta(m); setAttempts(a); setAgents(ag); setKnowledge(kn); setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // live updates: refetch (lightly debounced) whenever the DB changes
  useEffect(() => {
    let t = null;
    const unsub = subscribe(() => {
      clearTimeout(t);
      t = setTimeout(refresh, 400);
    });
    return () => { clearTimeout(t); unsub(); };
  }, [refresh]);

  const onSwitch = useCallback(async (task, run) => {
    await api.switchRun(task, run);
    refresh();
  }, [refresh]);

  return html`
    <${Header} meta=${meta} page=${page} setPage=${setPage} onSwitch=${onSwitch} />
    <main class="content">
      ${error && html`<div class="banner-error">backend error: ${error}</div>`}
      ${page === "overview" && html`
        <div class="page-cols">
          <section class="col-left"><${OverviewMain} meta=${meta} attempts=${attempts} /></section>
          <div class="col-divider"></div>
          <section class="col-right">
            <${Label}>Agents<//>
            <${AgentsPanel} agents=${agents} />
          </section>
        </div>`}
      ${page === "knowledge" && html`<${Knowledge} items=${knowledge} />`}
      ${page === "logs" && html`<${Logs} agents=${agents} />`}
    </main>`;
}

render(html`<${App} />`, document.getElementById("root"));
