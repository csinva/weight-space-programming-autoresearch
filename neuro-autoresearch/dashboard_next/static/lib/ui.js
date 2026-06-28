// Shared UI primitives for every section. Re-exports Preact + htm from the vendored
// standalone bundle, plus small components, formatters, and the color maps used to key
// agents and model families. Sections import ONLY from here (and lib/api.js).

import {
  html, render, Component, createContext,
  useState, useEffect, useRef, useMemo, useCallback, useContext, useLayoutEffect,
} from "../vendor/htm-preact-standalone.mjs";

export {
  html, render, Component, createContext,
  useState, useEffect, useRef, useMemo, useCallback, useContext, useLayoutEffect,
};

// ---- formatters ----
export function fmtNum(n, digits = 5) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toFixed(digits);
}

export function fmtParams(n) {
  if (n === null || n === undefined) return "—";
  n = Number(n);
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

export function fmtSeconds(s) {
  if (s === null || s === undefined) return "—";
  return Number(s).toFixed(1) + "s";
}

export function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function relTime(epoch) {
  if (!epoch) return "";
  const s = Math.max(0, Date.now() / 1000 - epoch);
  if (s < 60) return Math.round(s) + "s ago";
  if (s < 3600) return Math.round(s / 60) + "m ago";
  return Math.round(s / 3600) + "h ago";
}

// ---- color keys ----
// Ten distinct, muted hues for proposers p00..p09 (chart legend + agent accents).
export const AGENT_COLORS = [
  "#1a1d1e", "#2563eb", "#0891b2", "#7c3aed", "#b45309",
  "#15803d", "#be185d", "#4338ca", "#9a3412", "#0f766e",
];
export function agentColor(id) {
  const i = parseInt(String(id).replace(/\D/g, ""), 10);
  return AGENT_COLORS[(Number.isNaN(i) ? 0 : i) % AGENT_COLORS.length];
}

// Model family -> {label, color}. Keyed by prefix so minor version variants group sensibly.
export function modelStyle(model) {
  const m = (model || "").toLowerCase();
  if (m.includes("opus-4.8")) return { color: "#b45309", family: "Claude Opus" };
  if (m.includes("opus-4.7")) return { color: "#c2740a", family: "Claude Opus" };
  if (m.includes("opus-4.6")) return { color: "#d08a1f", family: "Claude Opus" };
  if (m.includes("claude"))   return { color: "#b45309", family: "Claude" };
  if (m.includes("gpt"))      return { color: "#15803d", family: "GPT" };
  if (m.includes("gemini"))   return { color: "#7c3aed", family: "Gemini" };
  return { color: "#727879", family: "model" };
}

const EFFORT_RANK = { low: 1, medium: 2, high: 3, xhigh: 4, max: 5 };
export function effortStyle(effort) {
  return { rank: EFFORT_RANK[(effort || "").toLowerCase()] || 0, max: 5 };
}

// ---- shared components ----
export function Label({ children }) {
  return html`<div class="label">${children}</div>`;
}

export function Badge({ kind = "outline", children }) {
  return html`<span class="badge badge-${kind}">${children}</span>`;
}

export function StatusBadge({ status }) {
  const s = (status || "").toLowerCase();
  if (s === "improved") return html`<${Badge} kind="dark">improved<//>`;
  if (s === "crashed")  return html`<span class="badge badge-soft" style="color:var(--red)">crashed</span>`;
  if (s === "active")   return html`<${Badge} kind="outline">active<//>`;
  if (s === "idle")     return html`<${Badge} kind="outline">idle<//>`;
  return html`<${Badge} kind="soft">baseline<//>`;
}
