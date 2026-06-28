// Tiny fetch + SSE helpers. All endpoints are read-only JSON (see CONTRACT.md).

async function getJSON(path) {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export const api = {
  meta:      () => getJSON("/api/meta"),
  attempts:  () => getJSON("/api/attempts"),
  agents:    () => getJSON("/api/agents"),
  knowledge: () => getJSON("/api/knowledge"),
  logsList:  () => getJSON("/api/logs"),
  log:       (id) => getJSON(`/api/logs/${id}`),
  switchRun: (task, run) =>
    fetch("/api/runs/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, run }),
    }).then((r) => r.json()),
};

// Subscribe to the server's SSE stream. `onEvent(type)` fires for any data-changing event
// (attempt:new / attempt:update / eval:update / note:update). Returns an unsubscribe fn.
export function subscribe(onEvent) {
  const es = new EventSource("/api/events");
  const fwd = (type) => () => onEvent(type);
  ["attempt:new", "attempt:update", "eval:update", "note:update"].forEach((t) =>
    es.addEventListener(t, fwd(t))
  );
  es.onerror = () => {}; // browser auto-reconnects
  return () => es.close();
}
