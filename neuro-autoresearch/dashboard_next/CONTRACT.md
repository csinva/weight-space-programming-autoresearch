# dashboard_next — API contract

The new dashboard is a self-contained Preact SPA (no build step) served by a Starlette
backend that reuses `dashboard._db` / `dashboard.events` / `dashboard.logs` and adds the
two new shapes the redesign needs (per-proposer **model+effort**, and the per-proposer
**best-model code+stats** gallery). It runs side-by-side with the old `dashboard/` on a
**different port (default 8090)**; the old one is untouched.

All endpoints are GET + read-only unless noted. Empty/no-run states return `[]` or zeros,
never 500.

## `GET /api/meta` — header + run selector
```json
{
  "task": "nsd",
  "run": "2026-06-26",
  "status": "active|idle|stopped",
  "alive": true,
  "total": 49,
  "scored": 41,
  "best_score": 0.36853,
  "best_index": 41,            // 1-based proposal order position of the best scored model
  "runs": {                    // for the run-selector dropdown
    "current": {"task": "nsd", "run": "2026-06-26"},
    "tasks": [ {"slug":"nsd","runs":[ {"run":"2026-06-26","status":"running|stopped","attempts":49,"is_latest":true} ]} ]
  }
}
```

## `GET /api/attempts` — every attempt, newest first (drives Overview chart + table)
```json
[ {
  "index": 49,                 // 1-based proposal order (chart x-axis = INDEX)
  "hash": "7pi2w",
  "agent": "p08",
  "title": "Gabor-opponent complex cells ...",
  "score": 0.36853,            // null if not yet scored
  "status": "improved|baseline|active|idle|crashed",
  "time": "2026-06-26T08:16:00-04:00"   // ISO; frontend formats "Jun 26, 08:16 AM"
} ]
```
Derived client-side: best-so-far line, scored/total counts, per-agent color filter.

## `GET /api/agents` — Overview right panel + the model/effort redesign
Ordered `p00..p09`.
```json
[ {
  "id": "p00",
  "model": "claude-opus-4.8",   // parsed from proposers/p00/proposer.log "(model/effort)" header (last session); roster round-robin fallback
  "effort": "medium",           // "" if unknown
  "status": "active|idle|stopped",
  "models_count": 5,
  "best_score": 0.1860,         // null if none scored
  "direction": "second-order bilinear / Gram texture ...",
  "prev_direction": "random projections / ...",
  "last_activity": 1750000000.0
} ]
```

## `GET /api/knowledge` — per-proposer BEST model: code (left) + stats (right)
One entry per proposer that has >=1 scored model, sorted by `test_corr` desc.
```json
[ {
  "agent": "p08",
  "model_id": "7pi2w",
  "title": "Gabor-opponent complex cells ...",   // = idea
  "insight": "Introduces translation-invariant ...",   // experiment_notes.insight
  "description": "Images are converted to luminance ...", // experiment_notes.description
  "n_params": 128460146,
  "seconds": 13.12,
  "test_corr": 0.36853,
  "roi_corr": null,
  "model": "claude-opus-4.8",   // which LLM authored it (from agents)
  "effort": "medium",
  "created_at": "2026-06-26T08:16:00-04:00",
  "source": "import torch\n..."  // full models/<id>.py text
} ]
```

## `GET /api/logs` — { "p00": [ {path,index,size_bytes,modified} ], ... }  (reused)
## `GET /api/logs/{agent_id}` — { agent_id, log_files:[], turns:[{index,entries:[{type,content,details}],timestamp}] }  (reused)
## `GET /api/events` — SSE stream (reused DBWatcher): events attempt:new / attempt:update / eval:update / note:update / heartbeat
## `POST /api/runs/switch` — body {task, run} -> {ok:true} ; switches the served run live

---
## Frontend section module interface (so sections are drop-in & parallelizable)
Each section lives in `static/sections/<name>.js` and default-exports a Preact component:
```js
import { html } from "../lib/ui.js";
export default function Overview({ data, actions }) { return html`...`; }
```
- `data` — the section's already-fetched payload(s) (shell fetches & passes in; see app.js SECTIONS map).
- `actions` — { selectRun(task,run), selectAgent(id), refresh() }.
- Shared primitives come from `../lib/ui.js`: `html`, `useState`, `useEffect`, `Card`, `Label`,
  `Badge`, `StatPill`, `fmtNum`, `fmtTime`, `MODEL_COLORS`, `AGENT_COLORS`.
- Each section may add `static/sections/<name>.css` (loaded by index.html) using the
  `.sx-<name>` class prefix to avoid collisions.
