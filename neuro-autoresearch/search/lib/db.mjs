// search/lib/db.mjs — read the search DB (search/state/results.db) via Node's built-in node:sqlite.
// Writes are owned by Python (search/db.py); this module only SELECTs.
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';

const NUM = ['test_corr', 'corr_V1', 'corr_hV4', 'corr_early', 'corr_ventral', 'corr_FFA',
             'corr_PPA', 'corr_EBA', 'corr_VWFA', 'train_corr', 'gap'];
const round4 = (x) => Math.round(x * 1e4) / 1e4;

export function openDb(dbPath) { return new DatabaseSync(dbPath); }

// All rows WITHOUT code (light) — for ranking / sampling / digest. Returns [] for a fresh
// namespace whose db/table doesn't exist yet (cold start of a new dataset).
export function readDb(dbPath) {
  const db = openDb(dbPath);
  let rows = [];
  try {
    rows = db.prepare(`SELECT id, idea, dataset, roi, status, ${NUM.join(', ')} FROM models`).all();
  } catch { /* no models table yet */ }
  db.close();
  return rows;
}

// Full source for a set of ids -> Map(id -> code). Each row stores a `path` pointer to its
// model.py on disk (not inlined source); read the file to recover the source.
export function getCode(dbPath, ids) {
  if (!ids.length) return new Map();
  const db = openDb(dbPath);
  let rows = [];
  try {
    rows = db.prepare(`SELECT id, path FROM models WHERE id IN (${ids.map(() => '?').join(',')})`).all(...ids);
  } catch { /* no models table yet */ }
  db.close();
  const out = new Map();
  for (const r of rows) {
    if (!r.path) continue;
    try { out.set(r.id, readFileSync(r.path, 'utf8')); } catch { /* file moved/missing */ }
  }
  return out;
}

// ok rows, best reward first.
export function rank(rows) {
  return rows.filter((r) => r.status === 'ok' && Number.isFinite(r.test_corr))
             .sort((a, b) => b.test_corr - a.test_corr);
}

// Compact summary for result.json / notebook bootstrap.
export function digest(rows, { roi = 'ventral' } = {}) {
  const ok = rank(rows);
  const champion = ok[0] ? { id: ok[0].id, corr: round4(ok[0].test_corr) } : null;
  const bestByRegion = {};
  for (const c of NUM) {
    if (!c.startsWith('corr_')) continue;
    const vals = ok.map((r) => r[c]).filter(Number.isFinite);
    if (vals.length) bestByRegion[c.slice(5)] = round4(Math.max(...vals));
  }
  return { champion, bestByRegion, nOk: ok.length, nTotal: rows.length, roi };
}
