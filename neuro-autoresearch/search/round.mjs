#!/usr/bin/env node
// search/round.mjs — ONE round of the open-ended search (plain code + spawned agents).
// READ -> SAMPLE -> SEED (parallel proposers) -> EVALUATE (evaluate.sh) -> SCRIBE -> result.json
//
//   node search/round.mjs --input search/state/round-input.json
//
// All the bulk (10 agents' outputs, model source, tracebacks) stays in this process and
// on disk; only the tiny result.json is meant for the conductor above it.
import { readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as DB from './lib/db.mjs';
import { buildSeedSets } from './lib/sample.mjs';
import { proposerPrompt, scribePrompt, NOTEBOOK_SKELETON } from './lib/prompts.mjs';

const REPO = dirname(dirname(fileURLToPath(import.meta.url)));
const P = (...a) => join(REPO, ...a);
const DBP = P('search', 'state', 'results.db');
const NOTEBOOK = P('search', 'state', 'notebook.md');
const STUDY = { nsd: 'NsdSubj01', nnn: 'NnnV1', tvsd: 'TvsdMonkeyF' };

const i = process.argv.indexOf('--input');
const cfg = i >= 0 ? JSON.parse(readFileSync(process.argv[i + 1], 'utf8')) : {};
const { round = 0, dataset = 'nsd', roi = 'ventral', imsize = 128,
        fleet = 10, topK = 5, randK = 5, directions = [] } = cfg;

const RUNTIME = `${process.env.COPILOT_ENV || '/home/t-ymehta/copilot-env'}/copilot/workflow/runtime/workflow.mjs`;
const { spawnAgent, mapPool } = await import(RUNTIME);

// 1. READ
const rowsBefore = DB.readDb(DBP);
const dig0 = DB.digest(rowsBefore, { roi });
const prevBest = dig0.champion ? dig0.champion.corr : -Infinity;
const notebook = existsSync(NOTEBOOK)
  ? readFileSync(NOTEBOOK, 'utf8')
  : NOTEBOOK_SKELETON.replace('<best model id + ventral test_corr, 1 line>',
      dig0.champion ? `${dig0.champion.id} -- ventral ${dig0.champion.corr}` : '(none yet)');

// 2. SAMPLE (+ attach source to the shared top set)
const seedSets = buildSeedSets(DB.rank(rowsBefore), { fleet, topK, randK, round });
const codeMap = DB.getCode(DBP, (seedSets[0]?.top || []).map((r) => r.id));
for (const s of seedSets) s.top = s.top.map((r) => ({ ...r, code: codeMap.get(r.id) || '' }));

// 3. SEED — parallel proposers; each writes search/state/queue/m_<id>.py
console.log(`[round ${round}] SEED ${fleet} proposers (topK=${topK} randK=${randK} dataset=${dataset}/${roi})`);
const seeded = await mapPool(
  seedSets,
  (s) => spawnAgent({ prompt: proposerPrompt({ seed: s, notebook, direction: directions[s.agent] || null }) }),
  { concurrency: Math.min(16, fleet) },
);
const nSeededOk = seeded.filter((r) => r && r.ok).length;
const queuedIds = readdirSync(P('search', 'state', 'queue')).filter((f) => f.endsWith('.py')).map((f) => f.slice(0, -3));

// 4. EVALUATE — sequential GPU; the existing Copilot auto-repair evaluator
console.log(`[round ${round}] EVALUATE ${queuedIds.length} queued (seeded ok ${nSeededOk}/${fleet})`);
execFileSync('bash', [P('search', 'evaluate.sh')], {
  cwd: REPO, stdio: 'inherit',
  env: { ...process.env, NEURO_DATASET: dataset, NEURO_ROI: roi,
         NEURO_IMSIZE: String(imsize), NEURO_STUDY: STUDY[dataset] || 'NsdSubj01' },
});

// 5. SCRIBE — rewrite the bounded notebook from (old notebook + this round's new rows)
const rowsAfter = DB.readDb(DBP);
const idset = new Set(queuedIds);
const newRows = rowsAfter.filter((r) => idset.has(r.id));
const scribe = await spawnAgent({ prompt: scribePrompt({ oldNotebook: notebook, newRows }) });
if (scribe && scribe.ok && scribe.text) {
  writeFileSync(NOTEBOOK, scribe.text.replace(/^```(?:markdown)?\n?/, '').replace(/\n?```\s*$/, '').trim() + '\n');
}

// 6. result.json — compact summary for the conductor
const dig1 = DB.digest(rowsAfter, { roi });
const newOk = newRows.filter((r) => r.status === 'ok' && Number.isFinite(r.test_corr))
                     .sort((a, b) => b.test_corr - a.test_corr);
const result = {
  round, roi, champion: dig1.champion, bestByRegion: dig1.bestByRegion,
  nQueued: queuedIds.length, nSeededOk, nOk: newOk.length,
  deltaBest: dig1.champion ? +(dig1.champion.corr - prevBest).toFixed(4) : 0,
  newIdeas: newOk.slice(0, 3).map((r) => `${(r.idea || '').trim()} (${r.test_corr.toFixed(4)})`),
};
writeFileSync(P('search', 'state', 'result.json'), JSON.stringify(result, null, 2));
console.log(`[round ${round}] done: champion=${dig1.champion?.corr} delta=${result.deltaBest} newOk=${result.nOk}/${result.nQueued}`);
