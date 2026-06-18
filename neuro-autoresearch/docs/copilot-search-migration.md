# Spec — migrate the open-ended search to a structured Copilot workflow

**Status:** IMPLEMENTED + smoke-validated (2026-06-18). DB migrated to SQLite (254 rows);
one live `fleet=1` NSD-ventral round ran end-to-end (SEED→EVALUATE→SCRIBE→result.json),
proposing a new `status=ok` model at ventral 0.3285. Entry point: `search/search.sh`.
The `lab/` directory has since been renamed to `search/`, with machine-written state
split out under `search/state/` (gitignored). The one deviation from the spec below: the
**database is SQLite (`search/state/results.db`)**, not CSV — `search/db.py` (Python
writes) + `search/lib/db.mjs` (`node:sqlite` reads). The old root `run.py` and
`lab/propose.js` are retired/deleted, superseded by the conductor + `search/round.mjs`.

## Goal

Move the open-ended untrained-model search off the **Claude Code** conductor +
dynamic-Workflow proposer, onto the **Copilot** `workflow` skill (unlimited credits),
and make the loop **structured** — the plan held in code, not a chat transcript.

Decisions locked (2026-06-18):
1. **Thin persistent conductor over a per-round code driver.** A long-running Copilot
   session (the conductor) owns the loop and carries cross-round context, but delegates
   ALL heavy work to `search/round.mjs` (code + spawned agents), so the fan-out never bloats
   its context. Its durable memory is on disk (`notebook.md` + `strategy.md`), so it is
   relaunchable losslessly. (See "Conductor + driver" below.)
2. **Memory model:** DB (full history) + a **bounded, rewritten** `notebook.md` (curated
   working memory) + a small `strategy.md` (the conductor's evolving plan), re-read each
   round. No growing append log; the conductor's live context holds only compact summaries.
3. **Evaluator:** keep the LLM evaluator with auto-repair (today's self-healing eval).
4. **Dataset scope:** port NSD-ventral 1:1 onto the new driver, but make dataset/ROI/
   imsize `--flags` so extending to `nnn`/`tvsd` later is trivial.
5. **Search face = open discovery:** thin proposer prompts (contract + seed models +
   notebook + "explore"), no hardcoded exploit/explore directives.
6. **Diversity by code:** a deterministic JS sampler builds each proposer's seed context
   from the DB (shared top-K + per-agent random-K), so exploration diversity is
   code-controlled and reproducible, not left to LLM temperature.
7. **Conductor agency (light touch):** each round the conductor may optionally hand any
   agent a **vague one-line direction** (a few words; default: none → full autonomy) and
   may tune the knobs `fleet` / `topK` / `randK`. It nudges from the high-level picture;
   the agents keep maximal freedom over the actual computation.

## Current state (the hybrid being replaced)

| Piece | Today | Engine |
| --- | --- | --- |
| Conductor / loop | old root `run.py` → persistent `claude -p … --continue` | **Claude Code** (replace) |
| PROPOSE (parallel) | `lab/propose.js` via Claude dynamic Workflow — N=8 × K=2 → `search/state/queue/` | **Claude Code** (replace) |
| EVALUATE (sequential) | `search/evaluate.sh` → `copilot -p … --model claude-opus-4.8`, one model/GPU, repairs bugs, appends `results.csv`, → `search/state/done/` | **Copilot** (keep) |
| Database | `search/state/results.csv` (every model: idea, per-region scores, full code); `search/state/ITER` round counter | unchanged |

The orchestration (conductor + proposer) is the unstructured part; the evaluator is
already Copilot and already structured enough.

## Copilot capability check (the "parallelize skill, headless" question)

Confirmed available in `~/.copilot/skills/`:
- **`orchestrate`** (`copilot-spawn`) — ~10 headless Opus workers in git worktrees,
  auto-merge. For code features; **not** our shape.
- **`workflow`** (`copilot-workflow`) — **the structured analog of Claude Code's
  dynamic workflows.** A Node `.mjs` driver holds the plan; `spawnAgent`/`mapPool`
  fan out up to 1000 fresh-context subagents (concurrency cap 16); `writeResult()` →
  `result.json`. Ships a **`loop-until-done`** template = our propose→evaluate→repeat
  loop. Launched headless: `copilot-workflow <slug> <driver.mjs> <repo> <timeout> -- <args>`
  (runs the driver in a detached tmux session for hours). **All headless. ✔**

Runtime API (`$COPILOT_ENV/copilot/workflow/runtime/workflow.mjs`):
`spawnAgent({prompt,model,effort,context,repoDir,worktree,timeout}) → {text,json,ok,code}`,
`mapPool(items,fn,{concurrency=16})`, `getArgs()`, `writeResult(value)`,
`MAX_CONCURRENCY=16`, `MAX_AGENTS=1000`.

## What the Node driver actually is

The per-round driver (`search/round.mjs`) is **plain Node.js — not a Copilot agent, not an
LLM.** It runs as a `node` process; it has variables, a loop, conditionals, and JSON; it
reads files and calls the runtime:
- `spawnAgent({prompt,effort,…})` — launches one **fresh, headless Copilot LLM worker**
  (isolated context) for a unit of work; returns its text/JSON.
- `mapPool(items, fn, {concurrency})` — runs ~10 such spawns in parallel.
- It aggregates results in plain JS and decides what to do next.

So a round is driven by **code, not a model**: the per-round mechanics (how many to spawn,
how to sample seeds, how to combine results) are deterministic JavaScript you can read and
re-run. Because a code process has **no context window**, the round itself cannot bloat or
drift; only the thin conductor above it holds cross-round state.

## Conductor + driver (long-running context, no bloat)

Two layers, split by where heavy context lives:

- **`search/round.mjs`** — does ONE round entirely as code + spawned agents (SEED →
  EVALUATE → SCRIBE) and writes a **tiny `result.json`** (round #, best/region, deltas,
  a couple of notable new ideas). All bulk — 10 agents' outputs, model source,
  tracebacks — stays inside this child process and on disk; none of it reaches the conductor.
- **The conductor** — a persistent Copilot session whose loop is: run `node search/round.mjs`,
  read the small `result.json`, update a one-paragraph rolling assessment + `strategy.md`,
  decide any knob change or stop, next round. It ingests only compact summaries, so its
  context grows slowly while it keeps genuine cross-round memory (spot plateaus, steer).

**Why this avoids bloat:** the fan-out bloat is quarantined in the child driver; the
conductor only sees per-round summaries. **Honest caveat:** an LLM context is a transcript,
so it still grows slowly. Two mitigations make it effectively unbounded: (a) the conductor
keeps durable memory in `strategy.md` + `notebook.md` and leans on those (not its
transcript) for detail; (b) all state is on disk, so the conductor is **disposable** —
relaunch it fresh anytime and it re-reads `strategy.md` + `notebook.md` with zero loss.

This refines (not reverses) the principle "code does the heavy lifting": the per-round
driver still owns all parallel work (no bloat); we add a *thin strategist conductor* for
cross-round reasoning, with its memory externalized to disk.

## Target architecture

The conductor (persistent Copilot session) owns the loop; `search/round.mjs` owns one round.

```
CONDUCTOR (persistent Copilot session) — reads strategy.md + notebook.md, then loops:
  each round:
    write search/state/round-input.json { round, dataset, roi, imsize, fleet, topK, randK,
                                 directions: [ "vague 1-liner or null", … ≤ fleet ] }
    node search/round.mjs --input search/state/round-input.json
    read search/state/result.json (compact) → update strategy.md + a 1-line rolling note
    decide: tune fleet/topK/randK? set/clear any vague directions? stop? → next round

search/round.mjs (ONE round, code + spawned agents):
  1. READ    — parse round-input.json, results.csv, notebook.md (plain JS).
  2. SAMPLE  — buildSeedSets(db,{fleet,topK,randK,round}): shared top-K + a per-agent
               random-K (seeded RNG) → diverse seed contexts. (code, not LLM)
  3. SEED    — mapPool `fleet` spawnAgent proposers (fresh context, no worktree). THIN
               prompt: model contract + its seed models + notebook.md + the agent's
               OPTIONAL one-line direction (if any) + "explore". Each writes a
               uniquely-named model.py → search/state/queue/.
  4. EVALUATE— child_process: bash search/evaluate.sh (existing Copilot auto-repair
               evaluator, sequential GPU; parameterized by dataset/roi/imsize).
               Appends results.csv, moves files to search/state/done/.
  5. SCRIBE  — ONE spawnAgent reads (old notebook.md + this round's new rows) and
               REWRITES notebook.md in place to the bounded skeleton (prune, don't append).
  6. write search/state/result.json (round #, best/region, deltas, notable ideas). Exit.
```

**Memory** = DB (complete append-only history, machine-written) + `notebook.md` (bounded
curated working memory) + `strategy.md` (the conductor's evolving plan). All on disk,
re-read each round — robust to context bloat, drift, and crash/restart.

## Notebook discipline (small, structured, open-ended)

`search/state/notebook.md` is **bounded curated state, not an append log.** The SCRIBE agent
REWRITES it each round to a fixed skeleton with a hard cap (~40 lines / small token
budget), pruning stale entries. Full history is always in `results.csv`.

Fixed skeleton:
- **Champion** — best model per region + score.
- **Working** — promising directions to push (capped, ≤~6 bullets).
- **Saturated / dead** — directions that stalled or hurt; don't re-try (capped).
- **Frontiers** — untried ideas worth exploring (capped).
- **Last round** — 1–2 lines on what just happened.

Simple (capped), structured (fixed sections), linear (rewritten, never an unbounded
log), yet open-ended: discovery lives in *Frontiers* + the thin proposer prompts, so the
agents — not hardcoded directives — drive the exploration.

## Diversity by code: the seed sampler (`buildSeedSets`)

A pure JS function in `round.mjs` assembles each proposer's seed context from the DB,
deterministically — so cross-agent diversity is **code-controlled, not LLM-temperature**:

```js
function buildSeedSets(db, { fleet = 10, topK = 5, randK = 5, round }) {
  const ok     = db.filter(r => r.status === 'ok' && Number.isFinite(+r.test_corr));
  const ranked = ok.sort((a, b) => b.test_corr - a.test_corr);
  const top    = ranked.slice(0, topK);          // shared anchor — proven leaders
  const pool   = ranked.slice(topK);             // everything else, for random draws
  return Array.from({ length: fleet }, (_, i) => {
    const rng  = mulberry32(round * 1000 + i);    // deterministic per (round, agent)
    const rand = sample(pool, randK, rng);        // distinct draw per agent → diversity
    return { agent: i, top, rand };               // each agent: 5 shared + its own 5 random
  });
}
```

- **`top` (shared):** every agent sees the proven leaders as an anchor (exploit signal).
- **`rand` (per-agent):** each agent draws its own random K from the rest of history —
  the diversity knob. Different agents study different prior work → diverse proposals.
- **Reproducible:** seeded RNG (`round*1000 + agent`) matches the project's fixed-seed ethos.
- **Knobs (code):** `topK`, `randK`, and an optional fleet **split** (e.g. half anchored
  on top, half pure-random) instead of every-agent-gets-both. Default = "5 shared + 5 random".
- **Context budget:** `top` rendered with full source (to build on); `rand` as
  idea + score + short snippet (to spark, not clone).

The SEED prompt is then thin: `model contract + render(top, rand) + notebook.md +
[optional vague direction] + "explore"`. The direction (if the conductor set one for this
agent) is a single nudging line; absent it, the agent has full autonomy.

## Interface contracts (freeze first — this is what lets the slices build in parallel)

Modules (ESM under `search/lib/`):
- **`db.mjs`** — `readDb(csvPath) → Row[]`; `rank(rows, {roi}) → Row[]` (ok rows, best-first
  by the roi's reward); `digest(rows, {roi}) → {champion:{id,corr}, bestByRegion:{}, nOk, nTotal}`.
- **`sample.mjs`** — `mulberry32(seed) → ()=>number`; `sample(pool, k, rng) → Row[]`;
  `buildSeedSets(ranked, {fleet, topK, randK, round}) → {agent, top:Row[], rand:Row[]}[]`.
- **`prompts.mjs`** — `MODEL_CONTRACT: string`; `NOTEBOOK_SKELETON: string`;
  `renderSeed({top, rand}) → string`; `proposerPrompt({seed, notebook, direction}) → string`
  (folds the optional one-liner); `scribePrompt({oldNotebook, newRows}) → string`.

Files on disk:
- **`search/state/round-input.json`** (conductor → round.mjs):
  `{ round:int, dataset, roi, imsize, fleet, topK, randK, directions:(string|null)[] }` —
  `directions[i]` is agent i's optional vague 1-liner; missing/null ⇒ full autonomy.
- **`search/state/result.json`** (round.mjs → conductor, kept tiny):
  `{ round, roi, champion:{id,corr}, bestByRegion:{}, nQueued, nOk, deltaBest, newIdeas:string[] }`.
- **`search/state/notebook.md`** — fixed skeleton (Champion / Working / Saturated / Frontiers / Last round), capped ~40 lines.
- **`search/state/strategy.md`** — the conductor's small evolving plan (hypothesis, what to try next, current knobs, stop condition).
- **`evaluate.sh` env contract** — reads `NEURO_DATASET`/`NEURO_ROI`/`NEURO_IMSIZE`
  (default nsd/ventral/128); consumes `search/state/queue/*.py`; appends `results.csv` via
  `record.py`; moves to `search/state/done/`.

`round.mjs` imports the three `lib/` modules + the runtime `spawnAgent`/`mapPool`. Each
slice is built against these signatures; integration only wires them together.

## Component changes

| File | Action |
| --- | --- |
| old root `run.py` | **retired/deleted** — the loop moves to the conductor (persistent Copilot session) |
| `search/round.mjs` | **new** — per-round driver (READ→SAMPLE→SEED→EVALUATE→SCRIBE→`result.json`); houses `buildSeedSets` |
| conductor prompt | **new** — the persistent Copilot session's standing instruction (loop `round.mjs`, curate `strategy.md`, decide knobs/stop) |
| `lab/propose.js` | **retired/deleted** — replaced by thin SEED prompts; seed models come from `buildSeedSets`, guidance from `notebook.md` (no hardcoded directives) |
| `search/evaluate.sh` | keep; parameterize the hardcoded `nsd/ventral/128` via env/args |
| `search/state/notebook.md` | **new** — scribe-maintained, **bounded**, rewritten each round (never appended) |
| `search/state/strategy.md` | **new** — the conductor's small evolving plan (its disk-backed cross-round memory) |
| `search/record.py`, `search/state/results.csv`, `search/state/queue/`, `search/state/done/`, `search/state/ITER` | unchanged (DB contract preserved) |
| `search/search.sh` | **new** — launcher: starts/resumes the conductor session with a wall-clock budget |
| `search/state/round-input.json`, `search/state/result.json` | **new** — the conductor↔round.mjs interface files (see contracts) |

Proposers add only uniquely-named files → **no git worktrees needed** for SEED
(`spawnAgent` plain file-drop), keeping the queue mechanism identical to today.

## Parameterization debt (NSD-ventral-first → multi-dataset later)

To go beyond NSD ventral, two NSD-specific spots must generalize (flagged, not done now):
- `search/evaluate.sh` hardcodes `NEURO_DATASET=nsd NEURO_ROI=ventral NEURO_IMSIZE=128`
  and the result path `runs/lab-eval/results/NsdSubj01__ventral__<id>.csv`.
- `search/record.py` + `search/state/results.csv` schema hardcode NSD region columns
  (`corr_V1..corr_VWFA`); `nnn`/`tvsd` have different region menus and no `corr_*` panel.
The driver itself stays dataset-agnostic; these two are the only NSD-coupled pieces.

## Open questions / risks

- **Round cadence vs. budget:** fixed `--rounds` vs. a wall-clock `--hours` budget the
  driver checks each round (mirrors today's `timeout`). Likely support both.
- **Fleet sizing / exploit-explore ratio:** today N=8 (2 exploit / 6 explore); the
  conductor can tune `fleet`/`topK`/`randK` from the trajectory (e.g. widen `randK` near
  saturation), or set vague directions to steer — both are its knobs.
- **Reward noise:** small-sample reward (esp. nnn 200-test) means <~0.005 deltas are
  noise; the SCRIBE/conductor should treat ties as ties when naming the champion to exploit.
- **Resumability:** since state is on disk, a killed run resumes by relaunching the
  conductor (it re-reads strategy.md/notebook.md/results.csv). No session id to preserve.
- **Cost:** spawnAgent proposers + the Copilot evaluator are all Copilot credits (the point).

## Implementation plan (parallel)

Built to run as disjoint-file slices in parallel (one agent each), after a tiny serial
setup that freezes the contracts. NSD-ventral only; `record.py`/schema generalization for
`nnn`/`tvsd` is deferred (see debt).

**Phase 0 — serial, tiny (the join key):** freeze the **Interface contracts** above;
create `search/lib/`; commit empty stubs for each module + the four disk files
(`round-input.json`, `result.json`, `notebook.md`, `strategy.md`) so every slice compiles
against fixed signatures. Ensure `.copilot-worktrees/` is gitignored if workers use them.

**Parallel wave — all disjoint files, one worker per slice:**

| Slice | Owns (files) | Does | Self-check |
| --- | --- | --- | --- |
| **A** | `search/lib/db.mjs` | parse results.csv, rank by reward, digest | run on the live 3.5 MB results.csv → prints champion + bestByRegion |
| **B** | `search/lib/sample.mjs` | `mulberry32`, `sample`, `buildSeedSets` | deterministic; per-agent rand sets distinct; no top/rand overlap |
| **C** | `search/lib/prompts.mjs` | `MODEL_CONTRACT` (port from the old proposer), `NOTEBOOK_SKELETON`, `renderSeed`, `proposerPrompt` (+optional direction), `scribePrompt` | snapshot the rendered strings |
| **D** | `search/round.mjs` | integrator: read input → A/B/C → mapPool spawnAgent SEED → exec evaluate.sh → spawnAgent SCRIBE → write result.json | codes against A/B/C contracts; wired at H |
| **E** | `search/evaluate.sh` | parameterize by `NEURO_DATASET/ROI/IMSIZE`; generalize result-path glob | one model evaluated on nsd ventral |
| **F** | `search/conductor.md` + `search/search.sh` + seed `search/state/strategy.md` | conductor standing prompt (loop, write round-input with optional directions + tuned knobs, read result, curate strategy.md, stop rule) + resume/budget launcher | dry-run prints the planned round-input.json |
| **G** | `search/bootstrap.mjs` | one-time build of `notebook.md` + `strategy.md` from current results.csv (reuses A + C) | produces a valid first notebook |

**Join — serial:**
- **H — integration + validation:** wire A–E; run ONE NSD-ventral round via
  `node search/round.mjs --input …` at `fleet=2` for speed; confirm queue → eval →
  results.csv → notebook rewrite → result.json. Then one full conductor round
  end-to-end; then a short timed multi-round run. Retire the old `run.py` + `lab/propose.js`.

**Dependencies:** A, B, C, E, F, G are mutually disjoint → fully parallel. D depends only
on the A/B/C *contracts* (frozen in Phase 0), so it is authored in parallel and integrated
at H. This maps cleanly onto the Copilot `orchestrate` skill (≈7 parallel workers + a
serial join), or I can build it directly.
