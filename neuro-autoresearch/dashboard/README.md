# `dashboard/` — live search dashboard

A real-time web dashboard for a search run, reusing the
[CORAL](https://github.com/Human-Agent-Society/CORAL) project's React frontend **unchanged**
and feeding it our data. CORAL's UI talks to a fixed REST + Server-Sent-Events contract and
does not care where the data comes from — so we keep their prebuilt frontend (vendored under
`static/`) and only swap in a backend that reads our `runs/<task>__<date>/search.db`.

## Run

```bash
pip install -e ".[dashboard]"               # one-time: starlette + uvicorn
python -m dashboard                  # serve the latest run
python -m dashboard --run runs/nsd__2026-06-24 --port 8080
```

Open the printed URL. It is **read-only**, so it is safe to run while a search is live; the
Overview chart, leaderboard, and header stats update on their own via SSE.

## What you get

- **Overview** — a prominent current **best score**, the search trajectory (every scored model,
  coloured by proposer, with the running-best line), the leaderboard, and per-proposer activity.
  Click a row to see its idea, metrics, and never-trained `model.py` source.
- **Knowledge** — Skills = each proposer's `direction.txt` (its self-chosen front); Notes = the
  champion models, each expandable to its full `model.py` source.
- **Logs** — each proposer's full agent transcript (reasoning + the commands it ran).
- Header **run selector** — switch between `runs/*__*` without restarting.

## How it maps (CORAL `Attempt` ⟵ our `models` row)

| CORAL field   | our column      |
|---------------|-----------------|
| `commit_hash` | `id`            |
| `agent_id`    | `proposer`      |
| `title`       | `summary`/`idea`|
| `score`       | `test_corr`     |
| `status`      | `status` (`ok`→`improved`/`baseline`, `error`→`crashed`, `running`→`active`, `queued`→`idle`) |
| `timestamp`   | `created_at`    |

`improved` marks each `ok` model that set a new running-best (highlighted in the chart).

## Files

| file          | role                                                                 |
|---------------|----------------------------------------------------------------------|
| `_db.py`      | the keystone: resolve/open a run's `search.db`, map rows → `Attempt` |
| `api.py`      | REST handlers (attempts, leaderboard, status, runs, config, dag, …)  |
| `events.py`   | SSE: polls `search.db` ~2s, emits `attempt:new` / `:update` / `eval:update` |
| `logs.py`     | parses `proposers/<pNN>/proposer.log` into the log-viewer shape       |
| `app.py`      | Starlette wiring + serves `static/` (the vendored CORAL frontend)    |
| `__main__.py` | `python -m dashboard` launcher                                |
| `static/`     | CORAL's prebuilt React bundle (vendored as-is)                       |

## Updating the vendored frontend

`static/` is a **prebuilt** copy of CORAL's React frontend with a few small source edits of ours:

- removed the **Lineage** tab (`App.tsx`, `Header.tsx`) — our search keeps no parent lineage;
- a prominent **best-score** banner on Overview (`pages/Overview.tsx`);
- **smaller** chart point markers (`components/ScoreChart.tsx`);
- dropped the always-zero **token in/out** line on agent cards (`pages/Overview.tsx`);
- rebranded the header + page title to **neuro-autoresearch** (`Header.tsx`, `index.html`).

To refresh: clone CORAL, re-apply the edits above in its `web/src`, run `npm run build` in
`web/`, and copy the resulting `coral/web/static/*` into this `static/` dir. We add no API
endpoints beyond the ones the frontend already calls.
