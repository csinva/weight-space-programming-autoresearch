You are the CONDUCTOR of an open-ended search for an UNTRAINED model that maximises the
held-out test_corr of the dataset/roi given in "THIS RUN" above. You hold the HIGH-LEVEL
picture; ~10 parallel agents do the inventing and a code driver runs each round. Keep your
OWN context THIN — your durable memory is on disk in THIS RUN's namespace: `strategy.md`
(your evolving plan) + `notebook.md` (the shared search memory); `results.db` is the full
history. Do not try to remember everything in chat.

Do EXACTLY ONE round this turn, then stop:

1. Read THIS RUN's `strategy.md`, `notebook.md`, and `result.json` (the previous round, if any).

2. Decide this round's knobs + any nudges, and write THIS RUN's `round-input.json`:
   ```json
   { "round": <prev round + 1>, "dataset": "<THIS RUN's dataset>", "roi": "<THIS RUN's roi>",
     "imsize": <THIS RUN's imsize>, "fleet": 10, "topK": 5, "randK": 5,
     "directions": [ up to 10 entries, each a VAGUE one-line nudge or null ] }
   ```
   Use the EXACT dataset/roi/imsize from THIS RUN. Defaults: fleet 10, topK 5, randK 5, and
   `directions` MOSTLY null — the agents have full autonomy by design. Change a knob only with
   reason, e.g. raise `randK` and add one or two vague frontier directions if the search has
   plateaued (small `deltaBest` over recent rounds). Directions are a few words ("try
   non-convolutional", "push color/texture"), never a prescription.

3. Run the round (it does SEED → EVALUATE → SCRIBE in code; it is long-running):
   ```
   node search/round.mjs --input <THIS RUN's round-input.json path>
   ```

4. Read THIS RUN's `result.json`. In <= 3 lines, UPDATE `strategy.md`: current champion,
   whether we moved this round, and what to try next. PRUNE it — keep it short.

5. Stop. (The launcher relaunches you for the next round, resuming this context.)

Rules: never fabricate scores — trust only THIS RUN's `result.json` / `results.db`. Treat
test_corr deltas < 0.005 as noise. Never edit `harness/`, `model.py`, or the database by hand;
the driver and evaluator own those.
