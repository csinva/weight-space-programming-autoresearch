export const meta = {
  name: 'lab-propose',
  description: 'Open-ended parallel proposal: agents read the database and invent new untrained NSD-ventral models, writing each as a model.py into lab/queue/.',
  phases: [{ title: 'Propose' }],
}

const N = 8   // parallel proposers per iteration
const K = 2   // models each proposer writes

const BRIEF = `You are an explorer in an OPEN-ENDED search for an UNTRAINED model that maximises held-out NSD VENTRAL test_corr. cwd = the repo root. Trained AlexNet scores 0.2838 (beat it); the target is 0.4257; the noise ceiling is ~0.50.

THE DATABASE -- lab/results.csv -- holds every model already tried: its idea, per-region scores (corr_V1..corr_VWFA), and FULL source code. READ IT FIRST and let it guide you (e.g. \`python -c "import pandas as p; d=p.read_csv('lab/results.csv'); print(d.sort_values('test_corr',ascending=False)[['id','idea','test_corr','corr_FFA','corr_PPA','corr_VWFA']].head(20).to_string())"\`, then open the source of the top models you want to learn from). Decide your own direction: build on what works, fix what stalls, or invent something new. You have TOTAL FREEDOM over the computation.

MODEL CONTRACT (harness format): write a self-contained file defining \`class Model\` with attributes \`name\`, \`idea\`, \`summary\`, \`n_params\`, and \`__call__(self, X)\` mapping a stimulus batch X of shape (N, D_in) -- flattened RGB pixels -- to activations (N, D_out); end with \`MODEL = Model()\`. Input is a 128x128 RGB image, so infer side = round(sqrt(D_in/3)) and reshape. Chunk the forward pass over N (e.g. bs<=128) and use torch.no_grad() to bound GPU memory.

THE ONLY TWO HARD RULES (they make the held-out score REAL -- everything else is unconstrained: any structure, any capacity, any operations):
1. UNTRAINED: every weight is analytic or random-but-fixed (use fixed seeds). You NEVER fit weights to data and NEVER read the neural target Y. (Only the harness's ridge readout is fit -- that is the eval, not your model.)
2. PER-SAMPLE: an image's activations depend ONLY on that image. NO statistic computed across the batch (no mean/std over dim 0, no batchnorm, no whitening over the set). Per-image normalisation only.

Write ${K} new models. For EACH: mint a fresh id with \`uuidgen | tr -d - | cut -c1-10\`, write the complete file to lab/queue/m_<id>.py, and make the FIRST line \`# idea: <a short label>\`. Do NOT run or evaluate them -- a separate evaluator scores the queue. Briefly list the ideas you queued.`

phase('Propose')
// Split the fleet: the first half EXPLOIT the proven leaders (the log-polar +
// steerable-pyramid texture family has topped the board for several rounds), the
// rest EXPLORE genuinely new mechanisms. Exploitation has been the search's gap.
const EXPLOIT = 2   // near saturation (~0.324); weight toward exploration for a regime change
const EXPLOIT_DIRECTIVE = `\n\nYou are proposer ${'${i + 1}'} of ${N}, in EXPLOIT mode. The current best (0.3244) keeps the proven 5-stream + all-three-conjunction + surround-divnorm champion UNCHANGED and only makes the READOUT spatial layout MODERATELY finer (an 8x8 SPP log-polar grid + denser eccentricity/angle global-form), staying under the harness ridge's ~100k-feature SVD limit. Findings on this axis: a MODERATE finer LOG-POLAR readout helps (0.3244); cartesian grids (0.318), over-dense log-polar/eccentricity grids (0.31-0.32 — the ridge OVERFITS when D_out is too large), and per-region band pooling (0.314) are all WORSE. There is a sweet spot. Your job: OPEN the 0.3244 model source and TUNE the readout grid resolution around that sweet spot — try a few nearby log-polar grid sizes (6x6, 7x7, 9x9 rings-by-sectors) and global-form densities, keeping total D_out comfortably under ~80k. Keep ALL proven streams/conjunctions/divnorm fixed; vary ONLY the spatial readout granularity. NOTE: gains are now within ~0.005 reward noise, so only a clearly >0.005 jump is real. Read the top models first: \`python -c "import pandas as p; d=p.read_csv('lab/results.csv').sort_values('corr_ventral',ascending=False).head(3); [print(r.id, r.corr_ventral, chr(10), r.code) for r in d.itertuples()]"\`.`
const EXPLORE_DIRECTIVE = `\n\nYou are proposer ${'${i + 1}'} of ${N}, in EXPLORE mode. Pick a direction distinct from the obvious and from what the database is already saturated with.`
const out = await parallel(Array.from({ length: N }, (_, i) => () =>
  agent(`${BRIEF}${(i < EXPLOIT ? EXPLOIT_DIRECTIVE : EXPLORE_DIRECTIVE).replace('${i + 1}', i + 1)}`,
    { label: `propose-${i + 1}-${i < EXPLOIT ? 'exploit' : 'explore'}`, phase: 'Propose' })
))
return { proposers: out.filter(Boolean).length }
