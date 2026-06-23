// search/lib/prompts.mjs — the search "face": model contract + the thin proposer prompt.
// Kept minimal on purpose — discovery is the agent's job. The ranked DB (sampled into each
// proposer's seed by buildSeedSets) is the ONLY shared cross-agent memory; there is no
// curated notebook and no conductor. The contract is parameterized per run by
// targetSpec(dataset, roi, imsize), so the same engine drives NSD / TVSD / Triple-N searches.

// Per-run framing: what to maximize, input resolution, the bar/ceiling, and any output cap.
export function targetSpec(dataset, roi, imsize) {
  const label = {
    nsd: `NSD subject-1 ${roi}-region fMRI`,
    tvsd: `TVSD macaque ${roi} spiking (monkey F)`,
    nnn: `Triple-N macaque ${roi} spiking`,
  }[dataset] || `${dataset} ${roi}`;
  const ceiling = {
    nsd: `a trained ImageNet AlexNet (fc7) read out the same way scores ~0.284 on ventral (the bar to BEAT); the noise ceiling is ~0.50`,
    tvsd: `the release's split-half noise ceiling averages ~0.55-0.63 (no trained-AlexNet bar is computed for tvsd) -- there is real headroom`,
    nnn: `an untrained hand-designed model already reaches ~0.23 on IT; the noise ceiling is not published`,
  }[dataset] || ``;
  // TVSD has ~22,248 train images; the GPU ridge SVD fails above ~48k output features at
  // that row count, so cap the model output for tvsd (nsd ~9000 / nnn 800 rows stay under it).
  const cap = dataset === 'tvsd' ? 30000 : null;
  return { dataset, roi, label, ceiling, imsize, cap };
}

function modelContract(t) {
  const capNote = t.cap
    ? `\n\nDATASET LIMIT: keep the model's OUTPUT dimension D_out under ~${t.cap}. This dataset has many training rows, and the harness ridge readout's GPU SVD fails above ~33k features at that row count -- a larger D_out is scored as an error, not a result.`
    : ``;
  return `You are an explorer in an OPEN-ENDED search for an UNTRAINED model that maximises held-out ${t.label} test_corr (mean per-target Pearson r of a fixed ridge readout). cwd = the repo root. For reference, ${t.ceiling}.

MODEL CONTRACT (harness format): write a self-contained file defining \`class Model\` with attributes \`name\`, \`idea\`, \`summary\`, \`n_params\`, and \`__call__(self, X)\` mapping a stimulus batch X of shape (N, D_in) -- flattened RGB pixels -- to activations (N, D_out); end with \`MODEL = Model()\`. Input is a ${t.imsize}x${t.imsize} RGB image, so infer side = round(sqrt(D_in/3)) and reshape. Chunk the forward pass over N (bs<=128) and use torch.no_grad() to bound GPU memory.

THE ONLY TWO HARD RULES (they make the held-out score REAL -- everything else is unconstrained: any structure, any capacity, any operations):
1. UNTRAINED: every weight is analytic or random-but-fixed (use fixed seeds). NEVER fit weights to data and NEVER read the neural target Y. (Only the harness ridge readout is fit -- that is the eval, not your model.)
2. PER-SAMPLE: an image's activations depend ONLY on that image. NO statistic across the batch (no mean/std over dim 0, no batchnorm, no whitening over the set). Per-image normalisation only.${capNote}`;
}

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(4) : '?');

function renderModel(r, withCode) {
  const head = `[${r.id}] score=${fmt(r.test_corr)} -- ${(r.idea || '').trim()}`;
  if (!withCode) return head;
  return `${head}\n\`\`\`python\n${(r.code || '').trim()}\n\`\`\``;
}

// top: full source (study + build on); rand: idea+score only (spark, not clone).
export function renderSeed({ top, rand }) {
  if (!top.length && !rand.length) {
    return `## Prior models\n(none yet -- this is a fresh search for this dataset; invent from first principles.)`;
  }
  const t = top.map((r) => renderModel(r, true)).join('\n\n');
  const e = rand.map((r) => renderModel(r, false)).join('\n');
  return `## Proven leaders (study their source; build on what works)\n${t}\n\n## Random prior models (for inspiration / cross-pollination)\n${e}`;
}

export function proposerPrompt({ seed, target, queueDir }) {
  return `${modelContract(target)}

${renderSeed(seed)}

Decide your OWN direction -- build on a leader, fix what stalls, or invent something new; you have TOTAL freedom over the computation. Write ONE new model: mint an id with \`uuidgen | tr -d - | cut -c1-10\`, write the complete file to ${queueDir}/m_<id>.py, and make the FIRST line \`# idea: <short label>\`. Do NOT run or evaluate it. Reply with the one idea you queued.`;
}
