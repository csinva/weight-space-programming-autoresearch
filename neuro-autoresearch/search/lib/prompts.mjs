// search/lib/prompts.mjs — the search "face": model contract, notebook skeleton, and the
// thin proposer / scribe prompts. Kept minimal on purpose — discovery is the agent's job.

export const MODEL_CONTRACT = `You are an explorer in an OPEN-ENDED search for an UNTRAINED model that maximises held-out NSD VENTRAL test_corr. cwd = the repo root. Trained AlexNet scores 0.2838 (beat it); the noise ceiling is ~0.50.

MODEL CONTRACT (harness format): write a self-contained file defining \`class Model\` with attributes \`name\`, \`idea\`, \`summary\`, \`n_params\`, and \`__call__(self, X)\` mapping a stimulus batch X of shape (N, D_in) -- flattened RGB pixels -- to activations (N, D_out); end with \`MODEL = Model()\`. Input is a 128x128 RGB image, so infer side = round(sqrt(D_in/3)) and reshape. Chunk the forward pass over N (bs<=128) and use torch.no_grad() to bound GPU memory.

THE ONLY TWO HARD RULES (they make the held-out score REAL -- everything else is unconstrained: any structure, any capacity, any operations):
1. UNTRAINED: every weight is analytic or random-but-fixed (use fixed seeds). NEVER fit weights to data and NEVER read the neural target Y. (Only the harness ridge readout is fit -- that is the eval, not your model.)
2. PER-SAMPLE: an image's activations depend ONLY on that image. NO statistic across the batch (no mean/std over dim 0, no batchnorm, no whitening over the set). Per-image normalisation only.`;

export const NOTEBOOK_SKELETON = `# Search notebook (rewritten each round -- keep <= ~40 lines)

## Champion
<best model id + ventral test_corr, 1 line>

## Working (push these)
- (promising directions, <= 6 bullets)

## Saturated / dead (don't re-try)
- (directions that stalled or hurt)

## Frontiers (untried, worth exploring)
- (open ideas)

## Last round
(1-2 lines: what was tried, what moved)`;

const fmt = (x) => (Number.isFinite(x) ? x.toFixed(4) : '?');

function renderModel(r, withCode) {
  const head = `[${r.id}] ventral=${fmt(r.test_corr)} -- ${(r.idea || '').trim()}`;
  if (!withCode) return head;
  return `${head}\n\`\`\`python\n${(r.code || '').trim()}\n\`\`\``;
}

// top: full source (study + build on); rand: idea+score only (spark, not clone).
export function renderSeed({ top, rand }) {
  const t = top.map((r) => renderModel(r, true)).join('\n\n');
  const e = rand.map((r) => renderModel(r, false)).join('\n');
  return `## Proven leaders (study their source; build on what works)\n${t}\n\n## Random prior models (for inspiration / cross-pollination)\n${e}`;
}

export function proposerPrompt({ seed, notebook, direction }) {
  const dir = direction ? `\n\nVAGUE DIRECTION (a nudge, not a constraint): ${direction}` : '';
  return `${MODEL_CONTRACT}

NOTEBOOK (the search's shared memory):
${notebook}

${renderSeed(seed)}${dir}

Decide your OWN direction -- build on a leader, fix what stalls, or invent something new; you have TOTAL freedom over the computation. Write ONE new model: mint an id with \`uuidgen | tr -d - | cut -c1-10\`, write the complete file to search/state/queue/m_<id>.py, and make the FIRST line \`# idea: <short label>\`. Do NOT run or evaluate it. Reply with the one idea you queued.`;
}

export function scribePrompt({ oldNotebook, newRows }) {
  const rows = newRows
    .map((r) => `[${r.id}] ventral=${fmt(r.test_corr)} status=${r.status} -- ${(r.idea || '').trim()}`)
    .join('\n') || '(no new results this round)';
  return `You are the SCRIBE for an open-ended model search. REWRITE the notebook below into the SAME fixed skeleton, <= ~40 lines. PRUNE stale/duplicate entries -- do NOT append; this is curated working state, not a log. Keep only what helps the NEXT round decide what to try.

CURRENT NOTEBOOK:
${oldNotebook}

THIS ROUND'S NEW RESULTS:
${rows}

Output ONLY the rewritten notebook markdown (sections: Champion / Working / Saturated-dead / Frontiers / Last round). No commentary.`;
}
