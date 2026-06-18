// search/lib/sample.mjs — deterministic seed-set construction (code-controlled diversity).

export function mulberry32(seed) {
  let s = seed >>> 0;
  return function () {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// k distinct items from pool, chosen with rng (partial Fisher–Yates).
export function sample(pool, k, rng) {
  const a = pool.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, Math.min(k, a.length));
}

// ranked = ok rows best-first (from db.rank). One seed set per agent:
// a SHARED top-K anchor + a per-agent RANDOM-K draw (the diversity knob).
export function buildSeedSets(ranked, { fleet = 10, topK = 5, randK = 5, round = 0 } = {}) {
  const top = ranked.slice(0, topK);
  const pool = ranked.slice(topK);
  return Array.from({ length: fleet }, (_, i) => ({
    agent: i,
    top,
    rand: sample(pool, randK, mulberry32(round * 1000 + i)),
  }));
}
