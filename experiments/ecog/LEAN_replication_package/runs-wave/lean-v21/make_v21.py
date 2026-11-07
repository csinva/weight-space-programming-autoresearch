"""Derive LEAN-v21.py from LEAN-v20.py by exact, asserted edits (bug fixes only)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', 'wave-trim', 'wave_encoders_lib', 'LEAN-v20.py')
DST = os.path.join(HERE, 'LEAN-v21.py')
s = open(SRC).read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    assert n == count, (n, old[:120])
    s = s.replace(old, new)


def cut(start, end, new, include_end=True):
    """Replace s[start_marker : end_marker(+len)] with new; both markers unique."""
    global s
    assert s.count(start) == 1, start[:80]
    i = s.index(start)
    j = s.index(end, i)
    assert s.count(end, i) >= 1
    j = j + len(end) if include_end else j
    s = s[:i] + new + s[j:]


# 1. module docstring
cut('"""\nHand-designed auditory front-end', 'comodulation) on top of the best 3-window x 6-tau gammatone base.\n"""\n', '''"""
Hand-designed auditory front-end: raw WAVEFORM -> patient1 ECoG high-gamma.

LEAN-v21 = LEAN-v20 with bugs fixed (no new features):
  - voicing gates are driven by the ACF voicing peak (0..1). v20 used the harmonic-comb
    strength (~6-10, a log-power difference), which saturated every gate at exactly 1.0;
    v20's flatness gates (0.2-0.6 thresholds on a flatness that never exceeds ~0.12 here)
    never opened and are removed;
  - identical filters are removed: the -0.5 cyc/ch copy of every +0.5 cyc/ch 2-D Gabor
    (sweep direction is undefined at the channel Nyquist), the slow bank's raw 2 Hz rate
    (already in the quadrature bank), and the 12 Hz quadrature cos/sin outputs at scales
    0 and 0.5 (numerically equal to 8 Hz, because the Gabor time window is < 1 frame);
  - dead columns are removed: second-order scattering paths below 4 Hz (no support on
    the 10.8 Hz FFT grid of a 4096-sample window) and short-window contrast octaves with
    fewer than 2 FFT bins;
  - interact_set defaults to 'v3e' (the scored configuration); unused code is removed.
"""
''')

# 2. class constants
rep("    SUB_WIN = 1024\n", "")
rep("        (4.0, 0.5),  (4.0, -0.5),\n", "        (4.0, 0.5),\n")
rep("        (8.0, 0.5),  (8.0, -0.5),\n", "        (8.0, 0.5),\n")
cut("    STMF_RATES_ADAPT = (\n", "        (8.0, 0.5), (8.0, -0.5),\n    )\n", """    # +0.5 cyc/ch only: at the channel Nyquist the +0.5 and -0.5 kernels are identical.
    QUAD_RATES = (2.0, 4.0, 8.0, 12.0)
    QUAD_SCALES = (0.0, 0.25, -0.25, 0.5)
    SLOW_RATES = (0.5, 1.0, 1.5, 2.0)  # raw 2 Hz duplicates QUAD; only its slow dyn is kept
    SCAT_FC2_MIN = 4.0  # 2nd-order scattering wavelets below this have no FFT-grid support
""")

# 3. __init__: unused filterbanks / bases
rep("        self.fb_sub = gammatone_powerbank(self.SUB_WIN, n_filters=n_filters).to(device)\n", "")
rep("        # Finer cochleagram (40 channels) on the MEDIUM 2048 window (q12)\n"
    "        self.fb_fine_med = gammatone_powerbank(2048, n_filters=40).to(device)\n", "")
rep("        self._cond_logged = False\n", "")
rep("        # Second-order modulation wavelets: 0.5..4096 Hz at Q2=1/oct\n"
    "        self.scat_fcs2 = log_space(0.5, 4096.0, Q2)\n",
    "        # Second-order modulation wavelets: 4..4096 Hz at Q2=1/oct. v20 started at 0.5 Hz,\n"
    "        # but the 0.5/1/2 Hz wavelets put <1e-4 weight on any non-DC bin of the 10.8 Hz grid.\n"
    "        self.scat_fcs2 = log_space(self.SCAT_FC2_MIN, 4096.0, Q2)\n")
cut("        # DCT-II basis (n_filters -> 13 cepstral coefs), closed-form\n",
    "        self.dct_basis = torch.from_numpy(dct.astype(np.float32)).to(device)\n", "")

# 4. unused DoG method
cut("    def _dog_channels(self, x: torch.Tensor, sigma_c: float, sigma_s: float) -> torch.Tensor:\n",
    "        return torch.nn.functional.conv1d(x.unsqueeze(1), dog, padding=k).squeeze(1)\n\n", "")

# 5. group-conditioning machinery -> interact_set default + quad-bank helper
cut("    # ---------- group-conditioning knobs (Stage 3) ----------\n",
    "        return 1\n\n", '''    # Stage-2 interaction set: 'v3e' (scored configuration) or 'none' (omit the 405 dims).
    interact_set: str = 'v3e'

    def _quad_bank(self, log_e: torch.Tensor, rates, scales, bin_edges, drop=()):
        """Quadrature 2-D Gabor bank averaged within frequency bins.

        Returns (raw, mags). raw is (T, n) with columns ordered rate -> scale -> bin ->
        (cos, sin, mag), omitting (rate, scale, phase) triples listed in `drop`; mags is
        (T, n_rates, n_scales, n_bins) with every magnitude."""
        cols, mags = [], []
        for wt in rates:
            m_r = []
            for wf in scales:
                c, s = self._stmf_quadrature(log_e, wt, wf)
                m_s = []
                for (lo, hi) in bin_edges:
                    cb = c[:, lo:hi].mean(dim=1, keepdim=True)
                    sb = s[:, lo:hi].mean(dim=1, keepdim=True)
                    mb = torch.sqrt(c[:, lo:hi] ** 2 + s[:, lo:hi] ** 2 + 1e-12).mean(dim=1, keepdim=True)
                    for ph, v in (('cos', cb), ('sin', sb), ('mag', mb)):
                        if (wt, wf, ph) not in drop:
                            cols.append(v)
                    m_s.append(mb)
                m_r.append(torch.cat(m_s, dim=1))
            mags.append(torch.stack(m_r, dim=1))
        return torch.cat(cols, dim=1), torch.stack(mags, dim=1)

''')

# 6. scattering: computed-but-unused blocks
cut("        # ---- DENSE S1 ANALYTIC: all n1 bands", "        S1_all_log = torch.pow(S1_all.clamp(min=0.0) + 1e-8, 1.0 / 3.0)\n\n", "")
cut("        # Per-j1 envelope time-centroid in [0,1]\n", "        s1_centroid = torch.cat(s1_centroid_blocks, dim=1)\n", "")
cut("        # Per-j1 envelope peak position within frame [0,1] — transient timing\n",
    "        peak_pos = torch.cat(peak_blocks, dim=1)\n", "")

# 7. group concatenation (no conditioning)
cut("    def _condition(self, groups: list) -> torch.Tensor:\n", "        return torch.cat(out, dim=1)\n",
    '''    def _concat_groups(self, groups: list) -> torch.Tensor:
        """Concatenate the named (name, tensor[T, d]) feature groups in order."""
        return torch.cat([t for _, t in groups], dim=1)
''')
rep("        feat = self._condition(groups)\n", "        feat = self._concat_groups(groups)\n")

# 8. phonetic docstring (code unchanged)
cut('    """Closed-form phonetic matched filters on the 20-channel long-window\n',
    '    Returns (T, 5)."""\n', '''    """Closed-form phonetic band contrasts on the 20-channel long-window
    log-cochleagram (ranges for n_filters=20, ERB-spaced 50-8000 Hz):

      - sibilant:  mean(ch14..19) - mean(ch0..5)       (3.1-8 kHz vs 50-450 Hz)
      - vowel:     mean(ch5..14)  - std over channels  (450 Hz-3.1 kHz, peaky spectrum)
      - nasal:     mean(ch1..5)   - mean(ch8..13)      (100-450 Hz vs 0.9-2.6 kHz)
      - plosive:   std over channels - mean over channels
      - fricative: mean(ch17..19) - mean(ch5..14)      (5.5-8 kHz vs 450 Hz-3.1 kHz)

    Returns (T, 5)."""
''')

# 9. Stage-2 history comments -> short description
rep("        # Toggled by self.interact_set: 'none', 'v2a', 'v2b', 'v2c', 'v2d'.\n",
    "        # Added when self.interact_set == 'v3e' (the default).\n")
cut("            # --- 'v2a' : core voicing-gated set (≈ 34 raw dims) ---\n",
    "            # of 3 separately-non-harmful adjustments that each hit 0.0777.\n",
    "            # 'v3e' (LEAN-v17): asymmetric (ReLU+, ReLU-, cube) transforms of 10 key signals,\n"
    "            # of their deviations at 3 time scales, and of phonetic/formant deviations; 6\n"
    "            # cross-lag products; phonetic x key-signal, phonetic x formant, key-signal x\n"
    "            # formant and onset/offset-gated phonetic products. 405 dims.\n")
rep("            # --- 'v2d' : add FORMANT × FLUX, MOMENTS × VOICING — moment-class gating ---\n", "")

# 10. contrast bands with too few FFT bins are skipped, not zero-filled
rep("            if p_o.shape[1] < 4:\n                contrast_list.append(torch.zeros(p_o.shape[0], 1, device=p_o.device))\n                continue\n",
    "            if p_o.shape[1] < 4:  # too few FFT bins for a top/bottom contrast\n                continue\n")
rep("            if p_so.shape[1] < 2:\n                contrast_short_list.append(torch.zeros(p_so.shape[0], 1, device=p_so.device))\n                continue\n",
    "            if p_so.shape[1] < 2:  # 62-125 Hz has 1 bin at 1024 samples: skipped (v20 wrote zeros)\n                continue\n")
rep("            if p_m.shape[1] < 2:\n                contrast_med_list.append(torch.zeros(p_m.shape[0], 1, device=p_m.device))\n                continue\n",
    "            if p_m.shape[1] < 2:  # too few FFT bins for a top/bottom contrast\n                continue\n")

# 11. quadrature bank + cross-rate energy
cut("        # LEAN-v18.q1-QUADBANK: DENSE 2-D quadrature MODULATION BANK on the\n",
    "        add('quad_xrate_energy_dyn', torch.cat(xrate_dyn, dim=1))\n", '''        # LEAN-v18.q1-QUADBANK: DENSE 2-D quadrature MODULATION BANK on the FINE
        # (40-channel) LONG log-cochleagram, keeping even (cos) and odd (sin) outputs
        # alongside the magnitude, averaged in three frequency bins (low/mid/high).
        # v21: scales (0, +/-0.25, +0.5), since the -0.5 kernel equals +0.5; and the 12 Hz
        # cos/sin outputs at scales 0 and 0.5 are dropped because they equal the 8 Hz ones
        # (sigma_t = period/2pi is < 1 frame at 8-12 Hz, so both kernels reduce to the same
        # 3-tap shape). 4 rates x 4 scales x 3 bins x 3 phases - 12 = 132 raw dims.
        # ============================================================
        log_e_fine_long = torch.log(torch.clamp(power_long @ self.fb_fine, min=1e-10))  # (T, 40)
        nF40 = log_e_fine_long.shape[1]
        bin_edges = [(0, nF40 // 3), (nF40 // 3, 2 * nF40 // 3), (2 * nF40 // 3, nF40)]
        quad_drop = {(12.0, wf, ph) for wf in (0.0, 0.5) for ph in ('cos', 'sin')}
        quad_raw, quad_mags = self._quad_bank(log_e_fine_long, self.QUAD_RATES,
                                              self.QUAD_SCALES, bin_edges, drop=quad_drop)
        add('quad_modbank', quad_raw)
        # LEAN-v18.q2/q14: multi-tau adapt-delta (phonetic 4.5, syllabic 15, prosodic 60)
        quad_dyn_blocks = []
        for tau in (4.5, 15.0, 60.0):
            quad_dyn_blocks.append(quad_raw - leaky_lowpass(quad_raw, tau_frames=tau))
        add('quad_modbank_dyn', torch.cat(quad_dyn_blocks, dim=1))
        # LEAN-v18.q15: ACAUSAL prosodic dyn -- deviation from the next ~2 s
        add('quad_modbank_acausal', quad_raw - acausal_leaky_lowpass(quad_raw, tau_frames=60.0))

        # LEAN-v18.q17: cross-rate ENERGY coincidence. Energy per (rate, bin) = sum over
        # scales of the magnitude; log1p of products across rate pairs (slow x fast
        # co-occurrence), 6 pairs x 3 bins = 18. v21: the sum runs over 4 distinct scales
        # (v20 counted the 0.5 scale twice through its -0.5 copy).
        energy_per_rate_bin = quad_mags.sum(dim=2)          # (T, 4 rates, 3 bins)
        xrate_pairs = [(0, 2), (0, 3), (1, 3), (1, 2), (2, 3), (0, 1)]
        xrate_raw = torch.cat([energy_per_rate_bin[:, i, :] * energy_per_rate_bin[:, j, :]
                               for (i, j) in xrate_pairs], dim=1)          # (T, 18)
        xrate_log = torch.log1p(xrate_raw.clamp(min=0.0))
        add('quad_xrate_energy', xrate_log)
        xrate_dyn = []
        for tau in (4.5, 15.0):
            xrate_dyn.append(xrate_log - leaky_lowpass(xrate_log, tau_frames=tau))
        add('quad_xrate_energy_dyn', torch.cat(xrate_dyn, dim=1))
''')

# 12. voicing gates
cut("        # LEAN-v18.q19: PERIODICITY-GATED cochleagram.",
    "        gate = torch.sigmoid((best_strength - 0.3) * 8.0)        # (T, 1) soft\n",
    '''        # LEAN-v18.q19: PERIODICITY-GATED cochleagram. Multiply each of 20 gammatone
        # channels (long-window log_e) by a soft "voiced" probability: voiced 20 +
        # complement 20 = 40, plus the syllabic (tau=15) deviation of the voiced part = 20.
        # v21 fix: the gate is driven by the ACF voicing peak hnr_peak (0..1). v20 used
        # best_strength (~6-10), so the gate was exactly 1.0 on every frame and these two
        # blocks duplicated cochleagram and adapt_delta.
        hnr_peak = hnr[:, 0:1]
        gate = torch.sigmoid((hnr_peak - 0.3) * 8.0)             # (T, 1) soft
''')
cut("        # LEAN-v18.q29: MULTI-THRESHOLD VOICING+FLATNESS gate SIGNALS.\n",
    "        gate_cat = torch.cat(gate_signals, dim=1)              # (T, 6)\n", '''        # LEAN-v18.q29: MULTI-THRESHOLD VOICING gate SIGNALS + multi-tau deviations.
        # v21 fix: driven by hnr_peak (0..1) instead of the saturated best_strength; v20's
        # flatness gates (thresholds 0.2-0.6 on log-flatness <= 0) never opened: removed.
        gate_cat = torch.cat([torch.sigmoid((hnr_peak - th) * 8.0)
                              for th in (0.15, 0.3, 0.6)], dim=1)   # (T, 3)
''')

# 13. slow quadrature bank
cut("        # LEAN-v18.q42: SLOW QUAD MODBANK at prosodic rates (1.0, 1.5 Hz).\n",
    "        add('slow_quad_modbank_dyn', slow_dyn)\n", '''        # LEAN-v18.q42-q45: SLOW quadrature bank at prosodic rates + 1 s deviation (tau=30).
        # v21: same 4 scales as the quadrature bank; the raw 2 Hz outputs are dropped
        # because they equal quad_modbank's 2 Hz outputs, but the 2 Hz tau=30 deviation (not
        # in quad_modbank_dyn) is kept. Raw 3 rates x 36 = 108; dyn 4 rates x 36 = 144.
        slow_all, _ = self._quad_bank(log_e_fine_long, self.SLOW_RATES, self.QUAD_SCALES, bin_edges)
        per_rate = slow_all.shape[1] // len(self.SLOW_RATES)
        keep = [i for i, r in enumerate(self.SLOW_RATES) if r not in self.QUAD_RATES]
        add('slow_quad_modbank', torch.cat([slow_all[:, i * per_rate:(i + 1) * per_rate]
                                            for i in keep], dim=1))
        add('slow_quad_modbank_dyn', slow_all - leaky_lowpass(slow_all, tau_frames=30.0))
''')

# 14. name / description
cut('model_shorthand_name = "LEAN-v20"', '    "plateau at 0.0775 confirmed."\n)\n', '''model_shorthand_name = "LEAN-v21"
model_description = (
    "LEAN-v21 = LEAN-v20 with bugs fixed, no new features: voicing gates driven by the ACF "
    "voicing peak (v20's were saturated at 1.0, duplicating cochleagram/adapt_delta); flatness "
    "gates removed (never opened); identical filters removed (-0.5 cyc/ch copies, slow-bank raw "
    "2 Hz = quad 2 Hz, 12 Hz quad cos/sin at scales 0/0.5 = 8 Hz); dead columns removed "
    "(scattering S2 paths < 4 Hz, short-window contrast octave with 1 FFT bin); interact_set "
    "defaults to v3e; unused code removed. Parent LEAN-v20: all-166 0.0808 / grid 0.0904."
)
''')

# 15. build_embedder / __main__
cut('def build_embedder(device: str = "cuda", cond_mode: str = "none",\n', '    return emb\n',
    '''def build_embedder(device: str = "cuda", interact_set: str = "v3e") -> WaveEmbedder:
    emb = WaveEmbedder(device=device)
    emb.interact_set = interact_set
    return emb
''')
rep('    parser.add_argument("--cond-mode", default="none")\n', '')
rep("    emb = build_embedder(device=args.device, cond_mode=args.cond_mode, interact_set=args.interact_set)\n",
    "    emb = build_embedder(device=args.device, interact_set=args.interact_set)\n")

for bad in ('best_strength - 0.3', 'flat - th', 'cond_mode', 'fb_sub', 'fb_fine_med', 'dct_basis',
            '_dog_channels', 'STMF_RATES_ADAPT', 'S1_all', 'peak_pos', 's1_centroid', '(4.0, -0.5)',
            "'v2a'", 'SLOW_SCALES', 'QUAD_SCALES = (0.0, 0.25, -0.25, 0.5, -0.5)'):
    assert bad not in s, bad
open(DST, 'w').write(s)
print(f'wrote {DST}: {s.count(chr(10))} lines')
