"""LEAN-v21 feature groups -> raw column ranges, sections, families, and the
deduplicated list of ridge fits for variance partitioning.

Order and sizes match LEAN-v21 (LEAN-v20 minus dead/duplicate columns) and are re-checked at runtime in
phase1_stats.py against the names/dims the embedder actually emits.
"""
from __future__ import annotations

import numpy as np

NDELAYS = 4

# (group, raw dim, section) in embedder output order; scattering is appended last.
GROUPS = [
    ('cochleagram', 60, 'A'), ('adapt_delta', 300, 'A'),
    ('pitch_salience', 11, 'B'), ('pitch_summary', 2, 'B'), ('pitch_dyn', 6, 'B'),
    ('spectral_moments', 4, 'C'), ('spectral_flatness', 1, 'C'), ('spectral_rolloff', 2, 'C'),
    ('hnr', 2, 'C'), ('phonetic', 5, 'C'),
    ('flux', 36, 'D'),
    ('modulation', 132, 'E'),
    ('stmf', 300, 'F'), ('stmf_mean_gabor', 15, 'F'),
    ('formants', 24, 'G'), ('formant_dyn', 9, 'G'),
    ('mod_depth', 40, 'H'), ('tonic', 300, 'H'), ('future_past', 260, 'H'),
    ('f0_dyn', 4, 'I'), ('hnr_dyn', 3, 'I'),
    ('interact_v3e', 405, 'J'),
    ('cpp', 4, 'K'), ('region_flat', 8, 'K'),
    ('spectral_contrast', 24, 'L'), ('contrast_short', 20, 'L'), ('contrast_med', 24, 'L'),
    ('cochlea_contrast', 20, 'L'), ('med_cochlea_contrast', 20, 'L'),
    ('short_cochlea_contrast', 20, 'L'), ('multitau_contrast', 10, 'L'),
    ('med_adapt_contrast', 5, 'L'), ('topbot', 10, 'L'), ('topbot_short', 10, 'L'),
    ('fut_cochlea_contrast', 5, 'L'), ('fut_cochlea_contrast_med', 5, 'L'),
    ('fut_med_contrast_6', 5, 'L'), ('fut_minus_cur_contrast', 5, 'L'), ('cube_fut_contrast', 5, 'L'),
    ('quad_modbank', 132, 'M'), ('quad_modbank_dyn', 396, 'M'), ('quad_modbank_acausal', 132, 'M'),
    ('quad_xrate_energy', 18, 'M'), ('quad_xrate_energy_dyn', 36, 'M'),
    ('cochlea_voiced_gated', 40, 'N'), ('cochlea_voiced_gated_dyn', 20, 'N'),
    ('gate_signals', 3, 'N'), ('gate_signals_dyn', 9, 'N'),
    ('slow_quad_modbank', 108, 'O'), ('slow_quad_modbank_dyn', 144, 'O'),
    ('scattering', 233, 'P'),
]
RAW_DIM = sum(d for _, d, _ in GROUPS)

SECTIONS = {
    'A': 'Cochleagram core', 'B': 'Pitch', 'C': 'Spectral shape', 'D': 'Spectral flux',
    'E': '1-D temporal modulation', 'F': 'Spectro-temporal modulation (STMF)', 'G': 'Formants',
    'H': 'Slow context + lookahead', 'I': 'Pitch/voicing dynamics', 'J': 'Interactions (v3e)',
    'K': 'Voice quality (CPP, region flatness)', 'L': 'Spectral/cochleagram contrast',
    'M': 'Quadrature modulation bank', 'N': 'Voicing gates', 'O': 'Slow quadrature bank',
    'P': 'Wavelet scattering',
}

FAMILIES = {
    'cochleagram_context': ('Cochleagram core + slow context + lookahead', 'AH'),
    'modulation': ('Modulation filters (1-D, STMF, quadrature, slow quadrature)', 'EFMO'),
    'pitch_voicing': ('Pitch, voicing, voice quality', 'BIKN'),
    'shape_flux_formants': ('Spectral shape, flux, formants', 'CDG'),
    'contrast': ('Spectral/cochleagram contrast', 'L'),
    'interactions': ('Interpretable interactions (v3e)', 'J'),
    'scattering': ('Wavelet scattering', 'P'),
}


def group_ranges():
    out, o = {}, 0
    for name, d, _ in GROUPS:
        out[name] = (o, o + d)
        o += d
    assert o == RAW_DIM, o
    return out


def raw_cols(group_names) -> np.ndarray:
    rng = group_ranges()
    return np.concatenate([np.arange(*rng[g]) for g in group_names]).astype(np.int64)


def delayed(raw: np.ndarray) -> np.ndarray:
    """Raw column indices -> indices in the delay-major (d*RAW_DIM + j) design matrix."""
    raw = np.sort(np.asarray(raw, dtype=np.int64))
    return np.concatenate([d * RAW_DIM + raw for d in range(NDELAYS)])


def units():
    """{level: [(unit_key, label, [group names])]} for the three partition levels."""
    sec_groups = {s: [g for g, _, ss in GROUPS if ss == s] for s in SECTIONS}
    return {
        'family': [(k, lab, [g for s in secs for g in sec_groups[s]])
                   for k, (lab, secs) in FAMILIES.items()],
        'section': [(s, f'{s}. {lab}', sec_groups[s]) for s, lab in SECTIONS.items()],
        'group': [(g, g, [g]) for g, _, _ in GROUPS],
    }


def fit_list():
    """Deduplicated fits: [(fit_id, raw column array)]. fit_id is 'full', 'alone:<cols-key>'
    or 'drop:<cols-key>', where the key is the sorted group-name tuple joined by '+'."""
    all_groups = [g for g, _, _ in GROUPS]
    fits = {'full': raw_cols(all_groups)}
    for level, us in units().items():
        for _, _, gs in us:
            key = '+'.join(sorted(gs))
            fits.setdefault(f'alone:{key}', raw_cols(gs))
            rest = [g for g in all_groups if g not in gs]
            fits.setdefault(f'drop:{key}', raw_cols(rest))
    # largest first so round-robin task assignment balances eigh cost
    return sorted(fits.items(), key=lambda kv: -len(kv[1]))


def fit_key(groups) -> str:
    return '+'.join(sorted(groups))


if __name__ == '__main__':
    fl = fit_list()
    print(f'{len(fl)} unique fits; raw dim {RAW_DIM}')
    for lvl, us in units().items():
        print(lvl, len(us), sum(len(raw_cols(gs)) for _, _, gs in us))
