"""Floor baseline for the waveform track: reproduce a log-mel from the raw audio.

This front-end re-derives a mel-spectrogram-like feature directly from the aligned
audio windows (Hann -> rFFT -> mel filterbank -> dB), with NO learned/estimated
parameters. Ridge on it should land near the mel-track's linear baseline (~0.044
on the speech-rich split), which both gives the waveform loop a sensible floor and
serves as a sanity check that the waveform framing is aligned to the high-gamma.

Exposes the embedder interface: `.win` (analysis window) and
`__call__(windows (T, win)) -> np.ndarray (T, 50)`.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


class WaveMelReproEmbedder:
    """Fixed STFT -> mel -> dB front-end (the waveform 'reproduce the melspec' floor)."""

    n_params = 0
    win = 1024  # match the mel pipeline's n_fft

    def __init__(self, device: str = "cpu", n_mels: int = 50,
                 f_min: int = 20, f_max: int = 8000, top_db: float = 70.0):
        self.device = device
        self.top_db = top_db
        self.hann = torch.hann_window(self.win, device=device)
        self.mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.win // 2 + 1, f_min=f_min, f_max=f_max, n_mels=n_mels,
            sample_rate=44100, norm=None, mel_scale="htk").to(device)  # (n_freqs, n_mels)

    @torch.no_grad()
    def __call__(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.device)  # (T, win)
        X = torch.fft.rfft(x * self.hann, n=self.win)        # (T, win//2+1)
        power = X.abs() ** 2
        mel = power @ self.mel_fb                            # (T, n_mels)
        db = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        db = torch.maximum(db, db.max() - self.top_db)       # AmplitudeToDB top_db clamp
        return db.float().cpu().numpy()
