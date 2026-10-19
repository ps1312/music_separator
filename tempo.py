#!/usr/bin/env python3
"""Lightweight tempo detection: spectral-flux onsets + autocorrelation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
N_FFT = 2048
HOP = 256
BLOCK = 512


def _decode_mono(path: Path) -> np.ndarray:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip()[-300:])
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < SAMPLE_RATE * 2:
        raise ValueError("Audio is too short to detect a tempo.")
    return audio


def _onset_envelope(audio: np.ndarray) -> np.ndarray | None:
    if audio.size < N_FFT:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP]
    n_frames = len(frames)
    if n_frames < 64:
        return None
    window = np.hanning(N_FFT).astype(np.float32)

    flux = np.zeros(n_frames, dtype=np.float32)
    previous = None
    for start in range(0, n_frames, BLOCK):
        chunk = frames[start : start + BLOCK] * window
        magnitude = np.log1p(np.abs(np.fft.rfft(chunk, axis=1)))
        if previous is not None:
            flux[start] = np.maximum(magnitude[0] - previous, 0).sum()
            if magnitude.shape[0] > 1:
                steps = np.maximum(np.diff(magnitude, axis=0), 0).sum(axis=1)
                flux[start + 1 : start + magnitude.shape[0]] = steps
        elif magnitude.shape[0] > 1:
            steps = np.maximum(np.diff(magnitude, axis=0), 0).sum(axis=1)
            flux[1 : magnitude.shape[0]] = steps
        previous = magnitude[-1]

    kernel = np.ones(16, dtype=np.float32) / 16
    baseline = np.convolve(flux, kernel, mode="same")
    envelope = np.maximum(flux - baseline, 0)
    if envelope.std() < 1e-8:
        return None
    return (envelope - envelope.mean()) / (envelope.std() + 1e-9)


def detect_bpm(path: Path) -> float | None:
    audio = _decode_mono(Path(path))
    envelope = _onset_envelope(audio)
    if envelope is None:
        return None

    n = len(envelope)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(envelope, size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    if autocorr[0] <= 0:
        return None
    autocorr /= autocorr[0]

    frame_rate = SAMPLE_RATE / HOP
    candidates = np.arange(50.0, 210.0, 0.1)
    lags = 60.0 * frame_rate / candidates
    valid = lags <= n - 2
    candidates, lags = candidates[valid], lags[valid]
    if not len(candidates):
        return None

    lower = np.floor(lags).astype(int)
    frac = lags - lower
    values = autocorr[lower] * (1 - frac) + autocorr[np.minimum(lower + 1, n - 1)] * frac

    # Broad prior around 120 BPM to resolve octave ambiguity.
    prior = np.exp(-0.5 * (np.log2(candidates / 120.0) / 0.8) ** 2)
    scores = values * prior
    best = int(np.argmax(scores))

    # Refine around the nearest autocorrelation peak with a parabolic fit.
    center = int(round(lags[best]))
    window = range(max(2, center - 2), min(n - 2, center + 2) + 1)
    peak = max(window, key=lambda index: autocorr[index])
    lag = float(peak)
    if 2 <= peak <= n - 3:
        y0, y1, y2 = autocorr[peak - 1], autocorr[peak], autocorr[peak + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag += float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    bpm = 60.0 * frame_rate / lag
    if bpm < 60:
        bpm *= 2
    elif bpm > 200:
        bpm /= 2
    return round(bpm, 1)
