"""
seizure_features.py
======================
Turns a raw signal (model output OR real EEG channel) into the same
small set of features in both cases, so simulated and real data can be
compared on equal footing. Also provides a simple, transparent
amplitude-threshold classifier for "ictal-like vs interictal-like"
epochs -- deliberately simple (not a black-box ML classifier) so you
can see exactly what decision rule is being applied and justify it in
a paper.
"""

import numpy as np
from scipy.signal import welch


def windowed_amplitude(signal, fs, window_sec=1.0, step_sec=0.5):
    """Sliding-window peak-to-peak amplitude -- the simplest possible
    'is this epoch big or small' feature."""
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    amps, centers = [], []
    for start in range(0, len(signal) - win, step):
        seg = signal[start:start + win]
        amps.append(seg.max() - seg.min())
        centers.append((start + win / 2) / fs)
    return np.array(centers), np.array(amps)


def band_power(signal, fs, low, high, nperseg=None):
    """Welch PSD, integrated over [low, high] Hz -- same function your
    original chb01 script used (power_band), so results are directly
    comparable to whatever you already computed on the real EEG."""
    nperseg = nperseg or min(2048, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(f >= low, f <= high)
    if idx.sum() < 2:
        return 0.0
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return trapz_fn(Pxx[idx], f[idx])


def dominant_frequency(signal, fs, fmax=30.0, nperseg=None):
    nperseg = nperseg or min(2048, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg)
    mask = f <= fmax
    return f[mask][np.argmax(Pxx[mask])]


def classify_epochs(signal, fs, window_sec=1.0, step_sec=0.5,
                     amplitude_threshold=None):
    """
    Returns (centers, amps, is_ictal_like) for a signal.

    If amplitude_threshold is None, it's set automatically as the
    midpoint between the signal's 10th and 90th amplitude percentiles
    -- a simple, data-driven default. In practice you should set this
    threshold from a calibration run (e.g. the midpoint between a
    known-resting simulation's amplitude and a known-oscillating one),
    not from the same signal you're about to classify.
    """
    centers, amps = windowed_amplitude(signal, fs, window_sec, step_sec)
    if amplitude_threshold is None:
        lo, hi = np.percentile(amps, [10, 90])
        amplitude_threshold = (lo + hi) / 2
    is_ictal_like = amps > amplitude_threshold
    return centers, amps, is_ictal_like, amplitude_threshold


def feature_summary(signal, fs, band=(3, 12)):
    """One-line feature summary dict -- use this to compare simulated
    vs. real segments side by side."""
    centers, amps = windowed_amplitude(signal, fs)
    return dict(
        mean_amplitude=float(np.mean(amps)),
        max_amplitude=float(np.max(amps)),
        band_power=float(band_power(signal, fs, *band)),
        dominant_freq_hz=float(dominant_frequency(signal, fs)),
    )
