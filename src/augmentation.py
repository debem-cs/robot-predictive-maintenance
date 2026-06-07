"""Synthetic fault augmentation.

Ports the competition's MATLAB failure-injection routine (briefing section 9) to
Python so we can inject extra, *labelled* thermal faults into the fault-free
training sequences.

Why this is sound here: the entire dataset - including the held-out test labels -
is produced by this same routine, so synthetic faults are drawn from the SAME
distribution as the real ones. That makes augmentation a principled way to
enlarge the tiny positive class. It is what finally lets motors 1-5 (one real
fault sequence each) be cross-validated instead of overfit on a single example.

Scope: we use the synthetic faults ONLY to enrich CALIBRATION of the unsupervised
detector (labels tune the knobs). The detector never trains on this data, so the
covariate-shift trap that sank the supervised attempts does not apply.

Injected signature (the routine's asymmetric triangle): a 1-degree-step ramp up
over the first quarter of the span to a peak ``max(start,end)+delta``, then a
slower ramp down over the remaining three quarters. The briefing states only the
qualitative shape is reliable (the transcription is best-effort), so we reproduce
that shape rather than the exact MATLAB index arithmetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _triangle_bump(n: int, delta: int) -> np.ndarray:
    """A 0 -> delta -> 0 asymmetric triangle of length ``n`` (integer-quantised).

    Rises in 1-degree steps over the first quarter, decays over the remaining
    three quarters - the routine's fast-heat / slow-cool morphology.
    """
    n_rise = max(1, n // 4)
    bump = np.empty(n)
    rise = np.arange(1, n_rise + 1)
    bump[:n_rise] = np.floor(rise / n_rise * delta)
    n_down = n - n_rise
    decay = np.arange(1, n_down + 1)
    bump[n_rise:] = delta - np.floor(decay / n_down * delta)
    return bump


def inject_failure(temp: np.ndarray, lo: int, hi: int, delta: int) -> np.ndarray:
    """Return a copy of ``temp`` with a triangular thermal fault ADDED over
    ``[lo, hi)``.

    Unlike the MATLAB routine (which overwrites the span with a clean triangle),
    we ADD the bump on top of the real signal so the motor's own noise and drift
    are preserved under the fault. That keeps synthetic faults as *hard* as real
    ones - a clean overwrite is trivially detectable and detunes the calibrator
    (it learns thresholds that then over-flag the noisy real data).
    """
    out = temp.astype(float).copy()
    n = hi - lo
    if n < 8 or delta < 1:
        return out
    out[lo:hi] = out[lo:hi] + _triangle_bump(n, int(delta))
    return out


def make_synthetic_fault_units(df: pd.DataFrame, motor_indices, *,
                               n_per_seq: int = 2, seed: int = 0,
                               peak_range=(3, 8), span_frac=(0.04, 0.15),
                               len_range=(120, 2500),
                               drop_sequences=frozenset(),
                               max_fault_frac: float = 0.5) -> list:
    """One synthetic fault unit per (motor, fault-free sequence, repeat).

    Faults are injected only into sequences where the motor is normal, so each
    unit carries a known label span plus realistic surrounding normal signal.
    Peak amplitude is sampled low-ish on purpose (the blind-test faults are
    subtler than the big training ones) so the calibrated knobs are tuned to
    catch the hard cases. Units carry a ``"#syn"`` tag on the source sequence id
    so the caller can fold by source to avoid leakage.
    """
    rng = np.random.default_rng(seed)
    units = []
    for m in motor_indices:
        temp_col = f"data_motor_{m}_temperature"
        label_col = f"data_motor_{m}_label"
        for seq in df["test_condition"].unique():
            if seq in drop_sequences:
                continue
            mask = (df["test_condition"] == seq).to_numpy()
            truth = df.loc[mask, label_col].to_numpy().astype(int)
            # Only inject where this motor is entirely normal.
            if truth.sum() > 0 or truth.mean() > max_fault_frac:
                continue
            temp = df.loc[mask, temp_col].to_numpy()
            n = len(temp)
            if not (len_range[0] <= n <= len_range[1]):
                continue
            for _ in range(n_per_seq):
                L = max(40, int(rng.uniform(*span_frac) * n))
                if L >= n:
                    continue
                lo = int(rng.integers(0, n - L))
                delta = int(rng.integers(peak_range[0], peak_range[1] + 1))
                t2 = inject_failure(temp, lo, lo + L, delta)
                lab = np.zeros(n, dtype=int)
                lab[lo:lo + L] = 1
                units.append({"motor": m, "seq": f"{seq}#syn",
                              "temp": t2, "truth": lab})
    return units
