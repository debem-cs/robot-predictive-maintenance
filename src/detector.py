"""
Residual-based fault detection - the heart of the solution.

A fault is an additive synthetic rise in the *temperature* channel (voltage and
position stay normal during a fault - verified on the training data).  So we
compute, per motor, an expected normal temperature and look at the residual

    r(t) = measured_temperature(t) - expected_temperature(t)

which is ~0 in normal operation and jumps POSITIVE during the injected anomaly
(an asymmetric triangle: fast rise, slow decay).

Two ways to get the "expected temperature" live in this codebase:

* ``detrend_residual`` (used by the main pipeline) - expected temperature is a
  robust *within-sequence* baseline (centred rolling median).  It needs no
  training and no cross-sequence generalisation, so it is immune to the
  train->test covariate shift that made the previous classification / learned-
  twin attempts collapse (they predicted ~0 faults on the blind test set).
* the learned regression twin in ``digital_twin.py`` - expected temperature is
  predicted from voltage/motion features.  Kept as a documented alternative.

Both feed the same detector below.  Detection uses hysteresis (a high threshold
to confirm a fault core, a low threshold to grow its ramp tails) with knobs
calibrated per motor:
  k_high   - sigmas above the per-sequence baseline to SEED a fault,
  k_low    - gentler sigmas to GROW the seed over the triangle's rising/falling
             edges (set == k_high for a plain single threshold),
  min_rise - absolute floor in degrees (kills sub-quantisation noise; the
             temperature sensor is integer-valued, so a 1 deg wobble is 1 LSB),
  min_run  - minimum consecutive flagged samples (faults are sustained),
  gap      - bridge short holes inside a fault region.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Robust per-sequence normalisation
# ---------------------------------------------------------------------------
def robust_baseline(r: np.ndarray, scale_floor: float = 0.5) -> tuple[float, float]:
    """Median and MAD-scale of a residual series, robust to the fault minority.

    ``scale_floor`` stops the scale collapsing to ~0 in very flat sequences
    (temperature is integer-quantised, so MAD is often exactly 0); without it a
    single 1-LSB wobble would score as a huge anomaly.
    """
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    scale = max(1.4826 * mad, scale_floor)
    return med, scale


# ---------------------------------------------------------------------------
# Run-length post-processing
#
# These run on the *handful of runs* in a sequence (via vectorised run-length
# encoding), not on every sample. Calibration calls them millions of times, so
# scanning only the O(#runs) transitions - not O(n) in a Python loop - is what
# keeps the grid search fast on hundreds of long sequences.
# ---------------------------------------------------------------------------
def _segments(mask: np.ndarray):
    """Start / end (exclusive) indices of every contiguous True run in ``mask``.

    Pads with False at both ends and reads transition positions, so a True run is
    a (start, end) pair of consecutive transitions - no per-sample Python loop and
    no int8/concatenate temporaries.
    """
    m = np.empty(mask.size + 2, dtype=bool)
    m[0] = m[-1] = False
    m[1:-1] = mask
    t = np.flatnonzero(m[1:] != m[:-1])
    return t[0::2], t[1::2]


def bridge_gaps(flag: np.ndarray, gap: int) -> np.ndarray:
    """Fill holes of <= ``gap`` zeros sitting between two flagged regions."""
    if gap <= 0:
        return flag
    flag = np.asarray(flag, dtype=bool)
    n = flag.size
    out = flag.copy()
    starts, ends = _segments(~flag)  # the False runs (holes)
    for a, b in zip(starts, ends):
        if a > 0 and b < n and (b - a) <= gap:  # interior hole only
            out[a:b] = True
    return out


def drop_short_runs(flag: np.ndarray, min_run: int) -> np.ndarray:
    """Remove flagged runs shorter than ``min_run`` (isolated noise spikes)."""
    if min_run <= 1:
        return flag
    flag = np.asarray(flag, dtype=bool)
    out = np.zeros_like(flag)
    starts, ends = _segments(flag)
    for a, b in zip(starts, ends):
        if (b - a) >= min_run:
            out[a:b] = True
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _grow_from_seeds(seed: np.ndarray, grow: np.ndarray) -> np.ndarray:
    """Hysteresis: keep every contiguous ``grow`` run that contains >=1 ``seed``.

    This is what lets us capture the *ramp tails* of the fault's asymmetric
    triangle (heat-up then ~2x slower cool-down). The strong core clears the high
    threshold (``seed``); the gentler rising/falling edges only clear the low
    threshold (``grow``) and would be missed by a single cutoff. Because growth
    is gated on a confirmed seed, precision is preserved - a region that never
    reaches the high threshold is never flagged.
    """
    flag = np.zeros(len(seed), dtype=bool)
    starts, ends = _segments(np.asarray(grow, dtype=bool))
    seed = np.asarray(seed, dtype=bool)
    for a, b in zip(starts, ends):
        if seed[a:b].any():
            flag[a:b] = True
    return flag


def detect_from_scored(rise: np.ndarray, score: np.ndarray, k_high: float,
                       k_low: float, min_rise: float, min_run: int,
                       gap: int) -> np.ndarray:
    """Hysteresis detection from PRECOMPUTED ``rise`` and ``score`` arrays.

    ``rise = residual - med`` and ``score = rise / scale`` depend only on the
    sequence + window, not on the thresholds - so the grid search computes them
    once per (sequence, window) and this fast path only does comparisons plus the
    vectorised run-length rules for each of the thousands of threshold combos.
    """
    seed = (score > k_high) & (rise > min_rise)
    grow = (score > k_low) & (rise > min_rise * 0.5)
    flag = _grow_from_seeds(seed, grow)
    flag = bridge_gaps(flag, gap)
    flag = drop_short_runs(flag, min_run)
    return flag.astype(int)


def detect_from_residual(residual: np.ndarray, k_high: float, k_low: float,
                         min_rise: float, min_run: int, gap: int,
                         scale_floor: float = 0.5, baseline=None) -> np.ndarray:
    """Turn one sequence's residual into a 0/1 fault label array (hysteresis).

    A point *seeds* a fault when it is simultaneously > ``k_high`` robust-sigmas
    AND > ``min_rise`` degrees above the sequence's normal residual level. The
    seed grows over neighbours clearing the gentler ``k_low`` / half ``min_rise``
    thresholds, then run-length rules apply. Set ``k_low == k_high`` for a plain
    single-threshold detector. Signed (positive-only): faults are temperature
    *rises*, so dips are never flagged. ``baseline`` may be a cached
    ``(med, scale)`` pair to skip the repeated median work.
    """
    if baseline is None:
        med, scale = robust_baseline(residual, scale_floor)
    else:
        med, scale = baseline
    rise = residual - med
    return detect_from_scored(rise, rise / scale, k_high, k_low, min_rise,
                              min_run, gap)


def prep_residual(temp: np.ndarray, window: int, scale_floor: float = 0.5):
    """Precompute ``(rise, score)`` for a sequence at one window.

    Cache this once per (sequence, window) and feed it to ``detect_from_scored``
    for every threshold combo in a grid search - removing the per-combo median,
    subtraction and division entirely."""
    residual = detrend_residual(temp, window)
    med, scale = robust_baseline(residual, scale_floor)
    rise = residual - med
    return rise, rise / scale


# ---------------------------------------------------------------------------
# The chosen "expected temperature" model: within-sequence robust baseline
# ---------------------------------------------------------------------------
def detrend_residual(temp: np.ndarray, window: int) -> np.ndarray:
    """Residual of temperature against its own centred rolling-median baseline.

    The rolling median tracks slow, legitimate warming/cooling but is barely
    moved by a short fault pulse, so the residual isolates the injected anomaly.
    Centred (uses both past and future) because we score whole sequences offline.
    """
    s = pd.Series(temp, dtype=float)
    base = s.rolling(window, center=True, min_periods=max(5, window // 4)).median()
    return (s - base).to_numpy()


def detect_sequence(temp: np.ndarray, window: int, k_high: float, k_low: float,
                    min_rise: float, min_run: int, gap: int) -> np.ndarray:
    """Full detrend-and-detect for a single sequence's temperature series."""
    return detect_from_residual(detrend_residual(temp, window),
                                k_high, k_low, min_rise, min_run, gap)


def detect_grouped(df: pd.DataFrame, temp_col: str, params: dict) -> np.ndarray:
    """Detect faults for many sequences stacked in ``df`` (grouped by sequence).

    Returns 0/1 labels aligned to ``df`` row order.
    """
    out = np.zeros(len(df), dtype=int)
    for _, idx in df.groupby("test_condition", sort=False).groups.items():
        pos = df.index.get_indexer(idx)
        out[pos] = detect_sequence(df.loc[idx, temp_col].to_numpy(), **params)
    return out
