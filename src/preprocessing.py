"""
Robust preprocessing + feature engineering for the robot predictive-maintenance
challenge.

Two responsibilities live here:

1. ``clean_motor_df``  - per-motor sensor cleaning.  The raw CSVs contain sensor
   glitches that are *physically impossible* (voltage = -25689, position =
   -32271, temperature = 258).  We blank those out and interpolate across them,
   then re-reference each signal to the start of the sequence so motors with
   different ambient baselines are comparable.  Crucially the temperature clip
   is set far above any genuine fault (real fault peaks ~75 deg) so the injected
   anomaly is **never** removed - only impossible glitches are.

2. ``engineer_twin_features`` - builds the load-history features the digital
   twin uses to predict each motor's temperature (rolling voltage / motion
   statistics, time, and cross-motor context).  The synthetic fault only
   perturbs the temperature channel, so these predictors stay clean during a
   fault and the model's residual exposes the injected rise.

The functions deliberately operate on the same wide dataframe layout produced by
``utility.read_all_test_data_from_path`` (columns ``data_motor_{m}_{signal}`` +
``time`` + ``test_condition``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Physically plausible ranges.  Anything outside is a sensor glitch.
# The upper temperature bound (150) sits well above the hottest genuine fault
# observed in training (~75 deg) so real anomalies survive cleaning.
VALID_RANGES = {
    "temperature": (10.0, 150.0),
    "voltage": (4000.0, 9000.0),
    "position": (-50.0, 1050.0),
}

SIGNALS = ("temperature", "voltage", "position")


def clean_motor_df(df: pd.DataFrame) -> None:
    """Clean a single motor's raw dataframe **in place**.

    Expects raw columns ``temperature``, ``voltage``, ``position`` (the format
    each ``data_motor_*.csv`` has before ``utility`` prefixes the columns).

    Steps per signal:
      * blank values outside the physical range (sensor glitches),
      * linearly interpolate across the gaps (much better than forward-fill for
        isolated spikes - it bridges instead of freezing a stale value),
      * back/forward fill any residual NaNs at the edges,
      * re-reference to a *robust* sequence start (median of the first valid
        samples) so a glitch on sample 0 can't corrupt the whole sequence.
    """
    for sig in SIGNALS:
        if sig not in df.columns:
            continue
        lo, hi = VALID_RANGES[sig]
        s = df[sig].where((df[sig] >= lo) & (df[sig] <= hi), np.nan)
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.ffill().bfill()
        df[sig] = s

    # Re-reference to the start of the sequence using a robust estimate of the
    # baseline (median of first 5 cleaned samples) so each motor starts at ~0
    # regardless of ambient temperature / sensor offset.
    for sig in SIGNALS:
        if sig not in df.columns:
            continue
        baseline = df[sig].iloc[:5].median()
        df[sig] = df[sig] - baseline


def _roll(group: pd.Series, window: int, fn: str) -> pd.Series:
    r = group.rolling(window, min_periods=1)
    out = getattr(r, fn)()
    return out.fillna(0.0)


def engineer_twin_features(df: pd.DataFrame,
                           windows=(20, 100, 300)) -> pd.DataFrame:
    """Add digital-twin predictor features for every motor, in place-ish.

    For motor ``m`` the temperature is driven by its *load history*, so we build
    rolling statistics of its voltage and of its motion (|position diff|) over a
    few time scales (2s / 10s / 30s at 10 Hz).  Instantaneous position / voltage
    of all motors are already present and provide cross-motor context.

    Returns the same dataframe with the new columns added.  All grouping is by
    ``test_condition`` so windows never leak across sequences.
    """
    g = df.groupby("test_condition", sort=False)
    for m in range(1, 7):
        vcol = f"data_motor_{m}_voltage"
        pcol = f"data_motor_{m}_position"
        if vcol not in df.columns:
            continue

        # Motion intensity: absolute position change per step.
        speed = g[pcol].diff().abs().fillna(0.0)
        df[f"data_motor_{m}_speed"] = speed
        speed_g = speed.groupby(df["test_condition"], sort=False)
        volt_g = g[vcol]

        for w in windows:
            df[f"data_motor_{m}_speed_rmean{w}"] = (
                speed_g.transform(lambda x: _roll(x, w, "mean")))
            df[f"data_motor_{m}_volt_rmean{w}"] = (
                volt_g.transform(lambda x: _roll(x, w, "mean")))
        # Short-window voltage volatility (electrical load fluctuation).
        df[f"data_motor_{m}_volt_rstd50"] = (
            volt_g.transform(lambda x: _roll(x, 50, "std")))

    # Fill any stray NaNs in non-label columns (rolling std edges, etc.).
    label_cols = [c for c in df.columns if c.endswith("_label")]
    non_label = [c for c in df.columns if c not in label_cols]
    df[non_label] = df[non_label].fillna(0.0)
    return df


def twin_feature_columns(motor_idx: int) -> list[str]:
    """Feature columns the twin uses to predict motor ``motor_idx`` temperature.

    Includes: time, every motor's instantaneous position & voltage (cross-motor
    context, all clean during a fault), and this motor's engineered load-history
    features.  Deliberately excludes *all* temperature columns so the model has
    to reconstruct temperature from the (fault-independent) electrical/motion
    signals - that is what makes the residual a clean fault detector.
    """
    cols = ["time"]
    for m in range(1, 7):
        cols += [f"data_motor_{m}_position", f"data_motor_{m}_voltage"]
    for w in (20, 100, 300):
        cols += [f"data_motor_{motor_idx}_speed_rmean{w}",
                 f"data_motor_{motor_idx}_volt_rmean{w}"]
    cols += [f"data_motor_{motor_idx}_speed",
             f"data_motor_{motor_idx}_volt_rstd50"]
    return cols
