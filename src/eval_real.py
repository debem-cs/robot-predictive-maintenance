#!/usr/bin/env python3
"""Validate knob choices on the REAL held-out additional faults.

The teacher's additional_data gives real fault sequences the calibration never
saw - the closest proxy to the hidden test we have. We score candidate knob sets
on it (per-motor pooled F1, Kaggle-style) to check whether the synthetic-tuned
W=500 finding holds on real unseen faults, and to find the real ceiling.
"""
from __future__ import annotations
import itertools, os, sys
import numpy as np
from sklearn.metrics import f1_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_pipeline as rp
import detector as det

WINDOWS = [150, 200, 300, 400, 500, 600, 800]


def precompute(units, windows):
    return [{w: det.prep_residual(u["temp"], w) for w in windows} for u in units]


def per_motor_f1(df, knobs_by_motor):
    out = {}
    for m in range(1, 7):
        units = rp.calibration_units(df, [m])
        if not units:
            out[m] = float("nan"); continue
        res = precompute(units, WINDOWS)
        truth = np.concatenate([u["truth"] for u in units])
        preds = np.concatenate([rp._predict_unit(res[i], knobs_by_motor[m])
                                for i in range(len(units))])
        out[m] = f1_score(truth, preds, zero_division=0)
    return out


def grid_ceiling(df, grid):
    """Per-motor best F1 achievable on this df (overfit upper bound)."""
    keys = list(grid); out = {}
    for m in range(1, 7):
        units = rp.calibration_units(df, [m])
        if not units:
            out[m] = (float("nan"), None); continue
        res = precompute(units, grid["window"])
        truth = np.concatenate([u["truth"] for u in units])
        best = None
        for combo in itertools.product(*grid.values()):
            p = dict(zip(keys, combo))
            if p["k_low"] > p["k_high"]:
                continue
            preds = np.concatenate([rp._predict_unit(res[i], p) for i in range(len(units))])
            f = f1_score(truth, preds, zero_division=0)
            if best is None or f > best[0]:
                best = (f, p)
        out[m] = best
    return out


def kn(window, k_high=4.0, k_low=1.0, min_rise=0.5, min_run=3, gap=3):
    return dict(window=window, k_high=k_high, k_low=k_low,
                min_rise=min_rise, min_run=min_run, gap=gap)


def main():
    df_extra = rp.load_tree(rp.EXTRA_DIR)
    print(f"extra: {len(df_extra)} rows, {df_extra['test_condition'].nunique()} sequences\n")

    # Shipped 0.79 knobs: M1-5 W=500, M6 W=300 (k_high=4 etc.)
    configs = {
        "W400 (old 0.75)": {m: kn(400 if m < 6 else 300) for m in range(1, 7)},
        "W500 (shipped 0.79)": {m: kn(500 if m < 6 else 300) for m in range(1, 7)},
        "W600": {m: kn(600 if m < 6 else 300) for m in range(1, 7)},
    }
    print("Per-motor F1 on REAL held-out additional faults:")
    print(f"  {'config':<22} " + " ".join(f"M{m}" for m in range(1, 7)) + "   mean")
    for name, knobs in configs.items():
        f = per_motor_f1(df_extra, knobs)
        mean = np.nanmean([f[m] for m in range(1, 7)])
        print(f"  {name:<22} " + " ".join(f"{f[m]:.2f}" for m in range(1, 7))
              + f"   {mean:.3f}")

    grid = {"window": WINDOWS, "k_high": [3.0, 3.5, 4.0, 5.0], "k_low": [1.0, 1.5, 2.0],
            "min_rise": [0.5, 1.0], "min_run": [3, 5], "gap": [3]}
    ceil = grid_ceiling(df_extra, grid)
    mean_c = np.nanmean([ceil[m][0] for m in range(1, 7)])
    print(f"\n  {'CEILING (fit on extra)':<22} "
          + " ".join(f"{ceil[m][0]:.2f}" for m in range(1, 7)) + f"   {mean_c:.3f}")
    print("\nPer-motor best window on real extra:")
    for m in range(1, 7):
        print(f"  M{m}: W={ceil[m][1]['window']} k_high={ceil[m][1]['k_high']} "
              f"k_low={ceil[m][1]['k_low']} rise={ceil[m][1]['min_rise']} "
              f"run={ceil[m][1]['min_run']}")


if __name__ == "__main__":
    main()
