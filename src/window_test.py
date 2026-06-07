#!/usr/bin/env python3
"""Quick check: does pushing the rolling-baseline window past 400 help on the
held-out generator-matched eval? (The sweep optimum sat at window=400, the grid
edge.)"""
from __future__ import annotations
import itertools, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import run_pipeline as rp
import detector as det
import sweep as sw

WINDOWS = [300, 400, 500, 600, 800, 1000]
GRID = {"window": WINDOWS, "k_high": [3.0, 3.5, 4.0, 5.0], "k_low": [1.0, 1.5, 2.0],
        "min_rise": [0.5, 1.0], "min_run": [3, 5], "gap": [3]}


def precompute(units, windows):
    return [{w: det.prep_residual(u["temp"], w) for w in windows} for u in units]


def main():
    df = rp.load(rp.TRAIN_DIR)
    for group in rp.CALIB_GROUPS:
        gname = "M" + "".join(map(str, group))
        norm = sw.normal_units(df, group)
        real = rp.calibration_units(df, group)
        eval_u = norm + sw.synth(df, group, sw.EVAL_SEEDS, **sw.EVAL_KW)
        eval_r = precompute(eval_u, WINDOWS)
        calib_u = real + sw.synth(df, group, sw.CALIB_SEEDS,
                                  peak_range=(2, 50), span_frac=(0.03, 0.25), n_per_seq=2)
        calib_r = precompute(calib_u, WINDOWS)
        best = sw.grid_search(calib_u, calib_r, list(range(len(calib_u))), GRID)
        ev = sw.macro_per_motor(eval_u, eval_r, list(range(len(eval_u))), best[0])
        print(f"{gname}: eval-F1={np.mean(list(ev.values())):.4f} "
              f"per-motor={[round(ev[m],3) for m in sorted(ev)]} knobs={best[0]}")


if __name__ == "__main__":
    main()
