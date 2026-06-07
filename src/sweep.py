#!/usr/bin/env python3
"""Offline, test-representative tuning for the augmented detector.

Now that the synthetic CV-F1 tracks the Kaggle leaderboard (0.75), we can tune
knobs against a synthetic *held-out* set instead of guessing. The competition's
generator draws peaks ``randi([2,50])``; our shipped 0.75 config calibrated on a
narrow +3-8C, which probably under-represents the test. This script:

  1. calibrates detection knobs under several candidate synthetic distributions
     (peak range / span / density) using a focused grid;
  2. scores each on the SAME held-out, generator-matched synthetic eval set
     (different seeds, peaks 2-50) - our best proxy for the hidden test;
  3. reports a 6-motor offline macro-F1 and ships the winner.

Calibration seeds and eval seeds are disjoint, so a config cannot win by
memorising its own synthetic draw.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_pipeline as rp  # noqa: E402
import augmentation as aug  # noqa: E402
import detector as det  # noqa: E402

# Focused grid around the known-good region (keeps the sweep affordable while we
# add a lot more synthetic data).
GRID_S = {
    "window": [150, 200, 300, 400],
    "k_high": [3.0, 3.5, 4.0, 5.0],
    "k_low": [1.0, 1.5, 2.0],
    "min_rise": [0.5, 1.0],
    "min_run": [3, 5],
    "gap": [3],
}

CALIB_SEEDS = (0, 1, 2)
EVAL_SEEDS = (50, 51, 52)
# Held-out eval = the true generator: peaks across the full randi[2,50] range,
# broad spans. This is the proxy for the hidden test.
EVAL_KW = dict(peak_range=(2, 50), span_frac=(0.03, 0.25), n_per_seq=2)

CANDIDATES = {
    "cur(3-8)":  dict(peak_range=(3, 8),  span_frac=(0.04, 0.15), n_per_seq=2),
    "mid(2-20)": dict(peak_range=(2, 20), span_frac=(0.03, 0.22), n_per_seq=2),
    "gen(2-50)": dict(peak_range=(2, 50), span_frac=(0.03, 0.25), n_per_seq=2),
    "gen-n4":    dict(peak_range=(2, 50), span_frac=(0.03, 0.25), n_per_seq=4),
}


def normal_units(df, group):
    return [u for u in rp.calibration_units(df, group) if u["truth"].sum() == 0]


def synth(df, group, seeds, **kw):
    units = []
    for s in seeds:
        units += aug.make_synthetic_fault_units(
            df, group, seed=s, drop_sequences=rp.DROP_SEQUENCES,
            max_fault_frac=rp.MAX_FAULT_FRAC_FOR_CALIB, **kw)
    return units


def grid_search(units, res, idx, grid):
    keys = list(grid)
    best = None
    for combo in itertools.product(*grid.values()):
        p = dict(zip(keys, combo))
        if p["k_low"] > p["k_high"]:
            continue
        s = rp._macro_objective(units, res, idx, p)
        if best is None or s > best[1]:
            best = (p, s)
    return best


def macro_per_motor(units, res, idx, params):
    """Mean across the group's motors of each motor's pooled F1 (Kaggle-style)."""
    from sklearn.metrics import f1_score
    by_motor = {}
    for i in idx:
        by_motor.setdefault(units[i]["motor"], []).append(i)
    pred = {i: rp._predict_unit(res[i], params) for i in idx}
    out = {}
    for m, ii in by_motor.items():
        truth = np.concatenate([units[i]["truth"] for i in ii])
        preds = np.concatenate([pred[i] for i in ii])
        out[m] = f1_score(truth, preds, zero_division=0)
    return out


def main():
    t0 = time.time()
    df = rp.load(rp.TRAIN_DIR)
    print(f"loaded {len(df)} rows ({time.time()-t0:.1f}s)\n")

    chosen = {}            # motor -> params
    per_motor_eval = {}    # motor -> eval F1 under chosen params
    for group in rp.CALIB_GROUPS:
        gname = "M" + "".join(map(str, group))
        base_norm = normal_units(df, group)
        real = rp.calibration_units(df, group)  # real faults + normal
        # Fixed held-out eval set for this group (generator-matched).
        eval_units = base_norm + synth(df, group, EVAL_SEEDS, **EVAL_KW)
        eval_res = rp._precompute_residuals(eval_units)
        eval_idx = list(range(len(eval_units)))

        print(f"=== group {gname} ===")
        results = []
        for name, kw in CANDIDATES.items():
            calib_units = real + synth(df, group, CALIB_SEEDS, **kw)
            calib_res = rp._precompute_residuals(calib_units)
            params, fit = grid_search(calib_units, calib_res,
                                      list(range(len(calib_units))), GRID_S)
            ev = macro_per_motor(eval_units, eval_res, eval_idx, params)
            ev_mean = float(np.mean(list(ev.values())))
            results.append((name, params, ev, ev_mean))
            print(f"  {name:9s} eval-F1={ev_mean:.4f} "
                  f"(per-motor {[f'{ev[m]:.2f}' for m in sorted(ev)]}) "
                  f"W={params['window']} kH={params['k_high']} kL={params['k_low']} "
                  f"rise={params['min_rise']} run={params['min_run']}")
        best = max(results, key=lambda r: r[3])
        print(f"  -> winner: {best[0]}  eval-F1={best[3]:.4f}\n")
        for m in group:
            chosen[m] = best[1]
        per_motor_eval.update(best[2])

    overall = float(np.mean([per_motor_eval[m] for m in range(1, 7)]))
    print("=" * 56)
    print("Offline 6-motor macro-F1 (held-out generator-matched eval): "
          f"{overall:.4f}")
    print("Per-motor:", {m: round(per_motor_eval[m], 3) for m in range(1, 7)})

    # Write submission with the chosen knobs.
    print("\nWriting submission with the chosen knobs ...")
    df_test = rp.load(rp.TEST_DIR)
    sub = pd.read_csv(rp.SAMPLE_SUB)
    for m in range(1, 7):
        col = f"data_motor_{m}_label"
        preds = det.detect_grouped(df_test, f"data_motor_{m}_temperature", chosen[m])
        for tc in df_test["test_condition"].unique():
            tmask = (df_test["test_condition"] == tc).to_numpy()
            sidx = sub.index[sub["test_condition"] == tc]
            p = preds[tmask]
            n = min(len(sidx), len(p))
            sub.loc[sidx[:n], col] = p[:n].astype(int)
        sub[col] = sub[col].replace(-1, 0).astype(int)
        print(f"  Motor {m}: {(sub[col]==1).sum()} faults "
              f"({(sub[col]==1).mean()*100:.1f}%)")
    out = os.path.normpath(os.path.join(HERE, "..", "submission_sweep.csv"))
    sub.to_csv(out, index=False)
    with open(os.path.join(HERE, "calibrated_params_sweep.json"), "w") as f:
        json.dump({str(k): v for k, v in chosen.items()}, f, indent=2)
    print(f"\nWrote {out}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
