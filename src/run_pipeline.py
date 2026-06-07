#!/usr/bin/env python3
"""
Main pipeline for the Robot Predictive-Maintenance challenge (Season 2026).

Method: per-motor temperature-residual anomaly detection.  The synthetic fault
adds a localised, asymmetric thermal pulse to the temperature channel while
voltage/position stay normal.  We model each motor's *expected* normal
temperature with a robust within-sequence rolling-median baseline, then flag
sustained positive deviations (see ``detector`` for the full rationale).

Why this design:
  * The previous classification pipeline collapsed under covariate shift - the
    blind-test faults are subtler than training, so fixed tree thresholds fired
    almost never (near-zero faults predicted -> F1 ~ 0 on several motors).
  * The previous regression-v2 pipeline used ABSOLUTE residuals + hand-tuned
    magnitudes that over-flagged 24-55% of points, destroying precision (0.2425).
  * Here detection is (a) signed (faults are temperature *rises*), (b) normalised
    per-sequence so it adapts to each sequence's noise/scale, and (c) purely
    within-sequence so it never depends on a model generalising across the
    train/test gap.  Every knob is calibrated against held-out fault sequences.

Steps:
  1. Load + robustly clean every sequence (physical-range clip + interpolation).
  2. Calibrate per-motor detection knobs by maximising F1 on the (minority-fault)
     training sequences.
  3. Detect faults on the blind test set and write a Kaggle-ready submission.

Run from the ``data_challenge`` directory:
    python src/run_pipeline.py                # calibrate + write submission
    python src/run_pipeline.py --no-submission  # calibrate / report only
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import preprocessing as pp  # noqa: E402
import detector as det  # noqa: E402
import augmentation as aug  # noqa: E402
from utility import read_all_test_data_from_path  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
TRAIN_DIR = os.path.join(DATA, "training_data") + os.sep
TEST_DIR = os.path.join(DATA, "testing_data") + os.sep
EXTRA_DIR = os.path.join(DATA, "additional_data")  # extra labelled training data
SAMPLE_SUB = os.path.join(DATA, "sample_submission.csv")
OUT_SUB = os.path.normpath(os.path.join(HERE, "..", "submission.csv"))

# Degenerate sequence: motors 2 & 4 labelled faulty for ALL 6652 rows with an
# inconsistent signature (fault temp < normal temp).  No usable normal baseline.
DROP_SEQUENCES = {"20240325_155003"}

# The detector assumes faults are a MINORITY within a sequence (it estimates the
# normal level from the per-sequence median) and the blind test set is sparse.
# Sequences with more faults than this are excluded from CALIBRATION only - they
# are unrepresentative - but the calibrated detector is still run on every test
# sequence.
MAX_FAULT_FRAC_FOR_CALIB = 0.5

# Class imbalance is the dominant difficulty here, and it bites at the SEQUENCE
# level: after dropping the degenerate sequence, motors 1-5 each have a fault in
# only ONE usable training sequence (20240426_140055), so per-motor knob tuning
# fits a single fault realisation and overfits (this is the train->test collapse).
# We therefore calibrate motors 1-5 JOINTLY with a SHARED knob set, pooling their
# scarce positives into 5 fault realisations instead of 1. Motor 6 (the gripper)
# has 5 usable fault sequences and a distinct signature, so it keeps its own knobs.
CALIB_GROUPS = [(1, 2, 3, 4, 5), (6,)]

# Detection-knob grid (hysteresis). window = rolling-baseline window (@10Hz);
# k_high seeds a fault, k_low grows its ramp tails; min_rise = absolute degree
# floor; min_run/gap = run-length post-processing. k_low is floored at 1.0 to
# avoid growing into noise (and to limit overfitting on test).
GRID = {
    # 500/600 added after an offline generator-matched sweep found the M1-5
    # optimum at window=500 (the old max of 400 was the grid edge); see src/sweep.py.
    "window": [150, 200, 300, 400, 500, 600],
    "k_high": [3.0, 4.0, 5.0, 6.0],
    # k_low <= k_high; k_low == k_high disables growth (= plain single threshold),
    # which some motors prefer (growth can over-extend and cost precision).
    "k_low": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0],
    "min_rise": [0.5, 1.0, 1.5, 2.0],
    "min_run": [3, 5, 10],
    "gap": [3],
}
_GRID_KEYS = list(GRID.keys())


def load(path: str) -> pd.DataFrame:
    """Load + clean a data directory into the wide per-motor dataframe."""
    df = read_all_test_data_from_path(path, pp.clean_motor_df, is_plot=False)
    return df.reset_index(drop=True)


def load_tree(base: str) -> pd.DataFrame:
    """Load + clean every sequence folder found anywhere under ``base``.

    Used for the additional training data, whose sequence folders are nested
    inside per-group sub-directories with differently-named spreadsheets, so we
    walk the tree and read each sequence directly (bypassing the xlsx the course
    loader expects). Empty/short folders are skipped."""
    from utility import read_all_csvs_one_test
    frames = []
    for root, _dirs, files in os.walk(base):
        if "data_motor_1.csv" not in files:
            continue
        tid = os.path.basename(root)
        try:
            df = read_all_csvs_one_test(root, tid, pp.clean_motor_df)
        except (ValueError, KeyError):
            continue  # empty/malformed sequence folder
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_units(df: pd.DataFrame, motor_indices):
    """One ``(temperature, truth)`` calibration unit per (motor, sequence).

    Drops the degenerate sequence and any sequence whose fault fraction exceeds
    ``MAX_FAULT_FRAC_FOR_CALIB`` (those violate the minority assumption the
    detector relies on). All-normal units are KEPT - they constrain the
    false-positive rate, which is what protects the F1=1.0 score on fault-free
    motors. Tagging each unit with its motor lets us pool motors for a shared
    knob set while still scoring per-motor."""
    units = []
    for m in motor_indices:
        temp_col = f"data_motor_{m}_temperature"
        label_col = f"data_motor_{m}_label"
        for seq in df["test_condition"].unique():
            if seq in DROP_SEQUENCES:
                continue
            mask = (df["test_condition"] == seq).to_numpy()
            truth = df.loc[mask, label_col].to_numpy().astype(int)
            if truth.mean() > MAX_FAULT_FRAC_FOR_CALIB:
                continue
            units.append({
                "motor": m, "seq": seq,
                "temp": df.loc[mask, temp_col].to_numpy(),
                "truth": truth,
            })
    return units


def _precompute_residuals(units):
    """Cache each unit's residual AND its robust baseline for every grid window
    once. Both depend only on the window, not on the detection thresholds, so the
    grid search (thousands of threshold combos over the same residuals) skips all
    the repeated rolling-median + MAD work - the dominant cost of calibration."""
    return [{w: det.prep_residual(u["temp"], w) for w in GRID["window"]}
            for u in units]


def _predict_unit(res_by_window, params):
    rise, score = res_by_window[params["window"]]
    return det.detect_from_scored(
        rise, score, params["k_high"], params["k_low"], params["min_rise"],
        params["min_run"], params["gap"])


# ---------------------------------------------------------------------------
# Parallel grid evaluation (calibration is embarrassingly parallel over the
# threshold combos). Workers hold the per-unit residual/baseline/truth payload in
# module globals set by the pool initializer, so only a small (idx, params) task
# crosses the process boundary per combo.
# ---------------------------------------------------------------------------
_PAR_RES = _PAR_TRUTH = _PAR_MOTOR = None


def _par_init(res, truth, motor):
    global _PAR_RES, _PAR_TRUTH, _PAR_MOTOR
    _PAR_RES, _PAR_TRUTH, _PAR_MOTOR = res, truth, motor


def _eval_one(idx, by_motor, p) -> float:
    """Macro objective for one knob set, reading the payload from globals."""
    w, kh, kl = p["window"], p["k_high"], p["k_low"]
    mr, run, gap = p["min_rise"], p["min_run"], p["gap"]
    f1s = []
    for ii in by_motor.values():
        parts = []
        for i in ii:
            rise, score = _PAR_RES[i][w]
            parts.append(det.detect_from_scored(rise, score, kh, kl, mr, run, gap))
        preds = np.concatenate(parts)
        truth = np.concatenate([_PAR_TRUTH[i] for i in ii])
        f1s.append(f1_score(truth, preds, zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def _eval_chunk(task) -> list:
    """Evaluate a batch of knob sets over one ``idx`` - so ``idx`` crosses the
    process boundary once per worker, not once per combo."""
    idx, params_list = task
    by_motor: dict = {}
    for i in idx:
        by_motor.setdefault(_PAR_MOTOR[i], []).append(i)
    return [_eval_one(idx, by_motor, p) for p in params_list]


def _valid_combos():
    out = []
    for c in itertools.product(*GRID.values()):
        p = dict(zip(_GRID_KEYS, c))
        if p["k_low"] <= p["k_high"]:  # k_low is the gentler grow threshold
            out.append(p)
    return out


def _grid(idx, executor, jobs=1) -> tuple[dict, float]:
    """Best knobs over ``idx`` by the macro objective; parallel if ``executor``."""
    combos = _valid_combos()
    if executor is None:
        scores = _eval_chunk((idx, combos))
    else:
        n = max(1, jobs)
        k = -(-len(combos) // n)  # ceil: contiguous chunks preserve combo order
        chunks = [combos[i:i + k] for i in range(0, len(combos), k)]
        scores = [s for part in executor.map(_eval_chunk, [(idx, c) for c in chunks])
                  for s in part]
    best = int(np.argmax(scores))  # first max, matching the serial argmax
    return combos[best], float(scores[best])


def _base_seq(unit) -> str:
    """Source sequence id of a unit (strips the synthetic ``#syn`` tag)."""
    return unit["seq"].split("#", 1)[0]


def _macro_objective(units, res_cache, idx, params) -> float:
    """Mean across motors of that motor's POOLED F1 over the units in ``idx``.

    This is exactly the competition metric (mean F1 across motor columns).
    Pooling *within* a motor is now well-conditioned because augmentation gives
    each motor many fault spans spread over sequences, so no single sequence
    dominates; averaging *across* motors keeps a subtle motor (M3) from being
    silently traded away for the high-fault-count motors. Each unit is predicted
    once, so cost does not blow up with the number of synthetic faults."""
    by_motor: dict = {}
    for i in idx:
        by_motor.setdefault(units[i]["motor"], []).append(i)
    pred = {i: _predict_unit(res_cache[i], params) for i in idx}
    f1s = []
    for ii in by_motor.values():
        truth = np.concatenate([units[i]["truth"] for i in ii])
        preds = np.concatenate([pred[i] for i in ii])
        f1s.append(f1_score(truth, preds, zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def _score(units, res_cache, idx, params) -> dict:
    truth = np.concatenate([units[i]["truth"] for i in idx])
    preds = np.concatenate([_predict_unit(res_cache[i], params) for i in idx])
    return {
        "f1": f1_score(truth, preds, zero_division=0),
        "precision": precision_score(truth, preds, zero_division=0),
        "recall": recall_score(truth, preds, zero_division=0),
        "n_true": int(truth.sum()), "n_pred": int(preds.sum()),
    }


def calibrate_group(df: pd.DataFrame, motor_indices, augment=True,
                    n_folds=4, seed=0, jobs=1) -> tuple[dict, dict]:
    """Calibrate ONE shared knob set for a group of motors, enriched with
    synthetic faults.

    Real fault sequences (few) and all-normal sequences (precision context) come
    from the labelled data; ``augment`` adds synthetic faults injected into the
    fault-free sequences so motors 1-5 get many fault realisations instead of one.

    Final knobs maximise the macro objective over ALL units. The honest estimate
    is **grouped K-fold by source sequence**: hold out a fold of source sequences
    (their normal *and* injected versions together, so nothing leaks), tune on the
    rest, score the held-out fold. That ``cv_f1`` is the imbalance-aware number;
    ``fit_f1`` is the optimistic in-sample value.

    ``jobs`` > 1 fans the grid search out over a process pool."""
    units = calibration_units(df, motor_indices)
    if augment:
        units = units + aug.make_synthetic_fault_units(
            df, motor_indices, seed=seed, drop_sequences=DROP_SEQUENCES,
            max_fault_frac=MAX_FAULT_FRAC_FOR_CALIB)
    res = _precompute_residuals(units)
    truth = [u["truth"] for u in units]
    motor = [u["motor"] for u in units]
    all_idx = list(range(len(units)))

    # Group source sequences into folds for the honest leakage-free CV estimate.
    bases = sorted({_base_seq(u) for u in units})
    rng = np.random.default_rng(seed)
    rng.shuffle(bases)
    folds = []
    for f in range(n_folds):
        val_bases = set(bases[f::n_folds])
        val_idx = [i for i in all_idx if _base_seq(units[i]) in val_bases]
        tr_idx = [i for i in all_idx if _base_seq(units[i]) not in val_bases]
        has_fault = lambda ix: any(units[i]["truth"].sum() > 0 for i in ix)
        if has_fault(tr_idx) and has_fault(val_idx):
            folds.append((tr_idx, val_idx))

    _par_init(res, truth, motor)  # also enables the serial path in this process
    executor = (cf.ProcessPoolExecutor(max_workers=jobs, initializer=_par_init,
                                       initargs=(res, truth, motor))
                if jobs and jobs > 1 else None)
    try:
        cv = []
        for tr_idx, val_idx in folds:
            p_f, _ = _grid(tr_idx, executor, jobs)
            cv.append(_macro_objective(units, res, val_idx, p_f))
        params, fit = _grid(all_idx, executor, jobs)
    finally:
        if executor is not None:
            executor.shutdown()

    info = _score(units, res, all_idx, params)  # pooled metrics, for reference
    info["fit_f1"] = fit
    info["cv_f1"] = float(np.mean(cv)) if cv else float("nan")
    info["n_fault_units"] = sum(units[i]["truth"].sum() > 0 for i in all_idx)
    info["n_units"] = len(units)
    return params, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-submission", action="store_true",
                    help="calibrate / report only, skip writing submission")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable synthetic-fault augmentation (calibrate on the real "
                         "fault sequences only). Augmentation is ON by default: it "
                         "lifted the Kaggle macro-F1 from 0.51 to 0.75. See docs/season_2_report.")
    ap.add_argument("--no-extra", action="store_true",
                    help="ignore the additional labelled training data (data/additional_data). "
                         "Included by default - it supplies real multi-sequence faults for "
                         "motors 1-5, directly addressing the imbalance.")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="parallel worker processes for the grid search "
                         "(default: CPU count - 1).")
    args = ap.parse_args()
    augment = not args.no_augment

    t0 = time.time()
    print("=" * 64)
    print("Loading + cleaning training data ...")
    df_train = load(TRAIN_DIR)
    if not args.no_extra and os.path.isdir(EXTRA_DIR):
        df_extra = load_tree(EXTRA_DIR)
        df_train = pd.concat([df_train, df_extra]).reset_index(drop=True)
        print(f"  + {len(df_extra)} rows from additional_data "
              f"({df_extra['test_condition'].nunique()} sequences)")
    print(f"  {len(df_train)} rows, {df_train['test_condition'].nunique()} sequences "
          f"({time.time()-t0:.1f}s)")

    print("\n" + "=" * 64)
    print(f"Grouped calibration (shared knobs; augment={augment}; jobs={args.jobs}; "
          "CV honest estimate)")
    print("=" * 64)
    best_params, cv_f1s = {}, []
    for group in CALIB_GROUPS:
        params, info = calibrate_group(df_train, group, augment=augment,
                                       jobs=args.jobs)
        for m in group:
            best_params[m] = params
        cv_f1s.append(info["cv_f1"])
        gname = ("motors " + ",".join(map(str, group)) if len(group) > 1
                 else f"motor {group[0]}")
        print(f"[{gname}] fit-F1={info['fit_f1']:.4f} | CV-F1={info['cv_f1']:.4f} | "
              f"{info['n_fault_units']} fault units / {info['n_units']} total")
        print(f"    knobs: W={params['window']} k_high={params['k_high']} "
              f"k_low={params['k_low']} rise={params['min_rise']} "
              f"run={params['min_run']} gap={params['gap']}")

    # Per-motor breakdown on the REAL labelled faults only (compare vs old tuning).
    print("\nPer-motor F1 on the REAL labelled faults under the calibrated knobs:")
    per_motor = []
    for m in range(1, 7):
        units_m = calibration_units(df_train, [m])
        res_m = _precompute_residuals(units_m)
        info_m = _score(units_m, res_m, list(range(len(units_m))), best_params[m])
        per_motor.append(info_m["f1"])
        print(f"  Motor {m}: F1={info_m['f1']:.4f} "
              f"(P={info_m['precision']:.3f} R={info_m['recall']:.3f}) "
              f"true={info_m['n_true']} pred={info_m['n_pred']}")
    print(f"\nMean per-motor F1 on the single real fault sequence: {np.mean(per_motor):.4f}")
    if augment:
        print(f"Mean synthetic CV-F1 (Kaggle proxy): {np.nanmean(cv_f1s):.4f}")
        print("The one real fault sequence (20240426_140055) is subtle and "
              "unrepresentative, so the number above UNDER-states test performance. "
              "The synthetic CV-F1 is the trustworthy proxy: it shares the test set's "
              "generator, and this augmented config scored macro-F1 = 0.75 on Kaggle "
              "(vs 0.51 before).")
    else:
        print(f"Motor 6 leave-one-fault-sequence-out CV-F1: {cv_f1s[-1]:.4f}")
        print("This is the --no-augment path; motors 1-5 cannot be cross-validated on "
              "real data (one fault sequence each). The default (augmented) run gives a "
              "Kaggle-tracking CV signal and scored 0.75 vs this path's unknown score.")

    if args.no_submission:
        return

    print("\n" + "=" * 64)
    print("Detecting on test data + writing submission ...")
    df_test = load(TEST_DIR)
    sub = pd.read_csv(SAMPLE_SUB)
    for m in range(1, 7):
        col = f"data_motor_{m}_label"
        preds = det.detect_grouped(df_test, f"data_motor_{m}_temperature",
                                   best_params[m])
        for tc in df_test["test_condition"].unique():
            tmask = (df_test["test_condition"] == tc).to_numpy()
            sidx = sub.index[sub["test_condition"] == tc]
            p = preds[tmask]
            n = min(len(sidx), len(p))
            sub.loc[sidx[:n], col] = p[:n].astype(int)
        sub[col] = sub[col].replace(-1, 0).astype(int)
        n_fault = int((sub[col] == 1).sum())
        print(f"  Motor {m}: {n_fault} faults ({n_fault/len(sub)*100:.1f}%)")

    sub.to_csv(OUT_SUB, index=False)
    # persist calibrated params for reproducibility / the notebook
    with open(os.path.join(HERE, "calibrated_params.json"), "w") as f:
        json.dump({str(k): v for k, v in best_params.items()}, f, indent=2)
    print(f"\nSubmission written to {OUT_SUB}  ({sub.shape[0]} rows)")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
