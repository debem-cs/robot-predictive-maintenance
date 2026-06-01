# Robot Predictive Maintenance — Season 2026

CentraleSupélec *Maintenance & Industry 4.0* Kaggle data challenge.
Detect a synthetic thermal fault ("abnormal temperature rise from a blockage")
injected into the temperature channel of 6 servo motors on an ArmPi FPV robot
arm, sampled at 10 Hz. Scored by **macro-averaged F1 across the 6 motors**.

## TL;DR

```bash
# from this directory, with the project .venv active
python src/run_pipeline.py
```
Produces `submission.csv` (14,157 rows, one 0/1 label per motor per timestamp).

**Kaggle history:** previous team `0.2425` → first version of this method
**`0.51`** → current version (hysteresis detection) **local macro-F1 ≈ 0.81**.
See [Honest caveat](#honest-caveat): local F1 is an optimistic upper bound, but
each design choice targets the train→test shift that sank the earlier attempts.

---

## What was wrong before, and the key insight

Two earlier pipelines (archived in [`docs/legacy/`](docs/legacy)):

| Attempt | Approach | What happened |
|---|---|---|
| `old_run_pipeline_classification.py` | SMOTE + RandomForest/HistGBR classifiers on engineered features | Looked great in CV, then predicted **≈0 faults** on the blind test set. The test faults are *subtler* than training, so the fixed tree thresholds never fired (covariate shift). |
| `old_run_pipeline_v2_regression.py` | Learned-twin regression, **absolute** residuals, hand-tuned thresholds | Over-corrected: flagged **24–55 %** of points to force recall, which destroyed precision → **0.2425**. |

The data analysis (in [`notebooks/`](notebooks) / reproducible from the scripts)
shows three facts that drive the redesign:

1. **The fault perturbs temperature only.** During a labelled fault, voltage and
   position stay normal; temperature rises ~3–8 °C as an asymmetric triangle
   (fast heat-up, slower cool-down). So a fault = a *positive, sustained bump* in
   temperature relative to what's expected.
2. **Severe, uninformative imbalance.** Only 7 of 23 training sequences contain
   any fault; one sequence (`20240325_155003`) is *entirely* faulty on motors 2
   & 4 with an inconsistent signature. Motors 3 and 5 have ~100 fault samples
   *total*. Class-balancing tricks (SMOTE) just amplified covariate shift.
3. **Temperature is integer-quantised (1 °C steps).** Motors 3 & 6 faults can be
   ~1 °C rises — a single quantisation step — so they sit near the detection
   floor. Expect those two motors to stay hard.

## The method

Per motor, anomaly detection on a **temperature residual**:

```
expected(t) = robust within-sequence baseline of temperature   (centred rolling median)
residual(t) = temperature(t) − expected(t)
```

The rolling median tracks slow legitimate warming but is barely moved by a short
fault pulse, so the residual isolates the injected anomaly. Detection uses
**hysteresis**, which exploits the fault's asymmetric-triangle morphology
(heat-up, then ~2× slower cool-down):

* a point **seeds** a fault when its residual is simultaneously `> k_high` robust
  σ above the sequence's normal level (per-sequence median/MAD z-score —
  **scale-invariant, so it adapts across the train→test gap**) **and** `> min_rise`
  °C above it (absolute floor that kills 1-LSB quantisation noise);
* the seed then **grows** over neighbours that clear the gentler `k_low`
  threshold, capturing the triangle's rising/falling edges that a single cutoff
  clips — this is where most of the recall gain over the first version comes
  from. Growth is gated on a confirmed seed, so precision is preserved.

Run-length rules then keep only sustained runs (`min_run`) and bridge short holes
(`gap`). Detection is **signed** — only temperature *rises* are faults. Setting
`k_low == k_high` disables growth (a plain single threshold); the calibrator
picks that automatically for motors where growth would over-extend (e.g. M4).

Crucially this is **purely within-sequence**: it never relies on a model
generalising from train to test, which is what made the previous attempts
collapse. It produces sane, non-zero predictions for every motor on the blind
test set (0.1–2.6 % flagged per motor).

### Handling the data problems
- **Outliers** — `preprocessing.clean_motor_df` blanks physically-impossible
  sensor glitches (voltage `-25689`, position `-32271`, temperature `258`) and
  *interpolates* across them (vs the old forward-fill, which froze stale values).
  The temperature clip (150 °C) sits far above any real fault (~75 °C) so genuine
  anomalies are never removed.
- **Imbalance** — sidestepped, not fought. The detector is unsupervised per
  sequence, so it needs no positive examples to *run*; labels are used only to
  *calibrate* the four knobs. The degenerate all-faulty and majority-fault
  sequences are excluded from calibration (they break the minority assumption and
  don't resemble the sparse test set) but the detector still runs on every test
  sequence.

### Calibration & local validation
`src/run_pipeline.py` grid-searches `(window, k_high, k_low, min_rise, min_run,
gap)` per motor to maximise F1 on the (minority-fault) training sequences:

| Motor | F1 | Precision | Recall | Note |
|------:|----:|----:|----:|---|
| 1 | 0.93 | 1.00 | 0.88 | base motor, clear faults |
| 2 | 0.93 | 1.00 | 0.86 | |
| 3 | 0.81 | 1.00 | 0.69 | hysteresis recovers the subtle ~1 °C ramps |
| 4 | 0.74 | 1.00 | 0.59 | single threshold (growth disabled) |
| 5 | 0.90 | 1.00 | 0.83 | |
| 6 | 0.52 | 0.61 | 0.45 | gripper, heterogeneous/weak faults |
| **mean** | **0.81** | | | (was 0.69 with the single-threshold version) |

#### Honest caveat
Most motors have only **one** fault sequence in training, so per-motor knob
tuning can overfit it; this 0.81 is an **upper bound**, not the expected Kaggle
score (the single-threshold version scored 0.69 locally / 0.51 on Kaggle). The
design choices (per-sequence normalisation, signed residual, hysteresis on the
fault morphology, no cross-sequence model) are what make it transfer better than
the prior attempts, more than the exact number.

## Repository layout

```
data_challenge/
├── README.md                 ← this file
├── submission.csv            ← latest generated submission
├── data/                     ← challenge data (flattened from the original double-nesting)
│   ├── training_data/        ← 23 sequences × 6 motor CSVs + Test conditions.xlsx
│   ├── testing_data/         ← 8 blind sequences + Test conditions.xlsx
│   ├── sample_submission.csv
│   └── raw_data_backup.zip   ← untouched original download
├── src/
│   ├── preprocessing.py      ← robust cleaning + (twin) feature engineering
│   ├── detector.py           ← residual detection + the chosen detrend baseline  ★ core
│   ├── digital_twin.py       ← learned-regression alternative (documented, not default)
│   ├── run_pipeline.py       ← calibrate + generate submission                   ★ entry point
│   ├── utility.py            ← course-provided data loaders / helpers
│   └── calibrated_params.json← per-motor knobs from the last run
├── notebooks/                ← exploratory + demo notebooks
└── docs/
    ├── briefing.pdf / season_2_report.md / briefing_extracted.txt
    └── legacy/               ← previous pipelines + the group-mate handoff (archived)
```

## Alternative considered: learned regression twin
`src/digital_twin.py` keeps the "predict temperature from voltage/motion, flag
the residual" learned model. It scores marginally lower out-of-fold **and
extrapolates poorly to unseen test trajectories** (its residual scale inflates
and buries real faults — e.g. motor 1 went to 0 predicted faults on test), so it
is not the default. The detrend baseline is the robust choice here.

## Ideas to push further
- **Full matched filter** — hysteresis already exploits the fault morphology;
  a template-correlation against the explicit 1:2 heat:cool triangle (a bank of
  templates at several durations) could squeeze more out of motors 3 & 6.
- **Load-aware suppression** — the synthetic fault leaves voltage/position
  unchanged, so a temperature rise *with* a matching load increase is more likely
  genuine. Gating flags on "rise unexplained by load" could lift precision (left
  out for now because it can't be validated without test labels and may regress).
- **Use `Test conditions.xlsx`** — payload / trajectory metadata could give
  condition-specific baselines.
- **Cross-validate calibration** when more labelled fault sequences are
  available, to remove the single-sequence overfitting caveat.
