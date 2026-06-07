# EXHAUSTIVE TECHNICAL SPECIFICATION & DATA DICTIONARY
**Subject:** Kaggle Competition - Robot Predictive Maintenance (Season 2026)
**Host Institution:** CentraleSupélec (Course: Maintenance and Industry 4.0)
**Document Version:** 2.0 (Expanded Architectural & Modeling Context)
**Intended Use:** High-density context provisioning for Large Language Models (LLMs), Automated ML Pipelines, and Data Science Engineering Teams.

---

## 1. PEDAGOGICAL & OPERATIONAL CONTEXT
This predictive maintenance challenge is the capstone evaluation for the "Maintenance and Industry 4.0" curriculum at CentraleSupélec. It bridges the gap between theoretical machine learning and practical, hardware-in-the-loop industrial applications. 

The core paradigm is **Condition-Based Maintenance (CBM)** transitioning into **Predictive Maintenance (PdM)**. Instead of relying on scheduled maintenance (which is inefficient) or reactive maintenance (which is costly), the goal is to use real-time telemetry to detect the *onset* of a failure mechanism—specifically, thermal runaway in servo motors—before catastrophic hardware failure occurs.

## 2. HARDWARE TOPOLOGY & KINEMATICS
The physical data-generating asset is the **ArmPi FPV** educational robot arm by Hiwonder. 
Understanding the hardware is critical for feature engineering:
* **Kinematic Chain:** The robot possesses 6 Degrees of Freedom (DoF). This means the movement and load are distributed across 6 individual serial servo motors. 
* **Base vs. End-Effector Load:** The motor at the base (Motor 1) supports the entire weight and inertia of the arm, while the motor at the gripper (Motor 6) only handles the payload. *Modeling Implication: The baseline temperature and current draw will vary significantly depending on the motor's position in the kinematic chain.*
* **Compute Node:** A Raspberry Pi 4B handles high-level processing.
* **Middleware:** ROS (Robot Operating System) Melodic is utilized. ROS operates on a Publisher/Subscriber model, meaning data is streamed asynchronously via ROS Topics.

## 3. DATA ACQUISITION & TELEMETRY PIPELINE
The raw data is bridged from the ROS environment to a structured format using a custom Matlab interface.
* **Sampling Frequency ($F_s$):** 10 Hz. 
* **Temporal Resolution:** $\Delta t = 0.1$ seconds. This is a relatively high frequency for thermal dynamics (which usually change slowly) but appropriate for capturing sudden spikes in electrical load (Voltage).
* **Data Structure:** The data is organized into multivariate time-series sets.

### 3.1. Feature Space Breakdown (Per Motor)
For every timestamp $t$, the system records the following for each of the 6 motors ($i \in \{1, 2, 3, 4, 5, 6\}$):
1.  **`Position_i` (Spatial/Angular State):** Represents the current angle or position command of the servo. Rapid changes in position indicate high acceleration, which correlates with higher power consumption.
2.  **`Temperature_i` (Thermal State):** The core diagnostic variable. Measured in internal units (likely Celsius, though unscaled in raw data). 
3.  **`Voltage_i` (Electrical Load):** Servos draw current proportional to the torque required. A drop in voltage or a spike in drawn power often precedes a temperature rise if the motor is working against an unexpected load.

*Total Feature Vector per timestamp:* 18 variables (3 features $\times$ 6 motors) + 1 Time index = 19 columns.

## 4. FAILURE PHYSICS & SYNTHETIC INJECTION PROTOCOL
In an industrial setting, a robotic actuator might fail due to bearing wear, gear stripping, or motor burnout. Here, the specific failure mode is **Abnormal Temperature Rise caused by physical blockage**.

* **The Physics:** If a servo is commanded to move to a position but is physically blocked, it enters a "stall" condition. The stall current ($I_{stall}$) is drawn continuously, leading to rapid Joule heating ($P = I^2R$) within the motor coils, causing a temperature spike.
* **The Simulation (Synthetic Injection):** To preserve the educational hardware, organizers synthetically injected this thermal anomaly into the dataset. 
* **Injection Algorithm Parameters:**
    * **$T_{start}$:** Randomly selected onset time of the blockage.
    * **$T_{end}$:** Randomly selected resolution time.
    * **$Temp_{peak}$:** Randomly selected maximum temperature threshold.
    * **The Thermodynamic Heuristic:** The synthetic curve dictates that the motor heats up twice as fast as it cools down. 
        * Heating Duration: $(T_{peak\_time} - T_{start}) = x$
        * Cooling Duration: $(T_{end} - T_{peak\_time}) = 2x$
        * *Modeling Implication:* The thermal decay curve has a longer tail than the attack curve. Models should look for asymmetric triangular spikes in the temperature time-series.

## 5. DATASET ARCHITECTURE & DIRECTORY TOPOGRAPHY
### 5.1. Training Environment (`train/`)
* Contains independent folders representing discrete operational runs (Tests).
* Each test contains continuous telemetry.
* **Ground Truth:** Includes a binary matrix of shape $(N_{timestamps}, 6)$, where `1` indicates the synthetic anomaly is active for that specific motor at that specific time.

### 5.2. Testing Environment (`test.zip`)
* Structurally identical to the training data.
* Contains the telemetry (Position, Temp, Voltage) but **strips the labels**.
* The goal is to infer the label matrix for these test files.

### 5.3. Contextual Metadata (`Test conditions.xlsx`)
* A critical auxiliary file containing qualitative or categorical variables about the test runs.
* May include factors like: Payload weight, ambient temperature, specific movement trajectories (e.g., pick-and-place vs. continuous rotation). 
* *Modeling Implication:* This data can be joined with the time-series data to create conditional baselines (e.g., expected temperature *given* a specific payload).

### 5.4. Submission Standard (`sample_submission.csv`)
* A flattened, strict tabular format mapping unique IDs (combining Test ID, Timestamp, and Motor ID) to the predicted binary label `[0, 1]`.

## 6. EVALUATION METRIC: THE AVERAGE F1 SCORE
The Kaggle leaderboard evaluates submissions using the **Macro-Averaged F1 Score** calculated across the 6 motors.

1.  **Why F1?** Anomaly detection is a classic imbalanced classification problem. 95%+ of the time, the robot is healthy (`Label = 0`). If a model lazily predicts `0` constantly, its Accuracy would be 95%, but it would be useless. F1 Score penalizes this.
2.  **The Math:** * $Precision = rac{True Positives}{True Positives + False Positives}$ (How many predicted failures were actual failures?)
    * $Recall = rac{True Positives}{True Positives + False Negatives}$ (How many actual failures did we successfully catch?)
    * $F1 = 2 	imes rac{Precision 	imes Recall}{Precision + Recall}$
3.  **Aggregation:** Kaggle computes $F1_1, F1_2, ... F1_6$ for each motor independently, then finalizes the score as $rac{1}{6} \sum_{i=1}^{6} F1_i$.

## 7. STRATEGIC MODELING CONSIDERATIONS (FOR PIPELINE DEVELOPMENT)
For an LLM or ML Engineer building the solution, the following steps are highly recommended:
1.  **Temporal Feature Engineering:** Raw values are less important than *rates of change*. Compute rolling means, rolling standard deviations, and the first derivatives (gradients) of the Temperature and Voltage variables over 1-second, 3-second, and 5-second windows.
2.  **Cross-Motor Correlation:** Do not model each motor in complete isolation. A spike in Motor 2 might cause a compensatory load in Motor 3. Feed the entire 18-feature vector into the model.
3.  **Algorithm Selection:**
    * *Baseline:* XGBoost or LightGBM using sliding-window engineered features.
    * *Advanced:* 1D Convolutional Neural Networks (1D-CNN) or Long Short-Term Memory (LSTM) networks, which natively handle the sequential nature of the 10Hz time-series data.

## 8. CALIBRATION CHANGELOG (imbalance-driven)

The dominant difficulty is **sequence-level class imbalance**. After dropping the
degenerate sequence `20240325_155003` (motors 2 & 4 labelled faulty for all 6652
rows, with an inconsistent signature — it has no usable normal baseline, so it is
excluded from calibration entirely), **motors 1–5 each have faults in only ONE
usable training sequence** (`20240426_140055`); motor 6 has 5.

### 8.1 Shared knobs + macro-F1 objective (`run_pipeline.py`)
Per-motor knob tuning on a single fault sequence overfits (the local→Kaggle
collapse). Mitigation, in `calibrate_group` / `_macro_grid`:
* **Motors 1–5 calibrate jointly** (`CALIB_GROUPS`) with one *shared* knob set,
  pooling 5 fault realisations instead of 1. Motor 6 keeps its own knobs.
* Selection maximises **macro F1** (mean of per-motor F1), matching the Kaggle
  metric. Pooled/micro F1 let the high-count motors dictate the threshold and
  silently zeroed Motor 3 (subtle ~1 °C ramps); macro averaging prevents that.
* Result on the real labelled faults: M1 0.93, M2 0.67, M3 0.81, M4 0.57,
  M5 0.73, M6 0.52 → **mean 0.71**. (Lower than the old per-motor 0.81, which was
  an overfit upper bound; this set is robust and motors 1–5 are no longer tuned
  to a single sequence.) Motor 6 leave-one-fault-sequence-out CV ≈ 0.22.

### 8.2 Synthetic augmentation — VALIDATED ON KAGGLE, ON by default (`augmentation.py`)
The competition's failure-injection routine (briefing §9) was ported to Python to
generate extra labelled faults in the fault-free sequences. This is
distribution-consistent: the whole dataset — **including the hidden test labels** —
is synthetic from the same routine, so the injected faults share the test set's
generator. Peaks are sampled at +3–8 °C (the README's observed real range), added
on top of the real noisy signal, over short spans.

**Kaggle result: macro-F1 0.51 → 0.75** with augmentation on (peaks +3–8 °C).

The decisive lesson is about *which local metric to trust*. There is only ONE real
training fault sequence for motors 1–5 (`20240426_140055`), and it is subtle and
unrepresentative — so per-motor F1 measured on it (0.48 augmented vs 0.71 without)
**under-states** test performance and pointed the wrong way. The **synthetic CV-F1
(~0.76) matched the Kaggle score (0.75) almost exactly**, because it is drawn from
the test's own generator. We now optimise against the synthetic CV-F1, the first
local metric on this project that tracks the leaderboard.

Caveat: a single leaderboard point, and the no-augment shared+macro config (§8.1)
has not itself been submitted, so the 0.51→0.75 gain is not yet attributed between
calibration (§8.1) and augmentation (§8.2). Submitting `submission_noaugment.csv`
would isolate it.

### 8.3 Offline tuning against a held-out generator-matched eval (`src/sweep.py`)
With a trustworthy local metric, knobs were tuned against a *held-out* synthetic
eval set drawn from the true generator (peaks `randi[2,50]`, disjoint seeds from
calibration). Findings:
* Broadening the calibration peak range beyond +3-8C to the full +2-50C does not
  change the chosen knobs but the optimum is robust.
* The rolling-baseline **window optimum for motors 1-5 is 500, not 400** (400 was
  the old grid edge); `src/window_test.py` confirms 500 beats 400/600/800/1000.
  Offline 6-motor macro-F1 improves 0.86 -> 0.87 (mostly M2/M3). The grid in
  `run_pipeline.py` was widened to include 500/600. Candidate `submission_w500.csv`
  differs from the 0.75 file in only ~1% of rows (a few extra M3/M4 detections).
* Note the offline eval (~0.86-0.87) sits above Kaggle (0.75): the synthetic eval
  is somewhat optimistic (span/position assumptions), so offline gains transfer
  only partially. Re-tuning the existing 6 knobs is essentially exhausted; the
  weakest motors offline are M3 and M6 (~0.80), which a shape-matched filter
  (the briefing's triangle) would target next.

**Kaggle confirmation:** W=500 scored **0.79** (vs 0.75 at W=400). The window
finding transferred to the leaderboard.

### 8.4 Additional labelled data + real validation (`data/additional_data/`)
The teacher supplied extra labelled training data (groups 1/6/7, 36 usable
sequences, 15 with faults). It **largely dissolves the imbalance**: motors 1-5 now
have 4-6 real fault sequences each (was 1) and M6 has 8. Loaded by
`run_pipeline.load_tree` (walks the tree; the per-group spreadsheets are named
inconsistently so we bypass them) and included by default (`--no-extra` to skip).

This finally enables **real held-out validation** (`src/eval_real.py`): scoring
knobs on the additional faults, which the calibration never saw.
* The window trend holds on REAL data: mean per-motor F1 W400=0.22 < W500=0.28 <
  W600=0.30; per-motor best windows run 500-800. So bigger window is robust, and
  W=600 is worth a Kaggle try.
* Reality check: the overfit ceiling on the additional faults is only ~0.38, far
  below the synthetic offline (0.86). The additional groups are different
  robots/conditions, so they are a much harder, more honest test. The Kaggle 0.79
  is higher partly because of the F1=1.0 empty-motor rule and because the hidden
  test resembles the original-robot distribution more than the other groups.

Calibrating on the combined real data (original + additional) is the principled
next step: it anchors the knobs on many real fault realisations per motor instead
of one, which is the actual fix for the imbalance.
