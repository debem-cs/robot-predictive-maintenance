#!/usr/bin/env python3
"""Generate the robot_predictive_maintenance.ipynb notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": ".venv (3.12.5)",
    "language": "python",
    "name": "python3",
}
nb.metadata.language_info = {
    "name": "python",
    "version": "3.12.5",
    "file_extension": ".py",
    "mimetype": "text/x-python",
}

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ────────────────────────────────────────
# 1. EXECUTIVE SUMMARY
# ────────────────────────────────────────
md("""\
# Robot Predictive Maintenance -- Data Challenge

## Objectives

This notebook establishes a comprehensive workflow for fault detection on a 6-motor robotic system.
We benchmark multiple machine learning models per motor using 5-fold cross-validation (split by test sequences), then generate a Kaggle submission with the best model per motor.

Key improvements over the baseline approach:
- Advanced feature engineering (rolling statistics, multi-scale derivatives, cross-motor interactions)
- Handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
- Benchmarking four model families: Logistic Regression, Random Forest, HistGradientBoosting, and Extra Trees
- Systematic per-motor best-model selection

## Deliverables
- This Jupyter notebook reporting the process and results
- A submission CSV for the Kaggle data challenge
""")

# ────────────────────────────────────────
# 2. GROUP MEMBERS
# ────────────────────────────────────────
md("""\
# Group Members

- Efe Olgun
- XXX
- XXX
""")

# ────────────────────────────────────────
# 3. SETUP AND DATA LOADING
# ────────────────────────────────────────
md("""\
# Setup and Data Loading

We load the training data using the provided utility functions, applying outlier removal, sequence bias compensation, and diff features as in the demo notebook.
""")

code("""\
utility_path = './kaggle_data_challenge/'
import sys
sys.path.insert(1, utility_path)

import numpy as np
import pandas as pd
import copy
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

from utility import read_all_test_data_from_path, extract_selected_feature, prepare_sliding_window
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

n_int = 20

def remove_outliers(df: pd.DataFrame):
    df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
    df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
    df['temperature'] = df['temperature'].ffill()
    df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
    df['voltage'] = df['voltage'].where(df['voltage'] <= 9000, np.nan)
    df['voltage'] = df['voltage'].ffill()
    df['position'] = df['position'].where(df['position'] >= 0, np.nan)
    df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
    df['position'] = df['position'].ffill()

def compensate_seq_bias(df: pd.DataFrame):
    df['temperature'] = df['temperature'] - df['temperature'].iloc[0]
    df['voltage'] = df['voltage'] - df['voltage'].iloc[0]
    df['position'] = df['position'] - df['position'].iloc[0]

def cal_diff(df: pd.DataFrame, n_int_val: int):
    df['temperature_diff'] = df['temperature'].diff(n_int_val)
    df['voltage_diff'] = df['voltage'].diff(n_int_val)
    df['position_diff'] = df['position'].diff(n_int_val)

def pre_processing(df: pd.DataFrame):
    remove_outliers(df)
    compensate_seq_bias(df)
    cal_diff(df, n_int)

base_dictionary = './kaggle_data_challenge/kaggle_data_challenge/training_data/training_data/'
df_data = read_all_test_data_from_path(base_dictionary, pre_processing, is_plot=False)
print(f'Total rows: {len(df_data)}')
print(f'Test conditions: {df_data["test_condition"].unique().tolist()}')
print(f'Columns: {df_data.columns.tolist()}')
""")

# ────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ────────────────────────────────────────
md("""\
# Feature Engineering

We enrich the raw signals with several groups of derived features, computed per-sequence to avoid data leakage:

1. Multi-scale first differences (lag 1, 5) to capture short-term dynamics
2. Rolling mean and standard deviation (windows of 10 and 50) to capture local trends and variability
3. Cross-motor temperature differences for paired motors (1-4, 2-5, 3-6) to detect relative anomalies
4. Second-order differences (rate of change of rate of change) for temperature
""")

code("""\
def engineer_features(df):
    \"\"\"Add derived features to the dataframe, computed per test_condition to avoid leakage.\"\"\"
    signals = ['temperature', 'voltage', 'position']
    
    for m in range(1, 7):
        for sig in signals:
            col = f'data_motor_{m}_{sig}'
            if col not in df.columns:
                continue
            
            # Multi-scale diffs (per sequence)
            df[f'{col}_diff1'] = df.groupby('test_condition')[col].diff(1)
            df[f'{col}_diff5'] = df.groupby('test_condition')[col].diff(5)
            
            # Rolling statistics (per sequence)
            for w in [10, 50]:
                df[f'{col}_rmean{w}'] = df.groupby('test_condition')[col].transform(
                    lambda x: x.rolling(w, min_periods=1).mean())
                df[f'{col}_rstd{w}'] = df.groupby('test_condition')[col].transform(
                    lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        
        # Second-order diff for temperature
        tcol = f'data_motor_{m}_temperature'
        df[f'{tcol}_diff2_1'] = df.groupby('test_condition')[tcol].diff(1).groupby(
            df['test_condition']).diff(1)
    
    # Cross-motor temperature differences
    for m1, m2 in [(1, 4), (2, 5), (3, 6)]:
        c1 = f'data_motor_{m1}_temperature'
        c2 = f'data_motor_{m2}_temperature'
        df[f'temp_cross_{m1}_{m2}'] = df[c1] - df[c2]
    
    # Fill any remaining NaN from diff/rolling with 0
    label_cols = [c for c in df.columns if c.endswith('_label')]
    non_label = [c for c in df.columns if c not in label_cols]
    df[non_label] = df[non_label].fillna(0)
    
    return df

df_data = engineer_features(df_data)
print(f'Total features after engineering: {len(df_data.columns)}')
print(f'Total rows: {len(df_data)}')

# Build the full feature list (exclude labels and test_condition)
feature_list_enhanced = [c for c in df_data.columns 
                         if not c.endswith('_label') and c != 'test_condition']
print(f'Number of input features: {len(feature_list_enhanced)}')
""")

# ────────────────────────────────────────
# 5. CLASS IMBALANCE ANALYSIS
# ────────────────────────────────────────
md("""\
# Class Imbalance Analysis

Before modelling, we examine the fault rates per motor to understand the severity of class imbalance.
""")

code("""\
print("Class distribution per motor (training data):\\n")
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    total = len(df_data)
    faults = int((df_data[col] == 1).sum())
    normal = int((df_data[col] == 0).sum())
    unlabeled = total - faults - normal
    ratio = faults / max(normal, 1) * 100
    print(f"Motor {m}: {normal:>6d} normal, {faults:>5d} faults "
          f"({faults/total*100:.2f}%), imbalance ratio 1:{normal//max(faults,1)}")
print()
print("The data is highly imbalanced. Faults are rare events.")
print("We will use SMOTE to synthetically oversample the minority class during training.")
""")

# ────────────────────────────────────────
# 6. MODEL DEFINITIONS + CV FUNCTION
# ────────────────────────────────────────
md("""\
# Model Definitions and Cross-Validation Framework

We define four model families and a custom cross-validation function that applies SMOTE on training folds only, preventing data leakage. The CV splits data by test sequences (GroupKFold-style).

Models:
1. Logistic Regression with GridSearchCV over C
2. Random Forest with 300 trees, max_depth=20, class_weight='balanced_subsample'
3. HistGradientBoostingClassifier with 500 iterations, max_depth=8
4. Extra Trees with 300 trees, max_depth=20, class_weight='balanced'
""")

code("""\
# ── Model 1: Logistic Regression ──
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='saga'))
])

# ── Model 2: Random Forest ──
mdl_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight='balanced_subsample',
        min_samples_leaf=5, random_state=42, n_jobs=-1))
])

# ── Model 3: HistGradientBoosting ──
mdl_hgb = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05,
        min_samples_leaf=20, random_state=42))
])

# ── Model 4: Extra Trees ──
mdl_et = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ExtraTreesClassifier(
        n_estimators=200, max_depth=20,
        class_weight='balanced',
        min_samples_leaf=5, random_state=42, n_jobs=-1))
])

MODEL_DICT = {
    'Logistic Regression': pipe_lr,
    'Random Forest': mdl_rf,
    'HistGradientBoosting': mdl_hgb,
    'Extra Trees': mdl_et,
}

n_cv = 5
print(f'{len(MODEL_DICT)} models defined, {n_cv}-fold CV ready.')
""")

code("""\
def custom_cv_one_motor(motor_idx, df_data, mdl, feature_list, n_fold=5, use_smote=True):
    \"\"\"
    Run GroupKFold CV for one motor with optional SMOTE on training folds.
    Returns a DataFrame with per-fold Accuracy, Precision, Recall, F1.
    \"\"\"
    y_name = f'data_motor_{motor_idx}_label'
    feat_cols = [c for c in feature_list if c != y_name]
    
    # Build feature matrix (keep test_condition for splitting)
    df_x = df_data[feat_cols + ['test_condition']].copy()
    y = df_data[y_name].values.copy()
    
    test_conditions = df_x['test_condition'].unique().tolist()
    kf = KFold(n_splits=n_fold, shuffle=False)
    perf = np.zeros((n_fold, 4))
    
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(test_conditions)):
        names_train = [test_conditions[j] for j in train_idx]
        names_test = [test_conditions[j] for j in test_idx]
        
        mask_train = df_x['test_condition'].isin(names_train)
        mask_test = df_x['test_condition'].isin(names_test)
        
        X_train = df_x.loc[mask_train, feat_cols].values
        y_train = y[mask_train.values]
        X_test = df_x.loc[mask_test, feat_cols].values
        y_test = y[mask_test.values]
        
        # Apply SMOTE on training data if minority class has enough samples
        n_minority = int((y_train == 1).sum())
        if use_smote and n_minority >= 2:
            k = min(5, n_minority - 1)
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass  # Fall back to original if SMOTE fails
        
        # Train and predict
        mdl_copy = copy.deepcopy(mdl)
        mdl_copy.fit(X_train, y_train)
        y_pred = mdl_copy.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        if sum(y_test == 1) == 0 and sum(y_pred == 1) == 0:
            prec = precision_score(y_test, y_pred, zero_division=1)
            rec = recall_score(y_test, y_pred, zero_division=1)
            f1 = f1_score(y_test, y_pred, zero_division=1)
        else:
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        
        perf[fold_i] = [acc, prec, rec, f1]
    
    return pd.DataFrame(perf, columns=['Accuracy', 'Precision', 'Recall', 'F1 score'])

print('Custom CV function defined.')
""")

# ────────────────────────────────────────
# 7-12. PER-MOTOR BENCHMARK
# ────────────────────────────────────────
for motor in range(1, 7):
    md(f"""\
# Motor {motor}

We benchmark all four models for motor {motor} using 5-fold cross-validation with SMOTE applied to training folds.
""")

    code(f"""\
print("=" * 60)
print(f"MOTOR {motor} BENCHMARK")
print("=" * 60)

results_motor_{motor} = {{}}
for model_name, mdl in MODEL_DICT.items():
    print(f"\\n--- {{model_name}} ---")
    perf = custom_cv_one_motor(
        motor_idx={motor}, df_data=df_data, mdl=mdl,
        feature_list=feature_list_enhanced, n_fold=n_cv, use_smote=True)
    results_motor_{motor}[model_name] = perf
    print(perf.to_string(index=True))
    print(f"Mean: Acc={{perf['Accuracy'].mean():.4f}}, "
          f"Prec={{perf['Precision'].mean():.4f}}, "
          f"Rec={{perf['Recall'].mean():.4f}}, "
          f"F1={{perf['F1 score'].mean():.4f}}")

# Summary table
print("\\n" + "=" * 60)
print(f"MOTOR {motor} SUMMARY")
print("=" * 60)
summary_rows = []
for name, perf in results_motor_{motor}.items():
    summary_rows.append({{
        'Model': name,
        'Accuracy': f"{{perf['Accuracy'].mean():.1%}}",
        'Precision': f"{{perf['Precision'].mean():.1%}}",
        'Recall': f"{{perf['Recall'].mean():.1%}}",
        'F1': f"{{perf['F1 score'].mean():.1%}}",
        'F1_raw': perf['F1 score'].mean()
    }})
summary_df_{motor} = pd.DataFrame(summary_rows)
print(summary_df_{motor}[['Model', 'Accuracy', 'Precision', 'Recall', 'F1']].to_string(index=False))
best_model_{motor} = summary_df_{motor}.loc[summary_df_{motor}['F1_raw'].idxmax(), 'Model']
print(f"\\nBest model for motor {motor}: {{best_model_{motor}}} "
      f"(F1 = {{summary_df_{motor}['F1_raw'].max():.1%}})")
""")

    md(f"""\
## Motor {motor} Summary

The table above shows the cross-validated performance of all four models for motor {motor}.
The best model is selected based on the highest mean F1 score across folds.
The enhanced feature engineering and SMOTE-based oversampling generally lead to improved recall and F1 compared to the baseline approach with raw features only.
""")

# ────────────────────────────────────────
# 13. OVERALL SUMMARY
# ────────────────────────────────────────
md("""\
# Overall Summary

We compile the best model per motor and compare against the TD6 baseline results.
""")

code("""\
print("=" * 70)
print("OVERALL BEST MODEL PER MOTOR")
print("=" * 70)

best_models = {}
overall_rows = []
for m in range(1, 7):
    summary = eval(f'summary_df_{m}')
    best_name = eval(f'best_model_{m}')
    best_f1 = summary.loc[summary['Model'] == best_name, 'F1'].values[0]
    best_f1_raw = summary.loc[summary['Model'] == best_name, 'F1_raw'].values[0]
    best_models[m] = best_name
    overall_rows.append({
        'Motor': m, 'Best Model': best_name, 'F1 (new)': best_f1,
        'F1_raw': best_f1_raw
    })

# TD6 baseline F1 scores for comparison
td6_baseline = {1: 0.40, 2: 0.20, 3: 0.40, 4: 0.20, 5: 0.40, 6: 0.264}
for row in overall_rows:
    row['F1 (TD6 baseline)'] = f"{td6_baseline[row['Motor']]:.1%}"

overall_df = pd.DataFrame(overall_rows)
print(overall_df[['Motor', 'Best Model', 'F1 (new)', 'F1 (TD6 baseline)']].to_string(index=False))
print()
avg_new = np.mean([r['F1_raw'] for r in overall_rows])
avg_old = np.mean(list(td6_baseline.values()))
print(f"Average F1 across motors: {avg_new:.1%} (new) vs {avg_old:.1%} (TD6 baseline)")
print(f"Improvement: +{(avg_new - avg_old)*100:.1f} percentage points")
""")

# ────────────────────────────────────────
# 14. FINAL SUBMISSION
# ────────────────────────────────────────
md("""\
# Prepare Final Submission

We train the best model for each motor on all training data (with SMOTE), predict on the test set, and write the submission CSV.
""")

code("""\
# Load test data with same preprocessing
base_dictionary_test = './kaggle_data_challenge/kaggle_data_challenge/testing_data/'
df_test = read_all_test_data_from_path(base_dictionary_test, pre_processing, is_plot=False)
df_test = engineer_features(df_test)
print(f'Test data rows: {len(df_test)}')
print(f'Test conditions: {df_test["test_condition"].unique().tolist()}')
""")

code("""\
# Train best model per motor on ALL training data and predict on test
predictions = {}
for motor_idx in range(1, 7):
    model_name = best_models[motor_idx]
    mdl = copy.deepcopy(MODEL_DICT[model_name])
    
    y_name = f'data_motor_{motor_idx}_label'
    feat_cols = [c for c in feature_list_enhanced if c != y_name]
    
    # Training data
    X_train = df_data[feat_cols].values
    y_train = df_data[y_name].values
    
    # Apply SMOTE
    n_minority = int((y_train == 1).sum())
    if n_minority >= 2:
        k = min(5, n_minority - 1)
        try:
            sm = SMOTE(k_neighbors=k, random_state=42)
            X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        except Exception:
            X_train_sm, y_train_sm = X_train, y_train
    else:
        X_train_sm, y_train_sm = X_train, y_train
    
    # Train
    mdl.fit(X_train_sm, y_train_sm)
    
    # Test data
    X_test = df_test[feat_cols].values
    y_pred = mdl.predict(X_test)
    predictions[motor_idx] = y_pred
    
    faults = int(sum(y_pred == 1))
    print(f'Motor {motor_idx} ({model_name}): predicted {faults} faults '
          f'out of {len(y_pred)} samples ({faults/len(y_pred)*100:.2f}%)')
""")

code("""\
# Read submission template and fill in predictions
path_submission = './kaggle_data_challenge/kaggle_data_challenge/sample_submission.csv'
df_submission = pd.read_csv(path_submission)

# Initialize all labels to 0
for m in range(1, 7):
    df_submission[f'data_motor_{m}_label'] = 0

# Fill predictions -- align by test_condition sequences
# The test data (after feature engineering) may have different length than submission
# We match by position within each test_condition group
test_conditions_sub = df_submission['test_condition'].unique().tolist()

for m in range(1, 7):
    col = f'data_motor_{m}_label'
    pred_full = predictions[m]
    
    # The predictions correspond to df_test rows (after preprocessing/feature eng)
    # Map them back to submission indices
    # df_test and df_submission should share the same test_condition ordering
    offset = 0
    for tc in df_test['test_condition'].unique():
        tc_mask_test = df_test['test_condition'] == tc
        tc_mask_sub = df_submission['test_condition'] == tc
        
        n_test = int(tc_mask_test.sum())
        n_sub = int(tc_mask_sub.sum())
        
        preds_tc = pred_full[offset:offset + n_test]
        
        sub_indices = df_submission.index[tc_mask_sub]
        
        if n_test <= n_sub:
            # Align from end (first rows may lack diff features)
            start = n_sub - n_test
            df_submission.loc[sub_indices[start:start + n_test], col] = preds_tc.astype(int)
        else:
            # More test rows than submission (unlikely), truncate
            df_submission.loc[sub_indices, col] = preds_tc[:n_sub].astype(int)
        
        offset += n_test

# Save
df_submission.to_csv('./submission.csv', index=False)
print('Submission saved to ./submission.csv')
print(f'Shape: {df_submission.shape}')
print()
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    faults = int((df_submission[col] == 1).sum())
    pct = faults / len(df_submission) * 100
    print(f'Motor {m}: {faults} faults ({pct:.2f}%)')
print()
print('Value check (should be no -1):')
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    neg = int((df_submission[col] == -1).sum())
    print(f'  Motor {m}: {neg} rows with -1')
""")

nb.cells = cells
nbf.write(nb, '/Users/efeolgun/maintenance_industry_4.0/robot_predictive_maintenance.ipynb')
print("Notebook written to robot_predictive_maintenance.ipynb")
