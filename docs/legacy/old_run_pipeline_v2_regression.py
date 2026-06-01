#!/usr/bin/env python3
"""
V2 Pipeline: Uses regression-residual fault detection (FaultDetectReg)
as intended by the competition design.

Key insight: The fault is a synthetic temperature rise injected into
the temperature channel. A regression model that predicts normal
temperature will show large residuals during faults.
"""
import sys, os, copy, warnings, json, time
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(1, './kaggle_data_challenge/')

from utility import (
    read_all_test_data_from_path, extract_selected_feature,
    prepare_sliding_window, FaultDetectReg, cal_classification_perf
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.metrics import f1_score

# ─── Preprocessing (matches the demo in utility.py) ──────────────────
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
    # Bias compensation
    df['temperature'] = df['temperature'] - df['temperature'].iloc[0]
    df['voltage'] = df['voltage'] - df['voltage'].iloc[0]
    df['position'] = df['position'] - df['position'].iloc[0]

def engineer_features(df):
    signals = ['temperature', 'voltage', 'position']
    for m in range(1, 7):
        for sig in signals:
            col = f'data_motor_{m}_{sig}'
            if col not in df.columns:
                continue
            df[f'{col}_diff10'] = df.groupby('test_condition')[col].diff(10)
            df[f'{col}_diff30'] = df.groupby('test_condition')[col].diff(30)
            df[f'{col}_diff50'] = df.groupby('test_condition')[col].diff(50)
            for w in [10, 30, 50]:
                df[f'{col}_rmean{w}'] = df.groupby('test_condition')[col].transform(
                    lambda x: x.rolling(w, min_periods=1).mean())
                df[f'{col}_rstd{w}'] = df.groupby('test_condition')[col].transform(
                    lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        tcol = f'data_motor_{m}_temperature'
        df[f'{tcol}_diff2_1'] = df.groupby('test_condition')[tcol].diff(1).groupby(
            df['test_condition']).diff(1)
    for m1, m2 in [(1,4),(2,5),(3,6)]:
        c1 = f'data_motor_{m1}_temperature'
        c2 = f'data_motor_{m2}_temperature'
        df[f'temp_cross_{m1}_{m2}'] = df[c1] - df[c2]
    label_cols = [c for c in df.columns if c.endswith('_label')]
    non_label = [c for c in df.columns if c not in label_cols]
    df[non_label] = df[non_label].fillna(0)
    return df

# ─── Load data ────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)
t0 = time.time()

base_train = './kaggle_data_challenge/kaggle_data_challenge/training_data/training_data/'
df_train = read_all_test_data_from_path(base_train, remove_outliers, is_plot=False)
df_train = engineer_features(df_train)
print(f'Training rows: {len(df_train)}, Time: {time.time()-t0:.1f}s')

base_test = './kaggle_data_challenge/kaggle_data_challenge/testing_data/'
df_test = read_all_test_data_from_path(base_test, remove_outliers, is_plot=False)
df_test = engineer_features(df_test)
print(f'Testing rows: {len(df_test)}')

# ─── Identify normal training sequences ──────────────────────────────
# From the briefing: 16 sequences are fully normal, 7 have faults.
# We train the regression model on NORMAL sequences only.
normal_test_ids = [
    '20240105_164214', '20240105_165300', '20240105_165972',
    '20240320_152031', '20240320_153841', '20240320_155664',
    '20240321_122650', '20240325_135213', '20240325_152902',
    '20240426_141190', '20240426_141532', '20240426_141602',
    '20240426_141726', '20240426_141938', '20240426_141980',
    '20240503_164435',
]

# We use ALL 18 features (position, temperature, voltage for each of 6 motors)
feature_list_all = [
    'time',
    'data_motor_1_position', 'data_motor_1_temperature', 'data_motor_1_voltage',
    'data_motor_2_position', 'data_motor_2_temperature', 'data_motor_2_voltage',
    'data_motor_3_position', 'data_motor_3_temperature', 'data_motor_3_voltage',
    'data_motor_4_position', 'data_motor_4_temperature', 'data_motor_4_voltage',
    'data_motor_5_position', 'data_motor_5_temperature', 'data_motor_5_voltage',
    'data_motor_6_position', 'data_motor_6_temperature', 'data_motor_6_voltage',
]

df_tr_normal = df_train[df_train['test_condition'].isin(normal_test_ids)]
print(f'\nNormal training rows: {len(df_tr_normal)} ({len(normal_test_ids)} sequences)')

# ─── Hyperparameter grid to search ───────────────────────────────────
# The FaultDetectReg approach has several knobs:
#   window_size, threshold, abnormal_limit, prediction_lead_time
# We also try different regression models.

# Regression models to try
reg_models = {
    'LinearRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', Ridge(alpha=1.0))
    ]),
    'HistGBR': Pipeline([
        ('scaler', StandardScaler()),
        ('reg', HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=10, random_state=42))
    ]),
}

# Hyperparameter configs to try
configs = [
    {'window_size': 5,  'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
    {'window_size': 10, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
    {'window_size': 10, 'threshold': 0.5, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
    {'window_size': 10, 'threshold': 1.0, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
    {'window_size': 10, 'threshold': 1.5, 'abnormal_limit': 5, 'pred_lead_time': 1, 'sample_step': 1},
    {'window_size': 20, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
]

# ─── Hardcoded best hyperparameters from previous grid search ──────
print("\n" + "=" * 60)
print("STEP 2: Using best models from previous run...")
print("=" * 60)

best_per_motor = {
    1: {'reg_name': 'Ridge', 'reg_mdl': reg_models['Ridge'], 
        'config': {'window_size': 20, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
        'f1': 0.5354, 'config_name': 'Ridge_w20_t0.8_a3'},
    2: {'reg_name': 'HistGBR', 'reg_mdl': reg_models['HistGBR'], 
        'config': {'window_size': 10, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
        'f1': 0.9387, 'config_name': 'HistGBR_w10_t0.8_a3'},
    3: {'reg_name': 'LinearRegression', 'reg_mdl': reg_models['LinearRegression'], 
        'config': {'window_size': 5, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
        'f1': 0.1338, 'config_name': 'LinearRegression_w5_t0.8_a3'},
    4: {'reg_name': 'HistGBR', 'reg_mdl': reg_models['HistGBR'], 
        'config': {'window_size': 10, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
        'f1': 0.6863, 'config_name': 'HistGBR_w10_t0.8_a3'},
    5: {'reg_name': 'HistGBR', 'reg_mdl': reg_models['HistGBR'], 
        'config': {'window_size': 10, 'threshold': 1.5, 'abnormal_limit': 5, 'pred_lead_time': 1, 'sample_step': 1},
        'f1': 0.1128, 'config_name': 'HistGBR_w10_t1.5_a5'},
    6: {'reg_name': 'Ridge', 'reg_mdl': reg_models['Ridge'], 
        'config': {'window_size': 20, 'threshold': 0.8, 'abnormal_limit': 3, 'pred_lead_time': 1, 'sample_step': 1},
        'f1': 0.7133, 'config_name': 'Ridge_w20_t0.8_a3'},
}

# ─── Overall validation summary ──────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
total_f1 = 0
for m in range(1, 7):
    info = best_per_motor[m]
    print(f"Motor {m}: {info['config_name']} -> F1={info['f1']:.4f}")
    total_f1 += info['f1']
print(f"\nMean F1 on faulty validation sequences: {total_f1/6:.4f}")

# ─── Generate submission ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Generating submission...")
print("=" * 60)

# Load sample submission
path_submission = './kaggle_data_challenge/kaggle_data_challenge/sample_submission.csv'
df_submission = pd.read_csv(path_submission)

# For each motor, train the best model on ALL normal data and predict test
for motor_idx in range(1, 7):
    info = best_per_motor[motor_idx]
    cfg = info['config']
    reg_name = info['reg_name']
    
    print(f"\nMotor {motor_idx} using {info['config_name']}...")
    
    # Train regression on normal training data
    x_tr_org, y_temp_tr_org = extract_selected_feature(
        df_data=df_tr_normal, feature_list=feature_list_all,
        motor_idx=motor_idx, mdl_type='reg')
    
    x_tr, y_temp_tr = prepare_sliding_window(
        df_x=x_tr_org, y=y_temp_tr_org,
        window_size=cfg['window_size'],
        sample_step=cfg['sample_step'],
        prediction_lead_time=cfg['pred_lead_time'],
        mdl_type='reg')
    
    mdl_fitted = copy.deepcopy(info['reg_mdl'])
    mdl_fitted.fit(x_tr, y_temp_tr)
    
    # Create detector
    detector = FaultDetectReg(
        reg_mdl=mdl_fitted,
        pre_trained=True,
        threshold=cfg['threshold'],
        abnormal_limit=cfg['abnormal_limit'],
        window_size=cfg['window_size'],
        sample_step=cfg['sample_step'],
        pred_lead_time=cfg['pred_lead_time'],
    )
    
    # Extract test features
    x_test_org, y_temp_test_org = extract_selected_feature(
        df_data=df_test, feature_list=feature_list_all,
        motor_idx=motor_idx, mdl_type='reg')
    
    # Ensure it's float to avoid pandas typecast errors when setting predicted values
    y_temp_test_org = y_temp_test_org.astype(float)
    
    # Predict with complement_truncation=True to match input length
    y_label_pred, y_temp_pred = detector.predict(
        df_x_test=x_test_org, y_response_test=y_temp_test_org,
        complement_truncation=True)
    
    # The predictions correspond to df_test rows
    # Map them to the submission dataframe by test_condition
    col = f'data_motor_{motor_idx}_label'
    
    # Get test condition ordering from df_test
    test_conditions_order = df_test['test_condition'].unique().tolist()
    
    offset_pred = 0
    for tc in test_conditions_order:
        tc_mask_test = df_test['test_condition'] == tc
        tc_mask_sub = df_submission['test_condition'] == tc
        
        n_test = int(tc_mask_test.sum())
        n_sub = int(tc_mask_sub.sum())
        
        preds_tc = y_label_pred[offset_pred:offset_pred + n_test]
        sub_indices = df_submission.index[tc_mask_sub]
        
        # Both should have same length since complement_truncation=True
        # and the test data rows == submission rows per test_condition
        if n_test == n_sub:
            df_submission.loc[sub_indices, col] = np.array(preds_tc).astype(int)
        elif n_test < n_sub:
            # Align from end
            start = n_sub - n_test
            df_submission.loc[sub_indices[:start], col] = 0
            df_submission.loc[sub_indices[start:], col] = np.array(preds_tc).astype(int)
        else:
            df_submission.loc[sub_indices, col] = np.array(preds_tc[:n_sub]).astype(int)
        
        offset_pred += n_test
    
    # Report
    n_faults = int((df_submission[col] == 1).sum())
    n_neg = int((df_submission[col] == -1).sum())
    print(f"  Predicted {n_faults} faults, {n_neg} remaining -1s")

# Final validation
print("\n" + "=" * 60)
print("SUBMISSION VALIDATION")
print("=" * 60)
print(f"Shape: {df_submission.shape}")
print(f"Columns: {list(df_submission.columns)}")
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    vals = sorted(df_submission[col].unique())
    n_faults = int((df_submission[col] == 1).sum())
    n_neg = int((df_submission[col] == -1).sum())
    pct = n_faults / len(df_submission) * 100
    print(f"Motor {m}: values={vals}, faults={n_faults} ({pct:.2f}%), remaining -1={n_neg}")

# Verify no -1s remain
any_neg = False
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    if (df_submission[col] == -1).any():
        any_neg = True
        print(f"WARNING: Motor {m} still has -1 values!")

if not any_neg:
    print("\nAll -1 values replaced successfully.")

# Save
df_submission.to_csv('./submission.csv', index=False)
print(f'\nSubmission saved to ./submission.csv')
print("DONE!")
