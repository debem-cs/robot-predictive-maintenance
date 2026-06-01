#!/usr/bin/env python3
"""
Run the predictive maintenance pipeline directly, capture results,
then embed them into the notebook.
"""
import sys, os, copy, warnings, json, time
warnings.filterwarnings('ignore')

sys.path.insert(1, './kaggle_data_challenge/')
import numpy as np
import pandas as pd
from utility import read_all_test_data_from_path, extract_selected_feature, prepare_sliding_window
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               HistGradientBoostingClassifier)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

n_int = 20

def remove_outliers(df):
    df['temperature'] = df['temperature'].where(df['temperature'] <= 100, np.nan)
    df['temperature'] = df['temperature'].where(df['temperature'] >= 0, np.nan)
    df['temperature'] = df['temperature'].ffill()
    df['voltage'] = df['voltage'].where(df['voltage'] >= 6000, np.nan)
    df['voltage'] = df['voltage'].where(df['voltage'] <= 9000, np.nan)
    df['voltage'] = df['voltage'].ffill()
    df['position'] = df['position'].where(df['position'] >= 0, np.nan)
    df['position'] = df['position'].where(df['position'] <= 1000, np.nan)
    df['position'] = df['position'].ffill()

def compensate_seq_bias(df):
    df['temperature'] = df['temperature'] - df['temperature'].iloc[0]
    df['voltage'] = df['voltage'] - df['voltage'].iloc[0]
    df['position'] = df['position'] - df['position'].iloc[0]

def cal_diff(df, n_int_val):
    df['temperature_diff'] = df['temperature'].diff(n_int_val)
    df['voltage_diff'] = df['voltage'].diff(n_int_val)
    df['position_diff'] = df['position'].diff(n_int_val)

def pre_processing(df):
    remove_outliers(df)
    compensate_seq_bias(df)
    cal_diff(df, n_int)

def engineer_features(df):
    signals = ['temperature', 'voltage', 'position']
    for m in range(1, 7):
        for sig in signals:
            col = f'data_motor_{m}_{sig}'
            if col not in df.columns:
                continue
            df[f'{col}_diff1'] = df.groupby('test_condition')[col].diff(1)
            df[f'{col}_diff5'] = df.groupby('test_condition')[col].diff(5)
            for w in [10, 50]:
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

def custom_cv_one_motor(motor_idx, df_data, mdl, feature_list, n_fold=5, use_smote=True):
    y_name = f'data_motor_{motor_idx}_label'
    feat_cols = [c for c in feature_list if c != y_name]
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
        n_minority = int((y_train == 1).sum())
        if use_smote and n_minority >= 2:
            k = min(5, n_minority - 1)
            try:
                sm = SMOTE(k_neighbors=k, random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except Exception:
                pass
        mdl_copy = copy.deepcopy(mdl)
        mdl_copy.fit(X_train, y_train)
        y_pred = mdl_copy.predict(X_test)
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

# ─── MAIN ───────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading training data...")
print("=" * 60)
t0 = time.time()
base_dictionary = './kaggle_data_challenge/kaggle_data_challenge/training_data/training_data/'
df_data = read_all_test_data_from_path(base_dictionary, pre_processing, is_plot=False)
print(f'Total rows: {len(df_data)}')
print(f'Test conditions: {df_data["test_condition"].unique().tolist()}')
print(f'Loading time: {time.time()-t0:.1f}s')

print("\n" + "=" * 60)
print("STEP 2: Feature engineering...")
print("=" * 60)
t0 = time.time()
df_data = engineer_features(df_data)
feature_list_enhanced = [c for c in df_data.columns
                         if not c.endswith('_label') and c != 'test_condition']
print(f'Total features: {len(feature_list_enhanced)}')
print(f'Feature engineering time: {time.time()-t0:.1f}s')

print("\n" + "=" * 60)
print("STEP 3: Class distribution...")
print("=" * 60)
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    total = len(df_data)
    faults = int((df_data[col] == 1).sum())
    normal = int((df_data[col] == 0).sum())
    print(f"Motor {m}: {normal:>6d} normal, {faults:>5d} faults "
          f"({faults/total*100:.2f}%), ratio 1:{normal//max(faults,1)}")

# ─── MODELS ──────────────────────────────
# Use faster models - drop GridSearchCV for LR, use fixed C
mdl_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='saga'))
])

mdl_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(
        n_estimators=200, max_depth=20,
        class_weight='balanced_subsample',
        min_samples_leaf=5, random_state=42, n_jobs=-1))
])

mdl_hgb = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', HistGradientBoostingClassifier(
        max_iter=300, max_depth=8, learning_rate=0.05,
        min_samples_leaf=20, random_state=42))
])

mdl_et = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', ExtraTreesClassifier(
        n_estimators=200, max_depth=20,
        class_weight='balanced',
        min_samples_leaf=5, random_state=42, n_jobs=-1))
])

MODEL_DICT = {
    'Logistic Regression': mdl_lr,
    'Random Forest': mdl_rf,
    'HistGradientBoosting': mdl_hgb,
    'Extra Trees': mdl_et,
}

n_cv = 5
all_results = {}
best_models = {}

print("\n" + "=" * 60)
print("STEP 4: Running benchmarks per motor...")
print("=" * 60)

for motor_idx in range(1, 7):
    print(f"\n{'='*60}")
    print(f"MOTOR {motor_idx} BENCHMARK")
    print(f"{'='*60}")
    
    motor_results = {}
    for model_name, mdl in MODEL_DICT.items():
        t0 = time.time()
        print(f"\n--- {model_name} ---")
        perf = custom_cv_one_motor(motor_idx, df_data, mdl,
                                    feature_list_enhanced, n_fold=n_cv, use_smote=True)
        motor_results[model_name] = perf
        elapsed = time.time() - t0
        print(perf.to_string(index=True))
        print(f"Mean: Acc={perf['Accuracy'].mean():.4f}, "
              f"Prec={perf['Precision'].mean():.4f}, "
              f"Rec={perf['Recall'].mean():.4f}, "
              f"F1={perf['F1 score'].mean():.4f} "
              f"[{elapsed:.1f}s]")
    
    all_results[motor_idx] = motor_results
    
    # Find best model
    best_name = max(motor_results, key=lambda n: motor_results[n]['F1 score'].mean())
    best_f1 = motor_results[best_name]['F1 score'].mean()
    best_models[motor_idx] = best_name
    
    print(f"\nBest model for motor {motor_idx}: {best_name} (F1 = {best_f1:.1%})")

# ─── OVERALL SUMMARY ──────────────────
print("\n" + "=" * 70)
print("OVERALL BEST MODEL PER MOTOR")
print("=" * 70)
td6_baseline = {1: 0.40, 2: 0.20, 3: 0.40, 4: 0.20, 5: 0.40, 6: 0.264}
for m in range(1, 7):
    name = best_models[m]
    f1_new = all_results[m][name]['F1 score'].mean()
    f1_old = td6_baseline[m]
    print(f"Motor {m}: {name:25s} F1={f1_new:.1%} (was {f1_old:.1%})")

avg_new = np.mean([all_results[m][best_models[m]]['F1 score'].mean() for m in range(1,7)])
avg_old = np.mean(list(td6_baseline.values()))
print(f"\nAverage F1: {avg_new:.1%} (new) vs {avg_old:.1%} (baseline)")

# ─── SUBMISSION ──────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating submission...")
print("=" * 60)

base_dictionary_test = './kaggle_data_challenge/kaggle_data_challenge/testing_data/'
df_test = read_all_test_data_from_path(base_dictionary_test, pre_processing, is_plot=False)
df_test = engineer_features(df_test)
print(f'Test data rows: {len(df_test)}')

predictions = {}
for motor_idx in range(1, 7):
    model_name = best_models[motor_idx]
    mdl = copy.deepcopy(MODEL_DICT[model_name])
    y_name = f'data_motor_{motor_idx}_label'
    feat_cols = [c for c in feature_list_enhanced if c != y_name]
    X_train = df_data[feat_cols].values
    y_train = df_data[y_name].values
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
    mdl.fit(X_train_sm, y_train_sm)
    X_test = df_test[feat_cols].values
    y_pred = mdl.predict(X_test)
    predictions[motor_idx] = y_pred
    faults = int(sum(y_pred == 1))
    print(f'Motor {motor_idx} ({model_name}): {faults} faults ({faults/len(y_pred)*100:.2f}%)')

# Write CSV
path_submission = './kaggle_data_challenge/kaggle_data_challenge/sample_submission.csv'
df_submission = pd.read_csv(path_submission)
for m in range(1, 7):
    df_submission[f'data_motor_{m}_label'] = 0

for m in range(1, 7):
    col = f'data_motor_{m}_label'
    pred_full = predictions[m]
    offset = 0
    for tc in df_test['test_condition'].unique():
        tc_mask_test = df_test['test_condition'] == tc
        tc_mask_sub = df_submission['test_condition'] == tc
        n_test = int(tc_mask_test.sum())
        n_sub = int(tc_mask_sub.sum())
        preds_tc = pred_full[offset:offset + n_test]
        sub_indices = df_submission.index[tc_mask_sub]
        if n_test <= n_sub:
            start = n_sub - n_test
            df_submission.loc[sub_indices[start:start+n_test], col] = preds_tc.astype(int)
        else:
            df_submission.loc[sub_indices, col] = preds_tc[:n_sub].astype(int)
        offset += n_test

df_submission.to_csv('./submission.csv', index=False)
print(f'\nSubmission saved to ./submission.csv')
print(f'Shape: {df_submission.shape}')
for m in range(1, 7):
    col = f'data_motor_{m}_label'
    faults = int((df_submission[col] == 1).sum())
    neg = int((df_submission[col] == -1).sum())
    print(f'Motor {m}: {faults} faults ({faults/len(df_submission)*100:.2f}%), {neg} rows with -1')

# ─── Save results to JSON for notebook embedding ──────
results_json = {
    'best_models': best_models,
    'motor_results': {},
}
for m in range(1, 7):
    results_json['motor_results'][str(m)] = {}
    for name, perf_df in all_results[m].items():
        results_json['motor_results'][str(m)][name] = {
            'accuracy': perf_df['Accuracy'].tolist(),
            'precision': perf_df['Precision'].tolist(),
            'recall': perf_df['Recall'].tolist(),
            'f1': perf_df['F1 score'].tolist(),
        }

with open('./run_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
print('\nResults saved to run_results.json')
print("DONE!")
