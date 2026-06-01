# Instructions for Your Group Mate

You will find the following files in this directory to continue working on the Predictive Maintenance project:
1. **`run_pipeline_v2.py`**: The main regression-residual pipeline (Digital Twin approach) that achieved the baseline 0.245 score.
2. **`run_pipeline.py`**: The alternative classification baseline (uses SMOTE, Feature Engineering, and classifiers like Random Forest/HistGBR).
3. **`submission.csv`**: The generated submission file ready for Kaggle (from the V2 pipeline).
4. **`kaggle_data_challenge/utility.py`**: Contains the core logic, including the `FaultDetectReg` class utilized by the V2 pipeline.

*(Note: The `kaggle_data_challenge` folder is symlinked so you don't have to duplicate the massive data files!)*

## How to Run and Verify the Pipeline

To run the pipeline and regenerate the results, you should navigate to the project root directory, activate the Python virtual environment (to ensure libraries like `scikit-learn` and `pandas` are loaded), and execute the script:

```bash
# Navigate to the main project directory (one level up from this folder)
cd ..

# Activate the virtual environment
source .venv/bin/activate

# Run the primary regression pipeline
python run_pipeline_v2.py
```
This will train the models using the pre-configured parameters, output the F1 scores for each motor based on the local cross-validation, and generate a fresh `submission.csv`. You can verify it works by checking the terminal output for the validation summary and ensuring the final generated `submission.csv` contains exactly 14,157 rows with no `-1` values.

## Guidelines for Improving Results with Powerful Models

The current V2 pipeline uses lightweight regression models (Ridge and basic HistGradientBoosting) to build a "Digital Twin" of the normal motor physics. If you want to push for a much higher score, you should try the following advanced strategies:

### 1. Upgrade to Advanced Gradient Boosting
Replace the `HistGradientBoostingRegressor` and `Ridge` models in `run_pipeline_v2.py` with more powerful, robust implementations like **XGBoost (`xgboost`)** or **LightGBM (`lightgbm`)**.
- These models handle non-linear relationships in the position-temperature-voltage dynamics much better.
- *Implementation detail:* You'll need to `pip install xgboost lightgbm`, import them into the script, and add them to the `reg_models` dictionary inside `run_pipeline_v2.py`.

### 2. Temporal Feature Engineering in the Digital Twin
The classification baseline (`run_pipeline.py`) contains a function called `engineer_features()` which calculates rolling means, standard deviations, and gradients over 1-second, 3-second, and 5-second windows. 
- Try extracting this function and feeding those 208 engineered features directly into your new XGBoost/LightGBM regressors in `run_pipeline_v2.py`. 
- **Important:** If you use the massive engineered feature vector, make sure to set `window_size=1` in the configurations so the pipeline doesn't slow down trying to artificially concatenate thousands of features!

### 3. Hyperparameter Tuning
We are currently using hardcoded parameters (e.g., `threshold`, `abnormal_limit`) inside `best_per_motor`. Motors 3 and 5 are particularly tricky because their synthetic faults in the test set are very subtle (this is why they suffer from covariate shift). 
- If you switch to XGBoost, you will need to re-tune these thresholds. Lowering the anomaly threshold (e.g. from 0.8 to 0.4) will drastically increase the number of flagged faults and improve Recall.
