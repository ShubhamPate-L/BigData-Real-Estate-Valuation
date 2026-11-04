"""
Phase 3: Scalable Machine Learning Pipeline at Enterprise Scale
- Strict Temporal Train / Validation / Test Splitting (Out-of-Time 2022 vs 2023)
- Model 1: Baseline Hedonic Econometric Regressor (Ridge Regression)
- Model 2: Scalable Enterprise LightGBM Point Regressor (RMSE Objective)
- Model 3, 4, 5: Novelty 2 - Quantile Valuation Risk Bands (10th, 50th, 90th percentiles)
- Evaluation: R2, RMSE (GBP), MAE (GBP), MdAPE (%), Coverage Ratio
- Serialization: Saves trained model binaries & metrics to models/
"""
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

FEATURE_MATRIX_PATH = os.path.join("data", "features", "feature_matrix.parquet")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Select relevant engineered features (Strictly avoid target leakage)
NUMERICAL_FEATURES = [
    "latitude", "longitude",
    "h3_sales_volume", "h3_median_price", "h3_mean_price", "h3_price_std",
    "h3_p25_price", "h3_p75_price", "h3_iqr_price",
    "h3_detached_ratio", "h3_new_build_ratio", "h3_freehold_ratio",
    "dist_to_london_km", "dist_to_birmingham_km", "dist_to_manchester_km",
    "dist_to_leeds_km", "dist_to_bristol_km", "dist_to_nearest_major_city_km",
    "is_london_commuter_belt",
    "boe_base_rate",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_new_build_int", "is_freehold",
    "type_detached", "type_semi", "type_terraced", "type_flat", "type_other"
]

CATEGORICAL_FEATURES = ["property_type", "is_new_build", "tenure", "county"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def calculate_metrics(y_true_real: np.ndarray, y_pred_real: np.ndarray, y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Computes comprehensive valuation metrics in GBP and log scale"""
    r2 = r2_score(y_true_log, y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mae = mean_absolute_error(y_true_real, y_pred_real)
    mdape = np.median(np.abs(y_true_real - y_pred_real) / y_true_real) * 100.0
    
    return {
        "r2_log_scale": round(float(r2), 4),
        "rmse_gbp": round(float(rmse), 2),
        "mae_gbp": round(float(mae), 2),
        "mdape_pct": round(float(mdape), 2)
    }

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Computes pinball (quantile) loss"""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))

def main():
    print("=" * 75)
    print("Phase 3: Scalable Machine Learning Pipeline & Quantile Risk Bands")
    print(f"Feature Source: {FEATURE_MATRIX_PATH}")
    print(f"Artifacts Destination: {MODELS_DIR}/")
    print("=" * 75)
    
    t_start = time.time()
    
    # 1. Load Data
    print("\n[1/6] Loading Feature Matrix...")
    cols_to_load = ALL_FEATURES + ["sale_year", "sale_month", "price", "log_price"]
    df = pd.read_parquet(FEATURE_MATRIX_PATH, columns=cols_to_load)
    print(f"      -> Loaded {len(df):,} records with {len(ALL_FEATURES)} input features in {time.time()-t_start:.2f}s.")
    
    # Cast categoricals for LightGBM
    for cat_col in CATEGORICAL_FEATURES:
        df[cat_col] = df[cat_col].astype("category")

    # 2. Strict Temporal Split
    print("\n[2/6] Executing Temporal Train / Validation / Test Splitting...")
    train_mask = (df["sale_year"] == 2022)
    val_mask = (df["sale_year"] == 2023) & (df["sale_month"] <= 6)
    test_mask = (df["sale_year"] == 2023) & (df["sale_month"] > 6)
    
    X_train = df.loc[train_mask, ALL_FEATURES]
    y_train = df.loc[train_mask, "log_price"].values
    y_train_price = df.loc[train_mask, "price"].values
    
    X_val = df.loc[val_mask, ALL_FEATURES]
    y_val = df.loc[val_mask, "log_price"].values
    y_val_price = df.loc[val_mask, "price"].values
    
    X_test = df.loc[test_mask, ALL_FEATURES]
    y_test = df.loc[test_mask, "log_price"].values
    y_test_price = df.loc[test_mask, "price"].values
    
    print(f"      -> Train Set (2022 Entire Year):        {len(X_train):,} records ({len(X_train)/len(df)*100:.1f}%)")
    print(f"      -> Val Set (2023 Q1 & Q2):              {len(X_val):,} records ({len(X_val)/len(df)*100:.1f}%)")
    print(f"      -> Test Set (2023 Q3 & Q4 Out-of-Time): {len(X_test):,} records ({len(X_test)/len(df)*100:.1f}%)")
    
    # 3. Model 1: Baseline Hedonic Econometric Regressor (Ridge)
    print("\n[3/6] Training Baseline Hedonic Model (Ridge on Normalized Features)...")
    t0 = time.time()
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(df.loc[train_mask, NUMERICAL_FEATURES])
    X_test_num_scaled = scaler.transform(df.loc[test_mask, NUMERICAL_FEATURES])
    
    ridge_model = Ridge(alpha=100.0, random_state=42)
    ridge_model.fit(X_train_num_scaled, y_train)
    
    ridge_pred_log = ridge_model.predict(X_test_num_scaled)
    ridge_pred_price = np.expm1(ridge_pred_log)
    ridge_metrics = calculate_metrics(y_test_price, ridge_pred_price, y_test, ridge_pred_log)
    print(f"      -> Hedonic Baseline trained in {time.time()-t0:.2f}s.")
    print(f"         Test R2: {ridge_metrics['r2_log_scale']:.4f} | MAE: GBP {ridge_metrics['mae_gbp']:,.0f} | MdAPE: {ridge_metrics['mdape_pct']:.2f}%")
    
    joblib.dump({"model": ridge_model, "scaler": scaler}, os.path.join(MODELS_DIR, "hedonic_ridge.joblib"))

    # 4. Model 2: Enterprise LightGBM Point Regressor
    print("\n[4/6] Training Enterprise LightGBM Point Regressor (RMSE Objective)...")
    t0 = time.time()
    lgb_point = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=600,
        learning_rate=0.08,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_point.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    lgb_pred_log = lgb_point.predict(X_test)
    lgb_pred_price = np.expm1(lgb_pred_log)
    lgb_metrics = calculate_metrics(y_test_price, lgb_pred_price, y_test, lgb_pred_log)
    print(f"      -> LightGBM Regressor trained in {time.time()-t0:.2f}s (Best Iteration: {lgb_point.best_iteration_}).")
    print(f"         Test R2: {lgb_metrics['r2_log_scale']:.4f} | MAE: GBP {lgb_metrics['mae_gbp']:,.0f} | MdAPE: {lgb_metrics['mdape_pct']:.2f}%")
    
    joblib.dump(lgb_point, os.path.join(MODELS_DIR, "lgbm_point_regressor.joblib"))
    
    # Feature Importance
    importances_df = pd.DataFrame({
        "feature": ALL_FEATURES,
        "importance_gain": lgb_point.booster_.feature_importance(importance_type="gain"),
        "importance_split": lgb_point.booster_.feature_importance(importance_type="split")
    }).sort_values(by="importance_gain", ascending=False)
    importances_df.to_csv(os.path.join(MODELS_DIR, "feature_importances.csv"), index=False)

    # 5. Model 3, 4, 5: Quantile Risk Bands (Novelty 2)
    print("\n[5/6] Training Quantile LightGBM Risk Bands (10th, 50th, 90th Percentiles)...")
    quantiles = [0.10, 0.50, 0.90]
    quantile_models = {}
    quantile_preds_log = {}
    quantile_preds_price = {}
    quantile_metrics = {}

    for q in quantiles:
        t_q = time.time()
        print(f"      -> Training Quantile Model alpha={q:.2f}...")
        lgb_q = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=350,
            learning_rate=0.1,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_q.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)]
        )
        pred_log = lgb_q.predict(X_test)
        pred_price = np.expm1(pred_log)
        
        quantile_models[q] = lgb_q
        quantile_preds_log[q] = pred_log
        quantile_preds_price[q] = pred_price
        
        loss = pinball_loss(y_test, pred_log, q)
        quantile_metrics[f"pinball_loss_q{int(q*100)}"] = round(loss, 5)
        print(f"         alpha={q:.2f} trained in {time.time()-t_q:.2f}s | Pinball Loss: {loss:.5f}")
        
        joblib.dump(lgb_q, os.path.join(MODELS_DIR, f"lgbm_quantile_q{int(q*100)}.joblib"))

    # Valuation Interval Diagnostics (10th vs 90th percentile)
    q10_price = quantile_preds_price[0.10]
    q50_price = quantile_preds_price[0.50]
    q90_price = quantile_preds_price[0.90]
    
    in_interval = (y_test_price >= q10_price) & (y_test_price <= q90_price)
    coverage_rate = np.mean(in_interval) * 100.0
    avg_interval_width = np.mean(q90_price - q10_price)
    median_interval_width = np.median(q90_price - q10_price)
    
    q50_metrics = calculate_metrics(y_test_price, q50_price, y_test, quantile_preds_log[0.50])
    
    print(f"\n      -> Quantile Uncertainty Band Evaluation:")
    print(f"         Expected Coverage: 80.0% | Empirical Coverage: {coverage_rate:.2f}%")
    print(f"         Average 80% Confidence Interval Width: GBP {avg_interval_width:,.0f}")
    print(f"         Median 80% Confidence Interval Width:  GBP {median_interval_width:,.0f}")

    # 6. Summary Report & JSON Serialization
    print("\n[6/6] Generating Benchmark Audit & Serializing Evaluation Metrics...")
    
    summary = {
        "dataset_split": {
            "total_records": len(df),
            "train_records_2022": int(train_mask.sum()),
            "val_records_2023_h1": int(val_mask.sum()),
            "test_records_2023_h2": int(test_mask.sum())
        },
        "hedonic_baseline_ridge": ridge_metrics,
        "lightgbm_point_regressor": lgb_metrics,
        "lightgbm_quantile_median_q50": q50_metrics,
        "quantile_risk_bands": {
            **quantile_metrics,
            "expected_coverage_pct": 80.0,
            "empirical_coverage_pct": round(float(coverage_rate), 2),
            "avg_interval_width_gbp": round(float(avg_interval_width), 2),
            "median_interval_width_gbp": round(float(median_interval_width), 2)
        },
        "top_10_features_by_importance": importances_df.head(10)[["feature", "importance_gain"]].to_dict(orient="records")
    }
    
    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print("PHASE 3 SCALABLE ML PIPELINE COMPLETE")
    print(f"  Total Runtime: {total_time:.2f}s")
    print("=" * 75)
    
    # Formatted Comparison Table
    print(f"\n{'Model Evaluation Benchmark Comparison (Out-of-Time Test Set: Q3-Q4 2023)':^75}")
    print("-" * 75)
    print(f"{'Model Architecture':<30} | {'R2 (Log)':<10} | {'MAE (GBP)':<15} | {'MdAPE (%)':<10}")
    print("-" * 75)
    print(f"{'Baseline Hedonic (Ridge)':<30} | {ridge_metrics['r2_log_scale']:<10.4f} | GBP {ridge_metrics['mae_gbp']:<11,.0f} | {ridge_metrics['mdape_pct']:<9.2f}%")
    print(f"{'LightGBM Point Regressor':<30} | {lgb_metrics['r2_log_scale']:<10.4f} | GBP {lgb_metrics['mae_gbp']:<11,.0f} | {lgb_metrics['mdape_pct']:<9.2f}%")
    print(f"{'LightGBM Quantile Median (q50)':<30} | {q50_metrics['r2_log_scale']:<10.4f} | GBP {q50_metrics['mae_gbp']:<11,.0f} | {q50_metrics['mdape_pct']:<9.2f}%")
    print("-" * 75)
    
    print("\nTop 10 Most Predictive Features:")
    for rank, row in enumerate(importances_df.head(10).itertuples(), 1):
        print(f"  {rank:2d}. {row.feature:<32} (Gain: {row.importance_gain:,.1f})")

if __name__ == "__main__":
    main()
