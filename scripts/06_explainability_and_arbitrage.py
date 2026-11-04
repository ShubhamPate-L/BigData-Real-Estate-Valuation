"""
Phase 4: Enterprise Explainability & Arbitrage Detection Engine
- Module 1: Native TreeSHAP Global & Local Attribution (Feature Contributions)
- Module 2: Robust Anomaly-Filtered Arbitrage / Hidden Gem Detector (Novelty 3 Upgrade)
- Module 3: DuckDB Geospatial Comparable Properties (Comps) Engine
- Output Artifacts:
  - models/shap_feature_importance.csv
  - data/features/arbitrage_opportunities.parquet
  - data/features/arbitrage_top50.csv
"""
import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
import duckdb
import joblib

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

FEATURE_MATRIX_PATH = os.path.join("data", "features", "feature_matrix.parquet")
PROPERTIES_CLEANED_PATH = os.path.join("data", "properties", "properties_cleaned.parquet")
MODELS_DIR = "models"
OUTPUT_DIR = os.path.join("data", "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORICAL_FEATURES = ["property_type", "is_new_build", "tenure", "county"]

def run_treeshap_analysis(model, X_sample: pd.DataFrame, feature_names: list):
    """
    Computes exact TreeSHAP contributions using LightGBM's native C++ engine.
    Returns global importance ranking and local explanation matrix.
    """
    print("\n[2/5] Computing Native TreeSHAP Feature Attributions (10,000 Sample)...")
    t0 = time.time()
    
    # pred_contrib=True returns (N, num_features + 1) where last col is base value (expected value)
    shap_matrix = model.predict(X_sample, pred_contrib=True)
    feature_shaps = shap_matrix[:, :-1]
    base_value = float(shap_matrix[0, -1])
    
    # Mean Absolute SHAP Impact
    mean_abs_shap = np.mean(np.abs(feature_shaps), axis=0)
    
    # Correlation with feature values to determine directionality (+/-)
    correlations = []
    for i, col in enumerate(feature_names):
        try:
            if col in CATEGORICAL_FEATURES:
                corr = 0.0
            else:
                col_vals = X_sample[col].values.astype(float)
                c = np.corrcoef(col_vals, feature_shaps[:, i])[0, 1]
                corr = float(c) if not np.isnan(c) else 0.0
        except Exception:
            corr = 0.0
        correlations.append(corr)

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(mean_abs_shap, 5),
        "directional_correlation": np.round(correlations, 3)
    }).sort_values(by="mean_abs_shap", ascending=False)
    
    out_csv = os.path.join(MODELS_DIR, "shap_feature_importance.csv")
    shap_df.to_csv(out_csv, index=False)
    print(f"      -> TreeSHAP computed in {time.time()-t0:.2f}s across {len(X_sample):,} observations.")
    print(f"      -> Expected Base Value (Log Price): {base_value:.4f} (~GBP {np.expm1(base_value):,.0f})")
    print(f"      -> Saved SHAP importance to: {out_csv}")
    
    return shap_df, shap_matrix, base_value

def run_arbitrage_detector(df_test: pd.DataFrame, model_point, model_q10, model_q50, feature_names: list):
    """
    Enterprise Anomaly-Filtered Arbitrage Engine (Novelty 3).
    Identifies statistically confirmed undervalued properties.
    """
    print("\n[3/5] Running Anomaly-Filtered Arbitrage & Hidden Gem Detector...")
    t0 = time.time()
    
    X_test = df_test[feature_names].copy()
    
    # Model predictions
    log_pred_point = model_point.predict(X_test)
    log_pred_q10 = model_q10.predict(X_test)
    log_pred_q50 = model_q50.predict(X_test)
    
    pred_fair_price = np.expm1(log_pred_q50)
    pred_point_price = np.expm1(log_pred_point)
    pred_floor_price = np.expm1(log_pred_q10)
    
    actual_price = df_test["price"].values
    discount_gbp = pred_fair_price - actual_price
    discount_pct = (discount_gbp / pred_fair_price) * 100.0
    
    # Robust Anomaly and Liquidity Filters
    # 1. Price >= GBP 40,000 (filters nominal transfers & contaminated deeds)
    # 2. Price <= GBP 4,000,000 (filters ultra-luxury speculative anomalies)
    # 3. Micro-market sales volume >= 15 (guarantees local liquidity & reliable comps)
    # 4. Actual price strictly below 10th percentile floor (90% confidence underpriced)
    # 5. Discount >= 20.0% below fair value
    valid_mask = (
        (actual_price >= 40000) &
        (actual_price <= 4000000) &
        (df_test["h3_sales_volume"].values >= 15) &
        (actual_price < pred_floor_price) &
        (discount_pct >= 20.0)
    )
    
    arb_df = df_test[valid_mask].copy()
    arb_df["pred_fair_price"] = np.round(pred_fair_price[valid_mask], 0).astype(int)
    arb_df["pred_floor_q10"] = np.round(pred_floor_price[valid_mask], 0).astype(int)
    arb_df["discount_gbp"] = np.round(discount_gbp[valid_mask], 0).astype(int)
    arb_df["discount_pct"] = np.round(discount_pct[valid_mask], 2)
    
    # Arbitrage Opportunity Score (AOS):
    # Balances discount margin with low local volatility (IQR/Median) and high market liquidity
    dispersion = np.clip(arb_df["h3_iqr_price"].values / np.maximum(arb_df["h3_median_price"].values, 1), 0.0, 0.8)
    liquidity_weight = np.log10(1.0 + arb_df["h3_sales_volume"].values)
    arb_df["arbitrage_score"] = np.round(arb_df["discount_pct"].values * (1.0 - dispersion) * liquidity_weight, 2)
    
    # Sort opportunities by highest Arbitrage Opportunity Score
    arb_df = arb_df.sort_values(by="arbitrage_score", ascending=False)
    
    # Save full arbitrage opportunities dataset
    out_parquet = os.path.join(OUTPUT_DIR, "arbitrage_opportunities.parquet")
    arb_df.to_parquet(out_parquet, index=False, compression="snappy")
    
    # Save top 50 audited gems
    display_cols = [
        "transaction_id", "sale_date", "postcode", "street", "city", "county",
        "property_type", "is_new_build", "tenure", "price", "pred_floor_q10",
        "pred_fair_price", "discount_gbp", "discount_pct", "h3_sales_volume", "arbitrage_score"
    ]
    avail_cols = [c for c in display_cols if c in arb_df.columns]
    top50 = arb_df[avail_cols].head(50)
    out_csv = os.path.join(OUTPUT_DIR, "arbitrage_top50.csv")
    top50.to_csv(out_csv, index=False)
    
    print(f"      -> Identified {len(arb_df):,} verified arbitrage candidates ({len(arb_df)/len(df_test)*100:.2f}% of test set) in {time.time()-t0:.2f}s.")
    print(f"      -> Full dataset saved to: {out_parquet}")
    print(f"      -> Top 50 audited deals saved to: {out_csv}")
    
    return arb_df, top50

def run_comps_engine(target_transaction: dict, con: duckdb.DuckDBPyConnection, limit: int = 5):
    """
    Vectorized DuckDB Comparable Property (Comps) Search Engine.
    Queries verified historical sales in the exact same micro-market hexagon.
    """
    t0 = time.time()
    postcode = target_transaction["postcode"]
    prop_type = target_transaction["property_type"]
    tenure = target_transaction["tenure"]
    h3_hex = target_transaction["h3_res7"]
    lat = target_transaction["latitude"]
    lon = target_transaction["longitude"]
    target_id = target_transaction["transaction_id"]
    
    query = f"""
        SELECT 
            transaction_id,
            street,
            locality,
            city,
            postcode,
            price AS comp_price,
            sale_date,
            property_type,
            tenure,
            ROUND(2 * 6371 * ASIN(SQRT(
                POWER(SIN(RADIANS(latitude - {lat}) / 2), 2) +
                COS(RADIANS({lat})) * COS(RADIANS(latitude)) * POWER(SIN(RADIANS(longitude - {lon}) / 2), 2)
            )), 2) AS distance_km
        FROM properties
        WHERE h3_res7 = '{h3_hex}'
          AND property_type = '{prop_type}'
          AND tenure = '{tenure}'
          AND transaction_id != '{target_id}'
        ORDER BY distance_km ASC, sale_date DESC
        LIMIT {limit};
    """
    comps = con.execute(query).df()
    elapsed_ms = (time.time() - t0) * 1000.0
    return comps, elapsed_ms

def main():
    print("=" * 75)
    print("Phase 4: Enterprise Explainability & Arbitrage Detection Engine")
    print(f"Feature Source:    {FEATURE_MATRIX_PATH}")
    print(f"Properties Source: {PROPERTIES_CLEANED_PATH}")
    print("=" * 75)
    
    t_start = time.time()
    
    # 1. Load Trained Models & Test Data
    print("\n[1/5] Loading Production Models & Out-of-Time Test Set...")
    model_point = joblib.load(os.path.join(MODELS_DIR, "lgbm_point_regressor.joblib"))
    model_q10 = joblib.load(os.path.join(MODELS_DIR, "lgbm_quantile_q10.joblib"))
    model_q50 = joblib.load(os.path.join(MODELS_DIR, "lgbm_quantile_q50.joblib"))
    feature_names = model_point.feature_name_
    
    # Load out-of-time test set (H2 2023)
    df_all = pd.read_parquet(FEATURE_MATRIX_PATH)
    test_mask = (df_all["sale_year"] == 2023) & (df_all["sale_month"] > 6)
    df_test = df_all[test_mask].copy()
    
    for cat in CATEGORICAL_FEATURES:
        df_test[cat] = df_test[cat].astype("category")
        
    print(f"      -> Loaded {len(df_test):,} test observations (H2 2023).")
    print(f"      -> Active Features ({len(feature_names)}): {', '.join(feature_names[:6])}...")

    # 2. Module 1: Native TreeSHAP Analysis
    sample_size = min(10000, len(df_test))
    X_sample = df_test[feature_names].sample(n=sample_size, random_state=42)
    shap_df, shap_matrix, base_val = run_treeshap_analysis(model_point, X_sample, feature_names)

    # 3. Module 2: Robust Arbitrage Detection
    arb_df, top50 = run_arbitrage_detector(df_test, model_point, model_q10, model_q50, feature_names)

    # 4. Module 3: Comps Engine Demonstration on #1 Top Arbitrage Deal
    print("\n[4/5] Testing DuckDB Geospatial Comps Engine on Top Arbitrage Opportunity...")
    top_gem = top50.iloc[0].to_dict()
    # retrieve lat/lon from df_test
    top_gem["latitude"] = df_test.loc[df_test["transaction_id"] == top_gem["transaction_id"], "latitude"].values[0]
    top_gem["longitude"] = df_test.loc[df_test["transaction_id"] == top_gem["transaction_id"], "longitude"].values[0]
    top_gem["h3_res7"] = df_test.loc[df_test["transaction_id"] == top_gem["transaction_id"], "h3_res7"].values[0]

    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE VIEW properties AS SELECT * FROM read_parquet('{FEATURE_MATRIX_PATH.replace(chr(92), '/')}');")
    
    comps, comps_latency = run_comps_engine(top_gem, con, limit=5)
    
    print(f"      -> Comps Query Latency: {comps_latency:.2f} ms")
    print(f"\n      Target Opportunity:")
    print(f"         Address:      {top_gem['street']}, {top_gem['city']} ({top_gem['postcode']})")
    print(f"         Actual Sale:  GBP {top_gem['price']:,}")
    print(f"         Fair Model:   GBP {top_gem['pred_fair_price']:,} (Discount: {top_gem['discount_pct']:.1f}%)")
    print(f"         AOS Score:    {top_gem['arbitrage_score']}")
    print(f"\n      Top 5 Local Comparable Properties in Same Hexagon:")
    for idx, comp in comps.iterrows():
        print(f"         {idx+1}. {comp['street']} ({comp['postcode']}) | Distance: {comp['distance_km']} km | Sale Price: GBP {int(comp['comp_price']):,} | Date: {str(comp['sale_date'])[:10]}")

    # 5. Local SHAP Attribution for the Top Opportunity
    print("\n[5/5] Local TreeSHAP Explanation for Top Arbitrage Deal...")
    top_gem_features = df_test[df_test["transaction_id"] == top_gem["transaction_id"]][feature_names]
    local_contrib = model_point.predict(top_gem_features, pred_contrib=True)[0]
    local_shap = local_contrib[:-1]
    
    local_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": top_gem_features.iloc[0].values,
        "shap_log_impact": np.round(local_shap, 4)
    })
    
    top_positive = local_df.sort_values(by="shap_log_impact", ascending=False).head(4)
    top_negative = local_df.sort_values(by="shap_log_impact", ascending=True).head(4)
    
    print(f"      Key Drivers Pushing Valuation UP (Positive Premiums):")
    for _, row in top_positive.iterrows():
        print(f"         + {row['feature']:<28} = {str(row['feature_value']):<12} (SHAP: +{row['shap_log_impact']:.4f})")
    print(f"      Key Drivers Pushing Valuation DOWN (Negative Discounts):")
    for _, row in top_negative.iterrows():
        print(f"         - {row['feature']:<28} = {str(row['feature_value']):<12} (SHAP: {row['shap_log_impact']:.4f})")

    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print("PHASE 4 EXPLAINABILITY & ARBITRAGE ENGINE COMPLETE")
    print(f"  Total Pipeline Latency: {total_time:.2f}s")
    print("=" * 75)
    
    # Summary of Global SHAP Top 10
    print(f"\nTop 10 Global TreeSHAP Feature Attributions:")
    for rank, row in enumerate(shap_df.head(10).itertuples(), 1):
        direction = "Positive (+)" if row.directional_correlation > 0.1 else ("Negative (-)" if row.directional_correlation < -0.1 else "Non-linear / Mixed")
        print(f"  {rank:2d}. {row.feature:<30} | Mean |SHAP|: {row.mean_abs_shap:.4f} | Direction: {direction}")

if __name__ == "__main__":
    main()

