"""
Phase 2: Advanced Feature Engineering at Scale
- Novelty 1 Upgrade: Uber H3 Hexagonal Micro-Market Spatial Indexing (Res 7 & Res 8)
- Geospatial Proximity to Major Economic Hubs (London, Birmingham, Manchester, etc.)
- Macroeconomic Integration: Official Bank of England Base Interest Rate Cycle
- Cyclical Temporal Features (Sine/Cosine month & day-of-week)
- Structural & Micro-Market Relative Ratios
- Target Variable Normalization (log_price)
- Output: Snappy-compressed data/features/feature_matrix.parquet
"""
import os
import time
import numpy as np
import pandas as pd
import duckdb
import h3

INPUT_PARQUET = os.path.join("data", "properties", "properties_cleaned.parquet").replace("\\", "/")
OUTPUT_DIR = os.path.join("data", "features")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "feature_matrix.parquet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Major UK Economic Hub Coordinates (Lat, Lon)
HUBS = {
    "london": (51.5074, -0.1278),
    "birmingham": (52.4862, -1.8904),
    "manchester": (53.4808, -2.2426),
    "leeds": (53.8008, -1.5491),
    "bristol": (51.4545, -2.5879)
}

# Official Bank of England Base Interest Rate (2022-2023)
BOE_RATES = {
    (2022, 1): 0.25, (2022, 2): 0.50, (2022, 3): 0.75, (2022, 4): 0.75,
    (2022, 5): 1.00, (2022, 6): 1.25, (2022, 7): 1.25, (2022, 8): 1.75,
    (2022, 9): 2.25, (2022, 10): 2.25, (2022, 11): 3.00, (2022, 12): 3.50,
    (2023, 1): 3.50, (2023, 2): 4.00, (2023, 3): 4.25, (2023, 4): 4.25,
    (2023, 5): 4.50, (2023, 6): 5.00, (2023, 7): 5.00, (2023, 8): 5.25,
    (2023, 9): 5.25, (2023, 10): 5.25, (2023, 11): 5.25, (2023, 12): 5.25
}

def haversine_vectorized(lats: np.ndarray, lons: np.ndarray, hub_lat: float, hub_lon: float) -> np.ndarray:
    """Vectorized Haversine distance in kilometers"""
    R = 6371.0  # Earth's radius in km
    dlat = np.radians(lats - hub_lat)
    dlon = np.radians(lons - hub_lon)
    a = (np.sin(dlat / 2.0) ** 2 + 
         np.cos(np.radians(hub_lat)) * np.cos(np.radians(lats)) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def main():
    print("=" * 70)
    print("Phase 2: Advanced Feature Engineering at Scale")
    print(f"Input:  {INPUT_PARQUET}")
    print(f"Output: {OUTPUT_PARQUET}")
    print("=" * 70)

    t0 = time.time()
    
    # 1. Load Cleaned Dataset into Pandas / Arrow
    print("\n[1/6] Loading Cleaned Properties Dataset...")
    df = pd.read_parquet(INPUT_PARQUET)
    n_rows = len(df)
    print(f"      -> Loaded {n_rows:,} records in {time.time()-t0:.2f}s.")

    # 2. Uber H3 Hexagonal Spatial Indexing
    print("\n[2/6] Computing Uber H3 Hexagonal Spatial Indexes (Res 7 & 8)...")
    t1 = time.time()
    lats = df['latitude'].values
    lons = df['longitude'].values
    
    # Fast list comprehension
    df['h3_res7'] = [h3.latlng_to_cell(lat, lon, 7) for lat, lon in zip(lats, lons)]
    df['h3_res8'] = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(lats, lons)]
    print(f"      -> Indexed {len(df['h3_res7'].unique()):,} Res-7 and {len(df['h3_res8'].unique()):,} Res-8 micro-markets in {time.time()-t1:.2f}s.")

    # 3. Micro-Market Aggregate Features via DuckDB
    print("\n[3/6] Calculating Hexagonal Micro-Market Metrics...")
    t2 = time.time()
    con = duckdb.connect()
    con.register("df_temp", df[['transaction_id', 'h3_res7', 'price', 'property_type', 'is_new_build', 'tenure']])
    
    h3_stats = con.execute("""
        SELECT 
            h3_res7,
            count(*) AS h3_sales_volume,
            round(median(price), 0)::BIGINT AS h3_median_price,
            round(avg(price), 0)::BIGINT AS h3_mean_price,
            round(stddev(price), 0)::BIGINT AS h3_price_std,
            round(quantile_cont(price, 0.25), 0)::BIGINT AS h3_p25_price,
            round(quantile_cont(price, 0.75), 0)::BIGINT AS h3_p75_price,
            round(count(CASE WHEN property_type = 'D' THEN 1 END) * 1.0 / count(*), 4) AS h3_detached_ratio,
            round(count(CASE WHEN is_new_build = 'Y' THEN 1 END) * 1.0 / count(*), 4) AS h3_new_build_ratio,
            round(count(CASE WHEN tenure = 'F' THEN 1 END) * 1.0 / count(*), 4) AS h3_freehold_ratio
        FROM df_temp
        GROUP BY h3_res7
    """).df()
    
    # Merge micro-market statistics back
    df = df.merge(h3_stats, on='h3_res7', how='left')
    
    # Fill standard deviation for single-sale hexagons
    df['h3_price_std'] = df['h3_price_std'].fillna(0)
    
    # Relative micro-market indicators
    df['price_to_h3_median_ratio'] = df['price'] / df['h3_median_price']
    df['h3_iqr_price'] = df['h3_p75_price'] - df['h3_p25_price']
    print(f"      -> Generated 10 spatial micro-market features in {time.time()-t2:.2f}s.")

    # 4. Geospatial Proximity to Economic Hubs
    print("\n[4/6] Computing Proximity to Major Urban Centers...")
    t3 = time.time()
    for city_name, (clat, clon) in HUBS.items():
        col_name = f"dist_to_{city_name}_km"
        df[col_name] = haversine_vectorized(lats, lons, clat, clon).round(2)
    
    # Nearest hub distance
    dist_cols = [f"dist_to_{c}_km" for c in HUBS.keys()]
    df['dist_to_nearest_major_city_km'] = df[dist_cols].min(axis=1)
    df['is_london_commuter_belt'] = (df['dist_to_london_km'] <= 65.0).astype(int)
    print(f"      -> Generated 7 proximity metrics in {time.time()-t3:.2f}s.")

    # 5. Temporal, Macroeconomic & Categorical Features
    print("\n[5/6] Encoding Macroeconomic & Cyclical Dimensions...")
    t4 = time.time()
    
    # Bank of England Base Rate
    date_tuples = list(zip(df['sale_year'], df['sale_month']))
    df['boe_base_rate'] = [BOE_RATES.get(dt, 3.50) for dt in date_tuples]

    # Cyclical Month & Day of Week Encodings
    df['month_sin'] = np.sin(2 * np.pi * df['sale_month'] / 12.0).round(4)
    df['month_cos'] = np.cos(2 * np.pi * df['sale_month'] / 12.0).round(4)
    df['dow_sin'] = np.sin(2 * np.pi * df['sale_day_of_week'] / 7.0).round(4)
    df['dow_cos'] = np.cos(2 * np.pi * df['sale_day_of_week'] / 7.0).round(4)

    # Binary Structural Encodings
    df['is_new_build_int'] = (df['is_new_build'] == 'Y').astype(int)
    df['is_freehold'] = (df['tenure'] == 'F').astype(int)

    # One-Hot Property Types
    df['type_detached'] = (df['property_type'] == 'D').astype(int)
    df['type_semi'] = (df['property_type'] == 'S').astype(int)
    df['type_terraced'] = (df['property_type'] == 'T').astype(int)
    df['type_flat'] = (df['property_type'] == 'F').astype(int)
    df['type_other'] = (df['property_type'] == 'O').astype(int)

    # Target Normalization
    df['log_price'] = np.log1p(df['price']).round(4)
    print(f"      -> Encodings & normalizations completed in {time.time()-t4:.2f}s.")

    # 6. Save Final Engineered Feature Matrix
    print("\n[6/6] Serializing ML-Ready Feature Matrix to Parquet...")
    t5 = time.time()
    df.to_parquet(OUTPUT_PARQUET, index=False, compression='snappy')
    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
    print(f"      -> Saved {OUTPUT_PARQUET} ({file_size_mb:.2f} MB) in {time.time()-t5:.2f}s.")

    # Final Verification & Summary
    print("\n" + "=" * 70)
    print("PHASE 2 FEATURE ENGINEERING COMPLETE")
    print(f"  Total Observations:     {len(df):,}")
    print(f"  Total Features Created: {len(df.columns)}")
    print(f"  Unique H3 Micro-Markets:{len(df['h3_res7'].unique()):,}")
    print(f"  Feature Matrix File:    {OUTPUT_PARQUET}")
    print(f"  Total Pipeline Runtime: {time.time()-t0:.2f}s")
    print("=" * 70)

    # Print feature overview
    print("\nFeature Matrix Columns Preview:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:<32} (dtype: {str(df[col].dtype)})")

if __name__ == "__main__":
    main()
