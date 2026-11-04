import duckdb
import time
import os

print("[CACHE] Precomputing H3 micro-markets lookup table...")
t0 = time.time()
con = duckdb.connect()

con.execute("""
    CREATE OR REPLACE TABLE h3_cache AS
    SELECT 
        h3_res7,
        ANY_VALUE(county) AS county,
        MAX(h3_sales_volume) AS h3_sales_volume,
        MAX(h3_median_price) AS h3_median_price,
        MAX(h3_mean_price) AS h3_mean_price,
        MAX(h3_price_std) AS h3_price_std,
        MAX(h3_p25_price) AS h3_p25_price,
        MAX(h3_p75_price) AS h3_p75_price,
        MAX(h3_iqr_price) AS h3_iqr_price,
        MAX(h3_detached_ratio) AS h3_detached_ratio,
        MAX(h3_new_build_ratio) AS h3_new_build_ratio,
        MAX(h3_freehold_ratio) AS h3_freehold_ratio
    FROM read_parquet('data/features/feature_matrix.parquet')
    GROUP BY h3_res7;
""")

out_path = os.path.join("data", "features", "h3_micro_markets.parquet")
con.execute(f"COPY h3_cache TO '{out_path.replace(chr(92), '/')}' (FORMAT PARQUET, CODEC 'SNAPPY');")
count = con.execute("SELECT count(*) FROM h3_cache;").fetchone()[0]
print(f"[CACHE] Saved {count:,} micro-markets to {out_path} in {time.time()-t0:.2f}s.")

