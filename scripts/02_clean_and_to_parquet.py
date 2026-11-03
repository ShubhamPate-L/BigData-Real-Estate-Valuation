"""
Phase 1 - Script 02: Clean, Geocode & Export Partitioned Parquet via DuckDB
- Loads HM Land Registry Price Paid Data (2022 + 2023)
- Loads UK Postcodes with Coordinates (1.7M postcodes)
- Cleans data, normalizes strings, extracts temporal columns
- Performs high-performance relational join on Postcode
- Exports snappy-compressed Parquet partitioned by sale_year
"""
import os
import time
import duckdb

RAW_DIR = os.path.join("data", "raw")
PROP_DIR = os.path.join("data", "properties")
os.makedirs(PROP_DIR, exist_ok=True)

def main():
    print("=" * 65)
    print("Phase 1: High-Performance Data Processing & Parquet Export")
    print("Engine: DuckDB In-Memory OLAP Vectorized Processing")
    print("=" * 65)
    
    t0 = time.time()
    con = duckdb.connect()

    # Configure DuckDB to use multi-threading
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='8GB';")

    print("[1/5] Loading & Standardizing UK Postcodes Coordinates...")
    postcodes_csv = os.path.join(RAW_DIR, "ukpostcodes.csv").replace("\\", "/")
    con.execute(f"""
        CREATE OR REPLACE TABLE postcodes_clean AS
        SELECT 
            REPLACE(UPPER(TRIM(postcode)), ' ', '') AS clean_postcode,
            AVG(latitude) AS latitude,
            AVG(longitude) AS longitude
        FROM read_csv(
            '{postcodes_csv}',
            header=true,
            columns={{'id': 'BIGINT', 'postcode': 'VARCHAR', 'latitude': 'DOUBLE', 'longitude': 'DOUBLE'}},
            auto_detect=false
        )
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        GROUP BY 1;
    """)
    n_postcodes = con.execute("SELECT count(*) FROM postcodes_clean").fetchone()[0]
    print(f"      -> Cleaned & Indexed {n_postcodes:,} unique postcodes in {time.time()-t0:.2f}s.")

    # 2. Ingest raw Price Paid CSVs
    print("\n[2/5] Ingesting HM Land Registry Raw Transactions (2022 & 2023)...")
    t1 = time.time()
    pp_files = [
        os.path.join(RAW_DIR, "pp-2022.csv").replace("\\", "/"),
        os.path.join(RAW_DIR, "pp-2023.csv").replace("\\", "/")
    ]
    files_str = ", ".join([f"'{f}'" for f in pp_files])

    con.execute(f"""
        CREATE OR REPLACE TABLE raw_transactions AS
        SELECT 
            column00 AS transaction_id,
            column01::BIGINT AS price,
            column02::TIMESTAMP AS sale_date,
            REPLACE(UPPER(TRIM(column03)), ' ', '') AS clean_postcode,
            column04 AS property_type,      -- D=Detached, S=Semi, T=Terraced, F=Flat, O=Other
            column05 AS is_new_build,       -- Y=New, N=Established
            column06 AS tenure,             -- F=Freehold, L=Leasehold
            column07 AS paon,
            column08 AS saon,
            column09 AS street,
            column10 AS locality,
            column11 AS city,
            column12 AS district,
            column13 AS county,
            column14 AS ppd_category,       -- A=Standard, B=Additional
            column15 AS record_status
        FROM read_csv(
            [{files_str}],
            header=false,
            auto_detect=true
        )
        WHERE column01::BIGINT >= 15000 AND column01::BIGINT <= 20000000
          AND column03 IS NOT NULL;
    """)
    n_trans = con.execute("SELECT count(*) FROM raw_transactions").fetchone()[0]
    print(f"      -> Ingested {n_trans:,} transactions in {time.time()-t1:.2f}s.")

    # 3. Join with Coordinates and Extract Temporal Dimensions
    print("\n[3/5] Geospatial Join & Temporal Feature Extraction...")
    t2 = time.time()
    con.execute("""
        CREATE OR REPLACE TABLE properties_enriched AS
        SELECT 
            t.transaction_id,
            t.price,
            t.sale_date,
            YEAR(t.sale_date)::INTEGER AS sale_year,
            MONTH(t.sale_date)::INTEGER AS sale_month,
            QUARTER(t.sale_date)::INTEGER AS sale_quarter,
            DAYOFWEEK(t.sale_date)::INTEGER AS sale_day_of_week,
            t.clean_postcode AS postcode,
            t.property_type,
            t.is_new_build,
            t.tenure,
            COALESCE(NULLIF(TRIM(t.street), ''), 'UNKNOWN') AS street,
            COALESCE(NULLIF(TRIM(t.locality), ''), 'UNKNOWN') AS locality,
            COALESCE(NULLIF(TRIM(t.city), ''), 'UNKNOWN') AS city,
            COALESCE(NULLIF(TRIM(t.district), ''), 'UNKNOWN') AS district,
            COALESCE(NULLIF(TRIM(t.county), ''), 'UNKNOWN') AS county,
            p.latitude,
            p.longitude
        FROM raw_transactions t
        INNER JOIN postcodes_clean p ON t.clean_postcode = p.clean_postcode;
    """)
    n_enriched = con.execute("SELECT count(*) FROM properties_enriched").fetchone()[0]
    print(f"      -> Successfully geocoded {n_enriched:,} properties ({n_enriched/n_trans*100:.2f}% match) in {time.time()-t2:.2f}s.")

    # 4. Export Partitioned Parquet
    print("\n[4/5] Exporting Partitioned Parquet (Snappy Columnar Storage)...")
    t3 = time.time()
    out_pattern = os.path.join(PROP_DIR, "properties_cleaned.parquet").replace("\\", "/")
    
    # We export a consolidated single partitioned parquet or multi-file parquet
    con.execute(f"""
        COPY properties_enriched 
        TO '{out_pattern}' 
        (FORMAT PARQUET, CODEC 'SNAPPY');
    """)
    
    # Also partition by sale_year for partitioned queries
    p2022 = os.path.join(PROP_DIR, "year=2022.parquet").replace("\\", "/")
    p2023 = os.path.join(PROP_DIR, "year=2023.parquet").replace("\\", "/")
    con.execute(f"""
        COPY (SELECT * FROM properties_enriched WHERE sale_year = 2022)
        TO '{p2022}'
        (FORMAT PARQUET, CODEC 'SNAPPY');
        
        COPY (SELECT * FROM properties_enriched WHERE sale_year = 2023)
        TO '{p2023}'
        (FORMAT PARQUET, CODEC 'SNAPPY');
    """)
    print(f"      -> Exported Parquet files in {time.time()-t3:.2f}s.")

    # 5. Summary & Storage Compression Stats
    print("\n[5/5] Storage Optimization & Compression Analysis:")
    print("-" * 65)
    raw_size_mb = (os.path.getsize(pp_files[0]) + os.path.getsize(pp_files[1])) / (1024 * 1024)
    parquet_size_mb = os.path.getsize(os.path.join(PROP_DIR, "properties_cleaned.parquet")) / (1024 * 1024)
    compression_ratio = raw_size_mb / parquet_size_mb
    
    print(f"  Raw CSV Size:         {raw_size_mb:.2f} MB")
    print(f"  Parquet Storage Size: {parquet_size_mb:.2f} MB")
    print(f"  Compression Ratio:    {compression_ratio:.2f}x storage saving!")
    print(f"  Total Cleaned Rows:   {n_enriched:,}")
    print(f"  Total Elapsed Time:   {time.time()-t0:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
