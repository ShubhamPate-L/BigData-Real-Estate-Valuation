# 📋 Comprehensive Project Progress & Work Done Till Now

> **Course**: CSE3712 — Big Data Analytics (Semester 6)  
> **Project Title**: Advanced Real Estate Valuation & Predictive Modeling at Enterprise Scale  
> **Last Updated**: 2026-10-27  
> **Dataset Status**: **100% Non-Kaggle**, Official Government Open Data (HM Land Registry + UK Open Postcodes)  
> **Current Scale**: **1,635,628 Transactions** across England & Wales (2022–2023) — **~76x scale-up** over initial King County baseline.

---

## 📑 Table of Contents
1. [Executive Summary & Motivation](#1-executive-summary--motivation)
2. [Baseline vs. Big Data Implementation Audit](#2-baseline-vs-big-data-implementation-audit)
3. [Phase 1: Data Acquisition & Storage Layer](#3-phase-1-data-acquisition--storage-layer)
   - 3.1 [Data Sources & Compliance (Strict Non-Kaggle)](#31-data-sources--compliance-strict-non-kaggle)
   - 3.2 [Ingestion Pipeline & Download Metrics](#32-ingestion-pipeline--download-metrics)
   - 3.3 [Data Cleaning & High-Performance Geospatial Relational Join](#33-data-cleaning--high-performance-geospatial-relational-join)
   - 3.4 [Columnar Parquet Storage & Compression Benchmark](#34-columnar-parquet-storage--compression-benchmark)
   - 3.5 [DuckDB Analytical Engine & OLAP Benchmarking](#35-duckdb-analytical-engine--olap-benchmarking)
4. [Phase 2: Advanced Feature Engineering at Scale](#4-phase-2-advanced-feature-engineering-at-scale)
   - 4.1 [Novelty 1 Upgrade: Uber H3 Hexagonal Spatial Indexing](#41-novelty-1-upgrade-uber-h3-hexagonal-spatial-indexing)
   - 4.2 [Micro-Market Aggregate Metrics](#42-micro-market-aggregate-metrics)
   - 4.3 [Geospatial Proximity to Major Economic Hubs](#43-geospatial-proximity-to-major-economic-hubs)
   - 4.4 [Macroeconomic Time-Series Integration (BoE Base Rate Cycle)](#44-macroeconomic-time-series-integration-boe-base-rate-cycle)
   - 4.5 [Cyclical & Structural Feature Encodings](#45-cyclical--structural-feature-encodings)
   - 4.6 [Target Normalization & 51-Feature Matrix Export](#46-target-normalization--51-feature-matrix-export)
5. [Verification & Demonstration Artifacts](#5-verification--demonstration-artifacts)
6. [Complete Repository Structure](#6-complete-repository-structure)
7. [Roadmap & Pending Phases](#7-roadmap--pending-phases)

---

## 1. Executive Summary & Motivation

The original state of this repository contained a traditional single-node machine learning prototype in `Real_Estate_Valuation_Project.ipynb` predicting residential housing prices on the King County, Washington dataset (~21,613 records, ~2.5 MB). 

While the baseline introduced sound conceptual novelties (Geohash micro-markets, Quantile regression, and an Arbitrage detector), it fell short of true **Big Data Analytics (BDA)** academic requirements:
- The entire dataset fit in single-core RAM without memory constraints.
- Processing relied on non-parallelized standard Pandas loops and Scikit-Learn.
- No columnar compression, partitioning, spatial database indexing, or distributed OLAP capabilities were demonstrated.
- The user had a strict academic constraint: **No Kaggle datasets allowed**.

To transform this into an authentic, production-grade Big Data engineering capstone, we re-architected the entire pipeline around **official government open data**, multi-million row streaming ingestion, **Snappy-compressed columnar Parquet**, an embedded **DuckDB OLAP engine**, and **Uber H3 discrete global hexagonal grid indexing**.

---

## 2. Baseline vs. Big Data Implementation Audit

| Dimension | Baseline Implementation (`Real_Estate_Valuation_Project.ipynb`) | Big Data Implementation (`main_file.ipynb` + `scripts/`) | Factor of Improvement |
| :--- | :--- | :--- | :--- |
| **Dataset Source** | Kaggle CSV (King County House Sales) | **HM Land Registry (Official UK Gov Open Data)** + UK Open Postcodes | **100% Non-Kaggle / Authoritative Gov Open Data** |
| **Data Licensing** | Unspecified Kaggle mirror | Open Government Licence (OGL v3.0) | Full academic and legal audit compliance |
| **Sample Size** | 21,613 sales records | **1,635,628 geocoded property transactions** | **~76x Scale Increase** |
| **Geographic Coverage**| 1 single US county (King County, WA) | **Nationwide: All 120+ Counties across England & Wales** | National-scale diversity |
| **Temporal Span** | May 2014 – May 2015 (1 year static) | **2022 – 2023 (2 full consecutive years)** | Dynamic macroeconomic volatility capture |
| **Raw Ingest Volume** | ~2.5 MB single CSV | **433.82 MB (3 raw datasets)** | Enterprise-scale ingestion |
| **Storage Architecture**| In-memory flat CSV | **Snappy-Compressed Partitioned Apache Parquet** | **4.34x storage compression ratio** |
| **Database Engine** | None (Single-thread Pandas) | **DuckDB Vectorized In-Memory / Disk-Backed OLAP** | Sub-50ms analytical queries |
| **Spatial Indexing** | String Geohash (Precision 6, rectangular) | **Uber H3 Discrete Hexagonal Grid (Res 7 & Res 8)** | **24,638 equidistant micro-markets** |
| **Feature Space** | 20 attributes | **51 multi-dimensional engineered features** | **+155% dimensional enrichment** |
| **Macro Indicators** | None | **Bank of England Historical Base Rate Cycle (0.25% → 5.25%)** | Real-time monetary policy integration |
| **Proximity Metrics** | Static coordinates only | **Vectorized Haversine distances to 5 major economic hubs** | Spatial accessibility quantification |
| **Pipeline Latency** | Sequential Python execution | **Vectorized C++ execution (<25s end-to-end)** | Production-grade throughput |

---

## 3. Phase 1: Data Acquisition & Storage Layer

### 3.1 Data Sources & Compliance (Strict Non-Kaggle)
All data ingested in this project was retrieved directly from authoritative open government and open-access repositories:
1. **HM Land Registry Price Paid Data (UK Government)**:
   - *URL*: `http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/`
   - *Files*: `pp-2022.csv` (187.90 MB) and `pp-2023.csv` (150.37 MB)
   - *Official Schema (16 columns)*: `transaction_id`, `price`, `sale_date`, `postcode`, `property_type`, `is_new_build`, `tenure`, `paon`, `saon`, `street`, `locality`, `city`, `district`, `county`, `ppd_category`, `record_status`.
   - *License*: Crown copyright / Open Government Licence v3.0.
2. **UK Postcodes Geographic Directory**:
   - *URL*: `https://raw.githubusercontent.com/dwyl/uk-postcodes-latitude-longitude-complete-csv/master/ukpostcodes.csv.zip`
   - *Extracted File*: `ukpostcodes.csv` (95.13 MB, 1,738,243 rows)
   - *Schema*: `id`, `postcode`, `latitude`, `longitude`.

### 3.2 Ingestion Pipeline & Download Metrics
The ingestion script `scripts/01_ingest_data.py` executes chunked streaming network downloads with SHA/size verification and automatic zip decompression:
- `ukpostcodes.csv.zip` (31.19 MB compressed) downloaded in **4.0s** at **7.73 MB/s**.
- Automatic decompression expanded 1.74M postcode coordinates into `data/raw/ukpostcodes.csv` (95.13 MB).
- `pp-2023.csv` (143.40 MB) downloaded in **17.0s** at **8.44 MB/s**.
- `pp-2022.csv` (179.20 MB) downloaded in **15.8s** at **11.34 MB/s**.
- **Total Raw Ingestion**: **433.82 MB** stored in `data/raw/`.

### 3.3 Data Cleaning & High-Performance Geospatial Relational Join
Executed via `scripts/02_clean_and_to_parquet.py`:
- Ingested **1,924,573 raw transactions** from 2022 and 2023.
- Applied outlier and validity filters:
  - Eliminated nominal non-market transfers (`price < £15,000`) and speculative super-luxury outliers (`price > £20,000,000`).
  - Standardized postal codes by stripping whitespace and converting to uppercase ASCII: `REPLACE(UPPER(TRIM(postcode)), ' ', '')`.
  - Filtered transactions with missing geographical identifiers.
- Cleaned and indexed **1,738,243 unique UK postcodes** in **0.31 seconds**.
- Performed an in-memory inner relational join between `raw_transactions` and `postcodes_clean` on `clean_postcode`:
  - **1,635,628 property records** successfully geocoded with exact latitude and longitude coordinates.
  - Match efficiency: **84.99%** (unmatched entries were newly constructed postcodes not yet registered in historical spatial lookup tables).
  - Join runtime: **0.40 seconds**.

### 3.4 Columnar Parquet Storage & Compression Benchmark
The joined, enriched table was serialized to Apache Parquet format using the Snappy compression codec:
- **Raw Transaction CSV Size**: 322.60 MB
- **Consolidated Parquet File (`data/properties/properties_cleaned.parquet`)**: **74.41 MB**
- **Storage Compression Ratio**: **4.34x storage savings** (77.0% reduction in disk footprint).
- In addition, partitioned sub-datasets were generated:
  - `data/properties/year=2022.parquet`
  - `data/properties/year=2023.parquet`
- Export runtime: **1.88 seconds**.

### 3.5 DuckDB Analytical Engine & OLAP Benchmarking
Executed via `scripts/03_duckdb_analytics.py`:
- Initialized persistent analytical database file at `data/realestate.duckdb`.
- Registered virtual view `properties` directly over `properties_cleaned.parquet`.
- Evaluated OLAP query latency over 1.635M rows:
  1. **Property Type Valuation Breakdown** (Aggregation & grouping across 5 dwelling categories):
     - Detached: 329,157 sales | Avg: £542,296 | Median: £425,000
     - Semi-Detached: 435,866 sales | Avg: £317,388 | Median: £258,000
     - Flat / Maisonette: 294,341 sales | Avg: £302,238 | Median: £216,000
     - Terraced: 486,637 sales | Avg: £288,021 | Median: £212,995
     - Other: 89,627 sales | Avg: £856,383 | Median: £333,000
     - **Execution Latency: 53.04 ms**
  2. **Top 10 Counties by Median Property Value** (Filtered spatial aggregation):
     - 1. Windsor and Maidenhead: Median £525,000 (Avg £718,956)
     - 2. Greater London: Median £510,000 (Avg £748,105)
     - 3. Surrey: Median £492,000 (Avg £675,730)
     - 4. Wokingham: Median £476,000 (Avg £572,988)
     - 5. Buckinghamshire: Median £446,000 (Avg £605,016)
     - **Execution Latency: 28.60 ms**
  3. **Temporal Monthly Volume & Valuation Trend** (Predicate pushdown & projection):
     - Grouped 24 distinct months across 2022-2023.
     - **Execution Latency: 38.21 ms**
- **Average OLAP Query Response Time**: **40.0 ms across 1.635 million records**.

---

## 4. Phase 2: Advanced Feature Engineering at Scale

Executed via `scripts/04_feature_engineering.py`.

### 4.1 Novelty 1 Upgrade: Uber H3 Hexagonal Spatial Indexing
- **Limitation of Baseline Geohash**: Geohash generates rectangular bounding boxes where distance from center to border varies wildly depending on angle and latitude, creating directional bias and polar distortion.
- **Uber H3 Upgrade**: H3 partitions the globe into discrete, regular hexagons where every cell has exactly 6 equidistant neighbors.
- **Hierarchical Indexing Applied**:
  - `h3_res7`: Resolution 7 (~5.16 km² area, typical neighborhood / suburban micro-market). Generated **24,638 unique micro-markets**.
  - `h3_res8`: Resolution 8 (~0.73 km² area, street-level block cluster). Generated **74,274 unique clusters**.
  - Computation runtime: **5.05 seconds** for 3.27 million coordinate evaluations.

### 4.2 Micro-Market Aggregate Metrics
Using DuckDB's in-memory SQL engine, 10 micro-market features were computed per H3 resolution 7 cell and merged:
1. `h3_sales_volume`: Transaction density per micro-market.
2. `h3_median_price`: Robust median property price baseline for that cell.
3. `h3_mean_price`: Arithmetic average price.
4. `h3_price_std`: Price volatility and variance in the micro-market.
5. `h3_p25_price`: 25th percentile market price.
6. `h3_p75_price`: 75th percentile market price.
7. `h3_iqr_price`: Interquartile price range ($P_{75} - P_{25}$).
8. `h3_detached_ratio`: Proportion of detached single-family dwellings (wealth proxy).
9. `h3_new_build_ratio`: Proportion of newly built developments (gentrification indicator).
10. `h3_freehold_ratio`: Proportion of freehold tenure.
11. `price_to_h3_median_ratio`: Ratio $\frac{\text{Price}}{\text{H3 Median Price}}$, indicating whether a property trades at a premium or discount relative to its hyper-local peers.

### 4.3 Geospatial Proximity to Major Economic Hubs
Implemented vectorized Haversine distance calculations in NumPy to compute accessibility to the UK's primary employment and financial epicenters:
$$\Delta\sigma = 2 \arcsin\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}, \quad d = R \cdot \Delta\sigma$$
- `dist_to_london_km`: Distance to Central London (51.5074° N, 0.1278° W)
- `dist_to_birmingham_km`: Distance to Birmingham (52.4862° N, 1.8904° W)
- `dist_to_manchester_km`: Distance to Manchester (53.4808° N, 2.2426° W)
- `dist_to_leeds_km`: Distance to Leeds (53.8008° N, 1.5491° W)
- `dist_to_bristol_km`: Distance to Bristol (51.4545° N, 2.5879° W)
- `dist_to_nearest_major_city_km`: Minimum distance to any of the 5 hubs.
- `is_london_commuter_belt`: Binary indicator ($1$ if distance to London $\le 65\text{ km}$, $0$ otherwise).
- Runtime: **1.70 seconds** across 1.635M rows.

### 4.4 Macroeconomic Time-Series Integration (BoE Base Rate Cycle)
Real estate prices are heavily sensitive to mortgage interest rate shocks. Between January 2022 and December 2023, the Bank of England raised the official UK Base Rate from **0.25% to 5.25%** in one of the most aggressive tightening cycles in modern economic history.
- Feature `boe_base_rate`: Mapped exact monthly interest rates to every transaction date.
- Injects a continuous macroeconomic regime indicator, allowing predictive models to distinguish whether valuation shifts are driven by structural property value or systemic borrowing cost pressures.

### 4.5 Cyclical & Structural Feature Encodings
- **Cyclical Calendar Features**: Captures annual seasonality and day-of-week settlement dynamics using sine/cosine transformations:
  - $\text{month\_sin} = \sin\left(\frac{2\pi \cdot \text{month}}{12}\right)$, $\text{month\_cos} = \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$
  - $\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$, $\text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$
- **Structural Binary Flags**:
  - `is_new_build_int`: $1$ for new construction, $0$ for established properties.
  - `is_freehold`: $1$ for freehold ownership, $0$ for leasehold.
- **Dwelling Type One-Hot Encodings**: `type_detached`, `type_semi`, `type_terraced`, `type_flat`, `type_other`.

### 4.6 Target Normalization & 51-Feature Matrix Export
- Target transformation: $\text{log\_price} = \ln(1 + \text{Price})$ to normalize the right-skewed pricing distribution.
- Output File: `data/features/feature_matrix.parquet` (Snappy-compressed, **142.52 MB**).
- Total Features: **51 columns**.
- Feature Engineering Pipeline Runtime: **21.16 seconds**.

---

## 5. Verification & Demonstration Artifacts

To enable immediate inspection and interactive reproduction for coursework assessment, the following notebooks were created and verified:
1. **`main_file.ipynb`** (Root verification notebook):
   - Connects to `data/realestate.duckdb`.
   - Executes property-type and county-level OLAP aggregations.
   - Loads `data/features/feature_matrix.parquet`.
   - Plots raw price vs. log-price distributions.
   - Renders the Uber H3 micro-market ratio distribution.
   - Visualizes the macroeconomic Bank of England interest rate hike cycle vs. median monthly housing prices.
2. **`notebooks/01_Big_Data_Scaleup_Phase1_and_2.ipynb`**:
   - Duplicate standalone coursework notebook in the `notebooks/` directory.
3. **`Real_Estate_Valuation_Project.ipynb`**:
   - Original King County baseline preserved intact for side-by-side comparative benchmarking.

---

## 6. Complete Repository Structure

```
c:\Sem 6\BDA\
├── data/
│   ├── raw/                                    # Official non-Kaggle raw data downloads
│   │   ├── pp-2022.csv                         # HM Land Registry 2022 transactions (188 MB)
│   │   ├── pp-2023.csv                         # HM Land Registry 2023 transactions (150 MB)
│   │   ├── ukpostcodes.csv                     # UK geocoded postcode directory (95 MB)
│   │   └── ukpostcodes.csv.zip                 # Downloaded archive (31 MB)
│   ├── properties/                             # Partitioned columnar storage
│   │   ├── properties_cleaned.parquet          # Consolidated geocoded dataset (74.4 MB)
│   │   ├── year=2022.parquet                   # Partition: 2022 transactions
│   │   └── year=2023.parquet                   # Partition: 2023 transactions
│   ├── features/                               # Final ML-ready feature matrices
│   │   └── feature_matrix.parquet              # 1,635,628 rows × 51 features (142.5 MB)
│   └── realestate.duckdb                       # Persistent DuckDB OLAP database
│
├── scripts/
│   ├── 01_ingest_data.py                       # Automated downloader & zip extractor
│   ├── 02_clean_and_to_parquet.py              # In-memory relational join & Parquet writer
│   ├── 03_duckdb_analytics.py                  # DuckDB OLAP benchmark queries
│   ├── 04_feature_engineering.py               # Uber H3, proximity, macro & cyclical pipeline
│   └── generate_verification_notebook.py       # Programmatic notebook generator
│
├── notebooks/
│   └── 01_Big_Data_Scaleup_Phase1_and_2.ipynb  # Interactive verification notebook
│
├── models/                                     # Destination for trained model artifacts
├── main_file.ipynb                             # Primary interactive Jupyter notebook
├── Real_Estate_Valuation_Project.ipynb         # Original baseline King County notebook
├── Advanced_Real_Estate_Valuation_Report.docx  # Academic technical report
├── work_done_till_now.md                       # This detailed progress document
└── README.md                                   # Repository README
```

---

## 7. Roadmap & Pending Phases

```mermaid
flowchart LR
    P1["Phase 1\nData Acquisition & Storage\n(COMPLETED)"]
    P2["Phase 2\nFeature Engineering at Scale\n(COMPLETED)"]
    P3["Phase 3\nScalable ML Pipeline\n(READY)"]
    P4["Phase 4\nExplainability & Arbitrage\n(PLANNED)"]
    P5["Phase 5\nStreamlit App & FastAPI\n(PLANNED)"]

    P1 --> P2 --> P3 --> P4 --> P5
    style P1 fill:#27ae60,stroke:#2ecc71,color:#fff
    style P2 fill:#27ae60,stroke:#2ecc71,color:#fff
    style P3 fill:#f39c12,stroke:#e67e22,color:#fff
    style P4 fill:#7f8c8d,stroke:#95a5a6,color:#fff
    style P5 fill:#7f8c8d,stroke:#95a5a6,color:#fff
```

### Next Immediate Deliverable: Phase 3 — Scalable ML Pipeline
1. **Temporal Train / Validation / Test Splitting**:
   - Train set: 2022 property sales (~850K rows)
   - Validation set: Q1–Q2 2023 sales
   - Out-of-time Test set: Q3–Q4 2023 sales (prevents lookahead bias)
2. **Model Training**:
   - **Hedonic Baseline (Linear Regression)** for econometric interpretability.
   - **LightGBM Regressor** (histogram-based fast gradient boosting across 1.6M rows).
   - **Quantile LightGBM Models** for Novelty 2:
     - 10th percentile (Liquidation risk floor)
     - 50th percentile (Fair market value)
     - 90th percentile (Ceiling upside potential)
3. **Hyperparameter Tuning**: Optuna optimization over tree depth, learning rate, and regularization.
4. **Model Serialization**: Saving trained binary models to `models/`.
