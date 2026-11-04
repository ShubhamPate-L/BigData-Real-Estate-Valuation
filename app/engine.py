"""
Valuation Engine & Comps Search Service
Provides low-latency inference, DuckDB spatial lookups, and micro-market enrichment.
"""
import os
import sys
import numpy as np
import pandas as pd
import duckdb
import joblib
import h3

MODELS_DIR = "models"
DATA_FEATURES = "data/features/feature_matrix.parquet"
DATA_PROPERTIES = "data/properties/properties_cleaned.parquet"
DATA_H3_CACHE = "data/features/h3_micro_markets.parquet"
ARBITRAGE_PATH = "data/features/arbitrage_top50.csv"

CATEGORICAL_FEATURES = ["property_type", "is_new_build", "tenure", "county"]

HUBS = {
    "london": (51.5074, -0.1278),
    "birmingham": (52.4862, -1.8904),
    "manchester": (53.4808, -2.2426),
    "leeds": (53.8008, -1.5491),
    "bristol": (51.4545, -2.5879)
}

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    return round(float(2 * R * np.arcsin(np.sqrt(a))), 2)

class ValuationEngine:
    def __init__(self):
        print("[ENGINE] Loading Machine Learning models from disk...")
        self.model_point = joblib.load(os.path.join(MODELS_DIR, "lgbm_point_regressor.joblib"))
        self.model_q10 = joblib.load(os.path.join(MODELS_DIR, "lgbm_quantile_q10.joblib"))
        self.model_q50 = joblib.load(os.path.join(MODELS_DIR, "lgbm_quantile_q50.joblib"))
        self.model_q90 = joblib.load(os.path.join(MODELS_DIR, "lgbm_quantile_q90.joblib"))
        self.feature_names = self.model_point.feature_name_
        
        # Connect to DuckDB
        print("[ENGINE] Initializing DuckDB spatial database...")
        self.con = duckdb.connect()
        self.con.execute(f"CREATE OR REPLACE VIEW properties AS SELECT * FROM read_parquet('{DATA_PROPERTIES}');")
        self.con.execute(f"CREATE OR REPLACE VIEW feature_matrix AS SELECT * FROM read_parquet('{DATA_FEATURES}');")
        
        # Load precomputed H3 micro-market lookup cache for instant O(1) enrichment
        print("[ENGINE] Loading H3 micro-market spatial indices cache...")
        if os.path.exists(DATA_H3_CACHE):
            self.h3_cache = pd.read_parquet(DATA_H3_CACHE).set_index("h3_res7")
        else:
            self.h3_cache = self.con.execute(f"""
                SELECT 
                    h3_res7, ANY_VALUE(county) AS county,
                    MAX(h3_sales_volume) AS h3_sales_volume, MAX(h3_median_price) AS h3_median_price,
                    MAX(h3_mean_price) AS h3_mean_price, MAX(h3_price_std) AS h3_price_std,
                    MAX(h3_p25_price) AS h3_p25_price, MAX(h3_p75_price) AS h3_p75_price,
                    MAX(h3_iqr_price) AS h3_iqr_price, MAX(h3_detached_ratio) AS h3_detached_ratio,
                    MAX(h3_new_build_ratio) AS h3_new_build_ratio, MAX(h3_freehold_ratio) AS h3_freehold_ratio
                FROM feature_matrix
                GROUP BY h3_res7;
            """).df().set_index("h3_res7")
        print(f"[ENGINE] Ready with {len(self.h3_cache):,} cached H3 micro-markets.")

    def lookup_postcode(self, postcode: str):
        clean_pc = postcode.strip().replace(" ", "").upper()
        res = self.con.execute(f"""
            SELECT postcode, street, city, county, latitude, longitude
            FROM properties
            WHERE postcode = '{clean_pc}'
            LIMIT 1;
        """).df()
        
        if len(res) == 0:
            return None
        return res.iloc[0].to_dict()

    def predict_valuation(self, postcode: str, property_type: str, is_new_build: str, tenure: str, asking_price: float = None):
        """
        Runs point and quantile valuation predictions for a property.
        property_type: 'D' (Detached), 'S' (Semi), 'T' (Terraced), 'F' (Flat), 'O' (Other)
        is_new_build: 'Y' (New), 'N' (Established)
        tenure: 'F' (Freehold), 'L' (Leasehold)
        """
        clean_pc = postcode.strip().replace(" ", "").upper()
        pc_info = self.lookup_postcode(clean_pc)
        
        if not pc_info:
            # Fallback centroid if unknown
            lat, lon = 51.5074, -0.1278
            county = "GREATER LONDON"
            city = "LONDON"
            street = "UNKNOWN"
        else:
            lat = pc_info["latitude"]
            lon = pc_info["longitude"]
            county = pc_info["county"]
            city = pc_info["city"]
            street = pc_info["street"]

        # H3 Hex indexing
        cell_res7 = h3.latlng_to_cell(lat, lon, 7)
        cell_res8 = h3.latlng_to_cell(lat, lon, 8)
        
        # Micro-market stats lookup
        if cell_res7 in self.h3_cache.index:
            stats = self.h3_cache.loc[cell_res7].to_dict()
        else:
            stats = {
                "h3_sales_volume": 50,
                "h3_median_price": 280000,
                "h3_mean_price": 310000,
                "h3_price_std": 60000,
                "h3_p25_price": 220000,
                "h3_p75_price": 370000,
                "h3_iqr_price": 150000,
                "h3_detached_ratio": 0.25,
                "h3_new_build_ratio": 0.05,
                "h3_freehold_ratio": 0.80
            }
            
        # Distances to hubs
        dist_london = haversine(lat, lon, HUBS["london"][0], HUBS["london"][1])
        dist_birm = haversine(lat, lon, HUBS["birmingham"][0], HUBS["birmingham"][1])
        dist_manc = haversine(lat, lon, HUBS["manchester"][0], HUBS["manchester"][1])
        dist_leeds = haversine(lat, lon, HUBS["leeds"][0], HUBS["leeds"][1])
        dist_bristol = haversine(lat, lon, HUBS["bristol"][0], HUBS["bristol"][1])
        dist_nearest = min(dist_london, dist_birm, dist_manc, dist_leeds, dist_bristol)
        is_commuter = 1 if dist_london <= 65.0 else 0

        # Construct single-row feature dict
        row = {
            "latitude": lat,
            "longitude": lon,
            "h3_sales_volume": int(stats["h3_sales_volume"]),
            "h3_median_price": int(stats["h3_median_price"]),
            "h3_mean_price": int(stats["h3_mean_price"]),
            "h3_price_std": int(stats["h3_price_std"]),
            "h3_p25_price": int(stats["h3_p25_price"]),
            "h3_p75_price": int(stats["h3_p75_price"]),
            "h3_iqr_price": int(stats["h3_iqr_price"]),
            "h3_detached_ratio": float(stats["h3_detached_ratio"]),
            "h3_new_build_ratio": float(stats["h3_new_build_ratio"]),
            "h3_freehold_ratio": float(stats["h3_freehold_ratio"]),
            "dist_to_london_km": dist_london,
            "dist_to_birmingham_km": dist_birm,
            "dist_to_manchester_km": dist_manc,
            "dist_to_leeds_km": dist_leeds,
            "dist_to_bristol_km": dist_bristol,
            "dist_to_nearest_major_city_km": dist_nearest,
            "is_london_commuter_belt": is_commuter,
            "boe_base_rate": 5.25,
            "month_sin": 0.0,
            "month_cos": 1.0,
            "dow_sin": 0.7818,
            "dow_cos": 0.6235,
            "is_new_build_int": 1 if is_new_build == "Y" else 0,
            "is_freehold": 1 if tenure == "F" else 0,
            "type_detached": 1 if property_type == "D" else 0,
            "type_semi": 1 if property_type == "S" else 0,
            "type_terraced": 1 if property_type == "T" else 0,
            "type_flat": 1 if property_type == "F" else 0,
            "type_other": 1 if property_type == "O" else 0,
            "property_type": property_type,
            "is_new_build": is_new_build,
            "tenure": tenure,
            "county": county
        }
        
        df_row = pd.DataFrame([row])
        for cat in CATEGORICAL_FEATURES:
            df_row[cat] = df_row[cat].astype("category")

        # Inferences
        pred_log_point = self.model_point.predict(df_row[self.feature_names])[0]
        pred_log_q10 = self.model_q10.predict(df_row[self.feature_names])[0]
        pred_log_q50 = self.model_q50.predict(df_row[self.feature_names])[0]
        pred_log_q90 = self.model_q90.predict(df_row[self.feature_names])[0]
        
        fair_value = int(np.round(np.expm1(pred_log_point)))
        floor_value = int(np.round(np.expm1(pred_log_q10)))
        ceiling_value = int(np.round(np.expm1(pred_log_q90)))
        
        # Comps
        comps = self.get_comps(lat, lon, cell_res7, property_type, tenure, limit=5)
        
        arbitrage_info = None
        if asking_price and asking_price > 0:
            discount_gbp = fair_value - asking_price
            discount_pct = (discount_gbp / fair_value) * 100.0
            is_underpriced = bool(asking_price < floor_value and discount_pct >= 20.0)
            arbitrage_info = {
                "asking_price": asking_price,
                "discount_gbp": int(discount_gbp),
                "discount_pct": round(discount_pct, 1),
                "is_arbitrage_deal": is_underpriced,
                "assessment": "High-Confidence Undervalued Gem" if is_underpriced else (
                    "Fair Market Price" if abs(discount_pct) <= 10 else (
                        "Overpriced / Premium" if discount_pct < -10 else "Minor Discount"
                    )
                )
            }

        return {
            "postcode": clean_pc,
            "street": street,
            "city": city,
            "county": county,
            "latitude": lat,
            "longitude": lon,
            "h3_cell": cell_res7,
            "fair_market_value_gbp": fair_value,
            "liquidation_floor_10pct_gbp": floor_value,
            "upside_ceiling_90pct_gbp": ceiling_value,
            "confidence_band_width_gbp": ceiling_value - floor_value,
            "arbitrage_analysis": arbitrage_info,
            "comparables": comps
        }

    def get_comps(self, lat: float, lon: float, h3_cell: str, property_type: str, tenure: str, limit: int = 5):
        query = f"""
            SELECT 
                street,
                city,
                postcode,
                price,
                sale_date,
                property_type,
                tenure,
                ROUND(2 * 6371 * ASIN(SQRT(
                    POWER(SIN(RADIANS(latitude - {lat}) / 2), 2) +
                    COS(RADIANS({lat})) * COS(RADIANS(latitude)) * POWER(SIN(RADIANS(longitude - {lon}) / 2), 2)
                )), 2) AS distance_km
            FROM feature_matrix
            WHERE h3_res7 = '{h3_cell}'
              AND property_type = '{property_type}'
              AND tenure = '{tenure}'
            ORDER BY distance_km ASC, sale_date DESC
            LIMIT {limit};
        """
        comps_df = self.con.execute(query).df()
        comps_list = []
        for _, r in comps_df.iterrows():
            comps_list.append({
                "street": str(r["street"]),
                "city": str(r["city"]),
                "postcode": str(r["postcode"]),
                "price_gbp": int(r["price"]),
                "sale_date": str(r["sale_date"])[:10],
                "distance_km": float(r["distance_km"])
            })
        return comps_list

    def get_top_arbitrage_deals(self, limit: int = 50):
        if os.path.exists(ARBITRAGE_PATH):
            df = pd.read_csv(ARBITRAGE_PATH)
            return df.head(limit).to_dict(orient="records")
        return []
