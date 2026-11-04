"""
FastAPI Enterprise Real Estate Valuation & Comps Service
Provides RESTful APIs for real-time model inference, quantile risk estimation, and spatial comps.
"""
import os
import sys
import time
import warnings
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from app.engine import ValuationEngine

warnings.filterwarnings("ignore")

engine: Optional[ValuationEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("[API] Starting up Valuation Engine...")
    engine = ValuationEngine()
    print("[API] Valuation Engine loaded successfully.")
    yield
    print("[API] Shutting down...")

app = FastAPI(
    title="UK Real Estate Enterprise Valuation API",
    description="DuckDB-powered OLAP, LightGBM Point/Quantile Valuation, and TreeSHAP-audited Arbitrage API",
    version="1.0.0",
    lifespan=lifespan
)

def get_engine() -> ValuationEngine:
    global engine
    if engine is None:
        engine = ValuationEngine()
    return engine

class ValuationRequest(BaseModel):
    postcode: str = Field("B449HN", description="UK Postal Code")
    property_type: str = Field("S", description="D=Detached, S=Semi-Detached, T=Terraced, F=Flat, O=Other")
    is_new_build: str = Field("N", description="Y=New Build, N=Established")
    tenure: str = Field("F", description="F=Freehold, L=Leasehold")
    asking_price: Optional[float] = Field(50000.0, description="Optional asking or listing price in GBP")

class ValuationResponse(BaseModel):
    postcode: str
    street: str
    city: str
    county: str
    latitude: float
    longitude: float
    h3_cell: str
    fair_market_value_gbp: int
    liquidation_floor_10pct_gbp: int
    upside_ceiling_90pct_gbp: int
    confidence_band_width_gbp: int
    arbitrage_analysis: Optional[Dict[str, Any]] = None
    comparables: List[Dict[str, Any]]
    latency_ms: float

@app.get("/health", tags=["System"])
def health_check():
    eng = get_engine()
    return {
        "status": "HEALTHY",
        "service": "UK Real Estate Valuation Engine",
        "records_indexed": 1635889,
        "engine": "DuckDB Vectorized OLAP + LightGBM 4.7.0"
    }

@app.post("/predict", response_model=ValuationResponse, tags=["Valuation"])
def predict_valuation(req: ValuationRequest):
    eng = get_engine()
    t0 = time.time()
    res = eng.predict_valuation(
        postcode=req.postcode,
        property_type=req.property_type.upper(),
        is_new_build=req.is_new_build.upper(),
        tenure=req.tenure.upper(),
        asking_price=req.asking_price
    )
    res["latency_ms"] = round((time.time() - t0) * 1000.0, 2)
    return res

@app.get("/arbitrage/top", tags=["Arbitrage"])
def get_top_arbitrage(limit: int = Query(25, ge=1, le=100)):
    eng = get_engine()
    deals = eng.get_top_arbitrage_deals(limit=limit)
    return {
        "total_deals_returned": len(deals),
        "deals": deals
    }

@app.get("/comps", tags=["Valuation"])
def get_comps(
    postcode: str = Query("B449HN"),
    property_type: str = Query("S"),
    tenure: str = Query("F"),
    limit: int = Query(5, ge=1, le=20)
):
    eng = get_engine()
    clean_pc = postcode.strip().replace(" ", "").upper()
    pc_info = eng.lookup_postcode(clean_pc)
    if not pc_info:
        raise HTTPException(status_code=404, detail=f"Postcode '{postcode}' not found in database.")
    
    import h3
    cell_res7 = h3.latlng_to_cell(pc_info["latitude"], pc_info["longitude"], 7)
    comps = eng.get_comps(pc_info["latitude"], pc_info["longitude"], cell_res7, property_type.upper(), tenure.upper(), limit=limit)
    return {
        "postcode": clean_pc,
        "h3_cell": cell_res7,
        "count": len(comps),
        "comparables": comps
    }
