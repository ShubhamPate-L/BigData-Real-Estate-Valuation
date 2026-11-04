"""
Streamlit Web Application: Enterprise Real Estate Valuation & Arbitrage Detection
Interactive Big Data dashboard powered by DuckDB, LightGBM, and Uber H3 hexagonal spatial indexing.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
from app.engine import ValuationEngine

st.set_page_config(
    page_title="UK Real Estate Enterprise Valuation Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-deal {
        background-color: #16a34a;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return ValuationEngine()

engine = get_engine()

# Header
st.title("🏠 Enterprise Real Estate Valuation & Arbitrage Platform")
st.markdown("""  
An enterprise Big Data platform that analyzes **1.63M+ UK residential property transactions** (HM Land Registry) to deliver real-time fair market valuations, **10%–90% risk confidence bands**, hyper-local comparable sales (comps), and automated detection of **undervalued real estate arbitrage opportunities**.
""")
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Scale-1.63M_Records-orange?style=for-the-badge", use_column_width=True)
    st.image("https://img.shields.io/badge/DuckDB-Vectorized_OLAP-yellow?style=for-the-badge", use_column_width=True)
    st.image("https://img.shields.io/badge/Uber_H3-24.6K_Hexagons-blue?style=for-the-badge", use_column_width=True)
    
    st.subheader("💡 Quick Demo Presets")
    preset = st.selectbox(
        "Choose a real listing from 2023:",
        [
            "Custom Input",
            "B449HN — Kings Road, Birmingham (79.2% Arbitrage Gem)",
            "DE216GB — Renfrew St, Derby (86.3% Discount)",
            "RM139JJ — Rainham, Greater London (89.5% Discount)",
            "SW1A1AA — Westminster, Central London",
            "M11AD — Central Manchester"
        ]
    )

tabs = st.tabs([
    "🔮 Real-Time Property Valuator & Comps",
    "💎 Arbitrage & Hidden Gems Explorer",
    "📊 Big Data OLAP Analytics"
])

# ---------------------------------------------------------
# TAB 1: Real-Time Valuation & Comps Engine
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Interactive Valuation & Uncertainty Risk Bands")
    
    # Pre-fill based on preset
    default_pc = "B449HN"
    default_type = "Semi-Detached"
    default_price = 50000.0
    
    if "B449HN" in preset:
        default_pc = "B449HN"
        default_type = "Semi-Detached"
        default_price = 50000.0
    elif "DE216GB" in preset:
        default_pc = "DE216GB"
        default_type = "Other"
        default_price = 60000.0
    elif "RM139JJ" in preset:
        default_pc = "RM139JJ"
        default_type = "Other"
        default_price = 50000.0
    elif "SW1A1AA" in preset:
        default_pc = "SW1A1AA"
        default_type = "Flat / Maisonette"
        default_price = 1200000.0
    elif "M11AD" in preset:
        default_pc = "M11AD"
        default_type = "Flat / Maisonette"
        default_price = 220000.0

    col_in1, col_in2, col_in3, col_in4 = st.columns([2, 2, 2, 2])
    with col_in1:
        postcode_input = st.text_input("UK Postal Code", value=default_pc).strip()
    with col_in2:
        prop_type_label = st.selectbox("Property Type", ["Semi-Detached", "Detached", "Terraced", "Flat / Maisonette", "Other"], index=["Semi-Detached", "Detached", "Terraced", "Flat / Maisonette", "Other"].index(default_type))
        type_code_map = {"Detached": "D", "Semi-Detached": "S", "Terraced": "T", "Flat / Maisonette": "F", "Other": "O"}
        property_type = type_code_map[prop_type_label]
    with col_in3:
        tenure_label = st.selectbox("Tenure", ["Freehold", "Leasehold"], index=0)
        tenure = "F" if tenure_label == "Freehold" else "L"
        new_build_label = st.checkbox("New Construction Build", value=False)
        is_new_build = "Y" if new_build_label else "N"
    with col_in4:
        asking_price = st.number_input("Asking / Listed Price (£)", min_value=0.0, value=default_price, step=5000.0)

    if st.button("🚀 Calculate Enterprise Valuation", type="primary"):
        with st.spinner("Executing DuckDB spatial indexing & LightGBM inference..."):
            val_result = engine.predict_valuation(
                postcode=postcode_input,
                property_type=property_type,
                is_new_build=is_new_build,
                tenure=tenure,
                asking_price=asking_price if asking_price > 0 else None
            )

        st.success(f"Valuation completed for **{val_result['street']}, {val_result['city']}** ({val_result['county']}) — H3 Hex Cell: `{val_result['h3_cell']}`")

        # Metric Cards
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f"""
            <div class="main-metric-card">
                <div class="metric-label">Liquidation Floor (Q10)</div>
                <div class="metric-value" style="color: #f87171;">£{val_result['liquidation_floor_10pct_gbp']:,}</div>
                <small style="color: #94a3b8;">Downside Risk Boundary</small>
            </div>
            """, unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"""
            <div class="main-metric-card" style="border-color: #3b82f6;">
                <div class="metric-label">Fair Market Value</div>
                <div class="metric-value" style="color: #60a5fa;">£{val_result['fair_market_value_gbp']:,}</div>
                <small style="color: #94a3b8;">LightGBM Regression Point</small>
            </div>
            """, unsafe_allow_html=True)
        with mcol3:
            st.markdown(f"""
            <div class="main-metric-card">
                <div class="metric-label">Upside Ceiling (Q90)</div>
                <div class="metric-value" style="color: #34d399;">£{val_result['upside_ceiling_90pct_gbp']:,}</div>
                <small style="color: #94a3b8;">Upper 90% Valuation Bound</small>
            </div>
            """, unsafe_allow_html=True)
        with mcol4:
            st.markdown(f"""
            <div class="main-metric-card">
                <div class="metric-label">80% Confidence Band</div>
                <div class="metric-value" style="color: #fbbf24;">£{val_result['confidence_band_width_gbp']:,}</div>
                <small style="color: #94a3b8;">Market Volatility Spread</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Arbitrage Assessment Alert
        arb = val_result.get("arbitrage_analysis")
        if arb:
            if arb["is_arbitrage_deal"]:
                st.markdown(f"""
                <div style="background-color: #14532d; border: 1px solid #22c55e; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                    <h4 style="color: #4ade80; margin: 0;">💎 VERIFIED ARBITRAGE OPPORTUNITY DETECTED!</h4>
                    <p style="color: #dcfce7; margin: 5px 0 0 0;">
                        Asking Price of <b>£{arb['asking_price']:,.0f}</b> is <b>{arb['discount_pct']}% below fair market value</b> (Savings of <b>£{arb['discount_gbp']:,}</b>).
                        The listing falls strictly below the 10th percentile liquidation floor of £{val_result['liquidation_floor_10pct_gbp']:,}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"Market Status: **{arb['assessment']}** | Discount vs Fair Value: **{arb['discount_pct']}%** (£{arb['discount_gbp']:,})")

        # Comps Table
        st.subheader("📍 Top 5 Local Comparable Properties (Same Hexagon)")
        comps = val_result.get("comparables", [])
        if comps:
            comps_df = pd.DataFrame(comps)
            comps_df.columns = ["Street", "City", "Postcode", "Price (£)", "Sale Date", "Distance (km)"]
            comps_df["Price (£)"] = comps_df["Price (£)"].apply(lambda x: f"£{x:,}")
            comps_df["Distance (km)"] = comps_df["Distance (km)"].apply(lambda x: f"{x:.2f} km")
            st.dataframe(comps_df, use_container_width=True)
        else:
            st.info("No immediate comps found in the same micro-market cell.")

# ---------------------------------------------------------
# TAB 2: Arbitrage & Hidden Gems Explorer
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("💎 National Arbitrage Explorer (Novelty 3)")
    st.markdown("Statistically verified underpriced property listings from H2 2023 filtered for minimum liquidity, price validity, and risk boundary clearance.")
    
    arb_deals = engine.get_top_arbitrage_deals(limit=50)
    if arb_deals:
        df_arb = pd.DataFrame(arb_deals)
        
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            counties = ["ALL"] + sorted(list(df_arb["county"].unique()))
            sel_county = st.selectbox("Filter by County", counties)
        with fcol2:
            min_disc = st.slider("Minimum Discount (%)", min_value=20, max_value=90, value=50, step=5)
        with fcol3:
            max_p = st.number_input("Maximum Price (£)", value=500000, step=25000)
            
        filtered_arb = df_arb[
            (df_arb["discount_pct"] >= min_disc) &
            (df_arb["price"] <= max_p)
        ]
        if sel_county != "ALL":
            filtered_arb = filtered_arb[filtered_arb["county"] == sel_county]
            
        st.write(f"Showing **{len(filtered_arb)}** verified opportunities matching criteria:")
        
        display_df = filtered_arb[[
            "street", "city", "county", "postcode", "price", "pred_floor_q10",
            "pred_fair_price", "discount_gbp", "discount_pct", "arbitrage_score"
        ]].copy()
        display_df.columns = ["Street", "City", "County", "Postcode", "Sold Price (£)", "Floor Q10 (£)", "Fair Value (£)", "Discount (£)", "Discount (%)", "AOS Score"]
        
        st.dataframe(
            display_df.style.format({
                "Sold Price (£)": "£{:,.0f}",
                "Floor Q10 (£)": "£{:,.0f}",
                "Fair Value (£)": "£{:,.0f}",
                "Discount (£)": "£{:,.0f}",
                "Discount (%)": "{:.1f}%",
                "AOS Score": "{:.2f}"
            }),
            use_container_width=True
        )
    else:
        st.warning("Arbitrage dataset not found. Please ensure Phase 4 script has been run.")

# ---------------------------------------------------------
# TAB 3: Big Data OLAP Analytics
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("📊 Enterprise DuckDB OLAP Analytics (1.63M Rows)")
    st.markdown("Live sub-50ms analytical queries directly over columnar Snappy Parquet files.")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("#### Median Property Price by Dwelling Type")
        type_res = engine.con.execute("""
            SELECT 
                CASE property_type 
                    WHEN 'D' THEN 'Detached' 
                    WHEN 'S' THEN 'Semi-Detached' 
                    WHEN 'T' THEN 'Terraced' 
                    WHEN 'F' THEN 'Flat / Maisonette' 
                    ELSE 'Other' 
                END AS dwelling,
                MEDIAN(price)::BIGINT AS median_price,
                COUNT(*) AS sales_count
            FROM properties
            GROUP BY 1
            ORDER BY median_price DESC;
        """).df()
        st.bar_chart(type_res.set_index("dwelling")["median_price"])
        
    with col_stat2:
        st.markdown("#### Bank of England Rate Hike Cycle vs Property Valuations")
        boe_trend = engine.con.execute("""
            SELECT 
                sale_year || '-' || LPAD(sale_month::VARCHAR, 2, '0') AS period,
                ROUND(MEDIAN(price), 0)::BIGINT AS median_property_price,
                AVG(boe_base_rate) AS boe_rate_pct
            FROM feature_matrix
            GROUP BY 1
            ORDER BY 1;
        """).df()
        st.line_chart(boe_trend.set_index("period")["median_property_price"])

    st.markdown("#### Top 10 Counties by Transaction Density")
    county_res = engine.con.execute("""
        SELECT county, COUNT(*) AS sales_volume, ROUND(MEDIAN(price), 0)::BIGINT AS median_price
        FROM properties
        WHERE county != 'UNKNOWN'
        GROUP BY county
        ORDER BY sales_volume DESC
        LIMIT 10;
    """).df()
    st.dataframe(county_res, use_container_width=True)

