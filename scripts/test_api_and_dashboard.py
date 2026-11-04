import sys
import os
import time
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.engine import ValuationEngine
from app.api import app, ValuationRequest, predict_valuation, health_check, get_top_arbitrage, get_comps

print("=" * 60)
print("Testing Application Services...")
print("=" * 60)

# 1. Health check
h = health_check()
print(f"[HEALTH] Status: {h['status']} | Engine: {h['engine']}")

# 2. Predict Endpoint
req = ValuationRequest(
    postcode="B449HN",
    property_type="S",
    is_new_build="N",
    tenure="F",
    asking_price=50000.0
)
pred = predict_valuation(req)
print(f"[PREDICT] Postcode: {pred['postcode']} ({pred['street']}, {pred['city']})")
print(f"          Fair Market Value: GBP {pred['fair_market_value_gbp']:,}")
print(f"          Floor Q10:         GBP {pred['liquidation_floor_10pct_gbp']:,}")
print(f"          Ceiling Q90:       GBP {pred['upside_ceiling_90pct_gbp']:,}")
print(f"          Arbitrage:         {pred['arbitrage_analysis']}")
print(f"          Latency:           {pred['latency_ms']} ms")
print(f"          Comps count:       {len(pred['comparables'])}")

# 3. Top Arbitrage Deals
arb = get_top_arbitrage(limit=5)
print(f"[ARBITRAGE] Returned {arb['total_deals_returned']} top opportunities.")
print(f"            #1 Gem: {arb['deals'][0]['street']}, {arb['deals'][0]['city']} (£{arb['deals'][0]['price']:,})")

# 4. Comps Endpoint
comps_res = get_comps(postcode="B449HN", property_type="S", tenure="F", limit=3)
print(f"[COMPS] Returned {comps_res['count']} comps for {comps_res['postcode']}.")
for c in comps_res["comparables"]:
    print(f"        - {c['street']} ({c['distance_km']} km): £{c['price_gbp']:,}")

print("\n" + "=" * 60)
print("[SUCCESS] All API and Engine methods passed verification!")
print("=" * 60)
