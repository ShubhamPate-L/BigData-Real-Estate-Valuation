"""
Launch the FastAPI Valuation & Comps Server
Access interactive Swagger UI at: http://127.0.0.1:8000/docs
"""
import sys
import os
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    print("=" * 65)
    print("Starting UK Real Estate Valuation & Comps API...")
    print("Interactive Documentation: http://127.0.0.1:8000/docs")
    print("Alternative Redoc:        http://127.0.0.1:8000/redoc")
    print("Health Endpoint:          http://127.0.0.1:8000/health")
    print("=" * 65)
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=False)

