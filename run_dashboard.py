"""
Launch the Streamlit Web Application
Access web UI at: http://localhost:8501
"""
import sys
import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    dashboard_path = os.path.join(BASE_DIR, "app", "dashboard.py")
    print("=" * 65)
    print("Starting Streamlit Web Dashboard...")
    print(f"Target file: {dashboard_path}")
    print("Opening in browser: http://localhost:8501")
    print("=" * 65)
    
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    subprocess.run(cmd)

