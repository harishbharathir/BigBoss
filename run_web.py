"""Launcher for BiggBoss CCTV Video Intelligence Web Dashboard.

Run this script to launch the local web server:
    python run_web.py
Then open your browser to:
    http://localhost:8000
"""

import sys
from pathlib import Path

# Add project root to sys.path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  BiggBoss Multi-Stream CCTV Video Intelligence Platform")
    print("  Enterprise HTML5 Web Dashboard running on FastAPI & Uvicorn")
    print("  Dashboard URL: http://localhost:8000")
    print("=" * 70)
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
