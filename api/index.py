"""
Serverless entrypoint for Vercel.
Exposes the FastAPI application instance `app`.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from api.detect import app
except Exception:
    from detect import app

__all__ = ["app"]
