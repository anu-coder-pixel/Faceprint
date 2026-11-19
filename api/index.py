"""
Serverless entrypoint for Vercel.
Exposes the FastAPI application instance `app`.
"""

from api.detect import app

__all__ = ["app"]
