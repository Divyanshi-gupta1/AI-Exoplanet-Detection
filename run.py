#!/usr/bin/env python3
"""ExoDip Standalone Web Application Launcher.

Starts the FastAPI backend server and serves the modern frontend at http://localhost:8000.
"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))

    print("=" * 65)
    print("🪐  ExoDip: AI Exoplanet Candidate Screening Web Application")
    print(f"🚀  Running at: http://localhost:{port}")
    print(f"📖  API Docs:   http://localhost:{port}/docs")
    print("=" * 65)

    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
