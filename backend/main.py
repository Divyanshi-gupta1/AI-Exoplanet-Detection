"""ExoDip FastAPI Backend Application."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.services.analysis import run_screening
from backend.services.io_helpers import (
    load_test_row,
    read_light_curve,
    sample_csv,
    test_set_size,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exodip")

app = FastAPI(
    title="ExoDip API",
    description="AI-Assisted Exoplanet Transit Candidate Screening API",
    version="1.0.0",
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class TestRowRequest(BaseModel):
    row_id: int = Field(..., ge=0, description="Row index from Kepler exoTest.csv dataset.")


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "app": "ExoDip", "version": "1.0.0"}


@app.get("/api/test-set-info")
def get_test_set_info() -> Dict[str, Any]:
    """Retrieve metadata about the held-out Kepler test dataset."""
    total = test_set_size()
    return {"total_rows": total, "max_row_id": max(0, total - 1)}


@app.get("/api/sample-csv")
def download_sample_csv() -> Response:
    """Download a valid demonstration Kepler light curve with an exoplanet transit."""
    content = sample_csv()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=kepler_sample_light_curve.csv"},
    )


@app.post("/api/analyze/file")
async def analyze_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and screen a light-curve file (CSV, TXT, or NPY)."""
    filename = file.filename or "uploaded_light_curve"
    logger.info(f"Received file upload: {filename}")

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        flux = read_light_curve(raw_bytes, filename=filename)
        logger.info(f"Successfully parsed {len(flux)} flux data points from {filename}")

        result = run_screening(flux, source_name=filename)
        return result

    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"Validation error parsing {filename}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        logger.error(f"Error processing {filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process light curve: {exc}")


@app.post("/api/analyze/test-row")
def analyze_test_row(req: TestRowRequest) -> Dict[str, Any]:
    """Screen an exoplanet transit observation from the held-out test dataset."""
    logger.info(f"Analyzing held-out test row: {req.row_id}")
    try:
        flux, source_name = load_test_row(req.row_id)
        result = run_screening(flux, source_name=source_name)
        return result
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe))
    except IndexError:
        raise HTTPException(status_code=400, detail=f"Row ID {req.row_id} is out of range.")
    except Exception as exc:
        logger.error(f"Error processing test row {req.row_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to screen test row: {exc}")


# Mount the frontend Single Page Application
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
