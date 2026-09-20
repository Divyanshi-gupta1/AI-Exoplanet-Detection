"""Input/Output and Data Loading Helpers for ExoDip."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"


def read_light_curve(raw_bytes: bytes, filename: str = "") -> np.ndarray:
    """Read a raw 1-D flux series from uploaded bytes.
    
    Supports 1-row, 1-col, Kepler/TESS format, multi-column formats, and NPY arrays.
    Exactly preserves the proven logic from app.py.
    """
    name = filename.lower()
    if name.endswith(".npy"):
        array = np.asarray(np.load(io.BytesIO(raw_bytes)), dtype=float)
        return array[:, -1].ravel() if array.ndim == 2 else array.ravel()

    try:
        frame = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8-sig", errors="replace")))
    except Exception:
        frame = pd.read_csv(io.StringIO(raw_bytes.decode("utf-8-sig", errors="replace")), sep=r"\s+")

    # Case 1: 1-row or 2-row CSV containing a series of flux columns (e.g. 3197 values, FLUX.1, FLUX.2, ...)
    if frame.shape[0] <= 2 and frame.shape[1] >= 30:
        start_col = 1 if "label" in str(frame.columns[0]).lower() else 0
        row_vals = pd.to_numeric(frame.iloc[0, start_col:], errors="coerce").dropna().to_numpy(dtype=float)
        if len(row_vals) >= 30:
            return row_vals

    col_map = {str(c).strip().lower(): c for c in frame.columns}

    # Case 2: Named flux column (prioritized)
    flux_col = None
    for candidate in ["flux", "pdcsap_flux", "sap_flux", "relative_flux", "norm_flux", "normalized_flux", "raw_flux"]:
        if candidate in col_map:
            flux_col = col_map[candidate]
            break

    # Fuzzy match: any column name containing 'flux' but NOT 'err' or 'unc'
    if flux_col is None:
        for low_c, orig_c in col_map.items():
            if "flux" in low_c and "err" not in low_c and "unc" not in low_c:
                flux_col = orig_c
                break

    if flux_col is not None:
        values = pd.to_numeric(frame[flux_col], errors="coerce").dropna().to_numpy(dtype=float)
    else:
        # Case 3: Pick the best numeric column, avoiding time/error/quality/cadence columns
        numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        candidate_cols = [
            c
            for c in numeric.columns
            if not any(bad in str(c).lower() for bad in ["time", "err", "qual", "cadence", "bjd", "index", "phase"])
        ]
        target_col = candidate_cols[-1] if candidate_cols else (numeric.columns[-1] if len(numeric.columns) else None)
        values = numeric[target_col].dropna().to_numpy(dtype=float) if target_col is not None else np.array([])

    if len(values) < 30:
        raise ValueError("Provide at least 30 numeric flux values. A CSV with a `flux` column is recommended.")
    return np.asarray(values, dtype=float)


def load_test_row(row_id: int) -> Tuple[np.ndarray, str]:
    """Loads a specific row from the Kepler exoTest.csv dataset."""
    test_csv = DATA_DIR / "exoTest.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_csv}")
    row = pd.read_csv(test_csv, nrows=1, skiprows=range(1, row_id + 1))
    return row.iloc[0, 1:].to_numpy(dtype=float), f"Test-set row {row_id}"


def test_set_size() -> int:
    """Returns the total number of rows available in exoTest.csv."""
    test_csv = DATA_DIR / "exoTest.csv"
    if not test_csv.exists():
        return 0
    return len(pd.read_csv(test_csv, usecols=["LABEL"]))


def sample_csv() -> str:
    """Generates a valid demo Kepler light curve CSV with confirmed exoplanet transit."""
    test_path = DATA_DIR / "exoTest.csv"
    if test_path.exists():
        row = pd.read_csv(test_path, nrows=1, skiprows=1)
        flux = row.iloc[0, 1:].to_numpy(dtype=float)
        return pd.DataFrame({"flux": flux}).to_csv(index=False)
    time = np.linspace(0, 6, 120)
    flux = 1 - 0.012 * np.exp(-((time - 3) ** 2) / 0.035) + 0.0005 * np.sin(time * 13)
    return pd.DataFrame({"time": time, "flux": flux}).to_csv(index=False)
