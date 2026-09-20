"""Input/Output and Data Loading Helpers for ExoDip (Pure-NumPy & Standard Library)."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"


def read_light_curve(raw_bytes: bytes, filename: str = "") -> np.ndarray:
    """Read a raw 1-D flux series from uploaded bytes without requiring pandas.
    
    Supports 1-row, 1-col, Kepler/TESS format, multi-column formats, and NPY arrays.
    """
    name = filename.lower()
    if name.endswith(".npy"):
        array = np.asarray(np.load(io.BytesIO(raw_bytes)), dtype=float)
        return array[:, -1].ravel() if array.ndim == 2 else array.ravel()

    text = raw_bytes.decode("utf-8-sig", errors="replace").strip()
    if not text:
        raise ValueError("Uploaded file is empty.")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("No valid lines found in file.")

    # Detect delimiter: comma, tab, or whitespace
    first_line = lines[0]
    if "," in first_line:
        reader = csv.reader(lines, delimiter=",")
    elif "\t" in first_line:
        reader = csv.reader(lines, delimiter="\t")
    else:
        reader = [line.split() for line in lines]

    raw_rows = [row for row in reader if row]
    if not raw_rows:
        raise ValueError("No data found in file.")

    # Check for horizontal 1-row or 2-row flux arrays (e.g. FLUX.1, FLUX.2, ...)
    if len(raw_rows) <= 2 and len(raw_rows[0]) >= 30:
        target_row = raw_rows[0] if len(raw_rows) == 1 else raw_rows[1]
        start_idx = 1 if any(word in str(target_row[0]).lower() for word in ["label", "id"]) else 0
        vals = []
        for item in target_row[start_idx:]:
            try:
                vals.append(float(item))
            except (ValueError, TypeError):
                continue
        if len(vals) >= 30:
            return np.array(vals, dtype=float)

    # Detect if row 0 is header
    header = [str(c).strip().lower() for c in raw_rows[0]]
    has_header = False
    for col_name in header:
        try:
            float(col_name)
        except ValueError:
            has_header = True
            break

    data_rows = raw_rows[1:] if has_header else raw_rows
    if not data_rows:
        raise ValueError("Provide at least 30 numeric flux values.")

    # Match prioritized flux column name
    flux_col_idx = None
    if has_header:
        for candidate in ["flux", "pdcsap_flux", "sap_flux", "relative_flux", "norm_flux", "normalized_flux", "raw_flux"]:
            if candidate in header:
                flux_col_idx = header.index(candidate)
                break
        if flux_col_idx is None:
            for idx, h in enumerate(header):
                if "flux" in h and "err" not in h and "unc" not in h:
                    flux_col_idx = idx
                    break

    # If no header or named flux column, pick the last non-excluded numeric column
    num_cols = len(data_rows[0])
    if flux_col_idx is None:
        if has_header:
            valid_indices = [
                idx for idx, h in enumerate(header)
                if not any(bad in h for bad in ["time", "err", "qual", "cadence", "bjd", "index", "phase"])
            ]
            flux_col_idx = valid_indices[-1] if valid_indices else (num_cols - 1)
        else:
            flux_col_idx = num_cols - 1

    values = []
    for row in data_rows:
        if flux_col_idx < len(row):
            try:
                val = float(row[flux_col_idx])
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
            except (ValueError, TypeError):
                continue

    if len(values) < 30:
        raise ValueError("Provide at least 30 numeric flux values. A CSV with a `flux` column is recommended.")

    return np.asarray(values, dtype=float)


def load_test_row(row_id: int) -> Tuple[np.ndarray, str]:
    """Loads a specific row from the Kepler exoTest.csv dataset using standard library csv."""
    test_csv = DATA_DIR / "exoTest.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_csv}")

    with open(test_csv, mode="r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for idx, row in enumerate(reader):
            if idx == row_id:
                # row[0] is LABEL, row[1:] are the 3197 flux readings
                flux = np.array([float(x) for x in row[1:]], dtype=float)
                return flux, f"Test-set row {row_id}"

    raise IndexError(f"Row ID {row_id} is out of range.")


def test_set_size() -> int:
    """Returns the total number of rows available in exoTest.csv."""
    test_csv = DATA_DIR / "exoTest.csv"
    if not test_csv.exists():
        return 0
    try:
        with open(test_csv, mode="r", encoding="utf-8-sig") as f:
            # count lines minus header
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def sample_csv() -> str:
    """Generates a valid demo Kepler light curve CSV with confirmed exoplanet transit."""
    test_path = DATA_DIR / "exoTest.csv"
    if test_path.exists():
        try:
            flux, _ = load_test_row(1)
            lines = ["flux"] + [str(v) for v in flux]
            return "\n".join(lines)
        except Exception:
            pass

    time = np.linspace(0, 6, 120)
    flux = 1 - 0.012 * np.exp(-((time - 3) ** 2) / 0.035) + 0.0005 * np.sin(time * 13)
    lines = ["time,flux"]
    for t, fx in zip(time, flux):
        lines.append(f"{t:.4f},{fx:.6f}")
    return "\n".join(lines)
