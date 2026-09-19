"""ExoDip Scientific Analysis Service.

Bridges incoming photometric flux into the existing scientific pipeline
(exoplanet_pipeline.py and exoplanet_detector.py) and computes visualization
telemetry (smoothing, dip thresholding, and transit morphology windows).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.ndimage import uniform_filter1d

# Ensure project root is on sys.path to import existing pipeline modules directly
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exoplanet_pipeline import predict_light_curve


def run_screening(flux: np.ndarray, source_name: str) -> Dict[str, Any]:
    """Execute the complete exoplanet candidate screening pipeline on 1-D flux data."""
    flux = np.asarray(flux, dtype=float).ravel()
    if len(flux) < 30:
        raise ValueError("A light curve must contain at least 30 numeric flux samples.")

    # 1. Run proven scientific exoplanet detection pipeline
    result = predict_light_curve(flux)

    # 2. Compute confidence score percentage (matches app.py logic)
    detected = result["prediction"] == "Planet"
    conf = result.get("confidence")
    if conf is not None:
        confidence_pct = float(conf * 100 if conf <= 1.0 else conf)
    else:
        prob = float(result.get("probability", 0.5))
        confidence_pct = float((prob if detected else (1.0 - prob)) * 100)

    # 3. Compute chart smoothing and transit dip detection (identical to make_chart in app.py)
    smooth_window = min(25, max(3, len(flux) // 20))
    smooth = uniform_filter1d(flux, size=smooth_window)
    threshold = smooth.mean() - 2.5 * smooth.std()
    dips = np.flatnonzero(smooth < threshold)

    # 4. Compute transit zoomed view geometry (identical to make_transit_view_chart in app.py)
    if len(dips) > 0:
        deepest_idx = int(dips[np.argmin(smooth[dips])])
        window = max(30, min(140, len(flux) // 10))
        zoom_start = int(max(0, deepest_idx - window))
        zoom_end = int(min(len(flux), deepest_idx + window))
        local_dips = [int(d) for d in dips if zoom_start <= d < zoom_end]
        has_deepest_dip = True
    else:
        deepest_idx = -1
        zoom_start = 0
        zoom_end = int(min(len(flux), 150))
        local_dips = []
        has_deepest_dip = False

    now_utc = datetime.now(timezone.utc)

    # Sanitize NaN/Inf in features if any
    clean_features = {
        k: (0.0 if not np.isfinite(v) else round(float(v), 6))
        for k, v in result.get("features", {}).items()
    }

    # Model scores table preparation
    model_scores = result.get("model_scores", {})

    return {
        "id": now_utc.isoformat(),
        "source": source_name,
        "timestamp": datetime.now().strftime("%b %d, %Y %H:%M"),
        "prediction": result["prediction"],
        "probability": round(float(result.get("probability", 0.0)), 4),
        "confidence": round(confidence_pct, 2),
        "model": result.get("model", "XGBoost"),
        "selection_metric": result.get("selection_metric", "Composite Comparison"),
        "metrics": result.get("metrics", {}),
        "model_probabilities": {
            m: round(float(p), 4) for m, p in result.get("model_probabilities", {}).items()
        },
        "model_scores": model_scores,
        "unavailable_models": result.get("unavailable_models", {}),
        "features": clean_features,
        "chart_data": {
            "flux": [round(float(v), 6) for v in flux],
            "smooth": [round(float(v), 6) for v in smooth],
            "dips": [int(d) for d in dips],
            "zoom": {
                "start": zoom_start,
                "end": zoom_end,
                "deepest_idx": deepest_idx,
                "has_deepest_dip": has_deepest_dip,
                "local_dips": local_dips,
            },
        },
    }
