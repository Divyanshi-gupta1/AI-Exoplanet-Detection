"""Held-out test-set error analysis for the saved RF and CNN models."""

import joblib
import numpy as np
import pandas as pd

from exoplanet_pipeline import DATA_DIR, extract_features


def _planet_labels(test_df):
    """Match the notebook label conversion: 1 -> non-planet, 2 -> planet."""
    return test_df["LABEL"].replace({1: 0, 2: 1}).to_numpy(dtype=int)


def _random_forest_report(raw_flux, labels):
    best_rf = joblib.load(DATA_DIR / "best_rf.pkl")
    feature_df = pd.DataFrame([extract_features(flux) for flux in raw_flux])
    model_input = feature_df.reindex(columns=best_rf.feature_names_in_, fill_value=0)
    probability = best_rf.predict_proba(model_input)[:, 1]
    misses = np.flatnonzero((labels == 1) & (probability < 0.5))
    return {
        "random_forest_misses": misses.tolist(),
        "random_forest_probabilities": probability[misses].tolist(),
    }


def _cnn_report(raw_flux, labels):
    try:
        import tensorflow as tf
    except ImportError as error:
        return {"cnn_misses": None, "cnn_error": f"TensorFlow is unavailable: {error}"}

    cnn = tf.keras.models.load_model(DATA_DIR / "cnn_exoplanet.keras", compile=False)
    expected_samples = cnn.input_shape[1]
    if raw_flux.shape[1] != expected_samples:
        return {
            "cnn_misses": None,
            "cnn_error": f"CNN expects {expected_samples} samples; exoTest.csv has {raw_flux.shape[1]}.",
        }
    normalized = (raw_flux - raw_flux.mean(axis=1, keepdims=True))
    normalized /= raw_flux.std(axis=1, keepdims=True) + 1e-8
    probability = cnn.predict(normalized[..., None].astype(np.float32), verbose=0).ravel()
    misses = np.flatnonzero((labels == 1) & (probability < 0.5))
    return {"cnn_misses": misses.tolist(), "cnn_probabilities": probability[misses].tolist()}


def run_error_analysis():
    """Return false-negative planet rows and probabilities for RF and CNN."""
    test_df = pd.read_csv(DATA_DIR / "exoTest.csv")
    labels = _planet_labels(test_df)
    raw_flux = test_df.iloc[:, 1:].to_numpy(dtype=float)
    report = _random_forest_report(raw_flux, labels)
    report.update(_cnn_report(raw_flux, labels))
    return report


if __name__ == "__main__":
    print(run_error_analysis())
