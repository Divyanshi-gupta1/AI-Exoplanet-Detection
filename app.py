"""Streamlit frontend for ExoDip: AI-assisted screening of exoplanet transit candidates.

This application provides candidate screening for potential exoplanet transits
using Box Least Squares (BLS) period searching, astrophysical feature extraction,
and an ensemble of machine learning classifiers (primarily tuned XGBoost).
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.ndimage import uniform_filter1d

from exoplanet_pipeline import DATA_DIR, predict_light_curve

st.set_page_config(
    page_title="ExoDip - AI Exoplanet Transit Screening",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def setup_state() -> None:
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("active_result", None)


def set_page(page: str) -> None:
    st.session_state.page = page


def inject_styles() -> None:
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

        :root {
            --bg-space: #020308;
            --bg-card: rgba(5, 7, 14, 0.90);
            --bg-card-hover: rgba(10, 14, 26, 0.95);
            --border-subtle: rgba(255, 255, 255, 0.09);
            --border-highlight: rgba(99, 102, 241, 0.4);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-purple: #818cf8;
            --accent-emerald: #34d399;
            --accent-rose: #f43f5e;
            --accent-amber: #fbbf24;
        }

        /* Deep-Space Canvas & Dense High-Fidelity Star Field */
        .stApp {
            color: var(--text-main);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 16px;
            background-color: var(--bg-space);
            background-image:
                /* Dense Micro Stars Layer 1 (500x500 tile) */
                radial-gradient(1px 1px at 25px 35px, rgba(255,255,255,0.85), transparent),
                radial-gradient(1px 1px at 75px 120px, rgba(200,225,255,0.7), transparent),
                radial-gradient(1.2px 1.2px at 140px 60px, rgba(255,255,255,0.9), transparent),
                radial-gradient(0.8px 0.8px at 180px 220px, rgba(165,180,252,0.65), transparent),
                radial-gradient(1px 1px at 230px 140px, rgba(255,255,255,0.75), transparent),
                radial-gradient(1.4px 1.4px at 290px 45px, rgba(254,240,138,0.85), transparent),
                radial-gradient(0.8px 0.8px at 340px 190px, rgba(255,255,255,0.6), transparent),
                radial-gradient(1px 1px at 390px 95px, rgba(56,189,248,0.75), transparent),
                radial-gradient(1.2px 1.2px at 440px 240px, rgba(255,255,255,0.8), transparent),
                radial-gradient(0.8px 0.8px at 485px 165px, rgba(192,132,252,0.65), transparent),
                /* Dense Micro Stars Layer 2 (650x650 tile) */
                radial-gradient(1px 1px at 40px 310px, rgba(255,255,255,0.7), transparent),
                radial-gradient(1.3px 1.3px at 95px 460px, rgba(56,189,248,0.8), transparent),
                radial-gradient(0.8px 0.8px at 165px 380px, rgba(255,255,255,0.65), transparent),
                radial-gradient(1.1px 1.1px at 245px 520px, rgba(254,240,138,0.75), transparent),
                radial-gradient(1px 1px at 320px 410px, rgba(255,255,255,0.8), transparent),
                radial-gradient(1.5px 1.5px at 385px 580px, rgba(165,180,252,0.85), transparent),
                radial-gradient(0.8px 0.8px at 460px 340px, rgba(255,255,255,0.7), transparent),
                radial-gradient(1.2px 1.2px at 530px 490px, rgba(255,255,255,0.85), transparent),
                radial-gradient(0.9px 0.9px at 595px 395px, rgba(56,189,248,0.65), transparent),
                radial-gradient(1.3px 1.3px at 625px 560px, rgba(255,255,255,0.75), transparent),
                /* Layer 3 Medium & Bright Stars (800x800 tile) */
                radial-gradient(1.8px 1.8px at 110px 180px, rgba(255,255,255,0.95), transparent),
                radial-gradient(1.5px 1.5px at 280px 320px, rgba(129,140,248,0.9), transparent),
                radial-gradient(2px 2px at 470px 140px, rgba(254,240,138,0.95), transparent),
                radial-gradient(1.6px 1.6px at 660px 420px, rgba(56,189,248,0.9), transparent),
                radial-gradient(1.7px 1.7px at 740px 210px, rgba(255,255,255,0.95), transparent),
                /* Ultra-Dark Deep Space Cosmic Canvas */
                radial-gradient(ellipse at 85% 15%, rgba(79, 70, 229, 0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 15% 85%, rgba(14, 116, 144, 0.035) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(217, 119, 6, 0.015) 0%, transparent 65%);
            background-size: 
                500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px, 500px 500px,
                650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px, 650px 650px,
                800px 800px, 800px 800px, 800px 800px, 800px 800px, 800px 800px,
                100% 100%, 100% 100%, 100% 100%;
        }

        /* Subtle animated sparkling stars overlay */
        .stApp:before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                radial-gradient(1.5px 1.5px at 80px 90px, rgba(255,255,255,0.95), transparent),
                radial-gradient(2px 2px at 210px 340px, rgba(192,132,252,0.95), transparent),
                radial-gradient(1.6px 1.6px at 360px 170px, rgba(56,189,248,0.95), transparent),
                radial-gradient(2.2px 2.2px at 510px 420px, rgba(254,240,138,0.95), transparent),
                radial-gradient(1.5px 1.5px at 640px 110px, rgba(255,255,255,0.95), transparent),
                radial-gradient(1.8px 1.8px at 780px 310px, rgba(165,180,252,0.95), transparent);
            background-size: 700px 520px;
            animation: starTwinkle 6s ease-in-out infinite alternate;
            z-index: 0;
        }

        @keyframes starTwinkle {
            0% { opacity: 0.3; }
            50% { opacity: 0.9; }
            100% { opacity: 0.4; }
        }

        @media (prefers-reduced-motion: reduce) {
            .stApp:before { animation: none !important; opacity: 0.5; }
        }

        /* Header & Chrome Clean-up */
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }
        .block-container { max-width: 1440px; padding-top: 0.9rem; padding-bottom: 2.8rem; }

        /* Typography (+1, +2 size bump) */
        h1, h2, h3, h4 { font-family: 'Outfit', sans-serif; letter-spacing: -0.025em; color: #fff; }
        h1 { font-size: 3.15rem !important; line-height: 1.08; font-weight: 800; }
        h2 { font-size: 1.85rem !important; font-weight: 700; margin-bottom: 0.4rem; }
        h3 { font-size: 1.32rem !important; font-weight: 600; }
        h4 { font-size: 1.1rem !important; font-weight: 600; }
        .eyebrow {
            color: var(--accent-cyan);
            font-weight: 700;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            margin-bottom: 0.45rem;
        }
        .lead { color: #cbd5e1; font-size: 1.15rem; line-height: 1.75; }
        .purple-gradient {
            background: linear-gradient(135deg, #a78bfa 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .muted { color: var(--text-muted); font-size: 0.92rem; line-height: 1.55; }
        .mono { font-family: 'DM Mono', monospace; font-size: 0.88rem; }

        /* ========================================================
           CLEAN NAVBAR: Completely borderless, box-free until hovered
           ======================================================== */
        div[data-testid="stHorizontalBlock"]:has(#nav-brand) {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            padding: 0.4rem 0 !important;
            margin-bottom: 2rem !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
            -webkit-backdrop-filter: none !important;
            align-items: center !important;
        }

        div[data-testid="stHorizontalBlock"]:has(#nav-brand) div[data-testid="stColumn"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        div[data-testid="stHorizontalBlock"]:has(#nav-brand) div[data-testid="stColumn"]:first-child {
            justify-content: flex-start !important;
        }

        .nav-brand-box {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0.2rem 0;
            white-space: nowrap;
        }
        .nav-brand-box .logo-icon {
            font-size: 1.65rem;
            filter: drop-shadow(0 0 10px rgba(129, 140, 248, 0.7));
        }
        .brand-name {
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 1.45rem;
            color: #ffffff;
            letter-spacing: -0.02em;
        }
        .brand-pill {
            font-size: 0.72rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.1em;
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 8px;
            padding: 3px 8px;
            margin-left: 4px;
        }

        /* Nav links: Completely borderless, box-free and transparent by default */
        div[class*="st-key-nav_"] button,
        div[data-testid="stHorizontalBlock"]:has(#nav-brand) div[data-testid="stButton"] button {
            border: 1px solid transparent !important;
            background: transparent !important;
            color: #94a3b8 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            min-height: 2.3rem !important;
            height: 2.3rem !important;
            border-radius: 8px !important;
            padding: 0 1.1rem !important;
            box-shadow: none !important;
            outline: none !important;
            transition: all 0.2s ease !important;
        }

        /* Active page link: Clean illuminated cyan text, NO box/pill outline */
        div[class*="st-key-nav_"] button[kind="primary"],
        div[data-testid="stHorizontalBlock"]:has(#nav-brand) div[data-testid="stButton"] button[kind="primary"] {
            background: transparent !important;
            border: 1px solid transparent !important;
            color: #38bdf8 !important;
            font-weight: 700 !important;
            box-shadow: none !important;
            text-shadow: 0 0 12px rgba(56, 189, 248, 0.4) !important;
        }

        /* ONLY ON HOVER: Show a sleek modern box outline */
        div[class*="st-key-nav_"] button:hover,
        div[class*="st-key-nav_"] button[kind="primary"]:hover,
        div[data-testid="stHorizontalBlock"]:has(#nav-brand) div[data-testid="stButton"] button:hover {
            color: #ffffff !important;
            background: rgba(30, 41, 59, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4) !important;
        }

        /* Glassmorphism Cards & Panels */
        .panel, .stat-card, .result-card, .feature-card, .workflow-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 1.35rem;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .feature-card:hover {
            border-color: var(--border-highlight);
            transform: translateY(-2px);
        }

        /* Planetary Stage Container */
        .orbit-stage-3d {
            position: relative;
            width: 100%;
            height: 420px;
            background: radial-gradient(circle at 50% 50%, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.75) 45%, #030712 90%);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 60px rgba(0, 0, 0, 0.85);
        }

        /* Feature Cards */
        .feature-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            border-radius: 10px;
            background: rgba(99, 102, 241, 0.16);
            color: var(--accent-purple);
            font-size: 1.35rem;
            margin-bottom: 0.8rem;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .feature-card h4 { margin: 0 0 0.4rem; font-size: 1.1rem; color: #fff; font-weight: 600; }
        .feature-card p { color: var(--text-muted); font-size: 0.90rem; line-height: 1.6; margin: 0; }

        /* Workflow Step Banner */
        .workflow-banner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 0.75rem 1.25rem;
            margin-bottom: 1.6rem;
            font-size: 0.92rem;
            font-weight: 500;
        }
        .wf-step { color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
        .wf-step.active { color: var(--accent-cyan); font-weight: 700; }
        .wf-arrow { color: rgba(255, 255, 255, 0.2); font-size: 0.85rem; }

        /* Results Display */
        .candidate-badge-candidate {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 0.45rem 1rem;
            border-radius: 20px;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.12rem;
            background: rgba(52, 211, 153, 0.15);
            border: 1px solid rgba(52, 211, 153, 0.45);
            color: #34d399;
            box-shadow: 0 0 20px rgba(52, 211, 153, 0.22);
        }
        .candidate-badge-non {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 0.45rem 1rem;
            border-radius: 20px;
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.12rem;
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.3);
            color: #94a3b8;
        }

        .metric-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin: 1.1rem 0 1.35rem;
        }
        .metric-cell {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            text-align: left;
        }
        .metric-cell-label {
            font-size: 0.76rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 3px;
        }
        .metric-cell-value {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            font-size: 1.25rem;
            color: #fff;
        }

        /* Model Badge Chip */
        .model-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.4rem 0.85rem;
            border-radius: 8px;
            background: rgba(99, 102, 241, 0.16);
            border: 1px solid rgba(99, 102, 241, 0.38);
            color: #c7d2fe;
            font-size: 0.88rem;
            font-weight: 600;
        }

        /* Standard Buttons (outside navbar) */
        div[data-testid="stButton"] button {
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.14);
            background: rgba(30, 41, 59, 0.65);
            color: #f1f5f9;
            font-weight: 600;
            font-size: 0.98rem;
            min-height: 2.65rem;
            transition: all 0.2s ease;
        }
        div[data-testid="stButton"] button:hover {
            border-color: var(--accent-purple);
            background: rgba(99, 102, 241, 0.25);
            color: #fff;
        }
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-color: #818cf8;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35);
            color: #fff;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        }

        .stDownloadButton button {
            border-radius: 8px !important;
            border: 1px solid rgba(255, 255, 255, 0.14) !important;
            background: rgba(15, 23, 42, 0.8) !important;
            color: #fff !important;
            font-size: 0.95rem !important;
        }

        /* Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .stTabs [data-baseweb="tab"] {
            color: var(--text-muted);
            font-weight: 500;
            font-size: 1rem;
            padding-bottom: 0.65rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent-cyan) !important;
            border-bottom-color: var(--accent-cyan) !important;
            font-weight: 700;
        }

        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #38bdf8 0%, #34d399 100%) !important;
            border-radius: 4px;
        }

        /* Footer */
        .app-footer {
            margin-top: 3.8rem;
            padding-top: 1.35rem;
            border-top: 1px solid var(--border-subtle);
            text-align: center;
            color: var(--text-muted);
            font-size: 0.88rem;
        }
        </style>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def test_set_size() -> int:
    return len(pd.read_csv(DATA_DIR / "exoTest.csv", usecols=["LABEL"]))


def read_light_curve(upload) -> np.ndarray:
    """Read a raw 1-D flux series; supports 1-row, 1-col, Kepler/TESS, and multi-column formats."""
    name = getattr(upload, "name", "").lower()
    raw = upload.getvalue()
    if name.endswith(".npy"):
        array = np.asarray(np.load(io.BytesIO(raw)), dtype=float)
        return array[:, -1].ravel() if array.ndim == 2 else array.ravel()

    try:
        frame = pd.read_csv(io.StringIO(raw.decode("utf-8-sig", errors="replace")))
    except Exception:
        frame = pd.read_csv(io.StringIO(raw.decode("utf-8-sig", errors="replace")), sep=r"\s+")

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


def load_test_row(row_id: int) -> tuple[np.ndarray, str]:
    row = pd.read_csv(DATA_DIR / "exoTest.csv", nrows=1, skiprows=range(1, row_id + 1))
    return row.iloc[0, 1:].to_numpy(dtype=float), f"Test-set row {row_id}"


def sample_csv() -> str:
    """A valid demo Kepler light curve with confirmed exoplanet transit."""
    test_path = DATA_DIR / "exoTest.csv"
    if test_path.exists():
        row = pd.read_csv(test_path, nrows=1, skiprows=1)
        flux = row.iloc[0, 1:].to_numpy(dtype=float)
        return pd.DataFrame({"flux": flux}).to_csv(index=False)
    time = np.linspace(0, 6, 120)
    flux = 1 - 0.012 * np.exp(-((time - 3) ** 2) / 0.035) + 0.0005 * np.sin(time * 13)
    return pd.DataFrame({"time": time, "flux": flux}).to_csv(index=False)


def make_chart(flux: np.ndarray, title: str = "Light Curve with Detected Transit Dips") -> tuple[go.Figure, np.ndarray, np.ndarray]:
    smooth = uniform_filter1d(flux, size=min(25, max(3, len(flux) // 20)))
    threshold = smooth.mean() - 2.5 * smooth.std()
    dips = np.flatnonzero(smooth < threshold)
    fig = go.Figure()
    fig.add_scatter(
        y=flux,
        mode="lines",
        name="Observed Flux",
        line=dict(color="#818cf8", width=1),
        opacity=0.65,
    )
    fig.add_scatter(
        y=smooth,
        mode="lines",
        name="Smoothed Trend",
        line=dict(color="#38bdf8", width=2),
    )
    if len(dips):
        fig.add_scatter(
            x=dips,
            y=smooth[dips],
            mode="markers",
            name="Detected Dips",
            marker=dict(color="#f43f5e", size=5, symbol="circle"),
        )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#03050a",
        plot_bgcolor="#03050a",
        height=400,
        margin=dict(l=15, r=15, t=42, b=15),
        legend=dict(orientation="h", y=1.05),
        xaxis_title="Time / Sample Index",
        yaxis_title="Normalized Relative Flux",
    )
    fig.update_xaxes(gridcolor="#1b2a42", zeroline=False)
    fig.update_yaxes(gridcolor="#1b2a42", zeroline=False)
    return fig, smooth, dips


def make_transit_view_chart(flux: np.ndarray, smooth: np.ndarray, dips: np.ndarray) -> go.Figure:
    """Zoomed view focused on detected transit dip morphology."""
    fig = go.Figure()
    if len(dips) > 0:
        deepest_idx = dips[np.argmin(smooth[dips])]
        window = max(30, min(140, len(flux) // 10))
        start = max(0, deepest_idx - window)
        end = min(len(flux), deepest_idx + window)
        x_axis = np.arange(start, end)
        fig.add_scatter(
            x=x_axis,
            y=flux[start:end],
            mode="markers+lines",
            name="Observed Data Points",
            line=dict(color="#818cf8", width=1),
            marker=dict(size=4, color="#818cf8"),
            opacity=0.6,
        )
        fig.add_scatter(
            x=x_axis,
            y=smooth[start:end],
            mode="lines",
            name="Smoothed Profile",
            line=dict(color="#38bdf8", width=2.5),
        )
        local_dips = dips[(dips >= start) & (dips < end)]
        if len(local_dips):
            fig.add_scatter(
                x=local_dips,
                y=smooth[local_dips],
                mode="markers",
                name="Transit Ingress/Egress",
                marker=dict(color="#f43f5e", size=7, symbol="triangle-down"),
            )
        title = f"Deepest Transit Dip (Centered at Sample #{deepest_idx})"
    else:
        n_show = min(len(flux), 150)
        x_axis = np.arange(n_show)
        fig.add_scatter(
            x=x_axis,
            y=flux[:n_show],
            mode="lines",
            name="Observed Flux",
            line=dict(color="#818cf8", width=1),
        )
        fig.add_scatter(
            x=x_axis,
            y=smooth[:n_show],
            mode="lines",
            name="Smoothed Profile",
            line=dict(color="#38bdf8", width=2),
        )
        title = "Transit View (No Significant Dips Detected)"

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#03050a",
        plot_bgcolor="#03050a",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.05),
        xaxis_title="Sample Index",
        yaxis_title="Normalized Relative Flux",
    )
    fig.update_xaxes(gridcolor="#1b2a42", zeroline=False)
    fig.update_yaxes(gridcolor="#1b2a42", zeroline=False)
    return fig


def add_history(source: str, result: dict, flux: np.ndarray) -> None:
    detected = result["prediction"] == "Planet"
    conf = result.get("confidence")
    if conf is not None:
        conf_pct = float(conf * 100 if conf <= 1.0 else conf)
    else:
        prob = float(result.get("probability", 0.5))
        conf_pct = float((prob if detected else (1.0 - prob)) * 100)

    item = {
        "id": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "prediction": result["prediction"],
        "probability": result["probability"],
        "confidence": conf_pct,
        "model": result["model"],
        "timestamp": datetime.now().strftime("%b %d, %Y %H:%M"),
        "result": result,
        "flux": flux.tolist(),
    }
    st.session_state.history.insert(0, item)
    st.session_state.active_result = item


def run_analysis(flux: np.ndarray, source: str) -> None:
    with st.status("Screening candidate light curve…", expanded=True) as status:
        st.write("✓ Reading raw photometry")
        st.write("✓ Normalizing flux & noise reduction")
        st.write("✓ Box Least Squares (BLS) periodic transit search")
        st.write("✓ Extracting transit depth, duration, and shape features")
        st.write("✓ Executing ML model ensemble (XGBoost, RF, CNN, SVM)")
        result = predict_light_curve(flux)
        metric_val = result.get("metrics", {}).get("best_composite_score")
        metric_str = f"{metric_val:.4f}" if isinstance(metric_val, (int, float)) else str(result.get("selection_metric", "Composite Benchmark"))
        st.write(f"✓ Primary decision model: {result['model']} (Composite benchmark: {metric_str})")
        status.update(label="Screening complete!", state="complete", expanded=False)
    add_history(source, result, flux)
    set_page("Results")
    st.rerun()


def nav() -> None:
    # Single unified horizontal block styled with :has(#nav-brand)
    cols = st.columns([2.8, 0.8, 0.85, 0.85, 0.8, 0.8])
    cols[0].markdown(
        "<div id='nav-brand' class='nav-brand-box'><span class='logo-icon'>🪐</span> <span class='brand-name'>ExoDip</span> <span class='brand-pill'>AI Screening</span></div>",
        unsafe_allow_html=True,
    )
    for col, page in zip(cols[1:6], ("Home", "Analyze", "History", "About", "Docs")):
        is_active = st.session_state.page == page
        if col.button(
            page,
            key=f"nav_{page}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            set_page(page)
            st.rerun()


def home_page() -> None:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown("<div class='eyebrow'>AI-POWERED EXOPLANET CANDIDATE SCREENING</div>", unsafe_allow_html=True)
        st.markdown(
            "<h1>Discover Worlds<br><span class='purple-gradient'>Beyond Our Own</span></h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:1.24rem;font-weight:600;color:#38bdf8;margin:0.5rem 0 0.8rem;'>"
            "Find the dip. Discover the world.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='lead'>Analyze stellar light curves for periodic transit signals "
            "and use machine learning to screen potential exoplanet candidates.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        cta_a, cta_b = st.columns(2)
        if cta_a.button("Analyze Data", type="primary", use_container_width=True):
            set_page("Analyze")
            st.rerun()
        if cta_b.button("Learn More", use_container_width=True):
            set_page("Docs")
            st.rerun()

    with right:
        components.html('''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body {
    background: #010206;
    overflow: hidden;
    width: 100%;
    height: 100%;
  }
  #c {
    display: block;
    width: 100%;
    height: 420px;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: inset 0 0 100px rgba(0, 0, 0, 0.99);
  }
</style>
</head>
<body>
<canvas id="c"></canvas>
<script>
  const canvas = document.getElementById('c');
  const ctx = canvas.getContext('2d');

  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width > 0 ? rect.width : (window.innerWidth || 640);
    canvas.height = 420;
  }
  resize();
  window.addEventListener('resize', resize);

  // Background stars
  const stars = [];
  const starPalette = ['#ffffff', '#fef08a', '#93c5fd', '#fed7aa', '#cbd5e1'];
  for (let i = 0; i < 190; i++) {
    stars.push({
      x: Math.random() * 850,
      y: Math.random() * 420,
      r: Math.random() * 1.1 + 0.2,
      color: starPalette[Math.floor(Math.random() * starPalette.length)],
      baseAlpha: Math.random() * 0.70 + 0.15,
      speed: Math.random() * 0.02 + 0.005,
      phase: Math.random() * Math.PI * 2
    });
  }

  // 3D INCLINED PERSPECTIVE — SCALED TO FIT ENTIRELY INSIDE THE BOX
  // The whole planetary system (star, orbits, planets, asteroid belt) fits with comfortable margins
  const TILT = -7 * Math.PI / 180;
  const cosT = Math.cos(TILT);
  const sinT = Math.sin(TILT);

  // Scaled orbits: fully contained within canvas boundaries (width ~600, height 420)
  const orbits = [
    { a: 88,  b: 33,  speed: 0.50, stroke: 'rgba(251, 191, 36, 0.45)', width: 1.2 }, // Orbit 1: Inner Rocky
    { a: 155, b: 58,  speed: 0.28, stroke: 'rgba(56, 189, 248, 0.40)', width: 1.2 }, // Orbit 2: Earth
    { a: 238, b: 89,  speed: 0.16, stroke: 'rgba(245, 158, 11, 0.38)', width: 1.3 }  // Orbit 3: Jupiter
  ];

  // Asteroid Belt: revolving ring between Earth and Jupiter (radius 182 to 206)
  const asteroids = [];
  for (let i = 0; i < 90; i++) {
    const angle = Math.random() * Math.PI * 2;
    const dist = Math.random() * 24 + 182;
    asteroids.push({
      dist: dist,
      bDist: dist * 0.375,
      angle: angle,
      speed: 0.20 + (Math.random() - 0.5) * 0.03,
      size: Math.random() * 1.0 + 0.35,
      color: Math.random() > 0.45 ? 'rgba(214, 211, 209, ' : 'rgba(251, 191, 36, ',
      alpha: Math.random() * 0.45 + 0.20
    });
  }

  // 3 Distinct Planets
  const planets = [
    {
      oi: 0,
      angle: 0.9,
      r: 6.0,
      type: 'rocky'
    },
    {
      oi: 1,
      angle: 2.7,
      r: 10.5,
      type: 'earth',
      rotation: 0
    },
    {
      oi: 2,
      angle: 5.1,
      r: 15.0,
      type: 'jupiter'
    }
  ];

  let lastTime = null;

  function render(now) {
    if (!lastTime) lastTime = now;
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    const W = canvas.width;
    const H = canvas.height;

    // FOCAL CENTER: Positioned in the exact center so the entire system fits completely inside the box
    const cx = W * 0.50;
    const cy = H * 0.50;

    // 1. Ultra-Dark Space Background with warm ambient stellar radiance
    const bg = ctx.createRadialGradient(cx, cy, 25, cx, cy, Math.max(W, H) * 0.85);
    bg.addColorStop(0, 'rgba(32, 22, 12, 0.48)');     // Subtle warm golden solar ambiance
    bg.addColorStop(0.30, 'rgba(6, 10, 20, 0.85)');   // Deep dark midnight
    bg.addColorStop(0.70, 'rgba(2, 4, 10, 0.98)');
    bg.addColorStop(1, '#010206');                     // Deep space black
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Subtle galactic star dust wash across the background
    ctx.save();
    ctx.rotate(-0.25);
    const dustGrad = ctx.createLinearGradient(0, -H, W * 1.4, H * 1.4);
    dustGrad.addColorStop(0, 'transparent');
    dustGrad.addColorStop(0.48, 'rgba(129, 140, 248, 0.022)');
    dustGrad.addColorStop(0.52, 'rgba(251, 191, 36, 0.020)');
    dustGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = dustGrad;
    ctx.fillRect(-W * 0.5, -H * 0.5, W * 2, H * 2);
    ctx.restore();

    // 2. Stars
    stars.forEach(s => {
      const alpha = s.baseAlpha + Math.sin(now * s.speed + s.phase) * 0.15;
      ctx.beginPath();
      ctx.arc(s.x * (W / 850), s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.globalAlpha = Math.max(0.10, Math.min(1.0, alpha));
      ctx.fill();
    });
    ctx.globalAlpha = 1.0;

    // 3. Complete, Unclipped 3D Perspective Orbit Rings
    orbits.forEach(orb => {
      ctx.beginPath();
      const steps = 110;
      for (let i = 0; i <= steps; i++) {
        const theta = (i / steps) * Math.PI * 2;
        const ox = orb.a * Math.cos(theta);
        const oy = orb.b * Math.sin(theta);
        const rx = ox * cosT - oy * sinT;
        const ry = ox * sinT + oy * cosT;
        const px = cx + rx;
        const py = cy + ry;

        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();

      ctx.strokeStyle = orb.stroke;
      ctx.lineWidth = orb.width;
      ctx.stroke();
    });

    // 4. Asteroid Belt Particles
    asteroids.forEach(a => {
      a.angle = (a.angle + a.speed * dt) % (Math.PI * 2);
      const ax = a.dist * Math.cos(a.angle);
      const ay = a.bDist * Math.sin(a.angle);
      const rx = ax * cosT - ay * sinT;
      const ry = ax * sinT + ay * cosT;
      const px = cx + rx;
      const py = cy + ry;

      ctx.beginPath();
      ctx.arc(px, py, a.size, 0, Math.PI * 2);
      ctx.fillStyle = a.color + a.alpha + ')';
      ctx.fill();
    });

    // 5. Gather Render Queue for Depth Sorting
    const renderQueue = [];

    // Central Star at (cx, cy)
    renderQueue.push({
      type: 'star',
      depth: 0,
      x: cx,
      y: cy
    });

    // Planets
    planets.forEach(pl => {
      const orb = orbits[pl.oi];
      pl.angle = (pl.angle + orb.speed * dt) % (Math.PI * 2);
      if (pl.type === 'earth') pl.rotation += 0.35 * dt;

      const ox = orb.a * Math.cos(pl.angle);
      const oy = orb.b * Math.sin(pl.angle);
      const rx = ox * cosT - oy * sinT;
      const ry = ox * sinT + oy * cosT;
      const px = cx + rx;
      const py = cy + ry;

      // Depth based on sine (negative = behind star, positive = in front)
      const depth = Math.sin(pl.angle);

      renderQueue.push({
        type: 'planet',
        planet: pl,
        depth: depth,
        x: px,
        y: py,
        r: pl.r
      });
    });

    // Sort by depth (farthest behind rendered first, closest foreground rendered last)
    renderQueue.sort((a, b) => a.depth - b.depth);

    // 6. Draw in Depth Order
    renderQueue.forEach(item => {
      if (item.type === 'star') {
        drawStar(item.x, item.y);
      } else {
        drawPlanet(item, cx, cy);
      }
    });

    requestAnimationFrame(render);
  }

  // === COMPLETE, UNCLIPPED CENTRAL STAR ===
  // Entire circular body is 100% visible inside the box with radiant warm glow
  function drawStar(sx, sy) {
    const starR = 30; // Fits perfectly in center with generous margin

    // Soft, warm coronal halo radiating into space
    const corona = ctx.createRadialGradient(sx, sy, starR * 0.7, sx, sy, starR * 2.4);
    corona.addColorStop(0, 'rgba(254, 215, 170, 0.85)');
    corona.addColorStop(0.30, 'rgba(249, 115, 22, 0.40)');
    corona.addColorStop(0.70, 'rgba(234, 88, 12, 0.12)');
    corona.addColorStop(1, 'transparent');
    ctx.beginPath();
    ctx.arc(sx, sy, starR * 2.4, 0, Math.PI * 2);
    ctx.fillStyle = corona;
    ctx.fill();

    // Single cohesive yellow-orange star body (clean, warm, radiant)
    const bodyGrad = ctx.createRadialGradient(sx, sy, 0, sx, sy, starR);
    bodyGrad.addColorStop(0, '#fef08a');     // Luminous warm yellow center
    bodyGrad.addColorStop(0.38, '#fbbf24');  // Radiant golden yellow
    bodyGrad.addColorStop(0.75, '#f59e0b');  // Rich yellow-orange
    bodyGrad.addColorStop(1.0, '#ea580c');   // Warm orange surface limb
    ctx.beginPath();
    ctx.arc(sx, sy, starR, 0, Math.PI * 2);
    ctx.fillStyle = bodyGrad;
    ctx.fill();

    // Delicate line outwards of its surface (clean, fine single outer ring)
    ctx.beginPath();
    ctx.arc(sx, sy, starR + 5, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.75)';
    ctx.lineWidth = 1.1;
    ctx.stroke();
  }

  // === VOLUMETRIC REALISTIC PLANETS ===
  function drawPlanet(item, starX, starY) {
    const { x, y, r, planet } = item;

    // Vector pointing from planet toward the central star
    // Reduced offset (0.28 instead of 0.48) to avoid a shiny specular hotspot
    const angleToStar = Math.atan2(starY - y, starX - x);
    const lightDx = Math.cos(angleToStar) * (r * 0.28);
    const lightDy = Math.sin(angleToStar) * (r * 0.28);

    ctx.save();
    ctx.translate(x, y);

    if (planet.type === 'earth') {
      // -------------------------------------------------------------
      // 1. AUTHENTIC 3D EARTH
      // -------------------------------------------------------------
      ctx.save();
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.clip(); // Mask to sphere

      // Ocean base with soft directional sunlight (matte, not shiny)
      const oceanGrad = ctx.createRadialGradient(lightDx, lightDy, r * 0.35, 0, 0, r);
      oceanGrad.addColorStop(0, '#2563eb');    // Muted sunlit blue (not bright/white)
      oceanGrad.addColorStop(0.40, '#1d4ed8'); // Deep blue sea
      oceanGrad.addColorStop(0.75, '#1e3a8a'); // Twilight ocean
      oceanGrad.addColorStop(1.0, '#020617');  // Night side shadow
      ctx.fillStyle = oceanGrad;
      ctx.fill();

      // Continents (Green landmasses)
      const rot = planet.rotation;
      ctx.fillStyle = 'rgba(34, 197, 94, 0.85)';
      ctx.beginPath();
      ctx.ellipse(lightDx * 0.5 + Math.cos(rot) * 2 - 1, lightDy * 0.5 - 2.5, r * 0.48, r * 0.35, 0.35, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(21, 128, 61, 0.80)';
      ctx.beginPath();
      ctx.ellipse(lightDx * 0.6 + Math.cos(rot + 1.2) * 1.5 + 1.5, lightDy * 0.6 + 3.0, r * 0.38, r * 0.46, -0.3, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = 'rgba(16, 185, 129, 0.75)';
      ctx.beginPath();
      ctx.arc(lightDx * 0.7 + 3.5, lightDy * 0.7 - 0.5, r * 0.18, 0, Math.PI * 2);
      ctx.fill();

      // Swirling White Cloud Belts
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
      ctx.lineWidth = Math.max(1.0, r * 0.14);
      ctx.beginPath();
      ctx.arc(lightDx * 0.35, lightDy * 0.35 - 3.0, r * 0.6, 0.2, Math.PI * 0.85);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.50)';
      ctx.lineWidth = Math.max(0.9, r * 0.12);
      ctx.beginPath();
      ctx.arc(lightDx * 0.45, lightDy * 0.45 + 2.5, r * 0.68, 0.4, Math.PI * 0.95);
      ctx.stroke();

      // Night-side shadow mask (terminator)
      const shadowGrad = ctx.createRadialGradient(-lightDx * 0.75, -lightDy * 0.75, r * 0.15, 0, 0, r);
      shadowGrad.addColorStop(0, 'rgba(2, 6, 23, 0.98)');
      shadowGrad.addColorStop(0.55, 'rgba(2, 6, 23, 0.75)');
      shadowGrad.addColorStop(1.0, 'transparent');
      ctx.fillStyle = shadowGrad;
      ctx.fill();

      ctx.restore();

      // Atmospheric Rayleigh scattering cyan halo
      ctx.beginPath();
      ctx.arc(0, 0, r + 0.6, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
      ctx.lineWidth = 1.1;
      ctx.stroke();

    } else if (planet.type === 'rocky') {
      // -------------------------------------------------------------
      // 2. INNER ROCKY PLANET (Mercury/Moon-like)
      // -------------------------------------------------------------
      ctx.save();
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.clip();

      const rockyGrad = ctx.createRadialGradient(lightDx, lightDy, r * 0.35, 0, 0, r);
      rockyGrad.addColorStop(0, '#94a3b8');    // Muted sunlit basalt (not bright/white)
      rockyGrad.addColorStop(0.42, '#64748b'); // Weathered rock
      rockyGrad.addColorStop(0.75, '#334155'); // Cratered shadow
      rockyGrad.addColorStop(1.0, '#020617');  // Deep shadow
      ctx.fillStyle = rockyGrad;
      ctx.fill();

      // Surface crater detail
      ctx.fillStyle = 'rgba(30, 41, 59, 0.45)';
      ctx.beginPath();
      ctx.arc(lightDx * 0.4 - 1, lightDy * 0.4 + 1, r * 0.22, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();

      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.20)';
      ctx.lineWidth = 0.8;
      ctx.stroke();

    } else if (planet.type === 'jupiter') {
      // -------------------------------------------------------------
      // 3. JUPITER-LIKE BANDED GAS GIANT
      // -------------------------------------------------------------
      ctx.save();
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.clip();

      // Base planetary gas gradient (matte, subdued)
      const jupGrad = ctx.createRadialGradient(lightDx, lightDy, r * 0.38, 0, 0, r);
      jupGrad.addColorStop(0, '#b8924a');    // Muted warm tan (not cream-white)
      jupGrad.addColorStop(0.40, '#9a7040'); // Amber atmospheric zone
      jupGrad.addColorStop(0.75, '#6b4423'); // Deep brown belt shadow
      jupGrad.addColorStop(1.0, '#1c0f05');  // Night side shadow
      ctx.fillStyle = jupGrad;
      ctx.fill();

      // Horizontal atmospheric belts (signature Jupiter look)
      ctx.fillStyle = 'rgba(107, 68, 35, 0.35)'; // Dark brown belt
      ctx.fillRect(-r, -r * 0.38, r * 2, r * 0.16);
      ctx.fillRect(-r, r * 0.08, r * 2, r * 0.22);
      ctx.fillRect(-r, r * 0.48, r * 2, r * 0.12);

      ctx.fillStyle = 'rgba(255, 248, 235, 0.22)'; // Light white/cream zone
      ctx.fillRect(-r, -r * 0.18, r * 2, r * 0.14);
      ctx.fillRect(-r, r * 0.32, r * 2, r * 0.12);

      // Great Red Spot / storm oval
      ctx.fillStyle = 'rgba(194, 65, 12, 0.65)';
      ctx.beginPath();
      ctx.ellipse(lightDx * 0.35 + 2.5, lightDy * 0.35 + r * 0.18, r * 0.22, r * 0.12, 0.05, 0, Math.PI * 2);
      ctx.fill();

      // Night-side shadow mask
      const shadowGrad = ctx.createRadialGradient(-lightDx * 0.75, -lightDy * 0.75, r * 0.15, 0, 0, r);
      shadowGrad.addColorStop(0, 'rgba(4, 3, 2, 0.98)');
      shadowGrad.addColorStop(0.55, 'rgba(4, 3, 2, 0.75)');
      shadowGrad.addColorStop(1.0, 'transparent');
      ctx.fillStyle = shadowGrad;
      ctx.fill();

      ctx.restore();

      // Delicate limb edge
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(229, 213, 181, 0.30)';
      ctx.lineWidth = 0.9;
      ctx.stroke();
    }

    ctx.restore();
  }

  requestAnimationFrame(render);
</script>
</body>
</html>''', height=430)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<h3>Core Capabilities</h3>", unsafe_allow_html=True)
    cols = st.columns(4)
    features = [
        ("🔭", "Transit Signal Detection", "Identify periodic brightness dips that may indicate planetary transits."),
        ("🧠", "Machine Learning Screening", "Use trained ML models to distinguish candidate signals from non-planetary patterns."),
        ("📈", "Light-Curve Analysis", "Visualize and inspect stellar brightness variations and detected transit characteristics."),
        ("🪐", "Candidate Insights", "Review metrics such as period, transit depth, duration, BLS power/SNR, and model outputs."),
    ]
    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f"<div class='feature-card'><div class='feature-icon'>{icon}</div><h4>{title}</h4><p>{desc}</p></div>",
            unsafe_allow_html=True,
        )


def analyze_page() -> None:
    st.markdown(
        """<div class='workflow-banner'>
            <span class='wf-step active'><b>1</b> Upload Light Curve</span>
            <span class='wf-arrow'>→</span>
            <span class='wf-step'><b>2</b> Preprocess & Filter</span>
            <span class='wf-arrow'>→</span>
            <span class='wf-step'><b>3</b> BLS Transit Search</span>
            <span class='wf-arrow'>→</span>
            <span class='wf-step'><b>4</b> ML Classification</span>
            <span class='wf-arrow'>→</span>
            <span class='wf-step'><b>5</b> Review Candidate</span>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='eyebrow'>Screening Workspace</div><h2>Analyze Light Curve</h2><p class='muted'>Submit raw photometric flux measurements for exoplanet transit candidate screening.</p>", unsafe_allow_html=True)

    left, right = st.columns([3.1, 1.25], gap="large")
    with left:
        with st.container(border=True):
            upload_tab, row_tab = st.tabs(["Upload Photometry (CSV, TXT, NPY)", "Pre-loaded Test Set Row"])

            with upload_tab:
                upload = st.file_uploader(
                    "Upload light-curve file",
                    type=["csv", "txt", "npy"],
                    label_visibility="collapsed",
                )

                if upload is not None:
                    # Parse and display file preview card
                    file_size_kb = len(upload.getvalue()) / 1024.0
                    size_str = f"{file_size_kb:.1f} KB" if file_size_kb < 1024 else f"{file_size_kb / 1024:.2f} MB"
                    try:
                        flux_preview = read_light_curve(upload)
                        num_points = len(flux_preview)
                        valid_status = True
                    except Exception as e:
                        flux_preview = None
                        num_points = 0
                        valid_status = False
                        parse_err = str(e)

                    st.markdown(
                        f"""<div class='workflow-card' style='margin: 0.75rem 0;'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <div>
                                    <div style='font-size:0.78rem; color:#94a3b8; text-transform:uppercase;'>File Details</div>
                                    <b style='font-size:1.12rem; color:#fff;'>{upload.name}</b>
                                </div>
                                <span class='model-chip'>{size_str}</span>
                            </div>
                            <div style='display:flex; gap:20px; margin-top:0.75rem; font-size:0.88rem;'>
                                <div>Data points: <b style='color:#38bdf8;'>{num_points if valid_status else "Error"}</b></div>
                                <div>Format: <b style='color:#a78bfa;'>{upload.name.split('.')[-1].upper()}</b></div>
                                <div>Status: <b style='color:{"#34d399" if valid_status else "#f43f5e"};'>{"Ready for screening" if valid_status else "Parse issue"}</b></div>
                            </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                    if valid_status:
                        if st.button("Analyze Light Curve", type="primary", use_container_width=True):
                            run_analysis(flux_preview, upload.name)
                    else:
                        st.error(f"Could not parse file: {parse_err}")
                else:
                    st.markdown(
                        """<div style='border:1px dashed #334155; border-radius:12px; padding:2rem 1.25rem; text-align:center; background:rgba(15, 23, 42, 0.4); margin:0.5rem 0;'>
                            <div style='font-size:2.4rem; margin-bottom:0.4rem;'>🔭</div>
                            <b style='color:#e2e8f0; font-size:1.05rem;'>Drag and drop your light-curve file here</b><br>
                            <span class='muted'>Supports CSV with flux column, Kepler 1-row CSVs, whitespace-separated TXT, or NPY arrays</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.download_button(
                    "⇩ Download Sample Kepler Transit CSV",
                    sample_csv(),
                    file_name="kepler_sample_light_curve.csv",
                    mime="text/csv",
                )

            with row_tab:
                maximum = test_set_size() - 1
                row_id = st.number_input(
                    "Held-out test dataset row ID",
                    min_value=0,
                    max_value=maximum,
                    value=0,
                    step=1,
                    help="Row index from Kepler exoTest.csv dataset.",
                )
                st.caption("Analyzes the specified held-out Kepler observation using the full BLS and XGBoost screening pipeline.")
                if st.button("Analyze Test-Set Row", type="primary"):
                    flux, source = load_test_row(int(row_id))
                    run_analysis(flux, source)

            with st.expander("⚙ Advanced Screening Configuration"):
                st.markdown(
                    """- **Transit Search Method:** Box Least Squares (Astropy BLS periodogram)
- **Period Search Grid:** 0.5 to 20.0 days (duration grid: 1 to 12 hours)
- **Primary Decision Model:** Tuned XGBoost Classifier (selected via composite cross-validation score)
- **Feature Pipeline:** 15 astrophysical and statistical indicators (energy, entropy, kurtosis, transit depth/duration)
- **Candidate Vetting Criteria:** Periodic transit signal + BLS SNR ≥ 3.0 + ML classification confidence"""
                )

    with right:
        st.markdown(
            """<div class='panel'>
                <p style='font-family:Outfit; font-weight:700; font-size:1.15rem; margin:0 0 0.6rem; color:#fff;'>Data Format Guide</p>
                <p class='muted'>For CSV files, include a numeric <code>flux</code> column. A <code>time</code> column is optional.</p>
                <pre class='mono' style='background:rgba(15,23,42,0.85); padding:0.75rem; border-radius:8px; border:1px solid rgba(255,255,255,0.06);'>time,flux\n0.0,1.0002\n0.02,0.9998\n0.04,0.9912\n0.06,0.9915</pre>
                <p class='muted' style='margin-top:0.75rem;'>Minimum 30 data points are required. Kepler 3197-cadence observations are fully supported.</p>
            </div>""",
            unsafe_allow_html=True,
        )


def result_page() -> None:
    active = st.session_state.active_result
    if not active:
        st.info("No analysis has been run yet. Submit a light curve to view results.")
        if st.button("Go to Analyze", type="primary"):
            set_page("Analyze")
            st.rerun()
        return

    result, flux = active["result"], np.asarray(active["flux"], dtype=float)
    fig, smooth, dips = make_chart(flux)
    detected = result["prediction"] == "Planet"

    conf = result.get("confidence")
    if conf is not None:
        confidence = float(conf * 100 if conf <= 1.0 else conf)
    else:
        prob = float(result.get("probability", 0.5))
        confidence = float((prob if detected else (1.0 - prob)) * 100)

    st.markdown(
        f"<div class='eyebrow'>Screening Assessment</div><h2>Candidate Screening Results</h2>"
        f"<p class='muted'>{active['source']} · Analyzed on {active['timestamp']}</p>",
        unsafe_allow_html=True,
    )

    header_a, header_b, header_c = st.columns([2.2, 1.1, 1.1])
    if header_b.button("← Back to Analyze", use_container_width=True):
        set_page("Analyze")
        st.rerun()

    report = {
        "source": active["source"],
        "timestamp": active["timestamp"],
        "screening_verdict": "Exoplanet Candidate" if detected else "Non-Candidate",
        "raw_prediction": result["prediction"],
        "planet_probability": result["probability"],
        "screening_confidence_pct": round(confidence, 2),
        "primary_model": result["model"],
        "selection_metric": result["selection_metric"],
        "metrics": result.get("metrics", {}),
        "model_probabilities": result["model_probabilities"],
        "features": result["features"],
    }
    header_c.download_button(
        "⇩ Download Report",
        json.dumps(report, indent=2),
        file_name="exodip_report.json",
        mime="application/json",
        use_container_width=True,
    )

    left, right = st.columns([1.05, 2.3], gap="large")

    with left:
        if detected:
            badge_html = "<div class='candidate-badge-candidate'>🪐 EXOPLANET CANDIDATE</div>"
            verdict_desc = "Candidate status indicates periodic transit-like signal characteristics warranting further astronomical follow-up."
        else:
            badge_html = "<div class='candidate-badge-non'>⚪ NON-CANDIDATE</div>"
            verdict_desc = "Signal does not exhibit sufficient periodic transit depth, SNR, or classifier confidence to qualify as a candidate."

        st.markdown(
            f"""<div class='result-card'>
                <div style='font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Screening Outcome</div>
                {badge_html}
                <p style='color:#cbd5e1; font-size:0.88rem; margin:0.85rem 0 1rem; line-height:1.55;'>{verdict_desc}</p>
                <div style='font-size:0.78rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em;'>Screening Confidence</div>
                <div style='font-family:Outfit; font-weight:800; font-size:2.35rem; color:#fff; margin:0.15rem 0;'>{confidence:.1f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence / 100.0, 0.0), 1.0))

        # Dynamic model agreement calculation
        model_probs = result.get("model_probabilities", {})
        planet_votes = sum(1 for p in model_probs.values() if p >= 0.5)
        total_votes = max(len(model_probs), 1)

        if planet_votes == total_votes or planet_votes == 0:
            agreement_text = "Unanimous (All models agree)"
            agreement_color = "#34d399"
        elif detected:
            agreement_text = f"Consensus ({planet_votes}/{total_votes} models agree)"
            agreement_color = "#38bdf8"
        else:
            agreement_text = f"Consensus ({total_votes - planet_votes}/{total_votes} models agree)"
            agreement_color = "#38bdf8"

        metrics_dict = result.get("metrics", {})
        bls_p = metrics_dict.get("bls_period_days")
        bls_snr = metrics_dict.get("bls_transit_snr")
        bls_dur = metrics_dict.get("bls_duration_hours")
        bls_depth = metrics_dict.get("bls_transit_depth", metrics_dict.get("transit_depth", 0.0))

        st.markdown(
            f"""<div class='result-card' style='margin-top:1rem;'>
                <p style='font-family:Outfit; font-weight:700; font-size:1.15rem; margin:0 0 0.75rem; color:#fff;'>Detection Telemetry</p>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.4rem;'><span class='muted'>Primary Model</span><b style='color:#fff;'>⭐ {result['model']}</b></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.4rem;'><span class='muted'>Model Agreement</span><b style='color:{agreement_color};'>{agreement_text}</b></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.4rem;'><span class='muted'>Detected Dips</span><b style='color:#fff;'>{len(dips)}</b></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.4rem;'><span class='muted'>Noise Std Dev</span><b style='color:#fff;'>{result['features']['noise_level']:.4g}</b></div>
                <div style='display:flex; justify-content:space-between; margin-bottom:0.4rem;'><span class='muted'>Negative Ratio</span><b style='color:#fff;'>{result['features']['negative_ratio']:.4g}</b></div>
            </div>""",
            unsafe_allow_html=True,
        )

    with right:
        # Signal Metrics Strip
        p_str = f"{bls_p:.2f} d" if bls_p is not None and bls_p > 0 else "N/A"
        snr_str = f"{bls_snr:.1f}" if bls_snr is not None and bls_snr > 0 else "0.0"
        dur_str = f"{bls_dur:.1f} h" if bls_dur is not None and bls_dur > 0 else f"{result['features']['avg_transit_duration']:.1f} pts"
        depth_pct = (bls_depth * 100) if bls_depth < 1.0 else bls_depth
        depth_str = f"{depth_pct:.3f}%" if bls_depth > 0 else "0.000%"

        st.markdown(
            f"""<div class='metric-strip'>
                <div class='metric-cell'><div class='metric-cell-label'>Period</div><div class='metric-cell-value'>{p_str}</div></div>
                <div class='metric-cell'><div class='metric-cell-label'>Transit Depth</div><div class='metric-cell-value'>{depth_str}</div></div>
                <div class='metric-cell'><div class='metric-cell-label'>Duration</div><div class='metric-cell-value'>{dur_str}</div></div>
                <div class='metric-cell'><div class='metric-cell-label'>BLS SNR</div><div class='metric-cell-value'>{snr_str}</div></div>
                <div class='metric-cell'><div class='metric-cell-label'>Dip Groups</div><div class='metric-cell-value'>{len(dips)}</div></div>
                <div class='metric-cell'><div class='metric-cell-label'>Decision Model</div><div class='metric-cell-value' style='color:#a78bfa;'>{result['model']}</div></div>
            </div>""",
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Full Light Curve",
            "🔍 Transit Dip View",
            "📊 Model Comparison & Consensus",
            "🔬 Extracted Features",
        ])

        with tab1:
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            zoom_fig = make_transit_view_chart(flux, smooth, dips)
            st.plotly_chart(zoom_fig, use_container_width=True)
            st.caption("Zoomed photometric profile centered on the deepest transit event, highlighting ingress and egress boundaries.")

        with tab3:
            model_scores = result.get("model_scores", {})
            if model_scores:
                st.markdown("<p style='font-family:Outfit; font-weight:700; font-size:1.15rem; color:#fff; margin-bottom:0.5rem;'>Model Benchmark Performance & Real-Time Verdict</p>", unsafe_allow_html=True)
                rows = []
                for mname, sc in model_scores.items():
                    prob_val = float(result["model_probabilities"].get(mname, 0.0))
                    if mname == result["model"]:
                        pred_label = "🪐 Candidate" if detected else "⚪ Non-Candidate"
                        disp_prob = result["probability"] if detected else prob_val
                    else:
                        pred_label = "🪐 Candidate" if prob_val >= 0.5 else "⚪ Non-Candidate"
                        disp_prob = prob_val
                    row = {"Model": ("⭐ " + mname) if mname == result["model"] else mname}
                    row.update({k: round(v, 4) for k, v in sc.items()})
                    row["Model Verdict"] = pred_label
                    row["Probability"] = f"{disp_prob * 100:.1f}%"
                    rows.append(row)
                score_df = pd.DataFrame(rows).sort_values("Composite", ascending=False, ignore_index=True)
                st.dataframe(score_df, use_container_width=True, hide_index=True)

            st.markdown("<p style='font-family:Outfit; font-weight:600; font-size:1.1rem; color:#fff; margin-top:1.25rem;'>Candidate Probability by Model</p>", unsafe_allow_html=True)
            prob_items = []
            for mname, p in result["model_probabilities"].items():
                if mname == result["model"] and detected:
                    p = result["probability"]
                prob_items.append({"Model": mname, "Candidate Probability (%)": round(p * 100, 1)})
            prob_df = pd.DataFrame(prob_items)
            st.bar_chart(prob_df.set_index("Model"), color="#818cf8")

            if result.get("unavailable_models"):
                st.caption("Unavailable for this input: " + "; ".join(f"{k} — {v}" for k, v in result["unavailable_models"].items()))

        with tab4:
            st.markdown("<p style='font-family:Outfit; font-weight:700; font-size:1.15rem; color:#fff; margin-bottom:0.75rem;'>Astrophysical & Statistical Features</p>", unsafe_allow_html=True)
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            feats = result["features"]

            with col_f1:
                st.markdown("<div class='workflow-card'><b>Statistical</b>", unsafe_allow_html=True)
                st.caption(f"Mean: {feats.get('mean', 0.0):.4g}")
                st.caption(f"Median: {feats.get('median', 0.0):.4g}")
                st.caption(f"Variance: {feats.get('variance', 0.0):.4g}")
                st.caption(f"Skewness: {feats.get('skewness', 0.0):.4g}")
                st.caption(f"Kurtosis: {feats.get('kurtosis', 0.0):.4g}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_f2:
                st.markdown("<div class='workflow-card'><b>Signal Dynamics</b>", unsafe_allow_html=True)
                st.caption(f"Signal Energy: {feats.get('signal_energy', 0.0):.4g}")
                st.caption(f"Spectral Entropy: {feats.get('spectral_entropy', 0.0):.4g}")
                st.caption(f"Negative Ratio: {feats.get('negative_ratio', 0.0):.4g}")
                st.caption(f"Noise Level: {feats.get('noise_level', 0.0):.4g}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_f3:
                st.markdown("<div class='workflow-card'><b>Transit Dips</b>", unsafe_allow_html=True)
                st.caption(f"Avg Depth: {feats.get('avg_transit_depth', 0.0):.4g}")
                st.caption(f"Avg Duration: {feats.get('avg_transit_duration', 0.0):.1f} pts")
                st.caption(f"Num Dips: {feats.get('num_dips', 0)}")
                st.caption(f"Min Flux: {feats.get('min_flux', 0.0):.4g}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col_f4:
                st.markdown("<div class='workflow-card'><b>BLS Parameters</b>", unsafe_allow_html=True)
                st.caption(f"Period: {p_str}")
                st.caption(f"Duration: {dur_str}")
                st.caption(f"Transit SNR: {snr_str}")
                st.caption(f"BLS Power: {metrics_dict.get('bls_power', 0.0):.4g}")
                st.markdown("</div>", unsafe_allow_html=True)


def history_page() -> None:
    st.markdown("<div class='eyebrow'>Session Storage</div><h2>Analysis History</h2><p class='muted'>Review light curves evaluated during this active browser session.</p>", unsafe_allow_html=True)

    total = len(st.session_state.history)
    planets = sum(item["prediction"] == "Planet" for item in st.session_state.history)
    rate_pct = (planets / total * 100.0) if total > 0 else 0.0

    cards = st.columns(3)
    cards[0].markdown(f"<div class='stat-card'><div class='metric-cell-label'>Total Analyses</div><div class='metric-cell-value'>{total}</div></div>", unsafe_allow_html=True)
    cards[1].markdown(f"<div class='stat-card'><div class='metric-cell-label'>Planet Candidates</div><div class='metric-cell-value' style='color:#34d399;'>{planets}</div></div>", unsafe_allow_html=True)
    cards[2].markdown(f"<div class='stat-card'><div class='metric-cell-label'>Candidate Rate</div><div class='metric-cell-value' style='color:#38bdf8;'>{rate_pct:.1f}%</div></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem; margin-bottom:0.75rem;'><span class='model-chip'>⭐ Active Decision Model: <b>XGBoost</b></span></div>", unsafe_allow_html=True)

    if st.session_state.history:
        rows = [
            {
                "Source": item["source"],
                "Verdict": "🪐 Candidate" if item["prediction"] == "Planet" else "⚪ Non-Candidate",
                "Confidence": f"{item['confidence']:.1f}%",
                "Decision Model": item.get("model", "XGBoost"),
                "Date": item["timestamp"],
            }
            for item in st.session_state.history
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col_h1, col_h2 = st.columns([3, 1])
        with col_h1:
            selected = st.selectbox(
                "Select a past run to view",
                range(len(st.session_state.history)),
                format_func=lambda i: f"{st.session_state.history[i]['source']} — {st.session_state.history[i]['timestamp']} ({st.session_state.history[i]['prediction']})",
            )
            if st.button("Open Selected Result", type="primary"):
                st.session_state.active_result = st.session_state.history[selected]
                set_page("Results")
                st.rerun()
        with col_h2:
            st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            if st.button("Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.active_result = None
                st.rerun()
    else:
        st.markdown(
            "<div class='panel'><p style='font-family:Outfit; font-weight:600; font-size:1.1rem; color:#fff; margin:0 0 0.35rem;'>No analyses yet</p>"
            "<p class='muted'>Submit a light curve in the Analyze tab to populate this session history.</p></div>",
            unsafe_allow_html=True,
        )


def about_page() -> None:
    st.markdown("<div class='eyebrow'>Scientific Context</div><h2>About ExoDip</h2>", unsafe_allow_html=True)

    st.markdown(
        """<div class='panel'>
            <p class='lead' style='color:#f1f5f9;'>
            <b>ExoDip</b> is an AI-assisted screening tool designed to help researchers, students, and astronomy enthusiasts analyze stellar light curves for signatures of transiting exoplanets. Using a combination of Box Least Squares (BLS) period searching, astrophysical feature extraction, and an ensemble of machine learning classifiers (including tuned XGBoost and Random Forest models), ExoDip evaluates raw photometry to identify candidate transit events.
            </p>
            <div style='background:rgba(239, 68, 68, 0.12); border-left:4px solid #f43f5e; padding:0.95rem 1.15rem; border-radius:6px; margin:1.35rem 0;'>
                <b style='color:#fca5a5; font-size:1.02rem;'>Scientific Notice:</b>
                <span style='color:#fecdd3; font-size:0.95rem; line-height:1.6;'>
                ExoDip is a candidate screening tool, not a confirmation pipeline. Exoplanet confirmation requires extensive follow-up observations, including high-resolution imaging, radial velocity spectroscopy, and statistical validation against astrophysical false positives (such as eclipsing binaries and background blended stars).
                </span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.5rem'></div><h3>Detection & Screening Pipeline</h3>", unsafe_allow_html=True)
    st.markdown(
        """<div class='workflow-banner' style='padding:1.15rem;'>
            <div style='text-align:center;'><b>1. Light Curve</b><br><span class='muted'>Time-series flux</span></div>
            <span class='wf-arrow'>→</span>
            <div style='text-align:center;'><b>2. Preprocessing</b><br><span class='muted'>Noise & detrending</span></div>
            <span class='wf-arrow'>→</span>
            <div style='text-align:center;'><b>3. Transit Detection</b><br><span class='muted'>BLS periodogram</span></div>
            <span class='wf-arrow'>→</span>
            <div style='text-align:center;'><b>4. Feature Extraction</b><br><span class='muted'>Morphology & entropy</span></div>
            <span class='wf-arrow'>→</span>
            <div style='text-align:center;'><b>5. Machine Learning</b><br><span class='muted'>XGBoost Ensemble</span></div>
            <span class='wf-arrow'>→</span>
            <div style='text-align:center;'><b style='color:#34d399;'>6. Candidate Screening</b><br><span class='muted'>Vetted candidate</span></div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.5rem'></div><h3>Core Technologies</h3>", unsafe_allow_html=True)
    t_cols = st.columns(4)
    techs = [
        ("Python 3", "Modern scientific runtime and high-performance computing environment."),
        ("Lightkurve & Astropy", "Photometric transit modeling, BLS search, and astronomical time-series tools."),
        ("NumPy & Pandas", "Array processing, mathematical transformations, and tabular metadata management."),
        ("Scikit-learn & XGBoost", "Supervised gradient boosting, random forests, and calibrated decision boundaries."),
    ]
    for col, (name, desc) in zip(t_cols, techs):
        col.markdown(f"<div class='stat-card'><b>{name}</b><p class='muted' style='margin-top:0.4rem;'>{desc}</p></div>", unsafe_allow_html=True)


def docs_page() -> None:
    st.markdown("<div class='eyebrow'>User & Methodological Guide</div><h2>Documentation</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚀 Getting Started",
        "📄 Input Formats",
        "⚙ How It Works",
        "📊 Understanding Results",
        "⚠️ Limitations",
        "❓ FAQ",
    ])

    with tab1:
        st.markdown(
            """### Getting Started with ExoDip
1. **Prepare Photometric Data:** Obtain a time series of stellar brightness (Kepler, TESS, or ground-based photometry).
2. **Navigate to Analyze:** Use the top navigation bar to select the **Analyze** page.
3. **Upload or Select:** Drop your CSV, TXT, or NPY file into the uploader, or choose a held-out test set row.
4. **Initiate Screening:** Click the **Analyze Light Curve** button.
5. **Inspect Detection Telemetry:** Review the detected dips, transit depth, duration, and BLS period.
6. **Export Findings:** Download the complete JSON telemetry report using the **Download Report** button.
"""
        )

    with tab2:
        st.markdown(
            """### Supported Input Formats
- **Standard CSV Format:** A comma-separated file with a `flux` column. A `time` column is optional.
```csv
time,flux
0.000,1.0002
0.020,0.9998
0.041,0.9912
0.061,0.9915
```
- **Kepler Test Set Row (exoTest format):** 1 row with 3197 whitespace or comma-separated flux values (`FLUX.1` to `FLUX.3197`).
- **Numpy Array (.npy):** 1-D array of float flux values or 2-D array of shape `(N, 1)`.
- **Whitespace-separated TXT:** Space-delimited numeric columns containing flux measurements.
"""
        )

    with tab3:
        st.markdown(
            """### How the Pipeline Works
- **Photometric Normalization:** Flux values are detrended and smoothed using a uniform box filter to isolate local fluctuations while preserving transit dip profiles.
- **Box Least Squares (BLS):** An Astropy BLS periodogram scans trial periods from 0.5 to 20 days with trial transit durations from 1 to 12 hours to search for periodic box-shaped occultations.
- **Feature Extraction:** 15 distinct morphological and statistical features are computed, including transit depth, duration, noise standard deviation, skewness, kurtosis, signal energy, and spectral entropy.
- **Ensemble Decision Logic:** Calibrated XGBoost and Random Forest models evaluate the feature vector alongside BLS transit significance to assign the final candidate classification.
"""
        )

    with tab4:
        st.markdown(
            """### Understanding Results
- **Period (Days):** The best-fit orbital period detected by the Box Least Squares periodogram.
- **Transit Depth (%):** The fraction of stellar flux blocked during transit mid-point:
$$\\delta = \\frac{\\Delta F}{F} \\approx \\left(\\frac{R_p}{R_*}\\right)^2$$
- **BLS SNR:** The signal-to-noise ratio of the transit dip relative to residual stellar noise. High SNR values (> 5.0) indicate statistically significant transit shapes.
- **Screening Confidence:** The model's calibrated probabilistic assessment that the signal matches bona fide planetary transit profiles rather than stellar variability or noise.
"""
        )

    with tab5:
        st.markdown(
            """### Important Limitations & Astrophysical Caveats
- **False Positives:** Eclipsing binary stars (EB), grazing binaries, and background blended stars can mimic planetary transits.
- **Stellar Activity:** Starspots, stellar flares, and rotation can introduce spurious periodic dips.
- **Confirmation Required:** Candidate status is an initial screening recommendation. Confirmation requires high-precision radial velocity spectroscopy (to measure planetary mass) and high-resolution imaging (to rule out background companions).
"""
        )

    with tab6:
        st.markdown(
            """### Frequently Asked Questions
**Q: What is the difference between an exoplanet candidate and a confirmed exoplanet?**
*A: A candidate exhibits transit-like signals in photometric data that have not yet undergone independent observational confirmation (such as spectroscopic mass determination).*

**Q: Why is XGBoost used as the primary decision model?**
*A: On our held-out benchmark evaluation, tuned XGBoost achieved the highest composite score (balanced accuracy, precision, recall, and ROC-AUC) across diverse light curve morphologies.*

**Q: Can I analyze multi-quarter Kepler or TESS full-frame images?**
*A: ExoDip accepts 1-D light curves extracted from calibrated photometry (such as Lightkurve PDCSAP flux).*
"""
        )


setup_state()
inject_styles()
nav()

pages = {
    "Home": home_page,
    "Analyze": analyze_page,
    "Results": result_page,
    "History": history_page,
    "About": about_page,
    "Docs": docs_page,
}
pages[st.session_state.page]()

st.markdown(
    """<div class='app-footer'>
        <b>ExoDip</b> • AI-assisted screening of exoplanet transit candidates • Developed for astrophysics research and candidate vetting.
    </div>""",
    unsafe_allow_html=True,
)
