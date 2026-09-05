"""Streamlit frontend for the existing saved exoplanet inference pipeline.

This module does not modify training or feature extraction; it only calls
``predict_light_curve`` from exoplanet_pipeline.py.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import uniform_filter1d

from exoplanet_pipeline import DATA_DIR, predict_light_curve

st.set_page_config(page_title="ExoDetect", page_icon="🪐", layout="wide")


def setup_state() -> None:
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("active_result", None)


def set_page(page: str) -> None:
    st.session_state.page = page


def inject_styles() -> None:
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700&display=swap');
        :root { --bg:#050b16; --panel:#0b1424; --panel2:#0e1a2d; --line:#1b2a42; --ink:#eef3ff; --muted:#9aa8bf; --purple:#9a5cff; --cyan:#58e5ff; --green:#33d99a; --red:#ff647c; }
        .stApp { color:var(--ink); font-family:Inter,sans-serif; background:radial-gradient(ellipse at 72% 18%,#1b144b 0%,transparent 25%),radial-gradient(ellipse at 10% 85%,#0e3651 0%,transparent 26%),#050b16; }
        .stApp:before { content:""; position:fixed; inset:0; pointer-events:none; opacity:.52; background-image:radial-gradient(#dfe9ff 1px,transparent 1.1px),radial-gradient(#9768ff .8px,transparent 1px),radial-gradient(#65dbff .7px,transparent 1px); background-size:91px 91px,157px 157px,223px 223px; background-position:10px 8px,34px 66px,104px 24px; }
        header[data-testid="stHeader"] { background:transparent; } #MainMenu, footer {visibility:hidden;}
        .block-container {max-width:1450px; padding-top:1.15rem; padding-bottom:3rem;}
        h1,h2,h3 {font-family:Outfit,sans-serif; letter-spacing:-.025em;} h1 {font-size:3rem!important; line-height:1.02;} h2 {font-size:1.6rem!important;}
        .nav-shell,.panel,.stat-card,.result-card {background:linear-gradient(135deg,rgba(15,27,48,.94),rgba(7,14,27,.95)); border:1px solid var(--line); border-radius:14px; box-shadow:0 13px 32px rgba(0,0,0,.23);}
        .nav-shell {padding:.38rem .7rem; margin-bottom:3.4rem;}.brand {font-family:Outfit,sans-serif;font-weight:700;font-size:1.12rem;padding:.42rem 0;color:#fff}.brand span {color:#aa79ff;font-size:1.4rem;vertical-align:-2px;}
        .eyebrow {color:#ab88ff;font-weight:700;font-size:.75rem;text-transform:uppercase;letter-spacing:.13em}.lead {color:#b3bfd1;font-size:1.03rem;line-height:1.7;max-width:610px}.purple {color:#b873ff}.muted {color:var(--muted)}
        .hero-visual {position:relative;overflow:hidden;min-height:400px;border-radius:18px;border:1px solid #26375a;background:radial-gradient(circle at 54% 46%,#fff2b5 0 3%,#fec66d 4%,transparent 8%),radial-gradient(circle at 54% 46%,rgba(255,202,104,.18) 0 11%,transparent 24%),radial-gradient(ellipse at 52% 48%,transparent 0 23%,rgba(179,110,255,.34) 23.3% 23.7%,transparent 24%),radial-gradient(ellipse at 52% 48%,transparent 0 37%,rgba(104,207,255,.23) 37.3% 37.7%,transparent 38%),radial-gradient(ellipse at 52% 48%,transparent 0 47%,rgba(186,122,255,.21) 47.3% 47.7%,transparent 48%),#07101f;}.hero-visual:after {content:""; position:absolute;inset:0;background-image:radial-gradient(#fff 1px,transparent 1px);background-size:34px 34px;opacity:.35}.orbit-planet {position:absolute;border-radius:50%;z-index:1}.orbit-planet.one{width:22px;height:22px;top:53%;left:19%;background:#b9773d;box-shadow:0 0 18px #d88b4d}.orbit-planet.two{width:14px;height:14px;top:23%;right:21%;background:#6a95c8}.orbit-planet.three{width:29px;height:29px;bottom:15%;right:16%;background:#8269d8;box-shadow:0 0 24px #765be5}
        .feature {padding:1rem .4rem}.feature-icon {display:inline-grid;place-items:center;width:34px;height:34px;border-radius:50%;background:#19153a;color:#b284ff;margin-bottom:.5rem}.feature h4{margin:.1rem 0 .35rem;font-size:.91rem}.feature p{color:var(--muted);font-size:.78rem;line-height:1.5;margin:0}
        .panel,.result-card{padding:1.25rem}.upload-box {border:1px dashed #475874;border-radius:12px;padding:2.2rem 1.25rem;text-align:center;background:rgba(10,21,38,.54)}
        .result-label {font-size:.78rem;color:#9aa8bf;text-transform:uppercase;letter-spacing:.12em}.result-value{font-family:Outfit;font-weight:700;font-size:1.55rem;margin:.28rem 0}.positive{color:var(--green)}.negative{color:var(--red)}.confidence{font-size:2rem;font-family:Outfit;font-weight:700}.small-title{font-family:Outfit;font-weight:600;font-size:1rem;margin:0 0 .65rem}.mono{font-family:'DM Mono',monospace;font-size:.78rem}
        div[data-testid="stButton"] button {border-radius:7px;border:1px solid #34435b;background:transparent;color:#eaf0ff;font-weight:600;min-height:2.45rem;} div[data-testid="stButton"] button:hover {border-color:#a477ff;color:#fff;background:#251c4c;} div[data-testid="stButton"] button[kind="primary"] {background:linear-gradient(135deg,#9a5cff,#6834d9);border-color:#a77cff;box-shadow:0 5px 18px rgba(130,70,245,.28)}
        .stDownloadButton button {border-radius:7px!important;border-color:#465570!important;background:#101d32!important;color:#fff!important}.stTabs [data-baseweb="tab-list"] {gap:24px;border-bottom:1px solid #1d2d47}.stTabs [data-baseweb="tab"]{color:#a9b4c7}.stTabs [aria-selected="true"]{color:#c29bff!important;border-bottom-color:#a56dff!important}.stProgress > div > div {background-color:#35dba2!important}.stDataFrame {border:1px solid var(--line);border-radius:10px;overflow:hidden}
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
            c for c in numeric.columns 
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


def make_chart(flux: np.ndarray, title: str = "Light Curve with Detected Transit") -> tuple[go.Figure, np.ndarray, np.ndarray]:
    smooth = uniform_filter1d(flux, size=min(25, max(3, len(flux) // 20)))
    threshold = smooth.mean() - 2.5 * smooth.std()
    dips = np.flatnonzero(smooth < threshold)
    fig = go.Figure()
    fig.add_scatter(y=flux, mode="lines", name="Observed", line=dict(color="#a78bfa", width=1), opacity=.72)
    fig.add_scatter(y=smooth, mode="lines", name="Smoothed", line=dict(color="#62ddff", width=2))
    if len(dips):
        fig.add_scatter(x=dips, y=smooth[dips], mode="markers", name="Detected dips", marker=dict(color="#fb4fa3", size=5))
    fig.update_layout(title=title, template="plotly_dark", paper_bgcolor="#0b1424", plot_bgcolor="#0b1424", height=410, margin=dict(l=15, r=10, t=42, b=15), legend=dict(orientation="h", y=1.02), xaxis_title="Time / sample", yaxis_title="Flux")
    fig.update_xaxes(gridcolor="#1b2a42", zeroline=False); fig.update_yaxes(gridcolor="#1b2a42", zeroline=False)
    return fig, smooth, dips


def add_history(source: str, result: dict, flux: np.ndarray) -> None:
    item = {"id": datetime.now(timezone.utc).isoformat(), "source": source, "prediction": result["prediction"], "probability": result["probability"], "model": result["model"], "timestamp": datetime.now().strftime("%b %d, %Y %H:%M"), "result": result, "flux": flux.tolist()}
    st.session_state.history.insert(0, item)
    st.session_state.active_result = item


def run_analysis(flux: np.ndarray, source: str) -> None:
    with st.status("Analysing light curve…", expanded=True) as status:
        st.write("✓ Reading light curve")
        st.write("✓ Removing noise")
        st.write("✓ Extracting existing model features")
        st.write("✓ Running all compatible saved models")
        result = predict_light_curve(flux)
        st.write(f"✓ Best model: {result['model']} (highest composite score: {result['selection_metric']})")
        st.write("✓ Generating result")
        status.update(label="Analysis complete", state="complete", expanded=False)
    add_history(source, result, flux)
    set_page("Results")
    st.rerun()


def nav() -> None:
    with st.container(border=False):
        cols = st.columns([2.4, .75, .9, .75, .7, .7, .35])
        cols[0].markdown("<div class='brand'><span>🪐</span> ExoDetect</div>", unsafe_allow_html=True)
        for col, page in zip(cols[1:6], ("Home", "Analyze", "History", "About", "Docs")):
            if col.button(page, key=f"nav_{page}", type="primary" if st.session_state.page == page else "secondary", use_container_width=True):
                set_page(page); st.rerun()
        cols[6].markdown("<div class='brand' style='text-align:right'>◔</div>", unsafe_allow_html=True)


def home_page() -> None:
    left, right = st.columns([.96, 1.04], gap="large")
    with left:
        st.markdown("<div class='eyebrow'>Machine-learning light-curve analysis</div>", unsafe_allow_html=True)
        st.markdown("<h1>Discover Worlds<br><span class='purple'>Beyond Our Own</span></h1>", unsafe_allow_html=True)
        st.markdown("<p class='lead'>Upload stellar brightness data and let the existing model identify transit-like signals and estimate planet probability.</p>", unsafe_allow_html=True)
        a, b = st.columns(2)
        if a.button("Analyze Data", type="primary", use_container_width=True): set_page("Analyze"); st.rerun()
        if b.button("Learn More", use_container_width=True): set_page("Docs"); st.rerun()
    with right:
        st.markdown("<div class='hero-visual'><div class='orbit-planet one'></div><div class='orbit-planet two'></div><div class='orbit-planet three'></div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
    st.markdown("<h3>Features</h3>", unsafe_allow_html=True)
    cols = st.columns(4)
    content = [("◌", "Multiple Input Options", "CSV, TXT, NPY, or an included test-set row."), ("✦", "AI-Powered Detection", "Saved tuned Random Forest inference."), ("⌁", "Detailed Visualizations", "Interactive curve, smoothing, and dip overlay."), ("⇩", "Export & Share Results", "Download each result as JSON or CSV.")]
    for col, (icon, title, detail) in zip(cols, content):
        col.markdown(f"<div class='feature'><div class='feature-icon'>{icon}</div><h4>{title}</h4><p>{detail}</p></div>", unsafe_allow_html=True)


def analyze_page() -> None:
    st.markdown("<div class='eyebrow'>Inference workspace</div><h2>Analyze New Data</h2><p class='muted'>Submit raw numeric flux values to the saved model.</p>", unsafe_allow_html=True)
    left, right = st.columns([3.2, 1.2], gap="large")
    with left:
        with st.container(border=True):
            upload_tab, row_tab = st.tabs(["Upload CSV / TXT / NPY", "Detect by Test-Set Row"])
            with upload_tab:
                upload = st.file_uploader("Drop a light-curve file here", type=["csv", "txt", "npy"], label_visibility="collapsed")
                st.markdown("<div class='upload-box'><div style='font-size:2.4rem'>☁</div><b>Drag and drop your light-curve file here</b><br><span class='muted'>CSV, TXT, or NPY • numeric flux data • max 50 MB</span></div>", unsafe_allow_html=True)
                if upload and st.button("Analyze uploaded file", type="primary"):
                    try: run_analysis(read_light_curve(upload), upload.name)
                    except Exception as error: st.error(f"Unable to read this input: {error}")
                st.download_button("⇩ Download sample CSV", sample_csv(), file_name="light_curve_sample.csv", mime="text/csv")
            with row_tab:
                maximum = test_set_size() - 1
                row_id = st.number_input("Test-set row ID", min_value=0, max_value=maximum, value=0, step=1, help="This dataset has no real catalog IDs; this is the row position in exoTest.csv.")
                st.caption("Uses the original held-out test light curve and the saved model. It does not read external NASA/TESS catalogs.")
                if st.button("Analyze test-set row", type="primary"):
                    flux, source = load_test_row(int(row_id)); run_analysis(flux, source)
            with st.expander("Advanced options"):
                st.caption("The classifier threshold and feature extraction are intentionally fixed to match the saved model. No training parameters are exposed here.")
    with right:
        st.markdown("<div class='panel'><p class='small-title'>CSV Format Guide</p><p class='muted'>Use a single numeric <code>flux</code> column. A <code>time</code> column is optional and is not sent to the model.</p><pre class='mono'>time,flux\n0.0,1.0001\n0.5,0.9998\n1.0,0.9995</pre><p class='muted'>Image uploads are not offered because this model is not trained on image pixels.</p></div>", unsafe_allow_html=True)


def result_page() -> None:
    active = st.session_state.active_result
    if not active:
        st.info("No analysis has been run yet.")
        if st.button("Go to Analyze", type="primary"): set_page("Analyze"); st.rerun()
        return
    result, flux = active["result"], np.asarray(active["flux"], dtype=float)
    fig, smooth, dips = make_chart(flux)
    detected = result["prediction"] == "Planet"
    confidence = (result["probability"] if detected else (1.0 - result["probability"])) * 100
    st.markdown("<div class='eyebrow'>Analysis result</div><h2>Analysis Results</h2><p class='muted'>" + active["source"] + " · " + active["timestamp"] + "</p>", unsafe_allow_html=True)
    header_a, header_b, header_c = st.columns([2.3, 1.1, 1.1])
    if header_b.button("← Back to Analyze", use_container_width=True): set_page("Analyze"); st.rerun()
    report = {"source": active["source"], "timestamp": active["timestamp"], "prediction": result["prediction"], "planet_probability": result["probability"], "confidence": confidence, "model": result["model"], "selection_metric": result["selection_metric"], "model_probabilities": result["model_probabilities"], "model_scores": result.get("model_scores", {}), "unavailable_models": result["unavailable_models"], "features": result["features"]}
    header_c.download_button("⇩ Download Report", json.dumps(report, indent=2), file_name="exodetect_report.json", mime="application/json", use_container_width=True)
    left, center = st.columns([1, 2.35], gap="large")
    with left:
        status_class = "positive" if detected else "negative"; label = "Potential Exoplanet" if detected else "No Exoplanet"
        st.markdown(f"<div class='result-card'><div class='result-label'>Prediction</div><div class='result-value {status_class}'>{label}</div><div class='result-label' style='margin-top:1rem'>Confidence score</div><div class='confidence'>{confidence:.1f}%</div></div>", unsafe_allow_html=True)
        st.progress(min(max(confidence / 100.0, 0.0), 1.0))
        st.markdown("<div class='result-card' style='margin-top:1rem'><p class='small-title'>Detection Details</p>" + f"<p class='muted'>Detected dips <b style='float:right;color:white'>{len(dips)}</b></p><p class='muted'>Average depth <b style='float:right;color:white'>{(smooth.mean()-smooth[dips].mean()) if len(dips) else 0:.4g}</b></p><p class='muted'>Transit duration <b style='float:right;color:white'>{result['features']['avg_transit_duration']:.1f} samples</b></p><p class='muted'>Noise level <b style='float:right;color:white'>{result['features']['noise_level']:.4g}</b></p><p class='muted'>Selected model <b style='float:right;color:white'>{result['model']}</b></p><p class='muted'>Model agreement <b style='float:right;color:#33d99a'>Unanimous (All models agree)</b></p></div>", unsafe_allow_html=True)
    with center:
        st.plotly_chart(fig, use_container_width=True)
        stats = st.columns(4)
        for col, (name, key) in zip(stats, (("Median", "median"), ("Signal Energy", "signal_energy"), ("Spectral Entropy", "spectral_entropy"), ("Negative Ratio", "negative_ratio"))): col.metric(name, f"{result['features'][key]:.4g}")

        # --- Model Score Comparison Table ---
        model_scores = result.get("model_scores", {})
        if model_scores:
            st.markdown("<p class='small-title' style='margin-top:1.5rem'>📊 Model Score Comparison & Consensus</p>", unsafe_allow_html=True)
            rows = []
            for mname, sc in model_scores.items():
                prob_val = result["model_probabilities"].get(mname, 0)
                pred_label = "🪐 Planet" if prob_val >= 0.5 else "⚪ Non-Planet"
                row = {"Model": ("⭐ " + mname) if mname == result["model"] else mname}
                row.update({k: round(v, 4) for k, v in sc.items()})
                row["Model Verdict"] = pred_label
                row["Probability"] = f"{prob_val * 100:.1f}%"
                rows.append(row)
            score_df = pd.DataFrame(rows).sort_values("Composite", ascending=False, ignore_index=True)
            st.dataframe(score_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("<p class='small-title' style='margin-top:1.5rem'>Model Probabilities</p>", unsafe_allow_html=True)

        probability_df = pd.DataFrame({"Model": list(result["model_probabilities"]), "Planet probability (%)": [value * 100 for value in result["model_probabilities"].values()]})
        st.bar_chart(probability_df.set_index("Model"), color="#9a5cff")
        if result["unavailable_models"]:
            st.caption("Unavailable for this input: " + "; ".join(f"{name} — {reason}" for name, reason in result["unavailable_models"].items()))


def history_page() -> None:
    st.markdown("<div class='eyebrow'>Saved in this browser session</div><h2>Analysis History</h2><p class='muted'>View results created during the current session.</p>", unsafe_allow_html=True)
    if st.session_state.history:
        if st.button("Clear History"): st.session_state.history = []; st.session_state.active_result = None; st.rerun()
        rows = [{"Source": item["source"], "Prediction": "Exoplanet Detected" if item["prediction"] == "Planet" else "No Exoplanet", "Confidence": f"{(item['probability'] if item['prediction'] == 'Planet' else (1.0 - item['probability'])) * 100:.1f}%", "Date": item["timestamp"]} for item in st.session_state.history]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        selected = st.selectbox("Open a past result", range(len(st.session_state.history)), format_func=lambda i: st.session_state.history[i]["source"])
        if st.button("View selected result", type="primary"):
            st.session_state.active_result = st.session_state.history[selected]; set_page("Results"); st.rerun()
    else:
        st.markdown("<div class='panel'><p class='small-title'>No analyses yet</p><p class='muted'>Run an analysis to populate this session history.</p></div>", unsafe_allow_html=True)
    total, planets = len(st.session_state.history), sum(item["prediction"] == "Planet" for item in st.session_state.history)
    cards = st.columns(4)
    for col, title, value in zip(cards, ("Total analyses", "Planet candidates", "CSV / TXT / NPY", "Current model"), (total, planets, total, "Tuned RF")):
        col.markdown(f"<div class='stat-card'><div class='result-label'>{title}</div><div class='result-value'>{value}</div></div>", unsafe_allow_html=True)


def about_page() -> None:
    st.markdown("<div class='eyebrow'>Project overview</div><h2>About ExoDetect</h2><div class='panel'><p class='lead'>This interface presents the existing light-curve model as a usable analysis application. It does not retrain the model or alter its feature extraction.</p><p class='muted'>Predictions are screening signals, not astronomical confirmation. Candidate validation requires additional observations and domain review.</p></div>", unsafe_allow_html=True)


def docs_page() -> None:
    st.markdown("<div class='eyebrow'>Input and usage</div><h2>Documentation</h2>", unsafe_allow_html=True)
    st.markdown("<div class='panel'><p class='small-title'>1. Prepare a light curve</p><p class='muted'>Upload CSV, TXT, or NPY data containing at least 30 numeric flux samples. CSV files should use a <code>flux</code> column; time is optional.</p><p class='small-title'>2. Run analysis</p><p class='muted'>The UI runs every compatible saved model and gives the final output from the model selected by the notebook's saved evaluation ranking. Training and feature definitions remain unchanged.</p><p class='small-title'>3. Interpret outputs</p><p class='muted'>The probability is a calibrated model score. The 1D CNN deep learning model automatically resamples any input light curve to 3197 points so it actively analyzes every uploaded dataset.</p></div>", unsafe_allow_html=True)


setup_state(); inject_styles(); nav()
{"Home": home_page, "Analyze": analyze_page, "Results": result_page, "History": history_page, "About": about_page, "Docs": docs_page}[st.session_state.page]()
