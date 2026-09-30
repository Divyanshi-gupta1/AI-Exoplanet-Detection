# 🔭 ExoDip — AI Exoplanet Transit Screening

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ensemble-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-blue)
![Astropy](https://img.shields.io/badge/Astropy-BLS_Periodogram-orange)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black?logo=vercel)
![License](https://img.shields.io/badge/License-Educational-green)

**An astronomical candidate screening engine combining Box Least Squares (BLS) transit modeling with supervised ensemble machine learning to detect exoplanet candidates from stellar photometry.**

[🌐 Live Demo](https://exodip.vercel.app/) · [📖 Architecture](#-detection-pipeline) · [📊 Dataset](#-dataset) · [🚀 Run Locally](#-installation--local-development)

</div>

---

## 🌌 Overview

**ExoDip** is an astronomical analysis tool designed to analyze stellar brightness time-series (light curves) for periodic transit signals characteristic of exoplanets passing in front of their host stars.

The platform bridges physics-based transit modeling (**Astropy Box Least Squares periodogram**) with a calibrated **multi-model machine learning ensemble** (Tuned XGBoost, Random Forest, SVM, Logistic Regression, and 1D CNN). It provides interactive light-curve visualizations, transit ingress/egress profiling, feature attribution, benchmark model comparisons, and exportable screening telemetry.

> **Scientific Notice:** ExoDip is a candidate screening and triage tool, not an observational confirmation pipeline. True exoplanet confirmation requires extensive astronomical follow-up (high-resolution imaging, radial velocity spectroscopy, and statistical false-positive validation against eclipsing binaries).

---

## ✨ Key Features

- 📤 **Multi-Format Photometric Ingestion**
  - Upload CSV time-series with a `flux` column (and optional `time` column).
  - Native support for Kepler 1-row CSV format (3,197 cadences, e.g., `FLUX.1` to `FLUX.3197`).
  - Whitespace-delimited `.txt` files and 1D/2D NumPy `.npy` arrays.
  - Direct download of sample Kepler exoplanet transit CSV for testing.
- 🔭 **Held-out Kepler Benchmark Mode**
  - Screen any of the 570 stars directly from the held-out NASA Kepler test dataset (`exoTest.csv`).
- 📐 **Astrophysical Transit Detection**
  - Astropy Box Least Squares (BLS) periodogram scanning trial periods (0.5 to 20.0 days) and transit durations (1 to 12 hours).
  - Automated flux detrending, baseline normalization, and noise standard deviation estimation.
  - Identification of transit depth ($\delta \approx (R_p / R_\star)^2$), duration, and transit ingress/egress boundaries.
- 🤖 **Ensemble Machine Learning Screening**
  - Composite multi-model evaluation: Tuned XGBoost (primary decision maker), Tuned Random Forest, Baseline Random Forest, Support Vector Machine (SVM), Logistic Regression, and 1D CNN.
  - Model agreement scoring (Unanimous, Majority, Consensus) with calibrated probabilities and benchmark metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
- 📊 **Responsive Interactive Plotly Charts**
  - **Full Light Curve**: Interactive zoomable time series with observed flux, smoothed trend, and flagged transit dip occurrences.
  - **Transit Dip View**: High-resolution zoom centered on the deepest transit event highlighting ingress and egress phases.
  - **Model Comparison**: Side-by-side probability bar chart with hover tooltips and full performance comparison table.
  - **Feature Inspection**: Deep breakdown of 15 statistical, dynamical, transit morphological, and BLS features.
  - **Mobile Optimized**: Shortened axis notation (`Norm. Flux`, `Time / Idx`, `Sample Idx`, `Prob. (%)`), dynamic automargin spacing, and explicit Axis Guide legends below every plot.
- 🪐 **Interactive Planetary Simulation**
  - Custom HTML5 Canvas 3D rendering of an exoplanetary system with dynamic star corona, multi-layered volumetric planets, atmospheric limb darkening, orbital inclinations, and an asteroid belt.
- 💾 **Client-Side Session History**
  - Local persistent history stored via browser `localStorage` with candidate summary metrics and one-click JSON report export.

---

## 🧠 Detection Pipeline

```
Raw Stellar Photometry (CSV / TXT / NPY / Kepler Row)
                        │
                        ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. Ingestion & Preprocessing                           │
 │    • Baseline normalization & detrending               │
 │    • Outlier handling & signal smoothing               │
 └──────────────────────┬─────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Box Least Squares (BLS) Transit Search              │
 │    • Period grid scan: 0.5 d to 20.0 d                 │
 │    • Duration grid: 1 h to 12 h                        │
 │    • BLS peak power, transit depth & SNR               │
 └──────────────────────┬─────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Morphological & Statistical Feature Extraction      │
 │    • Statistical: Mean, Median, Variance, Skew, Kurt   │
 │    • Dynamics: Signal Energy, Spectral Entropy, Noise  │
 │    • Transit Profile: Avg Depth, Duration, Num Dips    │
 └──────────────────────┬─────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────┐
 │ 4. Machine Learning Ensemble Screening                 │
 │    • Tuned XGBoost (Primary Decision Engine)           │
 │    • Random Forest & Tuned RF                          │
 │    • Support Vector Machine (SVM)                      │
 │    • Logistic Regression & 1D CNN                      │
 └──────────────────────┬─────────────────────────────────┘
                        │
                        ▼
 ┌────────────────────────────────────────────────────────┐
 │ 5. Screening Verdict & Diagnostic Telemetry            │
 │    • Candidate / Non-Candidate verdict                 │
 │    • Calibrated confidence score                       │
 │    • Model consensus & agreement diagnostics           │
 │    • Interactive Plotly charts & JSON export           │
 └────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | Vanilla JavaScript (ES6+), HTML5 Canvas (Planetary Simulation), CSS3 (Deep Space Observatory theme), Plotly.js |
| **Backend** | Python 3.12, FastAPI, Uvicorn, Pydantic |
| **Scientific & ML** | Astropy (BLS Periodogram), NumPy, Pandas, Scipy, Scikit-learn, XGBoost, Joblib, H5py |
| **Production Hosting** | **Frontend**: Vercel (`exodip.vercel.app`)<br>**Backend**: Render (`exodip.onrender.com`) |

---

## 📁 Repository Structure

```
AI-Exoplanet-Detection/
├── backend/
│   ├── main.py                # FastAPI endpoints, CORS, static mounting
│   └── services/
│       ├── analysis.py        # ML screening bridge & Plotly chart data generator
│       └── io_helpers.py      # Photometry parser (CSV, TXT, NPY, exoTest loader)
│
├── frontend/
│   ├── index.html             # Single Page Application (Home, Analyze, Results, History, About, Docs)
│   ├── app.js                 # SPA router, API integration, Plotly charts, session state
│   ├── planetary.js           # HTML5 Canvas 3D planetary simulation
│   ├── styles.css             # Observatory design system, responsive breakpoints, drawer nav
│   ├── config.js              # API base URL resolver (supports relative proxy & remote URLs)
│   └── vercel.json            # Vercel deployment & API rewrite proxy configuration
│
├── data/                      # Trained models and benchmark datasets
│   ├── best_rf.pkl            # Tuned Random Forest model
│   ├── rf.pkl                 # Baseline Random Forest model
│   ├── xgb.pkl                # Tuned XGBoost Classifier
│   ├── svm.pkl                # Support Vector Machine classifier
│   ├── log_reg.pkl            # Logistic Regression classifier
│   ├── voting.pkl             # Ensemble Voting classifier
│   ├── scaler.pkl             # StandardScaler for feature normalization
│   ├── selector.pkl           # Feature selection transform
│   ├── cnn_exoplanet.keras    # 1D Convolutional Neural Network weights
│   ├── exoTrain.csv           # Kepler training dataset (5,087 stars, Git LFS)
│   └── exoTest.csv            # Held-out Kepler evaluation dataset (570 stars, Git LFS)
│
├── exoplanet_detector.py      # Core detection engine: BLS, feature extraction & ensemble vetting
├── exoplanet_pipeline.py      # Data preprocessing & feature engineering pipeline
├── run.py                     # Local development entry point (FastAPI + static server)
├── pyproject.toml             # Project metadata & Python dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Installation & Local Development

### Prerequisites
- Python 3.12 or newer
- Git & [Git LFS](https://git-lfs.com/) (required to pull the Kepler datasets)

### 1. Clone the repository
```bash
git clone https://github.com/Divyanshi-gupta1/AI-Exoplanet-Detection.git
cd AI-Exoplanet-Detection

# Pull the Kepler dataset files stored via Git LFS
git lfs pull
```

### 2. Set up a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Run the application
```bash
python run.py
```
Open your browser at **`http://localhost:8000`** to view the application. Interactive API documentation is available at **`http://localhost:8000/docs`**.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Service health status and version |
| `GET` | `/api/test-set-info` | Metadata and row count for the held-out Kepler dataset |
| `GET` | `/api/sample-csv` | Download a valid Kepler light curve with an exoplanet transit |
| `POST` | `/api/analyze/file` | Screen an uploaded photometric light-curve file (`.csv`, `.txt`, `.npy`) |
| `POST` | `/api/analyze/test-row` | Screen a specific row index from the held-out Kepler test set |

### Example Request (`POST /api/analyze/file`)
```bash
curl -X POST http://localhost:8000/api/analyze/file \
  -F "file=@sample_light_curve.csv"
```

### Example Response
```json
{
  "source": "sample_light_curve.csv",
  "prediction": "Planet",
  "probability": 0.998,
  "confidence": "99.8%",
  "model": "XGBoost",
  "model_agreement": "Unanimous",
  "agreement_score": 1.0,
  "bls_period": 3.52,
  "bls_duration": 3.8,
  "bls_snr": 14.8,
  "metrics": {
    "has_transit_dip": true,
    "transit_depth": 0.012,
    "num_transit_groups": 3,
    "vetting_reason": "Periodic transit dip detected with BLS SNR 14.8 >= 3.0",
    "is_relative_flux": true
  },
  "chart_data": {
    "flux": [...],
    "smooth": [...],
    "dips": [...]
  }
}
```

---

## 📊 Dataset

The models are trained and benchmarked on observations from the **NASA Kepler Space Telescope**:

| Split | Stars | Confirmed Planets | Non-Planets | Cadences per Star |
|---|---|---|---|---|
| **Training** (`exoTrain.csv`) | 5,087 | 37 | 5,050 | 3,197 |
| **Held-out Test** (`exoTest.csv`) | 570 | 5 | 565 | 3,197 |

Each observation consists of **3,197 continuous flux measurements** sampled at Kepler's standard 29.4-minute long cadence. The dataset exhibits extreme astronomical class imbalance (< 1% transit positive), making precision-recall optimization and ensemble consensus crucial.

---

## 🤖 Model Benchmark Performance

Benchmark evaluation on the held-out Kepler test set (570 stars):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Composite Rank |
|---|---|---|---|---|---|---|
| **XGBoost (Primary)** | **99.65%** | **100.0%** | **80.0%** | **0.889** | **0.995** | **0.937** |
| **Tuned Random Forest** | 99.30% | 100.0% | 60.0% | 0.750 | 0.994 | 0.869 |
| **Baseline Random Forest** | 99.12% | 100.0% | 40.0% | 0.571 | 0.988 | 0.790 |
| **Support Vector Machine** | 99.12% | 100.0% | 40.0% | 0.571 | 0.957 | 0.780 |
| **1D CNN (NumPy Inference)** | 98.95% | 75.0% | 60.0% | 0.667 | 0.977 | 0.799 |
| **Logistic Regression** | 97.89% | 28.6% | 40.0% | 0.333 | 0.852 | 0.570 |

---

## 📚 Data Sources & References

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [MAST — Mikulski Archive for Space Telescopes](https://mast.stsci.edu/)
- [NASA Kepler Mission Documentation](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Astropy: Box Least Squares Periodogram](https://docs.astropy.org/en/stable/timeseries/bls.html)
- Kovács, G., Zucker, S., & Mazeh, T. (2002). *A box-fitting algorithm in the search for periodic transits*. A&A, 391, 369–377.

---

## 📜 License

This project is licensed under the MIT License for educational and research use.

---

## 👩‍💻 Author

**Divyanshi Gupta**  
[![GitHub](https://img.shields.io/badge/GitHub-Divyanshi--gupta1-181717?logo=github)](https://github.com/Divyanshi-gupta1)
