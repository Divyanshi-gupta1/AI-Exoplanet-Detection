
# 🔭 ExoDip — AI Exoplanet Transit Screening

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ensemble-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-classifier-blue)
![Vercel](https://img.shields.io/badge/Deployed-Vercel-black?logo=vercel)
![License](https://img.shields.io/badge/License-Educational-green)

**A high-precision exoplanet transit candidate screening engine combining ensemble ML with astronomical signal analysis.**

Upload a Kepler/TESS light curve → get a planet/non-planet verdict in seconds.

[🌐 Live Demo](https://ai-exoplanet-detection.vercel.app) · [📖 API Docs](#api-reference) · [🚀 Run Locally](#installation)

</div>

---

## 🌌 What is ExoDip?

ExoDip automates exoplanet transit detection by fusing **5 trained ML models** with **physics-based signal verification**. It analyzes stellar brightness time-series (light curves) to detect the characteristic dimming caused by a planet crossing in front of its host star.

The project was originally built as a Jupyter notebook, then evolved into a full-stack web application — a standalone **FastAPI backend** with a **vanilla JS single-page frontend** — deployable to Vercel with no extra infrastructure.

---

## ✨ Features

- 📤 **File upload** — drag-and-drop or browse any CSV / TXT / NPY light curve
- 🔭 **Kepler test-set mode** — analyze any row from the 570-star held-out Kepler dataset
- 🤖 **5-model ensemble** — Random Forest, Tuned RF, XGBoost, SVM, Logistic Regression + 1D CNN
- 📐 **Physics verification** — BLS periodogram, transit SNR, depth/duration consistency, two-pass detrending
- 📊 **Interactive charts** — full light-curve plot with detected dip window (Plotly.js, rendered client-side)
- 💾 **Session history** — all analyses stored locally in-browser with JSON export
- 🌑 **Deep-space UI** — starfield canvas, glassmorphism cards, smooth page transitions

---

## 🧠 Detection Pipeline

```
Light Curve (flux time-series)
         │
         ▼
   Preprocessing & Detrending
   (normalization, sigma-clipping, polynomial baseline)
         │
         ▼
   Feature Extraction (40+ features)
   ┌─────────────────────────────────┐
   │ Statistical  │ Frequency Domain │
   │ Transit Dip  │ Wavelet / FFT    │
   │ BLS Period   │ Rolling Stats    │
   └─────────────────────────────────┘
         │
         ▼
   Ensemble Voting (5 ML models + 1D CNN)
   ┌──────────────────────────────────────┐
   │ Random Forest  │ XGBoost            │
   │ Tuned RF       │ SVM                │
   │ Logistic Reg.  │ CNN (pure NumPy)   │
   └──────────────────────────────────────┘
         │
         ▼
   Physics Sanity Check (BLS, SNR, depth)
         │
         ▼
   🪐 PLANET  /  ⭐ NON-PLANET  +  Confidence %
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **ML / Science** | scikit-learn, XGBoost, scipy, astropy (BLS), h5py |
| **Frontend** | Vanilla JS (ES modules), HTML5 Canvas, Plotly.js CDN |
| **Deployment** | Vercel (serverless Python function) |
| **Data** | NASA Kepler Exoplanet Dataset (Kepler Space Telescope) |

---

## 📁 Project Structure

```
AI-Exoplanet-Detection/
├── backend/
│   ├── main.py                # FastAPI app — all API endpoints
│   └── services/
│       ├── analysis.py        # ML pipeline bridge + chart data prep
│       └── io_helpers.py      # Light-curve parser (CSV/TXT/NPY)
│
├── frontend/
│   ├── index.html             # Single-page application shell
│   ├── app.js                 # Routing, API client, Plotly charts
│   ├── planetary.js           # HTML5 Canvas 3D planet animation
│   └── styles.css             # Deep-space design system
│
├── data/
│   ├── best_rf.pkl            # Tuned Random Forest (primary model)
│   ├── rf.pkl                 # Baseline Random Forest
│   ├── xgb.pkl                # XGBoost classifier
│   ├── svm.pkl                # Support Vector Machine
│   ├── log_reg.pkl            # Logistic Regression
│   ├── voting.pkl             # Ensemble voting classifier
│   ├── scaler.pkl             # StandardScaler
│   ├── selector.pkl           # Feature selector
│   ├── cnn_exoplanet.keras    # 1D CNN weights (loaded via h5py/NumPy)
│   ├── exoTrain.csv           # Kepler training set — 5087 stars (Git LFS)
│   └── exoTest.csv            # Kepler test set — 570 stars  (Git LFS)
│
├── exoplanet_detector.py      # Core detection engine (models + BLS + CNN)
├── exoplanet_pipeline.py      # Feature extraction helpers
├── run.py                     # Local dev launcher
├── pyproject.toml             # Dependencies + Vercel entrypoint config
└── vercel.json                # Vercel function bundle config
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.12+
- Git (with [Git LFS](https://git-lfs.com/) for the CSV datasets)

### Clone & setup

```bash
git clone https://github.com/Divyanshi-gupta1/AI-Exoplanet-Detection.git
cd AI-Exoplanet-Detection

# Pull LFS files (large CSVs)
git lfs pull

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Run locally

```bash
python run.py
# → http://localhost:8000
```

The API docs are available at `http://localhost:8000/docs`.

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/test-set-info` | Kepler test-set row count |
| `GET` | `/api/sample-csv` | Download a demo light curve CSV |
| `POST` | `/api/analyze/file` | Screen an uploaded light-curve file |
| `POST` | `/api/analyze/test-row` | Screen a Kepler test-set row by index |

### Example — analyze a file

```bash
curl -X POST http://localhost:8000/api/analyze/file \
  -F "file=@my_lightcurve.csv"
```

### Example response

```json
{
  "label": "Planet",
  "confidence": 0.9976,
  "method": "ensemble_ml",
  "models_voted": 5,
  "bls_period_days": 3.71,
  "transit_depth_ppt": 12.4,
  "snr": 18.3,
  "source": "my_lightcurve.csv"
}
```

---

## 📊 Dataset

The models are trained on the **NASA Kepler Exoplanet Dataset**:

| Split | Stars | Confirmed Planets | Non-Planets |
|---|---|---|---|
| Training (`exoTrain.csv`) | 5,087 | 37 | 5,050 |
| Test (`exoTest.csv`) | 570 | 5 | 565 |

Each observation contains **3,197 flux measurements** from Kepler's 30-minute cadence photometry. Large CSV files are stored via **Git LFS**.

---

## 🤖 Model Performance

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Tuned Random Forest | 99.3% | 0.994 | 0.82 |
| XGBoost | 99.1% | 0.990 | 0.80 |
| Ensemble (Voting) | 99.3% | 0.993 | 0.82 |
| 1D CNN (NumPy) | — | 0.977 | 0.62 |

> Metrics are on the held-out Kepler test set (570 stars, class-imbalanced: 5 planets vs 565 non-planets).

---

## 🔮 Future Work

- [ ] Real-time TESS/MAST light curve fetching via API
- [ ] Habitability ranking for confirmed candidates
- [ ] Attention-based transformer for raw flux sequences
- [ ] Multi-planet system detection
- [ ] User authentication + cloud analysis history

---

## 📚 Data Sources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [MAST — Mikulski Archive for Space Telescopes](https://mast.stsci.edu/)
- [Kepler Mission](https://www.nasa.gov/mission_pages/kepler/main/index.html)
- [Kaggle Kepler Exoplanet Search Results](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)

---

## 📜 License

This project is intended for **educational and research purposes**.

---

## 👩‍💻 Author

**Divyanshi Gupta** — Computer Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-Divyanshi--gupta1-181717?logo=github)](https://github.com/Divyanshi-gupta1)
