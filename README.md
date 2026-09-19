# 🌌 AI-Based Exoplanet Detection using TESS Light Curves

An end-to-end machine learning pipeline that analyzes **TESS (Transiting Exoplanet Survey Satellite)** light curves to identify **exoplanet transit candidates** and distinguish them from **false positives** using **XGBoost**.

---

## 📖 Project Overview

The discovery of exoplanets is one of the most exciting fields in modern astronomy. Space missions like **TESS** monitor the brightness of millions of stars, searching for tiny periodic dips caused by planets passing in front of their host stars.

This project automates part of that detection process by:

- Downloading TESS light curves
- Extracting transit features using the **Box Least Squares (BLS)** algorithm
- Building a labeled dataset of confirmed exoplanets and false positives
- Training an **XGBoost** classifier
- Predicting the probability that a newly observed transit signal belongs to a genuine exoplanet

---

## 🚀 Features

- Download light curves directly from the TESS archive
- Automatic preprocessing of light curves
- Transit detection using Box Least Squares (BLS)
- Feature extraction including:
  - Transit Period
  - Transit Duration
  - Transit Depth
  - BLS Signal Power
  - Flux Standard Deviation
- Binary classification using XGBoost
- Probability prediction for new candidate stars
- Dataset generation for future model training

---

## 🛰️ Workflow

```text
TESS Light Curve
        │
        ▼
Light Curve Preprocessing
        │
        ▼
Box Least Squares (BLS)
        │
        ▼
Feature Extraction
        │
        ▼
Dataset Creation
        │
        ▼
XGBoost Classifier
        │
        ▼
Exoplanet Candidate / False Positive
```

---

## 📊 Dataset

The project uses publicly available data from the **NASA Exoplanet Archive**.

### Positive Class

- Confirmed TESS exoplanets

### Negative Class

- TESS False Positive (FP) objects

After feature extraction, both datasets are combined into a machine learning dataset.

---

## 🧠 Machine Learning Model

**Algorithm:** XGBoost Classifier

### Input Features

- Period
- Duration
- Transit Depth
- BLS Power
- Flux Standard Deviation

### Output

Probability that the observed transit signal corresponds to a genuine exoplanet candidate.

---

## 🛠 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Lightkurve
- Astropy
- Astroquery
- Scikit-learn
- XGBoost

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Divyanshi-gupta1/AI-Exoplanet-Detection.git

cd AI-Exoplanet-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
exoplanet_detection.ipynb
```

---

## 📈 Future Improvements

- Increase the training dataset size
- Add additional transit and stellar features
- Improve classification accuracy
- Deep learning on raw light curves
- Real-time candidate detection pipeline
- Habitability ranking for detected exoplanet candidates

---

## 📚 Data Source

- NASA Exoplanet Archive
- MAST (Mikulski Archive for Space Telescopes)
- TESS Mission

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👩‍💻 Author

**Divyanshi Gupta**

Computer Engineering Student
