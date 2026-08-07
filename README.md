# AI-Powered Parkinson's Disease Severity Assessment Using Voice, EEG, and Gait Analysis

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end multimodal Artificial Intelligence system for predicting Parkinson's Disease (PD) severity, estimating the UPDRS (Unified Parkinson's Disease Rating Scale) score, classifying severity into Mild, Moderate, or Severe levels, providing Explainable AI (XAI) insights via SHAP and Grad-CAM, and generating automated clinical PDF reports.

---

## 🌟 Key Features

1. **Multimodal Processing Pipelines**:
   - **Voice**: MFCCs, Delta/Delta-Delta, Pitch (F0), Jitter, Shimmer, HNR, Spectral Centroid/Roll-off/Contrast, ZCR, RMS, Formants.
   - **EEG**: Bandpass filtering, ICA artifact rejection, Delta/Theta/Alpha/Beta/Gamma relative band powers, Spectral Entropy, PLV (Phase Locking Value) functional connectivity.
   - **Gait**: IMU signal conditioning, step/stride length estimation, cadence, walking speed, swing/stance time, gait asymmetry, freeze-of-gait index, jerk smoothness.
2. **Deep Learning & Transformer Fusion**:
   - **Voice Encoder**: 1D-CNN with residual connections.
   - **EEG Encoder**: CNN + Multi-head Temporal Attention.
   - **Gait Encoder**: Bidirectional LSTM with temporal attention pooling.
   - **Multimodal Fusion**: Transformer cross-attention network with modality dropout for graceful handling of missing signals.
3. **Dual-Task Multi-Objective Learning**:
   - **Regression**: UPDRS total score estimation ($0.6 \times \text{MSE/Huber Loss}$).
   - **Classification**: Severity category ($0.4 \times \text{Cross Entropy Loss}$).
4. **Explainable AI (XAI)**:
   - SHAP summary, waterfall, and bar plots.
   - Grad-CAM 1D activation maps over feature sequences.
   - Cross-modal attention weight visualization.
5. **Production Deployment & REST API**:
   - **Streamlit App**: 7-page interactive glassmorphism UI dashboard.
   - **FastAPI**: Asynchronous REST endpoints (`/predict`, `/train`, `/report`, `/patient`, `/history`).
   - **PostgreSQL Database**: Async SQLAlchemy 2.0 schema for patient records, predictions, reports, and audit logs.
   - **Clinical PDF Reports**: Automated ReportLab PDF generator.

---

## 📁 Repository Structure

```
parkinson-severity-assessment/
├── api/                     # FastAPI REST API backend & endpoints
│   ├── main.py
│   ├── schemas.py
│   └── routers/             # predict, patients, reports, history
├── app/                     # Streamlit frontend application
│   ├── streamlit_app.py     # Entry point & dashboard landing
│   └── pages/               # 7-page interactive dashboard
├── data/                    # Datasets and preparation scripts
│   ├── raw/
│   ├── processed/
│   └── scripts/             # download_datasets.py
├── diagrams/                # Mermaid IEEE architecture diagrams
├── docker/                  # Dockerfile, Dockerfile.api, docker-compose.yml
├── src/                     # Core machine learning source code
│   ├── database/            # PostgreSQL SQL schema & SQLAlchemy ORM
│   ├── features/            # Feature engineering, scaling, selection, PCA
│   ├── models/              # PyTorch encoders (Voice, EEG, Gait, Fusion)
│   ├── processing/          # Voice, EEG, and Gait signal processing
│   ├── reports/             # Clinical PDF report generator (ReportLab)
│   ├── training/            # PyTorch training, loss functions, metrics
│   └── xai/                 # SHAP, Grad-CAM, attention visualizations
├── tests/                   # Pytest suite (processing, models, API)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and set up a Virtual Environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Prepare Data
Initialize directories and generate synthetic dataset for pipeline validation:
```bash
python data/scripts/download_datasets.py --synthetic
```

### 3. Run the Streamlit Application
Launch the web interface:
```bash
streamlit run app/streamlit_app.py
```
Open `http://localhost:8501` in your browser.

### 4. Run the FastAPI Backend
Start the asynchronous API server:
```bash
python api/main.py
```
Access interactive Swagger API documentation at `http://localhost:8000/docs`.

---

## 🐳 Docker Deployment

To spin up PostgreSQL, FastAPI, and Streamlit in containers:
```bash
cd docker
docker-compose up --build
```
- **Streamlit App**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8000/docs`
- **PostgreSQL Database**: `localhost:5432`

---

## 🧪 Testing & Verification

Execute the full automated test suite:
```bash
pytest tests/ -v
```

---

## 📄 IEEE Paper Methodology Mapping

| IEEE Section | Implemented Component | File Location |
| :--- | :--- | :--- |
| **Section III-A: Signal Acquisition & Processing** | Voice (MFCC, Pitch, Praat), EEG (MNE, PLV), Gait (IMU) | `src/processing/` |
| **Section III-B: Deep Feature Encoding** | Voice CNN, EEG CNN-Attention, Gait BiLSTM | `src/models/` |
| **Section III-C: Multimodal Transformer Fusion** | Cross-attention fusion with modality dropout | `src/models/fusion.py` |
| **Section III-D: Dual-Task Learning** | $0.6 \text{MSE} + 0.4 \text{CE}$ Loss | `src/training/losses.py` |
| **Section IV: Explainability & Clinical Utility** | SHAP, Grad-CAM, Attention maps, PDF reports | `src/xai/`, `src/reports/` |

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
