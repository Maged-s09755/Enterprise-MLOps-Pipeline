# 🚀 Enterprise MLOps Pipeline: Automated Drift Detection & Model Governance

An enterprise-grade, production-ready MLOps pipeline implementing continuous model training, experiment tracking, API serving, and automated real-time Data Drift monitoring.

## 🛠️ Architecture & Tech Stack
- **Experiment Tracking & Registry:** MLflow
- **Data Drift & Quality Monitoring:** Evidently AI (Kolmogorov-Smirnov statistical testing)
- **Model Serving API:** FastAPI & Uvicorn
- **Automated CI/CD:** GitHub Actions
- **Core Engine:** Python 3.10, Scikit-Learn, Pandas

## 🌟 Key Features
1. **Automated Experiment Tracking:** Logs metrics, model parameters, and serializes binaries into an SQLite-backed MLflow Registry.
2. **Real-time Drift Detection:** Detects data shifts across input feature distributions before model degradation occurs in production.
3. **Continuous Integration Pipeline:** Runs automated pipeline execution and statistical drift validation on every `push` event via GitHub Actions.
4. **High-Performance API:** Exposes REST endpoints with JSON response payloads for model predictions.

## 🚀 How to Run

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/Enterprise-MLOps-Pipeline.git](https://github.com/YOUR_USERNAME/Enterprise-MLOps-Pipeline.git)
cd Enterprise-MLOps-Pipeline
pip install -r requirements.txt
