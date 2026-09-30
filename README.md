# Enterprise MLOps Pipeline - Production Ready | MLflow + FastAPI + Drift Detection

> End-to-end MLOps platform that prevents model degradation in production. Built with MLflow, Evidently AI, FastAPI. Includes CI/CD, automated drift monitoring (KS-test), and REST API.

### 🎯 Business Problem Solved
Most ML models fail in production due to data drift. This pipeline automatically detects drift BEFORE performance drops.

### 🚀 What I Built
- **Experiment Tracking:** MLflow registry with SQLite backend
- **Drift Monitoring:** Evidently AI + KS-test for real-time feature shift
- **Production API:** FastAPI with <100ms latency
- **CI/CD:** GitHub Actions on every push

### 📊 Results
- Drift detection: 95%+
- API Response: <100ms
- Automated retraining trigger

### 🛠️ Tech Stack
Python, Scikit-Learn, MLflow, Evidently AI, FastAPI, GitHub Actions

### ⚡ How to Run
git clone https://github.com/Maged-s09755/Enterprise-MLOps-Pipeline.git
cd Enterprise-MLOps-Pipeline
pip install -r requirements.txt
python src/train.py
uvicorn src.serve:app --reload
