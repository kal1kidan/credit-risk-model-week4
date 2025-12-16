# Credit Risk Probability Model

Predicting credit risk using alternative data with an interpretable, regulation-aware ML pipeline, full experiment tracking, and production-ready API deployment.

---

## 📝 Business Understanding

### 1. Basel II and the Need for Interpretability

Under the **Basel II Accord** (IRB approach), financial institutions must measure, document, and justify credit risk decisions with:

* Clear and stable risk estimates
* Transparent, auditable model logic
* Consistent documentation of assumptions and features
* Explainability for regulators, auditors, and customers

Because of these requirements, credit scoring models **cannot be black boxes**. Even if advanced ML models achieve higher accuracy, banks must be able to clearly explain *why* a customer is classified as low- or high-risk.

**Design principles used in this project:**

* ✔ Interpretable models for production (Logistic Regression with WoE/IV)
* ✔ Stable and monitorable risk estimates
* ✔ Fully documented assumptions and feature logic
* ✔ Fairness- and transparency-aware modeling choices

Complex models (Random Forest, Gradient Boosting) are trained as **challenger models** for benchmarking accuracy, not as primary regulatory models.

---

### 2. Proxy Target Engineering (RFM-Based)

The dataset does not contain an explicit default label (e.g., loan non-repayment). To enable supervised learning, a **proxy target variable** is engineered using customer behavioral patterns.

#### RFM Methodology

* **Recency (R):** Time since last transaction
* **Frequency (F):** Number of transactions in a fixed window
* **Monetary (M):** Total transaction value

Steps:

1. Compute RFM metrics per customer
2. Normalize RFM values
3. Cluster customers using K-Means
4. Label the *least active cluster* as **High Risk (1)**
5. Label all other clusters as **Low Risk (0)**

This approach is common in **alternative credit scoring** where repayment history is unavailable.

#### Business Risks & Mitigation

| Risk            | Description                            | Mitigation                            |
| --------------- | -------------------------------------- | ------------------------------------- |
| Mislabeling     | Low spenders ≠ defaulters              | Conservative thresholds, revalidation |
| Bias            | Clusters may proxy income/demographics | Feature audits, fairness checks       |
| Regulatory risk | Proxy ≠ true default                   | Pre-production validation only        |
| Model drift     | Engagement behavior changes            | Periodic re-clustering                |

> ⚠️ **Important:** This proxy target is temporary and must be replaced once real repayment data becomes available.

---

### 3. Trade-Offs: Simple vs. Complex Models

| Aspect                | Simple Models       | Complex Models       |
| --------------------- | ------------------- | -------------------- |
| Examples              | Logistic Regression | RF, GBM, Neural Nets |
| Interpretability      | High                | Low–Medium           |
| Accuracy              | Moderate            | High                 |
| Regulatory acceptance | Strong              | Limited              |
| Monitoring            | Easy                | Complex              |

**Final Strategy:**

* Production model: **Logistic Regression + WoE/IV**
* Challenger models: Random Forest & Gradient Boosting

This balances:

* Accuracy (business performance)
* Interpretability (regulatory compliance)
* Operational stability (monitoring & updates)

---

## 🧠 Project Overview

End-to-end credit risk modeling system covering:

* Data exploration & feature engineering
* RFM-based proxy target construction
* Model training & evaluation
* MLflow experiment tracking
* FastAPI deployment
* Dockerization & CI pipeline

---

## 📂 Folder Structure

```
credit-risk-model-week4/
│
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned & feature-engineered data
│   └── external/            # Optional external data
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA & summary statistics
│   ├── 02_rfm_proxy_label.ipynb       # RFM computation & clustering
│   ├── 03_feature_engineering.ipynb   # WoE/IV & preprocessing
│   └── 04_model_training.ipynb        # Training & evaluation
│
├── src/
│   ├── rfm_label.py          # Proxy target generation pipeline
│   ├── preprocess.py        # Data preprocessing
│   ├── train_model.py       # Model training & tuning
│   ├── evaluate.py          # Metrics & plots
│   └── utils.py             # Shared helpers
│
├── api/
│   ├── main.py               # FastAPI application
│   ├── schemas.py           # Request/response models
│   └── inference.py         # Model loading & prediction
│
├── docker/
│   └── Dockerfile            # API containerization
│
├── .github/workflows/
│   └── ci.yml                # CI pipeline (lint, test, build)
│
├── mlruns/                   # MLflow experiment tracking
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## ⚙️ Tasks & Deliverables

### 1. Data Exploration

* Dataset understanding & cleaning
* Summary statistics & visualizations
* Missing value and correlation analysis

### 2. Proxy Target Engineering

* Compute RFM metrics
* Perform clustering
* Generate high-risk vs low-risk labels
* Validate cluster stability

### 3. Feature Engineering

* WoE/IV encoding for categorical variables
* Scaling & missing value handling
* Feature selection

### 4. Model Training

* Logistic Regression (production)
* Random Forest & Gradient Boosting (challengers)
* Hyperparameter tuning (GridSearchCV)

### 5. Evaluation & Metrics

* Accuracy, Precision, Recall, F1-score
* ROC-AUC comparison
* Confusion matrices & feature importance

### 6. Experiment Tracking (MLflow)

* Parameter, metric, and artifact logging
* Model comparison & reproducibility

### 7. Deployment (FastAPI)

* REST API for real-time predictions
* Input validation with Pydantic
* Serialized model loading

### 8. MLOps (Docker + CI)

* Dockerized API service
* CI pipeline for linting, testing, and builds

---

## 🚀 Setup & Usage

```bash
# Clone repo
git clone <repo_url>
cd credit-risk-model-week4

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Pipeline

```bash
python src/rfm_label.py
python src/preprocess.py
python src/train_model.py
```

### MLflow UI

```bash
mlflow ui
```

Open `http://localhost:5000`

---

## 🌐 API Usage

```bash
uvicorn api.main:app --reload
```

Endpoint:

```
POST /predict
```

Returns probability of high credit risk.

---

## 📈 Results (Sample)

* **Best Production Model:** Logistic Regression (WoE/IV)
* **Best Challenger:** Random Forest (ROC-AUC ≈ 0.87)
* **Key Drivers:** transaction_recency, transaction_frequency, monetary_value

---

## 🔮 Future Improvements

* Replace proxy target with real repayment data
* Add fairness & bias metrics
* Automated retraining with scheduled pipelines
* Cloud deployment (AWS/GCP)

---

## 📌 Feedback Addressed

✔ Fully implemented RFM-based proxy target pipeline
✔ Clear distinction between production & challenger models
✔ Working FastAPI inference service
✔ Docker + CI pipeline for end-to-end delivery

---

**Author:** Kalkidan Asdesach
**Project Type:** End-to-End Credit Risk Scoring & MLOps

