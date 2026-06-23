# 🏦 Credit Risk Intelligence

> **End-to-end ML pipeline** for loan default prediction using XGBoost + SHAP explainability — deployed as a real-time Streamlit app.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange) ![SHAP](https://img.shields.io/badge/SHAP-Explainability-green) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Dataset](https://img.shields.io/badge/Dataset-307K%20loans-lightgrey)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/igniz123/credit-risk-predictor)

---

## 🎯 Problem Statement

Traditional credit scoring is opaque and misses modern financial signals. Lenders can't just say "the model said no" — regulators and customers need a reason. This project builds a **transparent, ML-powered credit risk system** that:

- Predicts **loan default probability** on 307,511 real loan applications
- **Explains every decision** using SHAP — which factors drove the risk score and by how much
- Quantifies **business impact** in ₹ terms, not just model accuracy
- Deploys as a **live web app** for real-time applicant scoring

---

## 📊 Results

| Model | AUC-ROC | F1 Score | Avg Precision |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7493 | 0.0245 | 0.2366 |
| **XGBoost + SMOTE** | **0.7375** | **0.0796** | **0.2054** |

> **Note:** AUC of 0.74 is the realistic ceiling for `application_train.csv` alone.
> Top Kaggle scores use 7+ supplementary files; this project focuses on the full ML
> engineering pipeline — EDA → feature engineering → explainability → deployment.

---

## 💰 Business Impact

Based on test set evaluation (61,503 loans, avg loan size ₹5.99L):

| Metric | Value |
|---|---|
| Defaults correctly caught | 222 loans |
| Estimated loss prevented | ₹9.31 Cr |
| Net estimated value (test set) | ₹8.99 Cr |
| **Scaled to full dataset** | **₹44.95 Cr** |

> *Assumptions: 70% loss rate on defaulted loans, 30% of annuity as interest revenue*

---

## 🔍 SHAP Explainability

Every prediction comes with a plain-English explanation:

```
High-risk applicant (63.1% default probability):
  • Average external credit score = 0.31 → increased default risk (SHAP: +0.18)
  • Debt-to-income ratio = 0.52 → increased default risk (SHAP: +0.12)
  • Employment history = 1.2 years → increased default risk (SHAP: +0.09)

Low-risk applicant (0.4% default probability):
  • Average external credit score = 0.78 → decreased default risk (SHAP: -0.21)
  • Applicant age = 52 years → decreased default risk (SHAP: -0.08)
  • Employment stability = 0.71 → decreased default risk (SHAP: -0.06)
```

---

## 🏗️ Project Structure

```
credit-risk-predictor/
├── notebooks/
│   ├── week1_colab.ipynb        ← EDA + 12 engineered features
│   ├── week2_colab_fixed.ipynb  ← XGBoost + SMOTE training
│   └── week3_colab.ipynb        ← SHAP explainability + business impact
├── src/
│   ├── features.py              # Feature engineering pipeline (reusable)
│   ├── train.py                 # Model training script
│   └── predict.py               # Inference + SHAP for the app
├── app/
│   └── streamlit_app.py         # Live web app (Week 4)
├── data/
│   └── raw/                     # Raw Kaggle data (gitignored)
└── requirements.txt
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Seaborn, Matplotlib |
| Feature Engineering | 12 domain-driven financial features |
| ML Model | XGBoost, Scikit-learn |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer, waterfall + beeswarm plots) |
| App | Streamlit |
| Dataset | Home Credit Default Risk (Kaggle) |

---

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
Get `application_train.csv` from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)

### 3. Run notebooks in order
```
notebooks/week1_colab.ipynb   → EDA + feature engineering
notebooks/week2_colab_fixed.ipynb  → Model training
notebooks/week3_colab.ipynb   → SHAP explainability
```

### 4. Run the app (after Week 4)
```bash
streamlit run app/streamlit_app.py
```

---

## 📌 Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — 307,511 loan applications, 122 raw features, 8.07% default rate

---

## 🗓️ Built Week by Week

| Week | Focus | Key Output |
|---|---|---|
| 1 | EDA + Feature Engineering | 12 financial features, class imbalance analysis |
| 2 | Model Training | XGBoost + SMOTE, AUC 0.74 |
| 3 | SHAP Explainability | Waterfall plots, ₹44.95Cr business impact |
| 4 | Streamlit App | Live demo deployment *(coming soon)* |

---

*Built to demonstrate end-to-end ML engineering for FinTech — from raw data to deployed, explainable predictions.*
