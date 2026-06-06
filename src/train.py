"""
src/train.py
------------
Model training pipeline for Credit Risk Predictor.
Run this after Week 1 EDA notebook to train and save the model.

Usage:
    python src/train.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/processed/train_processed.csv'
MODEL_PATH = 'models/xgb_final.pkl'
SCALER_PATH = 'models/scaler.pkl'


def load_data():
    print('Loading processed data...')
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['TARGET'])
    y = df['TARGET']
    print(f'Shape: {X.shape} | Default rate: {y.mean():.2%}')
    return X, y


def train_baseline(X_train, y_train, X_test, y_test):
    """Quick logistic regression baseline."""
    print('\n── Baseline: Logistic Regression ──')
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_te_sc)[:, 1])
    print(f'AUC-ROC: {auc:.4f}')
    return auc


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with SMOTE and log to MLflow."""
    print('\n── XGBoost + SMOTE ──')

    # Apply SMOTE to balance classes
    print('Applying SMOTE...')
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'After SMOTE: {X_res.shape} | Default rate: {y_res.mean():.2%}')

    # XGBoost params (tuned for credit risk)
    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'scale_pos_weight': 1,          # SMOTE already balanced
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
    }

    with mlflow.start_run(run_name='xgb_smote'):
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(
            X_res, y_res,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Metrics
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc  = roc_auc_score(y_test, y_prob)
        f1   = f1_score(y_test, y_pred)
        ap   = average_precision_score(y_test, y_prob)

        mlflow.log_metrics({'auc_roc': auc, 'f1': f1, 'avg_precision': ap})
        mlflow.xgboost.log_model(model, 'xgb_model')

        print(f'AUC-ROC          : {auc:.4f}')
        print(f'F1 Score         : {f1:.4f}')
        print(f'Avg Precision    : {ap:.4f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))

    return model, auc


def save_model(model, scaler=None):
    os.makedirs('models', exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'\nModel saved → {MODEL_PATH}')
    if scaler:
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)


def main():
    X, y = load_data()

    # Train/test split — stratified to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {X_train.shape} | Test: {X_test.shape}')

    # Baseline
    train_baseline(X_train, y_train, X_test, y_test)

    # Main model
    model, auc = train_xgboost(X_train, y_train, X_test, y_test)

    save_model(model)
    print(f'\n✓ Training complete. Best AUC: {auc:.4f}')
    print('→ Next: run notebook 04_shap_explainability.ipynb')


if __name__ == '__main__':
    main()
