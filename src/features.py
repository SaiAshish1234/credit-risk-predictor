"""
src/features.py
---------------
Feature engineering pipeline for Credit Risk Predictor.
Used by notebooks AND the Streamlit app — single source of truth.
"""

import numpy as np
import pandas as pd


# ── Columns to drop (>40% missing in training data) ──────────────────────────
HIGH_MISSING_COLS = [
    'COMMONAREA_AVG', 'COMMONAREA_MODE', 'COMMONAREA_MEDI',
    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_MEDI',
    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'FLOORSMIN_AVG',
    'FLOORSMIN_MODE', 'FLOORSMIN_MEDI', 'YEARS_BUILD_AVG',
    'YEARS_BUILD_MODE', 'YEARS_BUILD_MEDI',
]

# ── Selected features for modelling ──────────────────────────────────────────
SELECTED_FEATURES = [
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN',
    'DEBT_TO_INCOME', 'CREDIT_TO_INCOME', 'ANNUITY_TO_CREDIT',
    'AGE_YEARS', 'EMPLOYMENT_YEARS', 'EMPLOYMENT_TO_AGE',
    'INCOME_PER_PERSON', 'LOAN_TO_VALUE',
    'DOCS_SUBMITTED', 'IS_YOUNG_APPLICANT',
    'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
    'DAYS_LAST_PHONE_CHANGE', 'REGION_RATING_CLIENT_W_CITY',
    'REG_CITY_NOT_WORK_CITY', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-driven financial risk features.
    Works on both train and test/app data.

    Parameters
    ----------
    df : Raw DataFrame (application_train.csv or single-row dict)

    Returns
    -------
    DataFrame with engineered features added
    """
    data = df.copy()

    # 1. Debt-to-Income — most important credit risk metric
    data['DEBT_TO_INCOME'] = data['AMT_ANNUITY'] / (data['AMT_INCOME_TOTAL'] + 1)

    # 2. Credit-to-Income — total loan burden relative to income
    data['CREDIT_TO_INCOME'] = data['AMT_CREDIT'] / (data['AMT_INCOME_TOTAL'] + 1)

    # 3. Annuity-to-Credit — indicates loan term length
    data['ANNUITY_TO_CREDIT'] = data['AMT_ANNUITY'] / (data['AMT_CREDIT'] + 1)

    # 4. Applicant age in years (DAYS_BIRTH is negative)
    data['AGE_YEARS'] = np.abs(data['DAYS_BIRTH']) / 365

    # 5. Employment length — 365243 is a special code for unemployed/pensioner
    data['DAYS_EMPLOYED'] = data['DAYS_EMPLOYED'].replace(365243, np.nan)
    data['EMPLOYMENT_YEARS'] = np.abs(data['DAYS_EMPLOYED']) / 365

    # 6. Employment-to-Age — fraction of life spent employed
    data['EMPLOYMENT_TO_AGE'] = data['EMPLOYMENT_YEARS'] / (data['AGE_YEARS'] + 1)

    # 7. Income per family member
    data['INCOME_PER_PERSON'] = data['AMT_INCOME_TOTAL'] / (data['CNT_FAM_MEMBERS'] + 1)

    # 8. Loan-to-Value ratio
    if 'AMT_GOODS_PRICE' in data.columns:
        data['LOAN_TO_VALUE'] = data['AMT_CREDIT'] / (data['AMT_GOODS_PRICE'] + 1)
    else:
        data['LOAN_TO_VALUE'] = np.nan

    # 9. External credit score aggregates (3 scores provided by Home Credit)
    ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in data.columns]
    if ext_cols:
        data['EXT_SOURCE_MEAN'] = data[ext_cols].mean(axis=1)
        data['EXT_SOURCE_MIN']  = data[ext_cols].min(axis=1)
        data['EXT_SOURCE_MAX']  = data[ext_cols].max(axis=1)

    # 10. Document submission count
    doc_cols = [c for c in data.columns if 'FLAG_DOCUMENT' in c]
    if doc_cols:
        data['DOCS_SUBMITTED'] = data[doc_cols].sum(axis=1)
    else:
        data['DOCS_SUBMITTED'] = 0

    # 11. Young applicant flag (< 27 years = statistically higher default)
    data['IS_YOUNG_APPLICANT'] = (data['AGE_YEARS'] < 27).astype(int)

    # 12. Registration recency
    if 'DAYS_REGISTRATION' in data.columns:
        data['REG_YEARS'] = np.abs(data['DAYS_REGISTRATION']) / 365

    return data


def clean_dataset(df: pd.DataFrame, drop_threshold: float = 0.40) -> pd.DataFrame:
    """
    Full cleaning pipeline: drop high-missing cols, encode categoricals, impute.

    Parameters
    ----------
    df             : Engineered DataFrame
    drop_threshold : Drop columns with more than this fraction missing

    Returns
    -------
    Clean DataFrame ready for train/test split
    """
    data = df.copy()

    # Drop ID column
    if 'SK_ID_CURR' in data.columns:
        data = data.drop(columns=['SK_ID_CURR'])

    # Drop high-missing columns
    missing_pct = data.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > drop_threshold].index.tolist()
    data = data.drop(columns=cols_to_drop, errors='ignore')

    # Encode categoricals
    cat_cols = data.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        data[col] = pd.Categorical(data[col]).codes

    # Impute numeric nulls with median
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())

    # Remove any infinities
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    return data


def get_feature_names() -> list:
    """Returns the list of selected feature names used by the model."""
    return SELECTED_FEATURES
