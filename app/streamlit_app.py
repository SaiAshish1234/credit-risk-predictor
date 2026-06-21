"""
app/streamlit_app.py
--------------------
Credit Risk Intelligence — Streamlit App
Clean, professional business UI

Run:
    streamlit run app/streamlit_app.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Credit Risk Intelligence',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── Clean business CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Clean white/light surface look */
[data-testid="stAppViewContainer"] {
    background: #F7F8FA;
}
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] .stMarkdown p {
    color: #374151;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 20px 0 6px 0;
}
/* Top bar */
.top-bar {
    background: #FFFFFF;
    border-bottom: 1px solid #E5E7EB;
    padding: 16px 0 14px 0;
    margin-bottom: 24px;
}
.top-bar-inner {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon {
    width: 36px; height: 36px;
    background: #1E40AF;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.brand-name {
    font-size: 17px;
    font-weight: 600;
    color: #111827;
    margin: 0;
}
.brand-sub {
    font-size: 12px;
    color: #9CA3AF;
    margin: 0;
}
/* Stat cards */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}
.stat-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 18px;
}
.stat-label {
    font-size: 12px;
    color: #6B7280;
    margin: 0 0 4px 0;
    font-weight: 500;
}
.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin: 0;
    line-height: 1.2;
}
.stat-sub {
    font-size: 11px;
    color: #9CA3AF;
    margin: 4px 0 0 0;
}
/* Risk badge */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}
.badge-low    { background: #DCFCE7; color: #15803D; }
.badge-medium { background: #FEF9C3; color: #854D0E; }
.badge-high   { background: #FEE2E2; color: #991B1B; }
/* Result card */
.result-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.result-card.border-low    { border-left: 4px solid #16A34A; }
.result-card.border-medium { border-left: 4px solid #CA8A04; }
.result-card.border-high   { border-left: 4px solid #DC2626; }
.result-title {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
    margin: 0 0 4px 0;
}
.result-sub {
    font-size: 13px;
    color: #6B7280;
    margin: 0;
}
/* Section titles */
.section-title {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #F3F4F6;
}
/* Flag items */
.flag-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
}
.flag-warn { background: #FFF7ED; border: 1px solid #FED7AA; }
.flag-ok   { background: #F0FDF4; border: 1px solid #BBF7D0; }
/* Summary rows */
.summary-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #F9FAFB;
    font-size: 13px;
}
.summary-key   { color: #6B7280; }
.summary-value { color: #111827; font-weight: 500; }
/* Info panel */
.info-panel {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 28px 32px;
    margin-top: 8px;
}
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 16px;
}
.info-step { padding: 14px; background: #F9FAFB; border-radius: 8px; }
.info-step-num {
    font-size: 11px;
    font-weight: 700;
    color: #1E40AF;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin: 0 0 4px 0;
}
.info-step-title {
    font-size: 14px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 2px 0;
}
.info-step-desc {
    font-size: 12px;
    color: #6B7280;
    margin: 0;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    paths = {
        'model'    : ['/content/drive/MyDrive/xgb_credit_risk.pkl',  'models/xgb_credit_risk.pkl'],
        'features' : ['/content/drive/MyDrive/selected_features.pkl','models/selected_features.pkl'],
        'explainer': ['/content/drive/MyDrive/shap_explainer.pkl',   'models/shap_explainer.pkl'],
        'labels'   : ['/content/drive/MyDrive/feature_labels.pkl',   'models/feature_labels.pkl'],
    }
    arts = {}
    for key, candidates in paths.items():
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    arts[key] = pickle.load(f)
                break
        if key not in arts:
            arts[key] = None
    return arts


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(raw):
    d = raw.copy()
    income  = d['AMT_INCOME_TOTAL']
    credit  = d['AMT_CREDIT']
    annuity = d['AMT_ANNUITY']
    goods   = d.get('AMT_GOODS_PRICE', credit)
    fam     = d.get('CNT_FAM_MEMBERS', 2)

    d['DEBT_TO_INCOME']    = annuity / (income + 1)
    d['CREDIT_TO_INCOME']  = credit  / (income + 1)
    d['ANNUITY_TO_CREDIT'] = annuity / (credit + 1)
    d['AGE_YEARS']         = abs(d['DAYS_BIRTH'])    / 365
    d['EMPLOYMENT_YEARS']  = abs(d['DAYS_EMPLOYED']) / 365
    d['EMPLOYMENT_TO_AGE'] = d['EMPLOYMENT_YEARS'] / (d['AGE_YEARS'] + 1)
    d['INCOME_PER_PERSON'] = income / (fam + 1)
    d['LOAN_TO_VALUE']     = credit / (goods + 1)
    d['IS_YOUNG_APPLICANT']= int(d['AGE_YEARS'] < 27)
    d['DOCS_SUBMITTED']    = 3

    ext = [d.get('EXT_SOURCE_1', np.nan),
           d.get('EXT_SOURCE_2', np.nan),
           d.get('EXT_SOURCE_3', np.nan)]
    valid = [e for e in ext if not np.isnan(e)]
    d['EXT_SOURCE_MEAN'] = np.mean(valid) if valid else 0.5
    d['EXT_SOURCE_MIN']  = np.min(valid)  if valid else 0.5
    return pd.DataFrame([d])


def risk_tier(prob):
    if prob < 0.15: return 'Low',    'low',    '●'
    if prob < 0.35: return 'Medium', 'medium', '●'
    return 'High', 'high', '●'


def get_flags(raw):
    flags = []
    dti  = raw['AMT_ANNUITY'] / (raw['AMT_INCOME_TOTAL'] + 1)
    age  = abs(raw['DAYS_BIRTH'])    / 365
    emp  = abs(raw['DAYS_EMPLOYED']) / 365
    cti  = raw['AMT_CREDIT'] / (raw['AMT_INCOME_TOTAL'] + 1)
    ext2 = raw.get('EXT_SOURCE_2', 0.5)

    if dti > 0.40:
        flags.append(('warn', f'Debt-to-income ratio is {dti*100:.0f}% — above the 40% safe threshold'))
    if ext2 < 0.40:
        flags.append(('warn', f'External credit score is low ({ext2:.2f}) — indicates prior credit issues'))
    if age < 27:
        flags.append(('warn', 'Applicant is under 27 — statistically higher default group'))
    if emp < 1:
        flags.append(('warn', 'Less than 1 year at current employer — income instability risk'))
    if cti > 5:
        flags.append(('warn', f'Loan is {cti:.1f}× annual income — high repayment burden'))
    if not flags:
        flags.append(('ok', 'No significant risk flags — applicant profile looks stable'))
    return flags


# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="top-bar-inner">
    <div class="brand-icon">🏦</div>
    <div>
      <p class="brand-name">Credit Risk Intelligence</p>
      <p class="brand-sub">Loan Default Prediction · XGBoost + SHAP</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

arts = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Financial Details**")
    income     = st.number_input('Annual Income (₹)', 50000, 10000000, 450000, step=10000)
    loan_amt   = st.number_input('Loan Amount (₹)',   10000, 5000000,  300000, step=10000)
    annuity_mo = st.number_input('Monthly EMI (₹)',   500,   150000,   14000,  step=500)
    goods_px   = st.number_input('Asset / Goods Price (₹)', 10000, 5000000, 250000, step=10000)

    st.markdown("**Applicant Profile**")
    age        = st.slider('Age', 18, 70, 35)
    emp_years  = st.slider('Years at Current Job', 0, 40, 4)
    family_sz  = st.slider('Family Members', 1, 10, 3)

    st.markdown("**Credit Scores (0 – 1)**")
    ext2 = st.slider('Primary Credit Score',   0.0, 1.0, 0.50, 0.01)
    ext3 = st.slider('Secondary Credit Score', 0.0, 1.0, 0.50, 0.01)
    ext1 = st.slider('Tertiary Credit Score',  0.0, 1.0, 0.50, 0.01)

    st.markdown("**Other**")
    own_car    = st.selectbox('Owns a Car',      ['No', 'Yes'])
    own_realty = st.selectbox('Owns Property',   ['Yes', 'No'])
    gender     = st.selectbox('Gender',          ['M', 'F'])

    st.divider()
    predict_btn = st.button('Run Assessment', type='primary', use_container_width=True)

# ── Landing state ─────────────────────────────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div class="stat-grid">
      <div class="stat-card">
        <p class="stat-label">Training Dataset</p>
        <p class="stat-value">307,511</p>
        <p class="stat-sub">loan applications</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Features Used</p>
        <p class="stat-value">36</p>
        <p class="stat-sub">financial signals</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Model AUC-ROC</p>
        <p class="stat-value">0.74</p>
        <p class="stat-sub">vs 0.75 baseline</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Business Value</p>
        <p class="stat-value">₹44.95 Cr</p>
        <p class="stat-sub">estimated (full dataset)</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-panel">
      <p style="font-size:16px; font-weight:600; color:#111827; margin:0 0 4px 0">
        How to use this tool
      </p>
      <p style="font-size:13px; color:#6B7280; margin:0">
        Fill in the applicant details in the sidebar and click Run Assessment.
      </p>
      <div class="info-grid">
        <div class="info-step">
          <p class="info-step-num">Step 1</p>
          <p class="info-step-title">Enter applicant details</p>
          <p class="info-step-desc">Income, loan amount, credit scores and profile in the sidebar</p>
        </div>
        <div class="info-step">
          <p class="info-step-num">Step 2</p>
          <p class="info-step-title">Model scores the loan</p>
          <p class="info-step-desc">XGBoost predicts default probability using 36 engineered features</p>
        </div>
        <div class="info-step">
          <p class="info-step-num">Step 3</p>
          <p class="info-step-title">SHAP explains the decision</p>
          <p class="info-step-desc">Waterfall chart shows which factors drove the risk score</p>
        </div>
        <div class="info-step">
          <p class="info-step-num">Step 4</p>
          <p class="info-step-title">Review risk flags</p>
          <p class="info-step-desc">Plain-English summary of key risk factors for the loan officer</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Prediction state ──────────────────────────────────────────────────────────
else:
    if arts['model'] is None:
        st.error('Model files not found. Make sure xgb_credit_risk.pkl is in the models/ folder.')
        st.stop()

    raw = {
        'AMT_INCOME_TOTAL'           : income,
        'AMT_CREDIT'                 : loan_amt,
        'AMT_ANNUITY'                : annuity_mo * 12,
        'AMT_GOODS_PRICE'            : goods_px,
        'DAYS_BIRTH'                 : -(age * 365),
        'DAYS_EMPLOYED'              : -(emp_years * 365),
        'DAYS_REGISTRATION'          : -1825,
        'DAYS_LAST_PHONE_CHANGE'     : -365,
        'DAYS_ID_PUBLISH'            : -1000,
        'CNT_FAM_MEMBERS'            : family_sz,
        'CNT_CHILDREN'               : max(0, family_sz - 2),
        'EXT_SOURCE_1'               : ext1,
        'EXT_SOURCE_2'               : ext2,
        'EXT_SOURCE_3'               : ext3,
        'FLAG_OWN_CAR'               : 1 if own_car  == 'Yes' else 0,
        'FLAG_OWN_REALTY'            : 1 if own_realty == 'Yes' else 0,
        'CODE_GENDER'                : 0 if gender == 'M' else 1,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'REGION_RATING_CLIENT'       : 2,
        'REG_CITY_NOT_WORK_CITY'     : 0,
        'NAME_EDUCATION_TYPE'        : 2,
        'NAME_INCOME_TYPE'           : 1,
        'NAME_CONTRACT_TYPE'         : 0,
        'OCCUPATION_TYPE'            : 3,
        'ORGANIZATION_TYPE'          : 5,
    }

    df_input = engineer_features(raw)
    features = arts['features'] if arts['features'] else list(df_input.columns)
    available = [f for f in features if f in df_input.columns]
    X_input = df_input[available].fillna(0)

    prob  = float(arts['model'].predict_proba(X_input)[0][1])
    label, tier, dot = risk_tier(prob)

    # ── Result summary card ───────────────────────────────────────────────────
    dti = (annuity_mo * 12) / (income + 1)
    cti = loan_amt / (income + 1)
    ext_mean = round((ext1 + ext2 + ext3) / 3, 2)

    st.markdown(f"""
    <div class="result-card border-{tier}">
      <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
        <div>
          <p class="result-title">Assessment Result</p>
          <p class="result-sub">Loan of ₹{loan_amt:,} · Applicant age {age}</p>
        </div>
        <span class="risk-badge badge-{tier}">{dot} {label} Risk — {prob*100:.1f}% default probability</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4 metric cards ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card">
        <p class="stat-label">Default Probability</p>
        <p class="stat-value">{prob*100:.1f}%</p>
        <p class="stat-sub">model confidence</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Debt-to-Income</p>
        <p class="stat-value">{dti*100:.1f}%</p>
        <p class="stat-sub">safe threshold: 40%</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Loan-to-Income</p>
        <p class="stat-value">{cti:.2f}×</p>
        <p class="stat-sub">annual income multiple</p>
      </div>
      <div class="stat-card">
        <p class="stat-label">Avg Credit Score</p>
        <p class="stat-value">{ext_mean}</p>
        <p class="stat-sub">0 = worst · 1 = best</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: SHAP | Flags + Summary ──────────────────────────────────
    col_shap, col_right = st.columns([3, 2])

    with col_shap:
        st.markdown('<p class="section-title">SHAP Explanation — what drove this score</p>',
                    unsafe_allow_html=True)
        if arts['explainer'] is not None:
            try:
                shap_vals = arts['explainer'](X_input)
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor('#FFFFFF')
                ax.set_facecolor('#FFFFFF')
                shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f'SHAP plot error: {e}')
        else:
            st.info('SHAP explainer not loaded. Save shap_explainer.pkl from the Week 3 notebook.')

    with col_right:
        st.markdown('<p class="section-title">Risk Flags</p>', unsafe_allow_html=True)
        for ftype, msg in get_flags(raw):
            icon = '⚠' if ftype == 'warn' else '✓'
            css  = 'flag-warn' if ftype == 'warn' else 'flag-ok'
            st.markdown(f'<div class="flag-item {css}"><span>{icon}</span><span>{msg}</span></div>',
                        unsafe_allow_html=True)

        st.markdown('<p class="section-title">Applicant Summary</p>', unsafe_allow_html=True)
        rows = [
            ('Annual Income',   f'₹{income:,}'),
            ('Loan Amount',     f'₹{loan_amt:,}'),
            ('Monthly EMI',     f'₹{annuity_mo:,}'),
            ('Age',             f'{age} years'),
            ('Employment',      f'{emp_years} years'),
            ('Family Size',     str(family_sz)),
            ('Owns Property',   own_realty),
            ('Owns Car',        own_car),
        ]
        for k, v in rows:
            st.markdown(f"""
            <div class="summary-row">
              <span class="summary-key">{k}</span>
              <span class="summary-value">{v}</span>
            </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px; padding-top:16px; border-top:1px solid #E5E7EB;
            font-size:12px; color:#9CA3AF; text-align:center;">
  Credit Risk Intelligence · Home Credit Default Risk Dataset ·
  <a href="https://github.com/SaiAshish1234/credit-risk-predictor"
     style="color:#1E40AF; text-decoration:none;">GitHub</a>
</div>
""", unsafe_allow_html=True)
