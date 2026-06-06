"""
app/streamlit_app.py
--------------------
Credit Risk Intelligence — Interactive Streamlit App
Run with: streamlit run app/streamlit_app.py
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib

from src.features import engineer_features, get_feature_names
from src.predict import predict_risk, risk_color

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Credit Risk Predictor',
    page_icon='🏦',
    layout='wide',
)

st.title('🏦 Credit Risk Intelligence')
st.markdown('*ML-powered loan default prediction with SHAP explainability*')
st.divider()

# ── Sidebar — Applicant Input ─────────────────────────────────────────────────
st.sidebar.header('📋 Applicant Details')

income       = st.sidebar.number_input('Annual Income (₹)', 100000, 10000000, 500000, step=50000)
loan_amt     = st.sidebar.number_input('Loan Amount (₹)',   50000,  5000000,  300000, step=50000)
annuity      = st.sidebar.number_input('Monthly Instalment (₹)', 1000, 200000, 15000, step=1000)
goods_price  = st.sidebar.number_input('Goods Price (₹)',   10000,  5000000,  250000, step=10000)
age          = st.sidebar.slider('Age (years)', 20, 70, 35)
employed_yrs = st.sidebar.slider('Employment Years', 0, 40, 5)
family_size  = st.sidebar.slider('Family Members', 1, 10, 3)
children     = st.sidebar.slider('No. of Children', 0, 5, 0)

st.sidebar.markdown('---')
st.sidebar.subheader('📊 External Credit Scores')
ext1 = st.sidebar.slider('Credit Score 1 (0–1)', 0.0, 1.0, 0.5, 0.01)
ext2 = st.sidebar.slider('Credit Score 2 (0–1)', 0.0, 1.0, 0.5, 0.01)
ext3 = st.sidebar.slider('Credit Score 3 (0–1)', 0.0, 1.0, 0.5, 0.01)

st.sidebar.markdown('---')
st.sidebar.subheader('🏠 Additional Info')
own_car    = st.sidebar.selectbox('Owns a Car?', ['Y', 'N'])
own_realty = st.sidebar.selectbox('Owns Real Estate?', ['Y', 'N'])
docs_count = st.sidebar.slider('Documents Submitted', 0, 10, 3)

predict_btn = st.sidebar.button('🔍 Predict Risk', type='primary', use_container_width=True)

# ── Build input DataFrame ─────────────────────────────────────────────────────
def build_input():
    return pd.DataFrame([{
        'AMT_INCOME_TOTAL'          : income,
        'AMT_CREDIT'                : loan_amt,
        'AMT_ANNUITY'               : annuity * 12,    # annualise
        'AMT_GOODS_PRICE'           : goods_price,
        'DAYS_BIRTH'                : -(age * 365),
        'DAYS_EMPLOYED'             : -(employed_yrs * 365),
        'DAYS_REGISTRATION'         : -1000,
        'DAYS_LAST_PHONE_CHANGE'    : -200,
        'CNT_FAM_MEMBERS'           : family_size,
        'CNT_CHILDREN'              : children,
        'EXT_SOURCE_1'              : ext1,
        'EXT_SOURCE_2'              : ext2,
        'EXT_SOURCE_3'              : ext3,
        'FLAG_OWN_CAR'              : 1 if own_car == 'Y' else 0,
        'FLAG_OWN_REALTY'           : 1 if own_realty == 'Y' else 0,
        'REGION_RATING_CLIENT_W_CITY': 2,
        'REG_CITY_NOT_WORK_CITY'    : 0,
        'DOCS_SUBMITTED'            : docs_count,
    }])

# ── Main content ──────────────────────────────────────────────────────────────
if not predict_btn:
    col1, col2, col3 = st.columns(3)
    col1.metric('Model', 'XGBoost + SMOTE')
    col2.metric('Training Data', '300K+ Loans')
    col3.metric('Explainability', 'SHAP')

    st.info('👈 Fill in the applicant details in the sidebar and click **Predict Risk** to get started.')

    st.markdown('''
    ### How it works
    1. **Input** applicant financial details
    2. **Model** scores default probability using XGBoost
    3. **SHAP** explains exactly which factors drove the decision
    4. **Risk tier** assigned: 🟢 Low / 🟡 Medium / 🔴 High
    ''')

else:
    try:
        raw_input = build_input()
        feat_input = engineer_features(raw_input)

        # Select only model features
        feature_cols = get_feature_names()
        available = [c for c in feature_cols if c in feat_input.columns]
        X_input = feat_input[available].fillna(0)

        result = predict_risk(X_input)

        prob  = result['probability']
        label = result['risk_label']
        color = risk_color(label)

        # ── Risk Score Card ───────────────────────────────────────────────────
        st.subheader('🎯 Risk Assessment')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Default Probability', f'{prob*100:.1f}%')
        c2.metric('Risk Level', label)
        c3.metric('DTI Ratio', f"{(annuity * 12 / income * 100):.1f}%")
        c4.metric('Credit-to-Income', f"{loan_amt / income:.2f}x")

        st.markdown(f'''
        <div style="background:{color}20; border-left: 4px solid {color};
                    padding: 14px 18px; border-radius: 8px; margin: 1rem 0;">
            <strong style="color:{color}; font-size:18px;">{label} Risk</strong><br>
            <span style="color:#555">Default probability: <strong>{prob*100:.1f}%</strong></span>
        </div>
        ''', unsafe_allow_html=True)

        # ── SHAP Waterfall Plot ───────────────────────────────────────────────
        st.subheader('🔍 Why this decision? (SHAP Explanation)')
        st.caption('Factors pushing toward default (red ↑) vs. away from default (blue ↓)')

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(result['shap_values'][0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Human-Readable Explanation ────────────────────────────────────────
        st.subheader('📝 Plain-English Summary')
        dti = annuity * 12 / income
        reasons = []
        if dti > 0.4:
            reasons.append(f"⚠️ High debt-to-income ratio ({dti*100:.0f}%) — repayments consume a large share of income")
        if ext2 < 0.4:
            reasons.append(f"⚠️ Low external credit score 2 ({ext2:.2f}) — indicates past credit issues")
        if age < 27:
            reasons.append("⚠️ Young applicant (< 27) — statistically higher default group")
        if employed_yrs < 1:
            reasons.append("⚠️ Very short employment history — income instability risk")
        if loan_amt / income > 5:
            reasons.append(f"⚠️ Loan is {loan_amt/income:.1f}x annual income — high burden")

        if not reasons:
            st.success('✅ No major risk flags detected. Applicant profile looks stable.')
        else:
            for r in reasons:
                st.warning(r)

    except FileNotFoundError:
        st.error('⚠️ Model not found. Please complete Week 2 training first: `python src/train.py`')
    except Exception as e:
        st.error(f'Error: {e}')
        st.info('Make sure you have trained the model (Week 2) before using the app.')
