# Credit Card Default Taiwan
# Machine Learning Classification Model
# Random Forest Classifier


# Modified interface layout and color scheme using Claude - 06/16/26


import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CC Default Risk Analyzer",
    page_icon="💳",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Google Font import ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ---- Root palette ---- */
:root {
    --navy:      #1E3A5F;
    --blue:      #2980B9;
    --blue-lt:   #D6EAF8;
    --green:     #1E8449;
    --green-lt:  #D5F5E3;
    --red:       #C0392B;
    --red-lt:    #FADBD8;
    --card:      #FFFFFF;
    --page:      #F0F4F8;
    --border:    #D5DDE8;
    --text:      #1E3A5F;
    --sub:       #5D7A99;
}

/* ---- Global reset ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* ---- Page background ---- */
.stApp {
    background: var(--page);
}

/* ---- Hero banner ---- */
.hero {
    background: linear-gradient(135deg, #1E3A5F 0%, #2471A3 100%);
    border-radius: 14px;
    padding: 36px 40px 30px;
    margin-bottom: 28px;
    color: #fff;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
    color: #fff;
}
.hero p {
    font-size: 0.95rem;
    opacity: 0.82;
    margin: 0;
    line-height: 1.6;
}
.hero-pills {
    display: flex;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.pill {
    background: rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.3px;
}

/* ---- Section label ---- */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--sub);
    margin-bottom: 12px;
    margin-top: 24px;
}

/* ---- Input card ---- */
.input-card {
    background: var(--card);
    border-radius: 12px;
    padding: 24px 28px;
    border: 1px solid var(--border);
    margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(30,58,95,0.06);
}
.input-card h4 {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--sub);
    margin: 0 0 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}

/* ---- Verdict card ---- */
.verdict-default {
    background: linear-gradient(135deg, #922B21 0%, #C0392B 100%);
    border-radius: 14px;
    padding: 32px 36px;
    text-align: center;
    color: #fff;
    margin: 8px 0;
}
.verdict-safe {
    background: linear-gradient(135deg, #1A6635 0%, #27AE60 100%);
    border-radius: 14px;
    padding: 32px 36px;
    text-align: center;
    color: #fff;
    margin: 8px 0;
}
.verdict-icon { font-size: 2.6rem; margin-bottom: 8px; }
.verdict-label {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 0;
}
.verdict-sub {
    font-size: 0.88rem;
    opacity: 0.85;
    margin-top: 6px;
}

/* ---- Prob bar card ---- */
.prob-card {
    background: var(--card);
    border-radius: 12px;
    padding: 22px 24px;
    border: 1px solid var(--border);
    margin: 8px 0;
    box-shadow: 0 1px 4px rgba(30,58,95,0.06);
}
.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.prob-row span { font-size: 0.85rem; font-weight: 600; }
.bar-track {
    background: #E8EFF7;
    border-radius: 6px;
    height: 10px;
    margin-bottom: 14px;
    overflow: hidden;
}
.bar-fill-green {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #27AE60, #1E8449);
}
.bar-fill-red {
    height: 10px;
    border-radius: 6px;
    background: linear-gradient(90deg, #E74C3C, #C0392B);
}

/* ---- Input table card ---- */
.params-card {
    background: var(--card);
    border-radius: 12px;
    padding: 22px 24px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 4px rgba(30,58,95,0.06);
}

/* ---- Streamlit element overrides ---- */
div[data-testid="stNumberInput"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
}
div[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
    border: 1.5px solid var(--border) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(41,128,185,0.15) !important;
}
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
button[kind="primary"], .stButton > button {
    background: var(--blue) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>💳 Credit Card Default Risk Analyzer</h1>
  <p>Enter a customer's credit profile below to generate an instant default prediction
  using a trained Random Forest classifier.</p>
  <div class="hero-pills">
    <span class="pill">📂 UCI ML Repository</span>
    <span class="pill">🌲 Random Forest Classifier</span>
    <span class="pill">🇹🇼 Taiwan Credit Data</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input Section ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Profile — Input Parameters</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown('<div class="input-card"><h4>💰 Account Overview</h4>', unsafe_allow_html=True)
    LIMIT_BAL = st.number_input(
        "Credit Card Limit (NT$)",
        value=None, min_value=0, placeholder="Enter Credit Card Limit", step=500
    )
    PAY_0 = st.selectbox(
        "Current Repayment Status",
        (-2,-1,1,2,3,4,5,6,7,8,9), 
        help=(
            "Repayment Status Codes\n\n"
            "-2 = No credit use\n"
            "-1 = Current (paid on time)\n"
            "1 = Past due 1 month\n"
            "2 = Past due 2 months\n"
            "3 = Past due 3 months\n"
            "4 = Past due 4 months\n"
            "5 = Past due 5 months\n"
            "6 = Past due 6 months\n"
            "7 = Past due 7 months\n"
            "8 = Past due 8 months\n"
            "9 = Past due 9+ months"
        )
    )
    BILL_AMT1 = st.number_input(
        "Current Statement Balance (NT$)",
        value=None, placeholder="Enter Current Statement Balance", step=500
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card"><h4>📅 Payment History</h4>', unsafe_allow_html=True)
    PAY_AMT1 = st.number_input(
        "Previous Month Payment (NT$)",
        value=None, min_value=0, placeholder="Enter Previous Month Payment", step=50
    )
    PAY_AMT2 = st.number_input(
        "Payment — 2 Months Prior (NT$)",
        value=None, min_value=0, placeholder="Enter 2 Months Prior Payment", step=50
    )
    PAY_AMT3 = st.number_input(
        "Payment — 3 Months Prior (NT$)",
        value=None, min_value=0, placeholder="Enter 3 Months Prior Payment", step=50
    )
    PAY_AMT4 = st.number_input(
        "Payment — 4 Months Prior (NT$)",
        value=None, min_value=0, placeholder="Enter 4 Months Prior Payment", step=50
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Build feature dataframe ───────────────────────────────────────────────────
df_input = pd.DataFrame({
    'LIMIT_BAL': [LIMIT_BAL],
    'PAY_0':     [PAY_0],
    'BILL_AMT1': [BILL_AMT1],
    'PAY_AMT1':  [PAY_AMT1],
    'PAY_AMT2':  [PAY_AMT2],
    'PAY_AMT3':  [PAY_AMT3],
    'PAY_AMT4':  [PAY_AMT4],
})

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('cc_rfc_model_jl.sav.bz2')

load_joblib_model = load_model()

# ── Predict ───────────────────────────────────────────────────────────────────
prediction   = load_joblib_model.predict(df_input)
predict_proba = load_joblib_model.predict_proba(df_input)

prob_no_default = predict_proba[0][0]
prob_default    = predict_proba[0][1]
is_default      = prediction[0] == 1

# ── Results Section ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Prediction Results</div>', unsafe_allow_html=True)

res_col, detail_col = st.columns([1, 1], gap="medium")

with res_col:
    if is_default:
        st.markdown(f"""
        <div class="verdict-default">
          <div class="verdict-icon">⚠️</div>
          <div class="verdict-label">DEFAULT RISK</div>
          <div class="verdict-sub">Model predicts this customer is likely to default</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-safe">
          <div class="verdict-icon">✅</div>
          <div class="verdict-label">NO DEFAULT</div>
          <div class="verdict-sub">Model predicts this customer will meet obligations</div>
        </div>
        """, unsafe_allow_html=True)

with detail_col:
    pct_safe    = round(prob_no_default * 100, 1)
    pct_default = round(prob_default * 100, 1)
    bar_safe    = round(prob_no_default * 100, 1)
    bar_def     = round(prob_default * 100, 1)

    st.markdown(f"""
    <div class="prob-card">
      <p style="font-size:0.78rem;font-weight:700;letter-spacing:1px;
                text-transform:uppercase;color:#5D7A99;margin:0 0 16px;">
        Probability Breakdown
      </p>

      <div class="prob-row">
        <span>✅ No Default</span>
        <span style="color:#1E8449;">{pct_safe}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill-green" style="width:{bar_safe}%"></div>
      </div>

      <div class="prob-row">
        <span>⚠️ Default</span>
        <span style="color:#C0392B;">{pct_default}%</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill-red" style="width:{bar_def}%"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Input Summary ─────────────────────────────────────────────────────────────
with st.expander("📋 View Input Parameter Summary", expanded=False):
    display_df = df_input.T.rename(columns={0: "Value"})
    display_df.index = [
        "Credit Limit (NT$)", "Payment Status", "Statement Balance (NT$)",
        "Payment −1 mo (NT$)", "Payment −2 mo (NT$)",
        "Payment −3 mo (NT$)", "Payment −4 mo (NT$)"
    ]
    st.dataframe(display_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8CA3BE;font-size:0.78rem;'>"
    "Data: UCI ML Repository · Default of Credit Card Clients Dataset · "
    "<a href='https://archive.uci.edu/dataset/350/default+of+credit+card+clients' "
    "style='color:#2980B9;'>Source</a></p>",
    unsafe_allow_html=True
)
