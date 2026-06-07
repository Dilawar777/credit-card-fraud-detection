import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1A56A0; }
    .sub-title   { font-size:1rem; color:#6B7280; margin-top:-10px; }
    .kpi-card    { background:#F0F7FF; border-left:4px solid #1A56A0;
                   padding:14px 18px; border-radius:8px; margin:4px 0; }
    .kpi-val     { font-size:1.8rem; font-weight:700; color:#1A56A0; }
    .kpi-lbl     { font-size:0.82rem; color:#6B7280; }
    .fraud-card  { background:#FEE2E2; border-left:4px solid #DC2626;
                   padding:14px 18px; border-radius:8px; }
    .legit-card  { background:#D1FAE5; border-left:4px solid #059669;
                   padding:14px 18px; border-radius:8px; }
    .fraud-val   { font-size:2rem; font-weight:700; color:#DC2626; }
    .legit-val   { font-size:2rem; font-weight:700; color:#059669; }
    .section     { font-size:1.1rem; font-weight:600; color:#1A1A2E;
                   border-bottom:2px solid #1A56A0;
                   padding-bottom:4px; margin:20px 0 12px; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model  = joblib.load("fraud_model.pkl")
    scaler = joblib.load("fraud_scaler.pkl")
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


model, scaler, feature_cols = load_model()

# ── Header ─────────────────────────────────────────────────
st.markdown('<p class="main-title">🔍 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Random Forest · class_weight balanced · 284,807 Real Transactions</p>', unsafe_allow_html=True)
st.markdown("")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="kpi-card"><div class="kpi-val">284,807</div><div class="kpi-lbl">Total Transactions</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card"><div class="kpi-val">492</div><div class="kpi-lbl">Fraud Cases</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi-card"><div class="kpi-val">0.17%</div><div class="kpi-lbl">Fraud Rate</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown('<div class="kpi-card"><div class="kpi-val">~85%</div><div class="kpi-lbl">Model F1 Score</div></div>', unsafe_allow_html=True)

st.markdown("")
st.divider()

tab1, tab2 = st.tabs(["🔮 Predict Transaction", "📊 About This Model"])

# ── TAB 1 — Predict ────────────────────────────────────────
with tab1:
    st.markdown('<p class="section">Enter Transaction Details</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        amount   = st.number_input("Transaction Amount ($)", 0.0, 10000.0, 150.0, step=10.0)
        time_val = st.number_input("Time (seconds since first transaction)", 0, 200000, 50000)
        v1  = st.slider("V1",  -5.0, 5.0, 0.0, 0.1)
        v2  = st.slider("V2",  -5.0, 5.0, 0.0, 0.1)
        v3  = st.slider("V3",  -5.0, 5.0, 0.0, 0.1)
        v4  = st.slider("V4",  -5.0, 5.0, 0.0, 0.1)
        v5  = st.slider("V5",  -5.0, 5.0, 0.0, 0.1)

    with col2:
        v14 = st.slider("V14 — strongest fraud signal", -10.0, 5.0, 0.0, 0.1)
        v17 = st.slider("V17", -5.0, 5.0, 0.0, 0.1)
        v12 = st.slider("V12", -5.0, 5.0, 0.0, 0.1)
        v10 = st.slider("V10", -5.0, 5.0, 0.0, 0.1)
        v11 = st.slider("V11", -5.0, 5.0, 0.0, 0.1)
        v7  = st.slider("V7",  -5.0, 5.0, 0.0, 0.1)
        v6  = st.slider("V6",  -5.0, 5.0, 0.0, 0.1)

    predict_btn = st.button("🔍 Analyse Transaction", type="primary", use_container_width=True)

    if predict_btn:
        # Build input row with median values as base
        row = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)

        # Scale amount and time
        amount_scaled = (amount - 88.0) / 250.0
        time_scaled   = (time_val - 94813.0) / 47488.0

        overrides = {
            "Amount_Scaled": amount_scaled,
            "Time_Scaled"  : time_scaled,
            "V1": v1, "V2": v2, "V3": v3, "V4": v4,
            "V5": v5, "V6": v6, "V7": v7,
            "V10": v10, "V11": v11, "V12": v12,
            "V14": v14, "V17": v17,
        }
        for col, val in overrides.items():
            if col in row.columns:
                row[col] = val

        prob     = model.predict_proba(row)[0][1]
        is_fraud = prob >= 0.5

        st.markdown("")
        r1, r2 = st.columns([1, 2])

        with r1:
            if is_fraud:
                st.markdown(
                    f'<div class="fraud-card">'
                    f'<div style="font-size:2.5rem">🚨</div>'
                    f'<div class="fraud-val">FRAUD ALERT</div>'
                    f'<div class="fraud-val" style="font-size:1.3rem">{prob*100:.1f}% probability</div>'
                    f'<div class="kpi-lbl" style="margin-top:8px">Strong fraud signals detected. Recommend blocking this transaction immediately.</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="legit-card">'
                    f'<div style="font-size:2.5rem">✅</div>'
                    f'<div class="legit-val">LEGITIMATE</div>'
                    f'<div class="legit-val" style="font-size:1.3rem">{(1-prob)*100:.1f}% confidence</div>'
                    f'<div class="kpi-lbl" style="margin-top:8px">Transaction appears normal. No action required.</div>'
                    f'</div>', unsafe_allow_html=True)

        with r2:
            fig, ax = plt.subplots(figsize=(5, 1.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            color = "#DC2626" if is_fraud else "#059669"
            ax.barh([""], [prob],  color=color,    height=0.4, zorder=3)
            ax.barh([""], [1],     color="#E5E7EB", height=0.4, zorder=2)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
            ax.set_xlabel("Fraud Probability", fontsize=9)
            ax.tick_params(left=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown(f"**Fraud probability:** `{prob*100:.2f}%`")
            st.markdown(f"**Decision threshold:** 50%")
            st.markdown(f"**Verdict:** {'🚨 Block transaction' if is_fraud else '✅ Allow transaction'}")

# ── TAB 2 — About ──────────────────────────────────────────
with tab2:
    st.markdown('<p class="section">How This Model Works</p>', unsafe_allow_html=True)

    st.markdown("""
    **Dataset:** European credit card transactions — September 2013
    - 284,807 total transactions over 2 days
    - Only 492 fraud cases (0.17%) — severely imbalanced

    **Model:** Random Forest Classifier
    - 100 decision trees trained on 80% of data
    - `class_weight='balanced'` used to handle severe class imbalance
    - Evaluated on 20% held-out test data

    **Why not just use Accuracy?**
    - A model that predicts "Legit" for everything gets 99.83% accuracy
    - But it catches zero fraud — useless in production
    - We use **Precision, Recall and F1 Score** instead
    - **Recall** = of all real fraud, how many did we catch?
    - **Precision** = when we flag fraud, how often are we right?
    - **F1** = balance between both

    **Features V1–V28** are PCA-transformed for privacy.
    **V14** is the strongest individual fraud signal in this dataset.
    """)

    st.markdown('<p class="section">Key Insight</p>', unsafe_allow_html=True)

    # Simple bar chart of feature importance
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 5))
        feat_imp.sort_values().plot(kind="barh", ax=ax, color="#1A56A0", alpha=0.85)
        ax.set_title("Top 15 Most Important Features", fontweight="bold", fontsize=13)
        ax.set_xlabel("Importance Score")
        for s in ["top","right"]:
            ax.spines[s].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.markdown(
    '<p style="text-align:center;color:#9CA3AF;font-size:0.8rem;">'
    'Built by <b>Dilawar Mahar</b> · Random Forest · Sukkur IBA University · '
    '<a href="https://github.com/Dilawar777" style="color:#1A56A0">GitHub</a></p>',
    unsafe_allow_html=True
)
