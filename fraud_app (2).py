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
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size:2.2rem; font-weight:700; color:#1A56A0; }
    .sub-title  { font-size:1rem; color:#6B7280; margin-top:-10px; }
    .kpi-card   { background:#F0F7FF; border-left:4px solid #1A56A0;
                  padding:14px 18px; border-radius:8px; margin:4px 0; }
    .kpi-val    { font-size:1.8rem; font-weight:700; color:#1A56A0; }
    .kpi-lbl    { font-size:0.82rem; color:#6B7280; }
    .fraud-card { background:#FEE2E2; border-left:4px solid #DC2626;
                  padding:14px 18px; border-radius:8px; }
    .legit-card { background:#D1FAE5; border-left:4px solid #059669;
                  padding:14px 18px; border-radius:8px; }
    .fraud-val  { font-size:2rem; font-weight:700; color:#DC2626; }
    .legit-val  { font-size:2rem; font-weight:700; color:#059669; }
    .section    { font-size:1.1rem; font-weight:600; color:#1A1A2E;
                  border-bottom:2px solid #1A56A0;
                  padding-bottom:4px; margin:20px 0 12px; }
    footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model        = joblib.load("fraud_model.pkl")
    scaler       = joblib.load("fraud_scaler.pkl")
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


model, scaler, feature_cols = load_model()

# ── Header ──────────────────────────────────────────────────
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

tab1, tab2, tab3 = st.tabs(["🔮 Predict Transaction", "🏆 Model Performance", "📖 About"])

# ── TAB 1 — PREDICT ─────────────────────────────────────────
with tab1:
    st.markdown('<p class="section">Enter Transaction Details</p>', unsafe_allow_html=True)
    st.info("💡 Tip: Move V14 to a very negative value (e.g. -5) to simulate a fraud transaction.")

    col1, col2 = st.columns(2)
    with col1:
        amount   = st.number_input("💵 Transaction Amount ($)", 0.0, 10000.0, 150.0, step=10.0)
        time_val = st.number_input("⏱ Time (seconds since first transaction)", 0, 200000, 50000)
        v1  = st.slider("V1",  -5.0, 5.0, 0.0, 0.1)
        v2  = st.slider("V2",  -5.0, 5.0, 0.0, 0.1)
        v3  = st.slider("V3",  -5.0, 5.0, 0.0, 0.1)
        v4  = st.slider("V4",  -5.0, 5.0, 0.0, 0.1)
        v5  = st.slider("V5",  -5.0, 5.0, 0.0, 0.1)

    with col2:
        v14 = st.slider("V14 🔴 Strongest fraud signal", -10.0, 5.0, 0.0, 0.1)
        v17 = st.slider("V17", -5.0, 5.0, 0.0, 0.1)
        v12 = st.slider("V12", -5.0, 5.0, 0.0, 0.1)
        v10 = st.slider("V10", -5.0, 5.0, 0.0, 0.1)
        v11 = st.slider("V11", -5.0, 5.0, 0.0, 0.1)
        v7  = st.slider("V7",  -5.0, 5.0, 0.0, 0.1)
        v6  = st.slider("V6",  -5.0, 5.0, 0.0, 0.1)

    predict_btn = st.button("🔍 Analyse Transaction", type="primary", use_container_width=True)

    if predict_btn:
        amount_scaled = (amount - 88.0) / 250.0
        time_scaled   = (time_val - 94813.0) / 47488.0

        row = pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols)
        for col, val in {
            "Amount_Scaled": amount_scaled, "Time_Scaled": time_scaled,
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
            "V6": v6, "V7": v7, "V10": v10, "V11": v11,
            "V12": v12, "V14": v14, "V17": v17,
        }.items():
            if col in row.columns:
                row[col] = val

        prob     = model.predict_proba(row)[0][1]
        is_fraud = prob >= 0.5

        st.markdown("")
        r1, r2 = st.columns([1, 2])
        with r1:
            if is_fraud:
                st.markdown(
                    f'<div class="fraud-card"><div style="font-size:2.5rem">🚨</div>'
                    f'<div class="fraud-val">FRAUD ALERT</div>'
                    f'<div class="fraud-val" style="font-size:1.3rem">{prob*100:.1f}% probability</div>'
                    f'<div class="kpi-lbl" style="margin-top:8px">Strong fraud signals. Recommend blocking immediately.</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="legit-card"><div style="font-size:2.5rem">✅</div>'
                    f'<div class="legit-val">LEGITIMATE</div>'
                    f'<div class="legit-val" style="font-size:1.3rem">{(1-prob)*100:.1f}% confidence</div>'
                    f'<div class="kpi-lbl" style="margin-top:8px">Transaction appears normal. No action required.</div>'
                    f'</div>', unsafe_allow_html=True)

        with r2:
            fig, ax = plt.subplots(figsize=(5, 1.5))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            color = "#DC2626" if is_fraud else "#059669"
            ax.barh([""], [prob],  color=color,    height=0.4, zorder=3)
            ax.barh([""], [1],     color="#E5E7EB", height=0.4, zorder=2)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
            ax.set_xlabel("Fraud Probability", fontsize=9)
            ax.tick_params(left=False, labelleft=False)
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()
            st.markdown(f"**Probability:** `{prob*100:.2f}%`  |  **Threshold:** 50%")
            if prob < 0.2:
                st.success("🟢 Low Risk")
            elif prob < 0.5:
                st.warning("🟡 Medium Risk")
            else:
                st.error("🔴 High Risk — block and investigate")

# ── TAB 2 — MODEL PERFORMANCE ───────────────────────────────
with tab2:
    st.markdown('<p class="section">Model Metrics</p>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="kpi-card"><div class="kpi-val">~85%</div><div class="kpi-lbl">F1 Score</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="kpi-card"><div class="kpi-val">~97%</div><div class="kpi-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="kpi-card"><div class="kpi-val">~83%</div><div class="kpi-lbl">Recall</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="kpi-card"><div class="kpi-val">~88%</div><div class="kpi-lbl">Precision</div></div>', unsafe_allow_html=True)

    st.markdown('<p class="section">Feature Importance</p>', unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#DC2626" if i < 3 else "#1A56A0" for i in range(len(feat_imp))]
        feat_imp.sort_values().plot(kind="barh", ax=ax, color=colors[::-1], alpha=0.85)
        ax.set_title("Top 15 Most Important Features", fontweight="bold", fontsize=13)
        ax.set_xlabel("Importance Score")
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<p class="section">Why Not Just Use Accuracy?</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    naive   = [99.83, 0, 0, 0, 50]
    ours    = [99.50, 88, 83, 85, 97]
    x = np.arange(len(metrics)); w = 0.35
    ax.bar(x - w/2, naive, w, label="Naive (predict all legit)", color="#9CA3AF", alpha=0.85)
    ax.bar(x + w/2, ours,  w, label="Our Random Forest",         color="#1A56A0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 115); ax.set_ylabel("Score (%)")
    ax.set_title("Our Model vs Naive Baseline", fontweight="bold")
    ax.legend()
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 3 — ABOUT ───────────────────────────────────────────
with tab3:
    st.markdown('<p class="section">About This Project</p>', unsafe_allow_html=True)
    st.markdown("""
    **Dataset:** European credit card transactions — September 2013
    - 284,807 total transactions · 492 fraud cases (0.17%) · severely imbalanced

    **Model:** Random Forest Classifier
    - 100 decision trees · `class_weight='balanced'` · trained on 80% of data

    **Tech Stack:** Python · Scikit-learn · Random Forest · Joblib · Streamlit · Pandas · Matplotlib

    **Project Steps:**
    1. Data Loading & first look
    2. Data Cleaning — removed duplicates, scaled Amount & Time
    3. EDA — visualised fraud patterns and top signal features
    4. Preprocessing — train/test split, handled 577:1 class imbalance
    5. Model Building — trained & compared 3 ML models
    6. Business Dashboard — charts for stakeholders
    7. Deployment — live on Streamlit Cloud
    """)

st.divider()
st.markdown(
    '<p style="text-align:center;color:#9CA3AF;font-size:0.8rem;">'
    'Built by <b>Dilawar Mahar</b> · CS Student · Sukkur IBA University · '
    '<a href="https://github.com/Dilawar777" style="color:#1A56A0">GitHub</a></p>',
    unsafe_allow_html=True
)
