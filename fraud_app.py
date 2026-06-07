import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score

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


@st.cache_data
def load_and_train():
    DATA_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"Could not load dataset: {e}")
        st.stop()

    df = df.drop_duplicates()
    scaler = StandardScaler()
    df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
    df["Time_Scaled"]   = scaler.fit_transform(df[["Time"]])
    df.drop(columns=["Amount","Time"], inplace=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Using class_weight='balanced' instead of SMOTE
       models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight="balanced", n_jobs=-1),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        results[name] = {
            "F1"       : f1_score(y_test, y_pred),
            "AUC"      : roc_auc_score(y_test, y_proba),
            "Recall"   : recall_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
        }
        trained[name] = model

    best_name  = max(results, key=lambda k: results[k]["F1"])
    best_model = trained[best_name]
    feat_imp   = None
    if hasattr(best_model, "feature_importances_"):
        feat_imp = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False).head(15)

    return trained, results, best_name, feat_imp, df, y, X.columns.tolist()


with st.spinner("Training 3 models on 284,000+ transactions... (about 60 seconds)"):
    trained, results, best_name, feat_imp, df, y, feature_cols = load_and_train()

fraud = df[df["Class"] == 1]
legit = df[df["Class"] == 0]

st.markdown('<p class="main-title">🔍 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning · Random Forest + XGBoost · class_weight balanced · 284,807 Real Transactions</p>', unsafe_allow_html=True)
st.markdown("")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown('<div class="kpi-card"><div class="kpi-val">284,807</div><div class="kpi-lbl">Total Transactions</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown('<div class="kpi-card"><div class="kpi-val">492</div><div class="kpi-lbl">Fraud Cases</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown('<div class="kpi-card"><div class="kpi-val">0.17%</div><div class="kpi-lbl">Fraud Rate</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-val">{results[best_name]["F1"]*100:.1f}%</div><div class="kpi-lbl">Best Model F1 Score</div></div>', unsafe_allow_html=True)

st.markdown("")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔮 Predict Transaction", "📊 Data Insights", "🤖 Model Performance"])

# ── TAB 1 ──────────────────────────────────────────────────
with tab1:
    st.markdown('<p class="section">Enter Transaction Details</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        amount       = st.number_input("Transaction Amount ($)", 0.0, 10000.0, 150.0, step=10.0)
        model_choice = st.selectbox("Choose Model", list(trained.keys()),
                                    index=list(trained.keys()).index(best_name))
        v1  = st.slider("V1",  -5.0, 5.0, 0.0, 0.1)
        v2  = st.slider("V2",  -5.0, 5.0, 0.0, 0.1)
        v3  = st.slider("V3",  -5.0, 5.0, 0.0, 0.1)
        v4  = st.slider("V4",  -5.0, 5.0, 0.0, 0.1)
        v5  = st.slider("V5",  -5.0, 5.0, 0.0, 0.1)

    with col2:
        time_val = st.number_input("Time (seconds since first transaction)", 0, 200000, 50000)
        v14 = st.slider("V14 — strongest fraud signal", -10.0, 5.0, 0.0, 0.1)
        v17 = st.slider("V17", -5.0, 5.0, 0.0, 0.1)
        v12 = st.slider("V12", -5.0, 5.0, 0.0, 0.1)
        v10 = st.slider("V10", -5.0, 5.0, 0.0, 0.1)
        v11 = st.slider("V11", -5.0, 5.0, 0.0, 0.1)

    predict_btn = st.button("🔍 Analyse Transaction", type="primary", use_container_width=True)

    if predict_btn:
        amount_scaled = (amount - df["Amount_Scaled"].mean()) / df["Amount_Scaled"].std()
        time_scaled   = (time_val - df["Time_Scaled"].mean()) / df["Time_Scaled"].std()

        row = pd.DataFrame([df[feature_cols].median().values], columns=feature_cols)
        for col, val in {
            "Amount_Scaled": amount_scaled, "Time_Scaled": time_scaled,
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
            "V14": v14, "V17": v17, "V12": v12, "V10": v10, "V11": v11
        }.items():
            if col in row.columns:
                row[col] = val

        prob     = trained[model_choice].predict_proba(row)[0][1]
        is_fraud = prob >= 0.5

        st.markdown("")
        r1, r2 = st.columns([1, 2])
        with r1:
            if is_fraud:
                st.markdown(
                    f'<div class="fraud-card"><div style="font-size:2.5rem">🚨</div>'
                    f'<div class="fraud-val">FRAUD ALERT</div>'
                    f'<div class="fraud-val" style="font-size:1.3rem">{prob*100:.1f}% probability</div>'
                    f'<div class="kpi-lbl" style="margin-top:8px">Strong fraud signals detected. Recommend blocking.</div>'
                    f'</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="legit-card"><div style="font-size:2.5rem">✅</div>'
                    f'<div class="legit-val">LEGITIMATE</div>'
                    f'<div class="legit-val" style="font-size:1.3rem">{(1-prob)*100:.1f}% confidence</div>'
                    f'<div class="kpi-lbl" style="margin-top:8px">Transaction appears normal. No action required.</div>'
                    f'</div>', unsafe_allow_html=True)

        with r2:
            fig, ax = plt.subplots(figsize=(5, 1.4))
            fig.patch.set_alpha(0); ax.set_facecolor("none")
            color = "#DC2626" if is_fraud else "#059669"
            ax.barh([""], [prob],  color=color,    height=0.4, zorder=3)
            ax.barh([""], [1],     color="#E5E7EB", height=0.4, zorder=2)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="#9CA3AF", linestyle="--", linewidth=1)
            ax.set_xlabel("Fraud Probability", fontsize=9)
            ax.tick_params(left=False, labelleft=False)
            for spine in ax.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
            st.markdown(f"**Model:** {model_choice}  |  **Probability:** `{prob*100:.2f}%`  |  **Threshold:** 50%")

# ── TAB 2 ──────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section">Dataset Overview</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["Legit","Fraud"], [len(legit), len(fraud)],
               color=["#1A56A0","#DC2626"], width=0.4, alpha=0.85)
        ax.set_title("Transaction Count by Class", fontweight="bold")
        ax.set_ylabel("Count")
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(legit["Amount_Scaled"], bins=50, color="#1A56A0", alpha=0.7, label="Legit", density=True)
        ax.hist(fraud["Amount_Scaled"], bins=50, color="#DC2626", alpha=0.7, label="Fraud", density=True)
        ax.set_title("Amount Distribution", fontweight="bold")
        ax.set_xlabel("Scaled Amount"); ax.legend()
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    feature_cols_v = [c for c in df.columns if c.startswith("V")]
    diff = (fraud[feature_cols_v].mean() - legit[feature_cols_v].mean()).abs()\
            .sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(diff.index, diff.values, color="#1A56A0", alpha=0.85)
    ax.set_title("Top 10 Fraud Signal Features", fontweight="bold")
    ax.set_ylabel("Absolute Mean Difference")
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ── TAB 3 ──────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section">Model Comparison</p>', unsafe_allow_html=True)
    metrics_df = pd.DataFrame(results).T.round(4)
    st.dataframe(metrics_df.style.highlight_max(color="#D1FAE5", axis=0), use_container_width=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    model_names = list(results.keys())
    x = np.arange(len(model_names)); w = 0.25
    ax.bar(x - w, [results[m]["F1"]     for m in model_names], w, label="F1 Score", color="#1A56A0", alpha=0.85)
    ax.bar(x,     [results[m]["AUC"]    for m in model_names], w, label="ROC-AUC",  color="#27AE60", alpha=0.85)
    ax.bar(x + w, [results[m]["Recall"] for m in model_names], w, label="Recall",   color="#F39C12", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0.7, 1.05); ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
    ax.legend()
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    if feat_imp is not None:
        st.markdown('<p class="section">Feature Importance</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        feat_imp.sort_values().plot(kind="barh", ax=ax, color="#1A56A0", alpha=0.85)
        ax.set_title(f"Top 15 Features — {best_name}", fontweight="bold")
        ax.set_xlabel("Importance Score")
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

st.divider()
st.markdown(
    '<p style="text-align:center;color:#9CA3AF;font-size:0.8rem;">'
    'Built by <b>Dilawar Mahar</b> · Random Forest + XGBoost · '
    '<a href="https://github.com/Dilawar777" style="color:#1A56A0">GitHub</a></p>',
    unsafe_allow_html=True
)
