from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score, recall_score, matthews_corrcoef, roc_auc_score
 
# --- 1. PATH CONFIGURATION ---
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mtask_3.1"
DATA_CSV = DATA_DIR / "yeast.csv"
 
st.set_page_config(page_title="Ensemble Learning | ML Studio", layout="wide")
 
# --- 2. DARK THEME ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(165deg, #050510 0%, #0a0f1e 35%, #0f172a 70%, #0c1222 100%);
        color: #e2e8f0;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(168, 85, 247, 0.25);
    }
    div[data-testid="column"] [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #c084fc;
        border-radius: 0.75rem;
        padding: 0.8rem;
    }
    .stTable, [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #E0E0E0 !important;
        background-color: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(168, 85, 247, 0.2) !important;
    }
    h1, h2, h3 { color: #f8fafc !important; }
    </style>
""", unsafe_allow_html=True)
 
# --- 3. DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_CSV.is_file():
        return None
    return pd.read_csv(DATA_CSV)
 
st.title("Ensemble Learning")
 
st.info("""
I implemented an Ensemble Learning framework to classify protein localization sites by combining multiple predictive models.
 
By utilizing Random Forest (Bagging) and AdaBoost (Boosting), the system leverages a "committee" of learners to improve overall classification accuracy and robustness. This approach addresses the limitations of individual decision trees by reducing variance and iteratively correcting classification errors, leading to a more reliable analysis of the protein dataset.
""")
 
df = load_data()
if df is None:
    st.error(f"Could not find yeast.csv in {DATA_DIR}")
    st.stop()
 
# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Algorithm Selection")
    model_type   = st.selectbox("Choose Ensemble Method", ["Random Forest", "AdaBoost"])
    st.header("Hyperparameters")
    n_estimators = st.slider("Number of Estimators", 10, 200, 100)
    if model_type == "Random Forest":
        max_depth    = st.slider("Max Depth", 1, 30, 10)
    else:
        learning_rate = st.slider("Learning Rate", 0.01, 2.0, 1.0, 0.05)
    target_col        = "name"
    feature_candidates = [c for c in df.columns if c != target_col]
    selected_features  = st.multiselect("Select Features", feature_candidates, default=feature_candidates)
 
if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()
 
# --- 5. PREPARE DATA ---
X = df[selected_features]
y = df[target_col]
le = LabelEncoder()
y_encoded = le.fit_transform(y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
scoring = {
    'f1_macro':     make_scorer(f1_score,        average='macro'),
    'recall_macro': make_scorer(recall_score,     average='macro'),
    'mcc':          make_scorer(matthews_corrcoef)
}
 
# --- 6. SELECTED MODEL ---
if model_type == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
else:
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
 
with st.spinner(f"Running {model_type}..."):
    cv_results = cross_validate(model, X, y_encoded, cv=skf, scoring=scoring)
    model.fit(X, y_encoded)
    y_probs = model.predict_proba(X)
    auc_val = roc_auc_score(y_encoded, y_probs, multi_class='ovr')
 
# --- 7. METRICS ---
st.subheader(f"{model_type} Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("F1 Score (Macro)",  f"{cv_results['test_f1_macro'].mean():.4f}")
m2.metric("Recall (Macro)",    f"{cv_results['test_recall_macro'].mean():.4f}")
m3.metric("MCC",               f"{cv_results['test_mcc'].mean():.4f}")
m4.metric("AUC (OvR)",         f"{auc_val:.4f}")
 
st.divider()
 
# --- 8. FEATURE IMPORTANCE ---
st.markdown(f"**{model_type} Feature Importance**")
importances = pd.DataFrame({
    'Feature':    selected_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)
 
fig_imp = px.bar(
    importances, x='Importance', y='Feature', orientation='h',
    template="plotly_dark", color_discrete_sequence=['#a855f7']
)
fig_imp.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=400
)
st.plotly_chart(fig_imp, use_container_width=True)
 
# --- 9. LIVE COMPARISON TABLE ---
with st.expander("Model Comparison (computed live on your features)"):
    st.caption("All scores computed fresh using 5-fold cross-validation on the current feature selection.")
 
    with st.spinner("Computing baseline and ensemble scores for comparison..."):
        # Baseline: single Decision Tree
        dt_baseline = DecisionTreeClassifier(criterion="entropy", random_state=42)
        dt_cv = cross_validate(dt_baseline, X, y_encoded, cv=skf, scoring=scoring)
        dt_baseline.fit(X, y_encoded)
        dt_auc = roc_auc_score(y_encoded, dt_baseline.predict_proba(X), multi_class='ovr')
 
        # Random Forest (fixed params for fair comparison)
        rf_compare = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_cv = cross_validate(rf_compare, X, y_encoded, cv=skf, scoring=scoring)
        rf_compare.fit(X, y_encoded)
        rf_auc = roc_auc_score(y_encoded, rf_compare.predict_proba(X), multi_class='ovr')
 
        # AdaBoost (fixed params for fair comparison)
        ada_compare = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada_cv = cross_validate(ada_compare, X, y_encoded, cv=skf, scoring=scoring)
        ada_compare.fit(X, y_encoded)
        ada_auc = roc_auc_score(y_encoded, ada_compare.predict_proba(X), multi_class='ovr')
 
    comparison_df = pd.DataFrame({
        "Model":    ["Decision Tree (Baseline)", "Random Forest", "AdaBoost"],
        "F1 Score": [
            f"{dt_cv['test_f1_macro'].mean():.4f}",
            f"{rf_cv['test_f1_macro'].mean():.4f}",
            f"{ada_cv['test_f1_macro'].mean():.4f}",
        ],
        "MCC": [
            f"{dt_cv['test_mcc'].mean():.4f}",
            f"{rf_cv['test_mcc'].mean():.4f}",
            f"{ada_cv['test_mcc'].mean():.4f}",
        ],
        "AUC": [
            f"{dt_auc:.4f}",
            f"{rf_auc:.4f}",
            f"{ada_auc:.4f}",
        ],
    })
    st.table(comparison_df)
 
# --- 10. SCIENCE EXPANDER ---
with st.expander("The Science Behind Ensembles"):
    if model_type == "Random Forest":
        st.markdown(r"""
**Random Forest** uses **Bagging** (Bootstrap Aggregating) — it builds $B$ independent trees, each trained on a random bootstrap sample of the data with a random subset of features. The final prediction averages all trees:
 
$$ \hat{f} = \frac{1}{B} \sum_{b=1}^{B} f_b(x) $$
 
This dramatically **reduces variance** compared to a single tree without increasing bias.
        """)
    else:
        st.markdown(r"""
**AdaBoost** uses **Boosting** — it builds trees sequentially where each new tree focuses more on the samples the previous tree got wrong. Misclassified samples are given higher weights so the next learner pays more attention to them.
 
The final prediction is a weighted vote of all weak learners:
 
$$ F(x) = \sum_{t=1}^{T} \alpha_t f_t(x) $$
 
Where $\alpha_t$ is the weight of learner $t$ based on its accuracy. This **reduces bias** iteratively.
        """)
 