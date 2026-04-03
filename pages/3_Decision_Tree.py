from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score, recall_score, matthews_corrcoef, roc_auc_score

# 1. Path Configuration
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mtask_3.0"
DATA_CSV = DATA_DIR / "yeast.csv"

st.set_page_config(
    page_title="Decision Tree | Protein Analysis",
    layout="wide",
)

# 2.Neon Dark Theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(165deg, #050510 0%, #0a0f1e 35%, #0f172a 70%, #0c1222 100%);
        color: #e2e8f0;
    }
    [data-testid="stHeader"] {
        background: transparent !important;
    }
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
    h1, h2, h3 { color: #f8fafc !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_CSV.is_file():
        return None
    return pd.read_csv(DATA_CSV)

st.title("Decision Tree")

# --- 3. BLUE DESCRIPTION BOX ---
st.info("""
I implemented a Decision Tree model to classify the localization site of proteins based on the given dataset. 

The model was trained to learn patterns between input features and protein localization labels, using recursive partitioning to create interpretable decision rules. This approach allows for a clear understanding of which biological features most significantly influence protein placement within a cell.
""")

df = load_data()

if df is None:
    st.error(f"Could not find yeast.csv in {DATA_DIR}")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Model Tuning")
    max_depth = st.slider("Max Depth", 1, 30, 10)
    min_split = st.slider("Min Samples Split", 2, 50, 5)
    target_col = "name"
    feature_candidates = [c for c in df.columns if c != target_col]
    selected_features = st.multiselect("Select Features", feature_candidates, default=feature_candidates)

if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()

# Logic
X = df[selected_features]
y = df[target_col]
le = LabelEncoder()
y_encoded = le.fit_transform(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, min_samples_split=min_split, random_state=42)

scoring = {
    'f1_macro': make_scorer(f1_score, average='macro'),
    'recall_macro': make_scorer(recall_score, average='macro'),
    'mcc': make_scorer(matthews_corrcoef)
}

with st.spinner("Training..."):
    cv_results = cross_validate(dt_model, X, y_encoded, cv=skf, scoring=scoring)
    dt_model.fit(X, y_encoded)
    y_probs = dt_model.predict_proba(X)
    auc_score = roc_auc_score(y_encoded, y_probs, multi_class='ovr')

# Metrics
st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("F1 Score (Macro)", f"{cv_results['test_f1_macro'].mean():.4f}")
m2.metric("Recall (Macro)", f"{cv_results['test_recall_macro'].mean():.4f}")
m3.metric("MCC", f"{cv_results['test_mcc'].mean():.4f}")
m4.metric("AUC (OvR)", f"{auc_score:.4f}")

st.divider()

# Visuals
col_tree, col_feat = st.columns([1.3, 0.7])

with col_tree:
    st.markdown("**Decision Logic Flow**")
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150) 
    fig.patch.set_facecolor('none') 
    
    plot_tree(dt_model, 
              feature_names=selected_features, 
              class_names=list(le.classes_), 
              filled=True, 
              rounded=True, 
              fontsize=6, 
              max_depth=3, 
              ax=ax,
              precision=2)
    
    plt.tight_layout() 
    st.pyplot(fig, clear_figure=True)
    st.caption("Showing top levels of decision logic for readability.")

with col_feat:
    st.markdown("**Feature Importance**")
    importances = pd.DataFrame({
        'Feature': selected_features, 
        'Importance': dt_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                      template="plotly_dark", color_discrete_sequence=['#a855f7'])
    
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(t=20, b=20, l=0, r=0)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with st.expander("The Science Behind the Model"):
    st.markdown(r"""
The model uses **Entropy** to determine the best splits:
$$ H(S) = - \sum_{i=1}^{C} p_i \log_2(p_i) $$
    """)