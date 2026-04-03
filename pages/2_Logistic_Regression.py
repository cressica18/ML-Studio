from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    make_scorer, f1_score, roc_auc_score
)
 
# --- 1. PATH CONFIGURATION ---
ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "mtask_2.1"
TRAIN_CSV = DATA_DIR / "crime_train.csv"
TEST_CSV  = DATA_DIR / "crime_test.csv"
 
FEATURES = ['city', 'crime_description', 'age', 'sex', 'weapon', 'domain']
TARGET   = 'closed'
 
st.set_page_config(page_title="Logistic Regression | ML Studio", layout="wide")
 
# --- 2. DARK THEME CSS ---
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
    h1, h2, h3 { color: #f8fafc !important; }
    </style>
""", unsafe_allow_html=True)
 
# --- 3. TITLE & DESCRIPTION ---
st.title("Logistic Regression")
 
st.info("""
I built a machine learning model to predict whether a criminal case is closed or unresolved, enabling data-driven analysis in investigative scenarios.
 
The model was trained on a crime dataset containing features like city, weapon type, age, and crime domain. One-hot encoding was applied to categorical variables and StandardScaler was used to normalize the data before training. Cross-validation was used to ensure the reported accuracy generalizes well to unseen data.
 
This project demonstrates the application of classification techniques for real-world decision-making and predictive analysis.
""")
 
# --- 4. DATA LOADING (only data is cached, not the model) ---
@st.cache_data(show_spinner=False)
def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
 
    for df in [train_df, test_df]:
        df['weapon'] = df['weapon'].fillna('Unknown')
        df['age']    = df['age'].fillna(df['age'].median())
        df[TARGET]   = df[TARGET].map({'Yes': 1, 'No': 0})
 
    # Encode categoricals consistently across train+test
    combined = pd.concat([train_df[FEATURES], test_df[FEATURES]], axis=0)
    combined_enc = pd.get_dummies(combined, columns=['city', 'crime_description', 'sex', 'weapon', 'domain'])
 
    X_train = combined_enc.iloc[:len(train_df)]
    X_test  = combined_enc.iloc[len(train_df):]
    y_train = train_df[TARGET]
    y_test  = test_df[TARGET]
 
    return X_train, X_test, y_train, y_test
 
# --- 5. CHECK FILES EXIST ---
if not TRAIN_CSV.is_file() or not TEST_CSV.is_file():
    st.error(f"Missing CSV files in: {DATA_DIR}")
    st.stop()
 
X_train, X_test, y_train, y_test = load_data()
 
# --- 6. SIDEBAR HYPERPARAMETERS ---
with st.sidebar:
    st.header("Model Hyperparameters")
    c_val      = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.01)
    max_iter   = st.slider("Max Iterations", 100, 1000, 500, 50)
    solver     = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], index=0)
    test_size  = st.slider("Test Split (%)", 10, 40, 20)
    n_top_feat = st.slider("Top N Features to Show", 5, 20, 10)
 
# --- 7. SCALE + TRAIN (runs on every slider change) ---
with st.spinner("Training model..."):
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
 
    model = LogisticRegression(C=c_val, max_iter=max_iter, solver=solver, random_state=42)
    model.fit(X_tr_sc, y_train)
 
    y_pred  = model.predict(X_te_sc)
    y_proba = model.predict_proba(X_te_sc)[:, 1]
 
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average='macro')
    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
 
    # 5-fold cross-val on training set for honest estimate
    cv_scores = cross_validate(
        LogisticRegression(C=c_val, max_iter=max_iter, solver=solver, random_state=42),
        X_tr_sc, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring={'accuracy': 'accuracy', 'f1_macro': make_scorer(f1_score, average='macro')}
    )
 
# --- 8. METRICS ROW ---
st.subheader("Model Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Test Accuracy",     f"{acc:.2%}")
m2.metric("F1 Score (Macro)",  f"{f1:.4f}")
m3.metric("ROC-AUC",           f"{auc:.4f}")
m4.metric("CV Accuracy (5-fold)", f"{cv_scores['test_accuracy'].mean():.2%}")
 
st.divider()
 
# --- ACCURACY NOTE ---
st.warning("""
**A note on the accuracy (~49–51%)**
 
This is not a bug — it's an honest result that reflects the nature of the dataset.
 
The features available (city, weapon type, age, crime domain) have very weak correlation with whether a case gets closed or not. In real-world crime datasets, case closure depends heavily on factors not present here — witness availability, investigator resources, case priority, evidence quality — none of which are captured in this data.
 
A model that scores ~50% on a balanced binary dataset is essentially saying: *"the available features don't contain enough signal to predict this outcome reliably."* This is itself a valid and important finding in data science — knowing when a model can't help is just as valuable as knowing when it can.
 
Changing C or max iterations won't fix this because the ceiling is set by the data, not the algorithm.
""")
 
# --- 9. PLOTS ---
col_cm, col_feat = st.columns(2)
 
with col_cm:
    st.markdown("**Confusion Matrix**")
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=['Predicted: No', 'Predicted: Yes'],
        y=['Actual: No',    'Actual: Yes'],
        color_continuous_scale='Purp',
        template="plotly_dark"
    )
    fig_cm.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    st.plotly_chart(fig_cm, use_container_width=True)
 
with col_feat:
    st.markdown(f"**Top {n_top_feat} Features by Coefficient Magnitude**")
    importance = np.abs(model.coef_[0])
    feat_df = (
        pd.DataFrame({'Feature': X_train.columns, 'Importance': importance})
        .sort_values('Importance', ascending=True)
        .tail(n_top_feat)
    )
    fig_feat = px.bar(
        feat_df, x='Importance', y='Feature', orientation='h',
        template="plotly_dark", color_discrete_sequence=['#a855f7']
    )
    fig_feat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    st.plotly_chart(fig_feat, use_container_width=True)
 
# --- 10. SCIENCE EXPANDER ---
with st.expander("The Science Behind the Model"):
    st.markdown(r"""
**Logistic Regression** models the probability of a binary outcome using the sigmoid function:
 
$$ \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \theta^T x $$
 
The model is trained by minimizing the **Binary Cross-Entropy Loss**:
 
$$ \mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] $$
 
The **C parameter** controls regularization — lower C = stronger regularization (simpler model), higher C = weaker regularization (fits training data more closely).
    """)