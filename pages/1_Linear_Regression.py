from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
 
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "MTask 2.0"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"
 
TARGET_COL   = "close"
AUTO_EXCLUDE = {"date", "symbols", "close"}
 
st.set_page_config(page_title="Linear Regression | ML Studio", layout="wide")
 
# --- 1. DARK THEME  ---
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
 
st.title("Linear Regression")
 
st.info("""
I built a machine learning model to predict stock closing prices, an important indicator of end-of-day market performance.
 
The model was trained using train.csv to learn patterns in the data and then used to predict closing prices on test.csv, which was kept separate for evaluation. You can switch between Gradient Descent (and watch the cost curve converge) and the Normal Equation for an instant closed-form solution. Performance is measured using R² and MSE.
 
This project demonstrates the use of machine learning for financial prediction and data-driven decision making.
""")
 
# --- 2. DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)
 
# --- 3. MATH HELPERS ---
def compute_cost(x, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((x @ theta - y) ** 2)
 
def gradient_descent(x, y, theta, lr, iters):
    m = len(y)
    cost_history = []
    for _ in range(iters):
        error    = x @ theta - y
        gradient = (1 / m) * (x.T @ error)
        theta    = theta - lr * gradient
        cost_history.append(compute_cost(x, y, theta))
    return theta, cost_history
 
def normal_equation_theta(x, y):
    theta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return theta.reshape(-1, 1), []          # empty list keeps return shape consistent
 
def align_test_features(test_df, feature_names):
    out = pd.DataFrame(index=test_df.index)
    for c in feature_names:
        out[c] = test_df[c] if c in test_df.columns else 0.0
    return out[feature_names]
 
def preprocess(x_train_raw, x_test_raw):
    non_zero = np.std(x_train_raw, axis=0) > 0
    x_tr = x_train_raw[:, non_zero]
    x_te = x_test_raw[:, non_zero]
    mean, std = np.mean(x_tr, axis=0), np.std(x_tr, axis=0)
    std[std == 0] = 1
    x_tr_s = (x_tr - mean) / std
    x_te_s = (x_te - mean) / std
    x_tr_d = np.concatenate([np.ones((x_tr_s.shape[0], 1)), x_tr_s], axis=1)
    x_te_d = np.concatenate([np.ones((x_te_s.shape[0], 1)), x_te_s], axis=1)
    return x_tr_d, x_te_d
 
# --- 4. FILE CHECK ---
if not TRAIN_CSV.is_file() or not TEST_CSV.is_file():
    st.error(f"Missing CSV files in: {DATA_DIR}")
    st.stop()
 
train_df, test_df = load_train_test()
numeric_candidates = [
    c for c in train_df.columns
    if c not in AUTO_EXCLUDE and pd.api.types.is_numeric_dtype(train_df[c])
]
 
# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("Model Options")
    selected_features = st.multiselect("Features", options=numeric_candidates, default=numeric_candidates)
    training_mode     = st.radio("Fit Method", ["Gradient Descent", "Normal Equation"])
    if "Gradient" in training_mode:
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.4f")
        iterations    = st.slider("Iterations", min_value=100, max_value=5000, value=1000, step=100)
 
if not selected_features:
    st.warning("Please select at least one feature.")
    st.stop()
 
# --- 6. PREPARE DATA ---
X_train_df = train_df[selected_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_train    = pd.to_numeric(train_df[TARGET_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
 
X_test_df  = align_test_features(test_df, selected_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_test     = pd.to_numeric(test_df[TARGET_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
 
x_train, x_test = preprocess(X_train_df.to_numpy(dtype=float), X_test_df.to_numpy(dtype=float))
 
# --- 7. TRAIN ---
with st.spinner("Training..."):
    if "Gradient" in training_mode:
        theta_init = np.zeros((x_train.shape[1], 1))
        final_theta, cost_history = gradient_descent(
            x_train, y_train, theta_init, float(learning_rate), int(iterations)
        )
    else:
        final_theta, cost_history = normal_equation_theta(x_train, y_train)
 
y_pred = x_test @ final_theta
 
# --- 8. METRICS ---
mse    = float(np.mean((y_test - y_pred) ** 2))
rmse   = float(np.sqrt(mse))
ss_res = float(np.sum((y_test - y_pred) ** 2))
ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
r2     = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
mae    = float(np.mean(np.abs(y_test - y_pred)))
 
st.subheader("Test Set Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² Score",  f"{r2:.4f}")
m2.metric("MSE",       f"{mse:,.4f}")
m3.metric("RMSE",      f"{rmse:,.4f}")
m4.metric("MAE",       f"{mae:,.4f}")
 
st.divider()
 
# --- 9. PLOTS ---
col_scatter, col_res = st.columns(2)
 
with col_scatter:
    fig_scatter = go.Figure()
    # Perfect prediction line
    mn, mx = float(y_test.min()), float(y_test.max())
    fig_scatter.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", name="Perfect Fit",
        line=dict(color="#f43f5e", dash="dash", width=1.5)
    ))
    fig_scatter.add_trace(go.Scatter(
        x=y_test.ravel(), y=y_pred.ravel(),
        mode="markers", name="Predictions",
        marker=dict(color="#3b82f6", opacity=0.5, size=4)
    ))
    fig_scatter.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Actual Close Price",
        yaxis_title="Predicted Close Price",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
 
with col_res:
    residuals = (y_test - y_pred).ravel()
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=y_pred.ravel(), y=residuals,
        mode="markers", name="Residuals",
        marker=dict(color="#10b981", opacity=0.5, size=4)
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="#f43f5e", line_width=1.5)
    fig_res.update_layout(
        title="Residual Plot",
        xaxis_title="Predicted Value",
        yaxis_title="Residual (Actual − Predicted)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400
    )
    st.plotly_chart(fig_res, use_container_width=True)
 
# --- 10. COST CURVE (only for gradient descent) ---
if "Gradient" in training_mode and cost_history:
    st.markdown("**Gradient Descent — Cost Convergence**")
    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=list(range(1, len(cost_history) + 1)),
        y=cost_history,
        mode="lines",
        name="Cost J(θ)",
        line=dict(color="#a855f7", width=2)
    ))
    fig_cost.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Cost J(θ)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=350
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption("A smoothly decreasing curve means your learning rate is well chosen. If it spikes or diverges, lower the learning rate.")
 
# --- 11. SCIENCE EXPANDER ---
with st.expander("The Science Behind the Model"):
    st.markdown(r"""
**Linear Regression** fits a straight line (or hyperplane) through the data by finding weights $\theta$ that minimize the cost function:
 
$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 $$
 
**Gradient Descent** iteratively updates weights by moving in the direction of steepest descent:
 
$$ \theta := \theta - \alpha \frac{\partial J(\theta)}{\partial \theta} = \theta - \frac{\alpha}{m} X^T (X\theta - y) $$
 
**Normal Equation** solves for the optimal $\theta$ directly in one step — no iterations needed:
 
$$ \theta = (X^T X)^{-1} X^T y $$
 
The tradeoff: Normal Equation is exact but slow for large datasets $O(n^3)$. Gradient Descent scales much better.
    """)