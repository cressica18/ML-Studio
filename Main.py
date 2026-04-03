import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ML Studio",
    page_icon="🧠",
    layout="wide",
)

# --- 1. UPDATED MODEL LIST (Cleaned & Specific) ---
MODEL_PAGES = [
    {
        "title": "Linear Regression",
        "file": "pages/1_Linear_Regression.py",
        "description": "Predicting stock closing prices using machine learning to evaluate end-of-day market performance based on historical trends.",
        "button_text": "Analyze Stocks →"
    },
    {
        "title": "Logistic Regression",
        "file": "pages/2_Logistic_Regression.py",
        "description": "Performs classification by estimating class probabilities. Optimized for predicting crime case status.",
        "button_text": "Predict Crime Case →"
    },
    {
        "title": "Decision Tree",
        "file": "pages/3_Decision_Tree.py",
        "description": "Uses rule-based splits to make interpretable predictions. Applied to protein sequence classification.",
        "button_text": "Classification of Proteins →"
    },
    {
        "title": "Tree Ensembles",
        "file": "pages/4_Tree_Ensembles.py",
        "description": "Combines many trees to improve stability and accuracy through parallel and sequential learning.",
        "button_text": "Bagging and Boosting →"
    },
    {
        "title": "KMeans Clustering",
        "file": "pages/5_KMeans_Clustering.py",
        "description": "Groups similar data points into clusters without labeled outputs. Identifying patterns in seed datasets.",
        "button_text": "Explore Seed Clusters →"
    },
]

# --- 2. CSS STYLING ---
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(165deg, #050510 0%, #0a0f1e 35%, #0f172a 70%, #0c1222 100%);
        color: #e2e8f0;
    }
    [data-testid="stHeader"] { background: rgba(15, 23, 42, 0.85); backdrop-filter: blur(8px); }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        border-right: 1px solid rgba(168, 85, 247, 0.25);
    }
    
    /* Clean CTA Button Styling */
    div.stButton > button {
        background: transparent !important;
        border: 1px solid #c084fc !important;
        color: #ffffff !important;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: rgba(192, 132, 252, 0.15) !important;
        border-color: #e879f9 !important;
        transform: translateY(-2px);
    }

    .hero-wrap {
        background: linear-gradient(125deg, #020617 0%, #0c1228 25%, #1e1b4b 55%, #581c87 85%, #8b5cf6 100%);
        padding: 2.5rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(168, 85, 247, 0.2);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .model-card {
        background: rgba(15, 23, 42, 0.4);
        border-radius: 0.75rem;
        padding: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        height: 180px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 3. HERO CONTENT ---
st.markdown(
    """
    <div class="hero-wrap">
        <h1 class="hero-title">ML Studio</h1>
        <p style="color: #94a3b8; font-size: 1.2rem;">A technical showcase of machine learning architectures implemented from the ground up.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.header("Navigation")
    for model in MODEL_PAGES:
        st.page_link(model["file"], label=model["title"])

# --- 5. FEATURED MODELS GRID ---
st.subheader("Model Architectures")
left_col, right_col = st.columns(2, gap="large")

for idx, model in enumerate(MODEL_PAGES):
    target_col = left_col if idx % 2 == 0 else right_col
    with target_col:
        # Card Content
        st.markdown(f"""
            <div class="model-card">
                <h3 style="margin-top:0;">{model['title']}</h3>
                <p style="color: #94a3b8;">{model['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive Button linked to the page
        if st.button(model["button_text"], key=f"btn_{idx}", use_container_width=True):
            st.switch_page(model["file"])
        
        st.write("") # Spacer

st.divider()
with st.expander("System Configuration"):
    st.info("Core Engine: Python 3.12 | Framework: Streamlit | Environment: .venv")