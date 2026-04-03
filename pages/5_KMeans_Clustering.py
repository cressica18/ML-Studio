import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --- 1. PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent

def find_data_file(filename):
    """Searches recursively for the data file to avoid folder name issues."""
    for path in BASE_DIR.rglob(filename):
        return path
    return None

DATA_PATH = find_data_file("seeds_data.csv")

st.set_page_config(
    page_title="K-Means Clustering | Seed Analysis",
    layout="wide",
)

# --- 2. NEON DARK THEME CSS ---
st.markdown(
    """
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
    [data-testid="stMetricValue"] { color: #f0abfc !important; }
    h1, h2, h3 { color: #f8fafc !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 3. DATA LOADING ---
@st.cache_data(show_spinner=False)
def load_data(path):
    if path and path.exists():
        return pd.read_csv(path)
    return None

if DATA_PATH:
    df = load_data(DATA_PATH)
else:
    st.error("Could not find seeds_data.csv")
    st.stop()

# --- 4. HEADER & BLUE DESCRIPTION BOX ---
st.title("K-Means Clustering")

st.info("""
I implemented K-Means clustering from scratch to group mixed seed data into distinct clusters. 

The model was used to identify natural groupings in the data, with techniques applied to determine the optimal number of clusters. Cluster centers were analyzed to interpret the characteristics of each group, and PCA was used to visualize the clustering results. Target labels were only used after clustering for evaluation and visualization.
""")

# --- 5. SIDEBAR PARAMETERS ---
with st.sidebar:
    st.header("Clustering Params")
    k_clusters = st.slider("Select K (Clusters)", 2, 6, 3)
    max_iters = st.number_input("Max Iterations", 10, 500, 100)
    
    st.header("Feature Selection")
    all_features = [c for c in df.columns if c != 'Class']
    selected_features = st.multiselect("Features", all_features, default=all_features)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

# --- 6. K-MEANS LOGIC ---
X = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def get_kmeans_clusters(data, k, iters):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) if len(data[labels==i]) > 0 else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids): break
        centroids = new_centroids
    return labels, centroids

with st.spinner("Calculating Clusters..."):
    cluster_labels, final_centroids = get_kmeans_clusters(X_scaled, k_clusters, max_iters)
    sil_score = silhouette_score(X_scaled, cluster_labels)

# Metrics
st.subheader("Clustering Metrics")
m1, m2 = st.columns(2)
m1.metric("Selected Clusters (K)", k_clusters)
m2.metric("Silhouette Score", f"{sil_score:.4f}")

st.divider()

# --- 7. PCA VISUALIZATION ---
st.subheader("Cluster Visualization (PCA)")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels.astype(str)
pca_df['Actual Class'] = df['Class'].astype(str)

fig_pca = px.scatter(
    pca_df, x='PC1', y='PC2', color='Cluster',
    symbol='Actual Class',
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
st.plotly_chart(fig_pca, use_container_width=True)

# --- 8. ANALYSIS & SCIENCE ---
col_left, col_right = st.columns(2)

with col_left:
    with st.expander("Cluster Centers (Original Scale)"):
        centers_orig = scaler.inverse_transform(final_centroids)
        centers_df = pd.DataFrame(centers_orig, columns=selected_features)
        centers_df.index.name = "Cluster ID"
        st.dataframe(centers_df, use_container_width=True)

with col_right:
    with st.expander("The Science: Unsupervised Learning"):
        st.markdown(r"""
        K-Means aims to minimize the **Within-Cluster Sum of Squares (WCSS)**:
        $$ J = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 $$
        Where $\mu_i$ is the centroid of cluster $C_i$. This is an iterative process of assignment and update.
        """)