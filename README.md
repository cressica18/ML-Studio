# 🧠 ML Studio

An interactive, multi-page web app that showcases machine learning models built from scratch — covering regression, classification, ensemble methods, and unsupervised learning. Each model was originally developed for real datasets provided by a university tech club, and is now wrapped in a clean, interactive UI for exploration and demonstration.

**[🚀 Live Demo →](https://your-app-link.streamlit.app)** ← replace with your deployed link

---

## 📸 Preview


>  [screentogif.com](https://www.screentogif.com/) 

---

## 🗂 Models Included

| Page | Model | Dataset | Task |
|------|-------|---------|------|
| Linear Regression | Gradient Descent + Normal Equation | Stock market data | Predict closing price |
| Logistic Regression | Sklearn LogisticRegression | Crime case data | Predict if case is closed |
| Decision Tree | DecisionTreeClassifier + 5-fold CV | Yeast protein data | Classify protein location |
| Tree Ensembles | Random Forest + AdaBoost | Yeast protein data | Classify protein location |
| K-Means Clustering | K-Means from scratch + PCA | Seeds dataset | Cluster seed varieties |

---

## ✨ Features

- **Interactive hyperparameter tuning** — sliders and dropdowns retrain models in real time
- **From-scratch implementations** — Linear Regression (gradient descent + normal equation) and K-Means built using only NumPy
- **Cross-validation** — honest performance estimates using StratifiedKFold throughout
- **Rich visualizations** — Plotly charts for actual vs predicted, residuals, confusion matrices, feature importance, PCA cluster plots, and cost convergence curves
- **Science expanders** — every page explains the math behind the model with LaTeX equations
- Consistent dark UI theme across all pages

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web framework |
| [scikit-learn](https://scikit-learn.org) | ML models & evaluation |
| [Pandas](https://pandas.pydata.org) | Data handling |
| [NumPy](https://numpy.org) | From-scratch math |
| [Plotly](https://plotly.com) | Interactive charts |
| [Matplotlib](https://matplotlib.org) | Decision tree visualization |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOURUSERNAME/ml-studio.git
cd ml-studio

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run Main.py
```

---

## 📁 Project Structure

```
ml-studio/
├── Main.py                      # Home page
├── pages/
│   ├── 1_Linear_Regression.py
│   ├── 2_Logistic_Regression.py
│   ├── 3_Decision_Tree.py
│   ├── 4_Tree_Ensembles.py
│   └── 5_KMeans_Clustering.py
├── MTask 2.0/              # Stock dataset
├── Mtask 2.1/              # Crime dataset
├── MTask 3.0/              # Yeast protein dataset
├── MTask 3.3/              # Seeds dataset
├── requirements.txt
└── README.md
```

---

##  Notes

- The **Logistic Regression** model scores ~49–51% accuracy. This is an honest result — the available features (city, weapon, age, domain) don't carry enough signal to predict case closure reliably. This is a valid data science finding: knowing when a model can't help is as important as knowing when it can.
- The **K-Means** implementation is built entirely from scratch using NumPy — no sklearn KMeans used.
- The **Linear Regression** gradient descent implementation is also from scratch, with a live cost convergence curve.

---

## 👤 Author

**Your Name**
- GitHub: [@cressica18]https://github.com/cressica18
- LinkedIn: https://www.linkedin.com/in/subhangi-banerjee-51b7772b1/