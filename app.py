"""
Streamlit app for the ML Assignment 2 (Telco Customer Churn classification).

Features:
  a. Dataset upload (CSV) — upload your own test data
  b. Model selection dropdown — choose among the 5 trained models
  c. Display of evaluation metrics
  d. Confusion matrix + classification report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Telco Churn Classifier", layout="wide")

MODEL_DIR = "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

TARGET_COL = "Churn"


@st.cache_resource
def load_model(model_name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
    return joblib.load(path)


@st.cache_data
def load_meta():
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        return json.load(f)


@st.cache_data
def load_default_test_data():
    return pd.read_csv("test_data.csv")


st.title("📉 Telco Customer Churn — Classification Demo")
st.markdown(
    "This app demonstrates 5 classification models trained on the "
    "**Telco Customer Churn** dataset (7,043 customers, 19 features). "
    "Upload your own test CSV (must include the `Churn` column with "
    "Yes/No values) or use the bundled sample test data."
)

# --- a. Dataset upload -------------------------------------------------
st.sidebar.header("1. Data")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(data)} rows from uploaded file.")
else:
    data = load_default_test_data()
    st.sidebar.info(f"Using bundled sample test_data.csv ({len(data)} rows).")

# --- b. Model selection --------------------------------------------------
st.sidebar.header("2. Model")
model_name = st.sidebar.selectbox("Choose a classification model", list(MODEL_FILES.keys()))

run_button = st.sidebar.button("Run Evaluation", type="primary")

st.subheader("Preview of data")
st.dataframe(data.head(10), use_container_width=True)

if run_button:
    if TARGET_COL not in data.columns:
        st.error(f"Uploaded CSV must contain a '{TARGET_COL}' column (Yes/No) to evaluate against.")
    else:
        pipe = load_model(model_name)

        eval_df = data.copy()
        y_true_raw = eval_df[TARGET_COL]
        # Accept either Yes/No strings or 0/1
        if y_true_raw.dtype == object:
            y_true = y_true_raw.map({"Yes": 1, "No": 0})
        else:
            y_true = y_true_raw

        X_eval = eval_df.drop(columns=[TARGET_COL])
        # customerID isn't used by the model; drop if present
        if "customerID" in X_eval.columns:
            X_eval = X_eval.drop(columns=["customerID"])

        y_pred = pipe.predict(X_eval)
        y_proba = pipe.predict_proba(X_eval)[:, 1]

        # --- c. Evaluation metrics ---------------------------------
        st.subheader(f"Evaluation Metrics — {model_name}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
        col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.4f}")
        col3.metric("Precision", f"{precision_score(y_true, y_pred):.4f}")
        col4.metric("Recall", f"{recall_score(y_true, y_pred):.4f}")
        col5.metric("F1 Score", f"{f1_score(y_true, y_pred):.4f}")
        col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.4f}")

        # --- d. Confusion matrix + classification report -------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.divider()

# --- Bonus: full comparison table across all 5 models ------------------
st.subheader("All-Model Comparison (on held-out test set used during training)")
try:
    comp_df = pd.read_csv(os.path.join(MODEL_DIR, "comparison_table.csv"))
    st.dataframe(comp_df, use_container_width=True)
except FileNotFoundError:
    st.info("Run model/train_models.py to generate comparison_table.csv")
