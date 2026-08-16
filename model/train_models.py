"""
train_models.py
----------------
The project builds five supervised classification models on the Telco Customer Churn dataset. Each model is assessed using a comprehensive set of metrics — Accuracy, AUC, Precision, Recall, F1‑score,
 and Matthews Correlation Coefficient (MCC). To ensure reproducibility and deployment readiness:
The trained models, along with the preprocessing pipeline, are saved.
A stratified hold‑out test set is preserved and exported (test_data.csv), serving as the default demo dataset in the Streamlit application.
A consolidated comparison of all models is generated and stored in model/comparison_table.csv.

Models:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier
  4. Gaussian Naive Bayes
  5. Random Forest(Ensemble)
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(HERE), "Telco-Customer-Churn.csv")

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# TotalCharges has some blank strings -> convert to numeric, impute later
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop the ID column (not predictive)
df = df.drop(columns=["customerID"])

# Target: Churn (Yes/No) -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

target = "Churn"
X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Total features: {len(numeric_cols) + len(categorical_cols)}, Instances: {len(df)}")

# ---------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the held-out test set (raw, pre-preprocessing) -> this is the
# "test_data.csv" required by the assignment and doubles as the
# default upload sample for the Streamlit app.
test_export = X_test.copy()
test_export[target] = y_test.values
test_export.to_csv(os.path.join(os.path.dirname(HERE), "test_data.csv"), index=False)

# ---------------------------------------------------------------
# 3. Shared preprocessing pipeline
# ---------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# ---------------------------------------------------------------
# 4. Define the 5 models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=150, random_state=RANDOM_STATE, max_depth=10
    ),
}

results = []
os.makedirs(HERE, exist_ok=True)

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    # GaussianNB / all these support predict_proba
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save the fitted pipeline (preprocessing + model together)
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(pipe, os.path.join(HERE, f"{fname}.joblib"), compress=3)

# ---------------------------------------------------------------
# 5. Save comparison table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(HERE, "comparison_table.csv"), index=False)
print("\n=== Comparison Table ===")
print(results_df.to_string(index=False))

# Save feature lists + column info for the Streamlit app
meta = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target": target,
}
with open(os.path.join(HERE, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nAll models trained and saved to model/")
