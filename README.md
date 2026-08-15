# Telco Customer Churn — Classification & Deployment

## a. Problem Statement
Customer churn (a subscriber discontinuing service) is one of the most
costly problems for telecom companies — acquiring a new customer is far
more expensive than retaining an existing one. This project builds and
compares 5 supervised classification models that predict whether a
telecom customer will churn (`Churn = Yes/No`) based on their
demographic profile, account information, and subscribed services. The
best-performing model is deployed as an interactive Streamlit web app
so evaluators can upload their own test data and see live predictions
and metrics.

## b. Dataset Description
- **Source:** [IBM Sample Data Sets — Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) (public, hosted on GitHub/Kaggle)
- **Instances:** 7,043 customers
- **Features:** 19 predictive features (4 numeric, 15 categorical) + 1 target column (`Churn`)
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Account info: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Target:** `Churn` — binary (`Yes` = customer left, `No` = customer stayed). ~26.5% positive class (moderately imbalanced).
- **Preprocessing:** `TotalCharges` blanks converted to numeric + median-imputed; numeric features standard-scaled; categorical features one-hot encoded; all wrapped in a single `sklearn` `Pipeline`/`ColumnTransformer` so the exact same preprocessing is applied at train and inference time.
- **Split:** 80% train / 20% stratified test (test set exported as `test_data.csv`, also used as the default sample in the Streamlit app).

## c. GitHub Repository Link
`<< PASTE YOUR GITHUB REPO URL HERE AFTER YOU PUSH >>`

## d. Models Used

### Comparison Table (on held-out 20% test set, 1,409 customers)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7750 | 0.7978 | 0.5882 | 0.5080 | 0.5452 | 0.3987 |
| kNN | 0.7779 | 0.8216 | 0.5845 | 0.5642 | 0.5741 | 0.4241 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Random Forest (Ensemble) | 0.8013 | 0.8404 | 0.6577 | 0.5241 | 0.5833 | 0.4601 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer — highest Accuracy, AUC, and F1. The churn decision boundary in this dataset is close to linear once categorical features are one-hot encoded (e.g. `Contract = Month-to-month` and low `tenure` strongly and additively raise churn probability), which suits a linear model well. Also the most interpretable via coefficients. |
| Decision Tree | Weakest of the five. A single tree overfits the training split's noise and doesn't generalize as well; even with `max_depth` capped at 8 it lags behind the ensemble and linear methods on every metric. |
| kNN | Middling performance. Distance-based similarity works reasonably once features are scaled, but the large number of one-hot encoded categorical dimensions (curse of dimensionality) dilutes the distance metric compared to numeric-heavy datasets. |
| Naive Bayes | Lowest precision and accuracy, but by far the **highest recall (0.84)** — it over-predicts churn, catching most true churners at the cost of many false alarms. The conditional-independence assumption is clearly violated here (e.g. `InternetService`, `OnlineSecurity`, `TechSupport` are correlated), which hurts calibration but doesn't stop it from being useful if the business goal is "don't miss a churner." |
| Random Forest (Ensemble) | Close second to Logistic Regression, with the best Precision and second-best AUC. Averaging many trees fixes most of the single Decision Tree's overfitting, but it still doesn't beat the simpler linear model on this particular dataset — a sign the true relationship is largely additive/linear rather than needing complex feature interactions. |
| **Overall Winner for this dataset** | **Logistic Regression** — best Accuracy (0.8055), AUC (0.8419), and F1 (0.6040), while remaining the simplest and most interpretable model. Random Forest is the runner-up and the best pick if precision is the top priority. |

## Repository Structure
```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out test split (also used as PDF's "test data")
│-- Telco-Customer-Churn.csv   # full source dataset
│-- model/
│   │-- train_models.py        # trains all 5 models + saves them
│   │-- comparison_table.csv
│   │-- meta.json
│   │-- logistic_regression.joblib
│   │-- decision_tree.joblib
│   │-- knn.joblib
│   │-- naive_bayes.joblib
│   │-- random_forest_ensemble.joblib
```

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # regenerates the trained models (already included)
streamlit run app.py
```

## Live Streamlit App
`<< PASTE YOUR DEPLOYED STREAMLIT CLOUD URL HERE >>`
