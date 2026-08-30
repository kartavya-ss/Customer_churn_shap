# Customer Churn Prediction with Explainable AI (SHAP)

Predicts which customers are likely to churn and explains **why** each
prediction was made — not just a risk score, but a clear, per-customer
breakdown of which factors drove that prediction.

## Problem

Telecom/subscription businesses lose significant revenue to customer churn.
Most churn models stop at producing a probability score, which isn't
directly actionable for a business team — this project adds an
explainability layer so predictions can be trusted and acted on with
specific reasoning per customer.

## Dataset

[Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
— 7,043 customers, 20 features (contract type, tenure, monthly charges,
services subscribed, etc.), binary churn label. Loaded via `kagglehub` for
reproducibility (no manual download/upload step needed).

## Approach

1. **Data Cleaning & Preprocessing** — handled missing values in
   `TotalCharges`, label-encoded categorical features, removed identifier
   columns.

2. **Class Imbalance Handling (SMOTE)** — the dataset is ~73%/27%
   imbalanced (non-churn/churn). Applied SMOTE (Synthetic Minority
   Oversampling) to the training set only, to avoid biasing the model
   toward the majority class without leaking synthetic data into evaluation.

3. **Model Benchmarking** — trained and compared three algorithms under
   identical conditions (same SMOTE-resampled data):

   | Model | AUC-ROC | Precision | Recall | F1 |
   |---|---|---|---|---|
   | Random Forest | 0.824 | 0.576 | 0.580 | 0.578 |
   | Logistic Regression | 0.822 | 0.523 | 0.717 | 0.605 |
   | XGBoost | 0.812 | 0.560 | 0.607 | 0.583 |

   **Random Forest** was selected for its highest AUC-ROC (best overall
   ranking ability across thresholds). Logistic Regression showed higher
   recall at the default threshold — a reasonable alternative pick if the
   business priority were maximizing churner capture over precision.

4. **Hyperparameter Tuning** — tuned Random Forest via `RandomizedSearchCV`
   (25 iterations, 5-fold CV, optimized for AUC-ROC), improving performance
   to:

   | Metric | Score |
   |---|---|
   | AUC-ROC | 0.825 |
   | Precision | 0.588 |
   | Recall | 0.618 |
   | F1 | 0.602 |

5. **Explainability with SHAP** — used `TreeExplainer` (optimized for
   tree-based models) to generate:
   - **Global explanations**: which features drive churn across the
     entire customer base (contract type, tenure, and tech support were
     top drivers).
   - **Local explanations**: per-customer waterfall plots showing exactly
     which features pushed an individual prediction up or down — e.g., a
     new customer (tenure = 1 month) on a month-to-month contract with no
     add-on services was flagged at 99.8% churn risk, with each
     contributing factor visible in the breakdown.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP,
matplotlib/seaborn.

## Key Takeaway

A churn probability alone isn't enough for a business to act on. By
pairing model predictions with SHAP-based explanations, this project
turns a black-box classifier into something a non-technical stakeholder
could use to understand *why* a specific customer is at risk — a
necessary step before any retention action can be taken.

## Files

- `churn_prediction_complete.ipynb` — full pipeline: data prep → model
  comparison → tuning → SHAP explainability
- `model_comparison.csv` — benchmark results across all three models
- `churn_model_final.pkl` — saved tuned Random Forest model

## Possible Extensions

- Estimate customer lifetime value independently of the churn model
  output (e.g., from historical spend) to prioritize retention efforts
  by both risk and revenue impact.
- Deploy as an interactive dashboard (Streamlit) for non-technical users
  to explore individual customer explanations.
