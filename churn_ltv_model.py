# =============================================================================
# Customer LTV and Churn Prediction Model
# =============================================================================
# What this project does:
#   - Builds a machine learning model to predict which users are likely to
#     churn within the next 30 days
#   - Uses Logistic Regression and Random Forest classifiers
#   - Evaluates model performance with precision, recall, and ROC-AUC
#   - Identifies the most important features that predict churn
#   - Estimates Customer Lifetime Value (LTV) per user segment
#
# Why this matters for product analytics roles:
#   LTV modeling and churn prediction are explicitly mentioned in the job
#   description. This project shows ability with scikit-learn (Python ML),
#   feature engineering, and translating model outputs into business actions.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, confusion_matrix, ConfusionMatrixDisplay)
import os
import warnings
warnings.filterwarnings("ignore")

np.random.seed(77)
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("PROJECT 3: Customer LTV and Churn Prediction Model")
print("=" * 60)


# -----------------------------------------------------------------------------
# STEP 1: Generate synthetic user behavior data
# -----------------------------------------------------------------------------
# Each row represents one paying customer and their usage behavior over
# the first 30 days after converting to paid.

N = 3000

# User features (these are the inputs to our model)
days_since_signup          = np.random.randint(30, 365, N)
logins_last_30d            = np.random.poisson(12, N)
features_used              = np.random.randint(1, 15, N)
integrations_connected     = np.random.poisson(1.5, N)
support_tickets_filed      = np.random.poisson(0.8, N)
invites_sent               = np.random.poisson(3, N)
plan_type                  = np.random.choice(["starter", "pro", "business"], N,
                                               p=[0.50, 0.35, 0.15])
mobile_app_used            = np.random.binomial(1, 0.45, N)
onboarding_completed       = np.random.binomial(1, 0.60, N)

# Monthly revenue per plan
plan_revenue = {"starter": 15, "pro": 45, "business": 99}
monthly_revenue = np.array([plan_revenue[p] for p in plan_type])

# Build churn label: high churn probability if low engagement, high tickets, no onboarding
churn_logit = (
    -2.5
    - 0.08 * logins_last_30d
    - 0.15 * features_used
    - 0.30 * integrations_connected
    + 0.40 * support_tickets_filed
    - 0.25 * invites_sent
    - 0.60 * onboarding_completed
    - 0.40 * mobile_app_used
    + 0.50 * (plan_type == "starter").astype(int)
    - 0.50 * (plan_type == "business").astype(int)
    + np.random.normal(0, 0.5, N)   # noise
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churned    = (np.random.random(N) < churn_prob).astype(int)

df = pd.DataFrame({
    "days_since_signup":       days_since_signup,
    "logins_last_30d":         logins_last_30d,
    "features_used":           features_used,
    "integrations_connected":  integrations_connected,
    "support_tickets_filed":   support_tickets_filed,
    "invites_sent":            invites_sent,
    "plan_type":               plan_type,
    "mobile_app_used":         mobile_app_used,
    "onboarding_completed":    onboarding_completed,
    "monthly_revenue":         monthly_revenue,
    "churned":                 churned,
})

print(f"\nDataset: {len(df):,} customers")
print(f"Overall churn rate: {df['churned'].mean():.1%}")
print(f"\nChurn rate by plan:")
print(df.groupby("plan_type")["churned"].mean().map("{:.1%}".format).to_string())


# -----------------------------------------------------------------------------
# STEP 2: Feature engineering and data prep
# -----------------------------------------------------------------------------

# One-hot encode the plan_type column (ML models need numbers, not text)
df_model = pd.get_dummies(df, columns=["plan_type"], drop_first=False)

FEATURE_COLS = [
    "days_since_signup", "logins_last_30d", "features_used",
    "integrations_connected", "support_tickets_filed", "invites_sent",
    "mobile_app_used", "onboarding_completed",
    "plan_type_pro", "plan_type_business",
]

X = df_model[FEATURE_COLS]
y = df_model["churned"]

# Split into 80% training, 20% test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features for logistic regression (RF doesn't need this, but good practice)
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining set: {len(X_train):,} users")
print(f"Test set:     {len(X_test):,} users")


# -----------------------------------------------------------------------------
# STEP 3: Train and evaluate two models
# -----------------------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42,
                                                   max_depth=6, min_samples_leaf=10),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred      = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_prob)
    results[name] = {"model": model, "y_pred": y_pred,
                      "y_pred_prob": y_pred_prob, "auc": auc}

    print(f"\n--- {name.upper()} ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))


# -----------------------------------------------------------------------------
# STEP 4: Visualizations
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# A: ROC curves for both models
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_pred_prob"])
    axes[0].plot(fpr, tpr, linewidth=2.2,
                 label=f"{name} (AUC = {res['auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random baseline")
axes[0].set_xlabel("False Positive Rate", fontsize=11)
axes[0].set_ylabel("True Positive Rate", fontsize=11)
axes[0].set_title("ROC Curve: Churn Prediction Models", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# B: Feature importance from Random Forest
rf_model     = results["Random Forest"]["model"]
importances  = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
importances  = importances.sort_values(ascending=True)
colors_fi    = ["#2563A8" if v >= importances.median() else "#A8C8F0"
                for v in importances]
importances.plot(kind="barh", ax=axes[1], color=colors_fi, edgecolor="white")
axes[1].set_title("Feature Importance (Random Forest)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Importance Score", fontsize=11)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# C: Confusion matrix for best model (Random Forest)
best_name   = max(results, key=lambda k: results[k]["auc"])
best_result = results[best_name]
cm = confusion_matrix(y_test, best_result["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Retained", "Churned"])
disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title(f"Confusion Matrix\n{best_name}", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/05_churn_model_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nChart saved: outputs/05_churn_model_results.png")


# -----------------------------------------------------------------------------
# STEP 5: LTV estimation by segment
# -----------------------------------------------------------------------------
# LTV = Monthly Revenue * Expected Months Retained
# Expected months retained = 1 / churn_rate (geometric series approximation)

print("\n--- LTV ESTIMATION BY PLAN ---")

ltv_summary = (
    df.groupby("plan_type")
      .agg(
          customers        = ("churned",         "count"),
          avg_monthly_rev  = ("monthly_revenue",  "mean"),
          churn_rate       = ("churned",           "mean"),
      )
      .assign(
          expected_months = lambda x: 1 / x["churn_rate"].clip(lower=0.01),
          estimated_ltv   = lambda x: x["avg_monthly_rev"] * x["expected_months"],
      )
)
print(ltv_summary.round(2).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ltv_sorted = ltv_summary.sort_values("estimated_ltv", ascending=True)
colors_ltv = ["#A8C8F0", "#3B7DD8", "#2563A8"]
bars = ax.barh(ltv_sorted.index, ltv_sorted["estimated_ltv"],
               color=colors_ltv, height=0.45)
for bar, val in zip(bars, ltv_sorted["estimated_ltv"]):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"${val:,.0f}", va="center", fontsize=11, fontweight="bold")
ax.set_xlabel("Estimated LTV ($)", fontsize=11)
ax.set_title("Estimated Customer LTV by Plan Type", fontsize=13, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}"))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/06_ltv_by_plan.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nChart saved: outputs/06_ltv_by_plan.png")


# -----------------------------------------------------------------------------
# STEP 6: Identify at-risk accounts
# -----------------------------------------------------------------------------

rf = results["Random Forest"]["model"]
df["churn_probability"] = rf.predict_proba(X[FEATURE_COLS])[:, 1]
df["risk_segment"] = pd.cut(df["churn_probability"],
                              bins=[0, 0.30, 0.60, 1.0],
                              labels=["Low Risk", "Medium Risk", "High Risk"])

at_risk = df[df["risk_segment"] == "High Risk"].sort_values("churn_probability", ascending=False)
print(f"\nHigh-risk accounts identified: {len(at_risk):,} "
      f"({len(at_risk)/len(df):.1%} of customers)")
print(f"Potential monthly revenue at risk: "
      f"${at_risk['monthly_revenue'].sum():,}")

df.to_csv("outputs/churn_predictions.csv", index=False)
print("\nFull predictions saved: outputs/churn_predictions.csv")

print("\n" + "=" * 60)
print("Project complete. Charts and predictions saved to outputs/")
print("=" * 60)
