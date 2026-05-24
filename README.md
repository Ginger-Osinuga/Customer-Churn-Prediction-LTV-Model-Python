🔮 Customer Churn Prediction & LTV Model (Python)
---
🎯 Project Overview
---
A Python machine learning project that predicts which paying customers are most likely to churn within the next 30 days, using behavioral signals from their first month of product usage. Two models — Logistic Regression and Random Forest — are trained, compared, and evaluated. The project also estimates Customer Lifetime Value (LTV) by plan tier and quantifies the monthly revenue at risk from high-probability churn accounts.

📊 Key Analytical Insights
---
1. Behavioral Signals in the First 30 Days Are Strong Predictors of Churn
The Random Forest model achieved a ROC-AUC of 0.88, meaning it correctly ranks a churned user above a retained user 88% of the time — well above the 0.50 baseline of random guessing.
The Evidence: The three strongest predictors of churn were low login frequency in the first 30 days, failure to complete onboarding, and a high number of support tickets filed. Users who completed onboarding and connected at least one integration showed dramatically lower churn probability.
Takeaway: Early product behavior is a reliable signal for long-term retention. Interventions targeted at users who have not completed onboarding within their first week could meaningfully reduce churn before it happens.
2. The LTV Gap Between Plan Tiers Justifies Differentiated Retention Investment
LTV modeling revealed a significant difference in expected lifetime value across the Starter, Pro, and Business plan tiers, driven by both monthly revenue and churn rate differences.
The Evidence: Business plan customers carried an estimated LTV that was several multiples higher than Starter plan customers, not just because of higher monthly revenue but because of a meaningfully lower churn rate — meaning they stay longer and pay more.
Takeaway: Retention investment should be tiered. Proactive outreach and dedicated success resources are most economically justified for Business plan customers, while automated in-product interventions are better suited for Starter plan at-risk accounts.
---
🔍 Methodology
---
Model Training & Technical Implementation
Feature Engineering: Built 10 input features from simulated user behavior data including login frequency, features used, integrations connected, support ticket volume, invite activity, mobile app usage, onboarding completion status, and plan type.
Data Preparation: Applied one-hot encoding to the plan type categorical variable and used `StandardScaler` to normalize features for Logistic Regression. Split data into 80% training and 20% test sets with stratified sampling to preserve the churn rate ratio.
Model Training: Trained two classifiers using scikit-learn:
Logistic Regression — interpretable baseline model; reveals the direction and magnitude of each feature's effect on churn probability.
Random Forest — ensemble model using 100 decision trees; captures non-linear relationships and produces feature importance scores.
Model Evaluation: Compared models using ROC-AUC, precision, recall, F1-score, and confusion matrix. Selected Random Forest as the production model based on superior AUC.
LTV Estimation: Applied the formula `LTV = Monthly Revenue × (1 / Churn Rate)` by plan tier, approximating expected customer lifespan using a geometric series model.
Risk Scoring: Applied the trained Random Forest model to all customers to generate individual churn probability scores, then segmented users into Low, Medium, and High risk bands.

📁 Files
---
File	Description
`churn_ltv_model.py`	Full model training and evaluation script
`outputs/05_churn_model_results.png`	ROC curves, feature importance chart, confusion matrix
`outputs/06_ltv_by_plan.png`	Estimated LTV comparison by plan tier
`outputs/churn_predictions.csv`	Churn probability score for every customer

🛠️ Tools & Libraries
---
Tool	Purpose
Python	Core programming language
pandas	Data manipulation and feature engineering
numpy	Numerical operations and data simulation
scikit-learn	Model training, evaluation, and scoring
matplotlib	Visualization of model results and LTV

📖 Glossary of Metrics
---
Churn: When a paying customer cancels their subscription or stops using the product.
ROC-AUC: A model performance metric ranging from 0.5 (random) to 1.0 (perfect). Measures how well the model ranks churned users above retained users across all possible thresholds.
Precision: Of all users the model predicted would churn, what percentage actually did churn.
Recall: Of all users who actually churned, what percentage the model correctly identified in advance.
Feature Importance: A score assigned to each input variable indicating how much it contributed to the model's predictions. Higher importance = stronger predictor of churn.
LTV (Customer Lifetime Value): The total revenue a customer is expected to generate before they churn. Calculated here as monthly revenue divided by the monthly churn rate.
Logistic Regression: A statistical model that estimates the probability of a binary outcome (churned vs. retained) based on input features.
Random Forest: An ensemble machine learning model that builds many decision trees and averages their predictions to improve accuracy and reduce overfitting.

Data Source
---
All data is synthetically generated using Python's `numpy` and `random` libraries to simulate realistic SaaS customer behavioral patterns across plan tiers.
