# Profit-Driven Customer Churn Prediction (Credit Cards)

End-to-end, profit-optimized customer churn prediction app for a credit card portfolio, built with Python, Streamlit, XGBoost, Plotly, and Scikit-Learn.

Instead of just maximizing accuracy or AUC, this project is designed the way a business would actually use it: to decide which customers to proactively retain based on Lifetime Value (LTV) and retention offer cost, and to maximize total profit from those decisions.

---

## Why This Project Is Different

### 1. Profit-First Optimization (Not Just Accuracy)

Most churn projects stop at "my AUC is 0.86". This app goes further:

► Uses a user-defined business payoff matrix based on:
   ▪ LTV = expected profit from keeping a customer.
   ▪ Offer Cost = cost of giving a retention incentive (discount, cashback, etc.).

► For any chosen threshold t, it computes:
   ▪ True Positives (TP): churners you correctly target.
   ▪ False Positives (FP): non-churners you unnecessarily target.
   ▪ False Negatives (FN): churners you miss and lose.

► Calculates total portfolio profit:

$$\text{Profit}(t) = TP(t) \cdot (LTV - Cost) - FP(t) \cdot Cost - FN(t) \cdot LTV$$

The app searches for the threshold that maximizes this profit, not the one that maximizes accuracy, F1, or AUC.

---

### 2. Dynamic Threshold Optimization in the UI

► User inputs or adjusts:
   ▪ Average Customer LTV
   ▪ Retention Offer Cost

► App recomputes the optimal probability threshold in real-time.

► Immediate visibility into how:
   ▪ The confusion matrix changes.
   ▪ The profit curve moves.
   ▪ The recommended targeting strategy evolves.

This makes the model feel like a decision engine, not just a static prediction tool.

---

### 3. Visual "What-If" Analysis for Analysts and Product Teams

The app offers an interactive What-If sandbox:

► Select a customer or define a synthetic profile.

► Adjust key behavioral or engagement features:
   ▪ Total_Trans_Ct — transaction count.
   ▪ Total_Revolving_Bal — revolving balance.
   ▪ Contacts_Count_12_mon — support calls.
   ▪ Months_Inactive_12_mon — recent inactivity.

► Interactive charts show:
   ▪ How churn probability changes as you move a feature.
   ▪ Which levers are most effective at reducing churn risk.

This turns the model into a storytelling tool you can use with non-technical stakeholders.

---

## Tech Stack

Core Language: Python
Modeling: XGBoost, Scikit-Learn, Probability Calibration (Isotonic)
App Layer: Streamlit
Visualization: Plotly, Chart.js
Data: BankChurners.csv-style customer-level churn dataset
Serialization: Joblib (trained model & pipeline)

Project Structure:

```
.
├── churn.py               # Streamlit app
├── BankChurners.csv       # Sample / training data
├── Churn_Prdiction.ipynb  # Training + EDA notebook
├── models/                # Saved pipelines, encoders, calibrated models
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Installation & Setup

Assuming you have Python 3.8+ installed.

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "Customer Churn Prediction"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App Locally

```bash
streamlit run churn.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### 4. Deploy on Streamlit Community Cloud

1. Push your repo to GitHub (see Deployment Guide below).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click New app and specify:
   - Repository: <your-username>/<your-repo-name>
   - Branch: main
   - Main file path: churn.py
4. Click Deploy — your app will be live in minutes.


---

## Application Overview

### 1. Executive Metrics Dashboard

► Portfolio-level churn risk distribution (pie chart showing churned vs. retained customers).
► Current vs. optimal threshold comparison with dynamic updates.
► Profit curve vs. threshold showing how profit changes across different probability thresholds.
► High-level KPIs: Total customers, Attrition rate, ROC-AUC score, and projected annual profit.
► Feature importance visualization using SHAP values.

### 2. Customer Inspector

► Customer features input form with three sections:
   ▪ Demographics: Gender, age, dependent count, education, marital status, income category.
   ▪ Account & Behavior: Card category, months on book, relationship count, credit limit, revolving balance, utilization ratio.
   ▪ Transaction Behavior: Transaction amount, transaction count, months inactive, support contacts, quarterly changes.

► Risk Assessment gauge showing churn probability with a visual semi-circle indicator.

► Decision recommendation badge:
   ▪ Green: "Low Risk - No Action Required"
   ▪ Red: "High Risk - Recommend Retention Offer"

► What-If Sensitivity Analysis: Select a feature and sweep it across a range to visualize how it impacts churn probability.

### 3. Bulk Strategy Simulator

► Decision Threshold Simulator:
   ▪ Adjust probability threshold from 0.01 (target everyone) to 0.99 (very selective).
   ▪ See real-time updates to all business metrics.

► Confusion Matrix Metrics:
   ▪ Saved Customers (TP), Wasted Offers (FP), Lost Customers (FN), True Negatives (TN).
   ▪ Color-coded cards for quick visual reference.

► Total Profit Display: Aggregated profit at the selected threshold.

► Confusion Matrix Visualization: 2x2 grid showing the breakdown of predictions vs. actual outcomes.

► Profit Breakdown Chart: Bar chart showing financial impact by outcome type (TP, FP, FN, TN).

► Ideal for scenario planning with stakeholders on different business assumptions.

### 4. Dynamic Business Parameters

► Left Sidebar with collapsible on mobile:
   ▪ Customer LTV: Set expected lifetime value of a retained customer (default: $1000).
   ▪ Retention Offer Cost: Set the cost of each retention action (default: $100).
   ▪ Optimize Threshold button: Automatically finds the profit-maximizing threshold.
   ▪ Model Info: Displays XGBoost calibration status, ROC-AUC score, and total customer count.

► All dashboard metrics update in real-time as you adjust LTV and Cost.

► Threshold optimization runs instantly to reflect the new profit-maximizing point.

---

## Business Logic: How Profit Is Calculated

At the heart of the app is a decision rule:

For each customer, if calibrated churn probability p ≥ t, we send a retention offer; otherwise, we do not.

Where t is a probability threshold chosen to maximize profit.

We define:
• LTV = expected profit from a retained customer (e.g., $500 or $1000).
• Cost = cost of the retention offer (e.g., $50 or $100).

Based on the confusion matrix, each outcome has a financial payoff:

► True Positive (TP): Customer would churn, we target them, they stay.
   Profit per TP = LTV - Cost

► False Positive (FP): Customer would not churn, we target them unnecessarily.
   Profit per FP = -Cost

► False Negative (FN): Customer would churn, we do not target, we lose them.
   Profit per FN = -LTV

► True Negative (TN): Customer would not churn, we do not target.
   Profit per TN = 0 (no cost, no additional gain)

Over N customers, total profit at threshold t is:

$$\text{Profit}(t) = TP(t) \cdot (LTV - Cost) - FP(t) \cdot Cost - FN(t) \cdot LTV$$

The app evaluates this profit across a grid of thresholds (e.g., 0.01 to 0.99) and selects the threshold that maximizes profit. This is what you see in the dashboard.

Key Points:

► If LTV is high and Cost is low, the app tends to recommend a low threshold (treat more people, tolerate more FPs) because missing churners (FNs) is very expensive.

► If LTV is modest and Cost is high, the app prefers a higher threshold (treat fewer people) to avoid overspending on non-churners.

---

## Model & Pipeline

► Base model: XGBoost classifier trained on a cleaned version of BankChurners.csv.

► Preprocessing:
   ▪ Categorical encoding (e.g., one-hot encoding of card type, education, income category).
   ▪ Numeric scaling / imputation as needed.

► Probability Calibration: Isotonic Calibration (e.g., via CalibratedClassifierCV) to convert XGBoost scores into well-calibrated probabilities.

► Scikit-Learn Pipeline to chain:
   1. Preprocessing
   2. XGBoost estimator
   3. Calibration step

The trained pipeline is saved (e.g., with Joblib) and loaded in churn.py so the Streamlit app can safely process raw inputs without data leakage.

---

## Future Improvements

► Customer-Specific LTV: Replace a single average LTV with segmented or individual LTVs (e.g., based on CLV modeling), so thresholding is even more targeted.

► Uplift Modeling: Move from churn prediction ("who will churn") to uplift prediction ("who will respond to the retention offer").

► Time-to-Churn Modeling: Use survival models to factor when churn is likely to happen, not just if.

► Model Explainability: Add SHAP-based explanations in the UI to show local feature attributions for each prediction.

► Auto-Thresholding per Segment: Derive different thresholds per segment (e.g., by product, geography, or risk band).

► A/B Testing Integration: Export decisions and outcomes into a real A/B experiment to validate profit gains.

► CSV Upload & Batch Prediction: Allow users to upload a CSV of customers and get predictions + recommendations in bulk.

► Export Reports: Generate and download Excel reports with predictions, confusion matrix, and profit breakdown.
