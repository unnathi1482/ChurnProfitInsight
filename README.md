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

## Deployment Guide

### Prerequisites

► Git installed and configured on your machine.
► A GitHub account.
► Your project folder initialized with the files listed above.

### Step 1: Initialize Git

```bash
cd "path/to/Customer Churn Prediction"
git init
git add .
git commit -m "Initial commit: churn prediction Streamlit app"
git branch -M main
```

### Step 2: Create a GitHub Repository

1. Go to https://github.com and click New repository.
2. Name it (e.g., customer-churn-streamlit).
3. Do NOT initialize with README, .gitignore, or license (you already have these locally).
4. Click Create repository.

### Step 3: Push to GitHub

GitHub will show you instructions. Run:

```bash
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### Step 4: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io.
2. Sign in with your GitHub account.
3. Click New app.
4. Paste your repository URL and choose main branch, churn.py as main file.
5. Click Deploy.

Your app will be live at:
```
https://<your-repo-name>-<hash>.streamlit.app
```

---

## Key Features

### Responsive Design

The app is fully responsive and works seamlessly on all devices:

► Desktop (1200px+): Full 4-column metric layouts, fixed sidebar navigation.
► Tablet (768px–1024px): 2-column grids, responsive padding, collapsible sidebar.
► Mobile (< 768px): Single-column layouts, hamburger menu for sidebar, optimized font sizes.

All interactive components (charts, forms, buttons) are touch-friendly on mobile and tablet devices. The UI uses:

• Tailwind CSS for responsive grid systems.
• Media queries for breakpoints at 480px, 768px, and 1024px.
• Responsive typography with text-xs md:text-sm lg:text-base patterns.
• Flexible sidebar that toggles on mobile and remains fixed on desktop.

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

---

## Tech Stack & Architecture

Backend: Python 3.8+, Streamlit, XGBoost, Scikit-Learn, Pandas, NumPy
Frontend: Streamlit (no separate frontend framework)
UI/Styling: Tailwind CSS (via CDN), Chart.js, Font Awesome icons
Visualization: Built-in Chart.js for responsive charts (profit curve, attrition distribution, feature importance)
Serialization: Joblib (model artifact storage)
Deployment: Streamlit Community Cloud (GitHub-integrated)
Data: CSV-based (BankChurners dataset)

Responsive Design Implementation:

• Tailwind CSS utility classes for responsive grids (grid-cols-2 md:grid-cols-4, etc.)
• Media queries for mobile (@media (max-width: 768px), @media (max-width: 480px))
• Hamburger menu toggle for mobile sidebar navigation
• Touch-optimized inputs and buttons

---

## How to Talk About This Project in an Interview

This project is designed as a portfolio-ready, end-to-end churn solution:

You can highlight:

► How you translate model outputs into business decisions.
► How you use probability calibration to make financial math meaningful.
► How dynamic thresholding tied to LTV and cost drives actual profit.
► How the What-If analysis makes the model interpretable and actionable for non-technical stakeholders.
► How you optimized the UI for mobile and tablet devices with responsive Tailwind CSS and Chart.js.
► How you deployed to Streamlit Community Cloud for instant sharing with stakeholders.

See SUMMARY.md for a deeper Interview Prep & Logic Guide (business math, feature logic, and STAR-based Q&A).
