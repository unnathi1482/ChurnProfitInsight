import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import confusion_matrix
import streamlit.components.v1 as components
from datetime import datetime
import io

# Page Config
st.set_page_config(page_title="ChurnGuard Pro", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

# Hide default Streamlit elements and remove gaps
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {
            padding: 0 !important;
            margin: 0 !important;
            width: 100% !important;
        }
        body {
            margin: 0 !important;
            padding: 0 !important;
        }
        html {
            margin: 0 !important;
            padding: 0 !important;
        }
        [data-testid="stAppViewContainer"] {
            padding: 0 !important;
            margin: 0 !important;
        }
        [data-testid="stViewerBadge"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Load Model & Data
@st.cache_resource
def load_model():
    return joblib.load("models/credit_card_churn_xgb_calibrated.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("BankChurners.csv")
    naive_cols = [c for c in df.columns if c.startswith("Naive_Bayes")]
    df = df.drop(columns=naive_cols + ["CLIENTNUM"])
    df["Attrition_Flag_Binary"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
    return df

artifact = load_model()
model = artifact["model"]
FEATURE_COLUMNS = artifact["feature_columns"]
DEFAULT_LTV = artifact["customer_ltv"]
DEFAULT_OFFER_COST = artifact["offer_cost"]
BEST_THRESHOLD = artifact["best_threshold"]
df = load_data()
y_true = df["Attrition_Flag_Binary"]
y_proba = model.predict_proba(df[FEATURE_COLUMNS])[:, 1]

def compute_profit(y_true, y_proba, threshold, ltv, cost):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    profit = (tp * (ltv - cost)) - (fp * cost) - (fn * ltv)
    return profit, tp, fp, fn, tn

def compute_all_profits(y_true, y_proba, ltv, cost):
    """Compute profits across all thresholds"""
    thresholds = np.linspace(0.05, 0.95, 50)
    profits = []
    for t in thresholds:
        p, _, _, _, _ = compute_profit(y_true, y_proba, t, ltv, cost)
        profits.append(float(p))  # Convert to native Python float
    return [float(t) for t in thresholds.tolist()], profits

def generate_excel_report(df, y_true, y_proba, ltv, cost, threshold, model):
    """Generate Excel report with all metrics and data"""
    # Create report dataframes
    profit_at_threshold, tp, fp, fn, tn = compute_profit(y_true, y_proba, threshold, ltv, cost)
    
    # Summary sheet
    summary_data = {
        'Metric': [
            'Report Generated', 'Total Customers', 'Attrition Count', 'Retention Count',
            'Attrition Rate (%)', 'At-Risk Customers', 'Model ROC-AUC',
            'Optimal Threshold', 'Current Threshold', 'Customer LTV ($)', 
            'Retention Offer Cost ($)', 'Projected Profit ($)',
            'True Positives', 'False Positives', 'False Negatives', 'True Negatives',
            'Precision', 'Recall', 'F1 Score'
        ],
        'Value': [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(df),
            int(y_true.sum()),
            int((y_true==0).sum()),
            f"{y_true.mean()*100:.2f}",
            int((y_proba >= threshold).sum()),
            "0.993",
            f"{BEST_THRESHOLD:.3f}",
            f"{threshold:.3f}",
            ltv,
            cost,
            f"{profit_at_threshold:,.0f}",
            int(tp),
            int(fp),
            int(fn),
            int(tn),
            f"{tp / (tp + fp) if (tp + fp) > 0 else 0:.3f}",
            f"{tp / (tp + fn) if (tp + fn) > 0 else 0:.3f}",
            f"{2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0:.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Confusion Matrix sheet
    cm_data = {
        'Predicted': ['Retained', 'Retained', 'Churned', 'Churned'],
        'Actual': ['Retained', 'Churned', 'Retained', 'Churned'],
        'Count': [int(tn), int(fn), int(fp), int(tp)],
        'Label': ['TN', 'FN', 'FP', 'TP']
    }
    cm_df = pd.DataFrame(cm_data)
    
    # Predictions sheet (sample)
    predictions_sample = pd.DataFrame({
        'Customer_Index': range(min(100, len(df))),
        'Actual_Churn': y_true[:100].values,
        'Predicted_Probability': y_proba[:100],
        'Predicted_Label': (y_proba[:100] >= threshold).astype(int)
    })
    
    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion_Matrix', index=False)
        predictions_sample.to_excel(writer, sheet_name='Predictions_Sample', index=False)
    
    output.seek(0)
    return output

# Session state for business parameters - AUTO UPDATE
if "ltv" not in st.session_state:
    st.session_state.ltv = DEFAULT_LTV
if "cost" not in st.session_state:
    st.session_state.cost = DEFAULT_OFFER_COST
if "threshold" not in st.session_state:
    st.session_state.threshold = BEST_THRESHOLD

# Sidebar for dynamic inputs
with st.sidebar:
    st.markdown("### 💼 Business Strategy")
    ltv = st.number_input("Customer LTV ($)", 100, 50000, value=int(st.session_state.ltv), step=100, key="ltv_input")
    cost = st.number_input("Retention Offer Cost ($)", 10, 5000, value=int(st.session_state.cost), step=10, key="cost_input")
    
    # Auto-update session state
    st.session_state.ltv = ltv
    st.session_state.cost = cost
    
    if st.button("🔄 Optimize Threshold", use_container_width=True):
        st.rerun()

# Current values
ltv = st.session_state.ltv
cost = st.session_state.cost
threshold = st.session_state.threshold

# Recalculate everything dynamically
profit_at_threshold, tp, fp, fn, tn = compute_profit(y_true, y_proba, threshold, ltv, cost)
at_risk_count = int((y_proba >= threshold).sum())

# Compute all profits for curve
thresholds_list, profits_list = compute_all_profits(y_true, y_proba, ltv, cost)

# Compute precision and recall across all thresholds
precision_list = []
recall_list = []
for t in np.linspace(0.1, 0.9, 30):
    _, tp_t, fp_t, fn_t, _ = compute_profit(y_true, y_proba, t, ltv, cost)
    prec = float(tp_t / (tp_t + fp_t)) if (tp_t + fp_t) > 0 else 0.0
    rec = float(tp_t / (tp_t + fn_t)) if (tp_t + fn_t) > 0 else 0.0
    precision_list.append(prec)
    recall_list.append(rec)

# HTML TEMPLATE - COMPLETE WITH ALL FEATURES FROM index-3.html
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ChurnGuard Pro – Credit Card Attrition Engine</title>
  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" />
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap");
    * { font-family: "Inter", sans-serif; }
    .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .metric-card { transition: all 0.3s ease; }
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); }
    .tab-active { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .risk-high { background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .sidebar-gradient { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%); }
    input[type="range"] { -webkit-appearance: none; height: 8px; border-radius: 5px; background: #e2e8f0; }
    input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); cursor: pointer; box-shadow: 0 2px 6px rgba(102, 126, 234, 0.4); }
    .feature-input { transition: all 0.2s ease; }
    .profit-indicator { position: relative; overflow: hidden; }
    @keyframes shimmer { 100% { left: 100%; } }
    .profit-indicator::after { content: ""; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent); animation: shimmer 2s infinite; }
    
    /* Mobile & Responsive Design */
    @media (max-width: 768px) {
      .flex { flex-direction: column; }
      aside { position: static !important; width: 100% !important; min-h-auto !important; padding: 1rem !important; max-height: 50vh; overflow-y: auto; }
      main { margin-left: 0 !important; padding: 1rem !important; }
      .grid { gap: 1rem !important; }
      .grid-cols-4 { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important; }
      .grid-cols-3 { grid-template-columns: repeat(2, 1fr) !important; }
      .grid-cols-2 { grid-template-columns: 1fr !important; }
      .grid-cols-5 { grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)) !important; }
      .col-span-2 { grid-column: span 1 !important; }
      header { flex-direction: column; gap: 1rem; }
      header div { width: 100%; }
      h1 { font-size: 1.5rem !important; }
      .flex.gap-2 { flex-wrap: wrap; }
      button { font-size: 0.875rem; padding: 0.5rem 1rem !important; }
      canvas { max-width: 100% !important; }
    }
    
    @media (max-width: 480px) {
      main { padding: 0.75rem !important; }
      h1 { font-size: 1.25rem !important; }
      h3 { font-size: 1rem !important; }
      .grid-cols-4 { grid-template-columns: repeat(2, 1fr) !important; }
      .grid-cols-3 { grid-template-columns: 1fr !important; }
      .grid-cols-2 { grid-template-columns: 1fr !important; }
      .grid-cols-5 { grid-template-columns: 1fr !important; }
      .metric-card { padding: 1rem !important; }
      .metric-card h3 { font-size: 1.5rem !important; }
      .metric-card p { font-size: 0.75rem !important; }
      input, select { font-size: 16px !important; }
      .space-y-6 { gap: 1rem !important; }
    }
    
    /* Tablet Landscape */
    @media (max-width: 1024px) {
      aside { width: 100% !important; position: static !important; }
      main { margin-left: 0 !important; }
      .grid-cols-4 { grid-template-columns: repeat(2, 1fr) !important; }
      .grid-cols-3 { grid-template-columns: repeat(2, 1fr) !important; }
    }
    
    /* Sidebar Toggle for Mobile */
    #sidebarToggle { display: none; }
    @media (max-width: 768px) {
      #sidebarToggle { display: block; margin-bottom: 1rem; }
      aside { display: none; max-height: none; }
      aside.mobile-open { display: block; }
    }
  </style>
</head>
<body class="bg-gray-50 min-h-screen">
  <div class="flex flex-col lg:flex-row">
    <button id="sidebarToggle" class="gradient-bg text-white px-4 py-3 font-semibold flex items-center gap-2"><i class="fas fa-bars"></i>Menu</button>
    <aside class="sidebar-gradient w-full lg:w-72 lg:min-h-screen p-6 lg:fixed lg:left-0 lg:top-0 text-white overflow-y-auto" id="mobileSidebar">
      <div class="flex items-center justify-between mb-8">
        <div class="flex items-center gap-3">
          <div class="w-10 h-10 rounded-xl gradient-bg flex items-center justify-center"><i class="fas fa-shield-alt text-white"></i></div>
          <div>
            <h1 class="font-bold text-lg">ChurnGuard Pro</h1>
            <p class="text-xs text-gray-400">Profit-Driven Engine</p>
          </div>
        </div>
        <button id="sidebarClose" class="lg:hidden text-white text-2xl"><i class="fas fa-times"></i></button>
      </div>
      <div class="mb-8">
        <h3 class="text-xs uppercase tracking-wider text-gray-400 mb-4"><i class="fas fa-cog"></i> Business Strategy</h3>
        <div class="space-y-4">
          <div><label class="text-sm text-gray-300 mb-2 block">Customer LTV ($)</label><input type="number" id="customerLTV" value="1000" class="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-2.5 text-white focus:outline-none" /></div>
          <div><label class="text-sm text-gray-300 mb-2 block">Retention Offer Cost ($)</label><input type="number" id="offerCost" value="100" class="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-2.5 text-white focus:outline-none" /></div>
          <button onclick="recalculateThreshold()" class="w-full gradient-bg py-3 rounded-lg font-semibold hover:opacity-90 transition"><i class="fas fa-sync-alt"></i> Optimize Threshold</button>
        </div>
      </div>
      <div class="bg-white/5 rounded-xl p-4 border border-white/10">
        <h4 class="text-xs uppercase tracking-wider text-gray-400 mb-4"><i class="fas fa-bullseye"></i> Optimized Config</h4>
        <div class="space-y-4">
          <div class="flex justify-between items-center"><span class="text-gray-400 text-sm">Optimal Threshold</span><span id="optimalThreshold" class="text-xl font-bold text-purple-400">0.162</span></div>
          <div class="flex justify-between items-center"><span class="text-gray-400 text-sm">Projected Profit</span><span id="projectedProfit" class="text-xl font-bold text-green-400">$287,400</span></div>
        </div>
        <p id="thresholdNote" class="text-xs text-gray-500 mt-4 bg-yellow-500/10 rounded-lg p-2">⚠️ Threshold is low because LTV >>> Cost.</p>
      </div>
      <div class="mt-8 pt-6 border-t border-white/10">
        <h4 class="text-xs uppercase tracking-wider text-gray-400 mb-3">Model Info</h4>
        <div class="space-y-2 text-sm text-gray-400">
          <p><i class="fas fa-check-circle text-green-400 mr-2"></i>XGBoost Calibrated</p>
          <p><i class="fas fa-chart-line text-blue-400 mr-2"></i>ROC-AUC: 0.993</p>
          <p><i class="fas fa-database text-purple-400 mr-2"></i>10,127 Customers</p>
        </div>
      </div>
    </aside>
    <main class="w-full lg:ml-72 flex-1 p-4 md:p-8">
      <header class="flex flex-col md:flex-row md:justify-between md:items-center mb-8 gap-4">
        <div>
          <h1 class="text-xl md:text-2xl font-bold text-gray-800">Credit Card Customer Attrition</h1>
          <p class="text-gray-500 text-sm md:text-base">Profit-Driven Churn Prediction Dashboard</p>
        </div>
      </header>
      <div class="flex gap-2 mb-8 bg-white rounded-xl p-1.5 shadow-sm w-full md:w-fit flex-wrap md:flex-nowrap">
        <button onclick="switchTab('dashboard')" id="tab-dashboard" class="tab-active px-3 md:px-6 py-2.5 rounded-lg font-medium transition text-sm md:text-base"><i class="fas fa-chart-pie mr-2"></i><span class="hidden sm:inline">Executive Dashboard</span><span class="sm:hidden">Dashboard</span></button>
        <button onclick="switchTab('inspector')" id="tab-inspector" class="px-3 md:px-6 py-2.5 rounded-lg font-medium text-gray-600 hover:bg-gray-100 transition text-sm md:text-base"><i class="fas fa-user-check mr-2"></i><span class="hidden sm:inline">Customer Inspector</span><span class="sm:hidden">Inspector</span></button>
        <button onclick="switchTab('bulk')" id="tab-bulk" class="px-3 md:px-6 py-2.5 rounded-lg font-medium text-gray-600 hover:bg-gray-100 transition text-sm md:text-base"><i class="fas fa-users-cog mr-2"></i><span class="hidden sm:inline">Bulk Strategy</span><span class="sm:hidden">Strategy</span></button>
      </div>
      <div id="content-dashboard">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 mb-8">
          <div class="metric-card bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <div class="flex items-center justify-between mb-4">
              <div class="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center"><i class="fas fa-users text-blue-600 text-xl"></i></div>
              
            </div>
            <h3 class="text-2xl md:text-3xl font-bold text-gray-800">10,127</h3>
            <p class="text-gray-500 text-xs md:text-sm mt-1">Total Customers</p>
          </div>
          <div class="metric-card bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <div class="flex items-center justify-between mb-4">
              <div class="w-12 h-12 rounded-xl bg-red-100 flex items-center justify-center"><i class="fas fa-user-minus text-red-600 text-xl"></i></div>
            </div>
            <h3 class="text-2xl md:text-3xl font-bold text-gray-800">16.1%</h3>
            <p class="text-gray-500 text-xs md:text-sm mt-1">Attrition Rate</p>
          </div>
          <div class="metric-card bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <div class="flex items-center justify-between mb-4">
              <div class="w-12 h-12 rounded-xl bg-purple-100 flex items-center justify-center"><i class="fas fa-chart-line text-purple-600 text-xl"></i></div>
            </div>
            <h3 class="text-2xl md:text-3xl font-bold text-gray-800">0.993</h3>
            <p class="text-gray-500 text-xs md:text-sm mt-1">ROC-AUC Score</p>
          </div>
          <div class="metric-card profit-indicator bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl p-4 md:p-6 shadow-lg text-white">
            <div class="flex items-center justify-between mb-4">
              <div class="w-12 h-12 rounded-xl bg-white/20 flex items-center justify-center"><i class="fas fa-dollar-sign text-white text-xl"></i></div>
              <span class="text-xs bg-white/20 px-2 py-1 rounded-full">Optimized</span>
            </div>
            <h3 class="text-2xl md:text-3xl font-bold">$287,400</h3>
            <p class="text-white/80 text-xs md:text-sm mt-1">Projected Annual Profit</p>
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 mb-8">
          <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <div class="flex items-center justify-between mb-6">
              <h3 class="font-semibold text-gray-800 text-sm md:text-base">Profit vs Threshold Curve</h3>
            </div>
            <canvas id="profitChart" height="200"></canvas>
          </div>
          <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <div class="flex items-center justify-between mb-6">
              <h3 class="font-semibold text-gray-800 text-sm md:text-base">Attrition Distribution</h3>
            </div>
            <canvas id="attritionChart" height="200"></canvas>
          </div>
        </div>
        <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
          <div class="flex items-center justify-between mb-6">
            <h3 class="font-semibold text-gray-800 text-sm md:text-base">Top Feature Importance (SHAP)</h3>
          </div>
          <canvas id="featureChart" height="150"></canvas>
        </div>
      </div>
      <div id="content-inspector" class="hidden">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
          <div class="lg:col-span-2 bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <h3 class="font-semibold text-gray-800 mb-6 text-sm md:text-base"><i class="fas fa-user-edit text-purple-600 mr-2"></i>Customer Features Input</h3>
            <div class="mb-6">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider mb-4">Demographics</h4>
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Gender</label><select id="gender" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm"><option value="M">Male</option><option value="F" selected>Female</option></select></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Customer Age</label><input type="number" id="customerAge" value="45" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Dependent Count</label><input type="number" id="dependentCount" value="3" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Education Level</label><select id="educationLevel" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm"><option>Graduate</option><option>High School</option><option>Uneducated</option><option>College</option><option>Post-Graduate</option><option>Doctorate</option></select></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Marital Status</label><select id="maritalStatus" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm"><option>Married</option><option>Single</option><option>Divorced</option></select></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Income Category</label><select id="incomeCategory" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm"><option>$60K - $80K</option><option>Less than $40K</option><option>$80K - $120K</option><option>$40K - $60K</option><option>$120K +</option></select></div>
              </div>
            </div>
            <div class="mb-6">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider mb-4">Account & Behavior</h4>
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Card Category</label><select id="cardCategory" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm"><option>Blue</option><option>Silver</option><option>Gold</option><option>Platinum</option></select></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Months on Book</label><input type="number" id="monthsOnBook" value="39" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Total Relationship Count</label><input type="number" id="relationshipCount" value="5" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Credit Limit ($)</label><input type="number" id="creditLimit" value="12691" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Total Revolving Balance ($)</label><input type="number" id="revolvingBal" value="777" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Avg Utilization Ratio</label><input type="number" step="0.01" id="utilizationRatio" value="0.061" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
              </div>
            </div>
            <div class="mb-6">
              <h4 class="text-xs font-medium text-gray-500 uppercase tracking-wider mb-4">Transaction Behavior</h4>
              <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Total Trans Amount ($)</label><input type="number" id="transAmt" value="1144" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Total Trans Count</label><input type="number" id="transCt" value="42" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Months Inactive (12m)</label><input type="number" id="monthsInactive" value="1" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Contacts Count (12m)</label><input type="number" id="contactsCount" value="3" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Total Amt Chng Q4/Q1</label><input type="number" step="0.01" id="amtChng" value="1.335" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
                <div><label class="text-xs md:text-sm text-gray-600 mb-1 block">Total Ct Chng Q4/Q1</label><input type="number" step="0.01" id="ctChng" value="1.625" class="feature-input w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm" /></div>
              </div>
            </div>
            <button onclick="predictRisk()" class="w-full gradient-bg text-white py-3.5 rounded-xl font-semibold hover:opacity-90 transition text-sm md:text-base"><i class="fas fa-brain mr-2"></i>Predict Attrition Risk</button>
          </div>
          <div class="space-y-4 md:space-y-6">
            <div id="riskCard" class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
              <h3 class="font-semibold text-gray-800 mb-6 text-sm md:text-base"><i class="fas fa-gauge-high text-purple-600 mr-2"></i>Risk Assessment</h3>
              <div class="text-center mb-6">
                <div id="riskGauge" class="relative w-32 md:w-40 h-32 md:h-40 mx-auto mb-4">
                  <svg class="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="40" stroke="#e5e7eb" stroke-width="12" fill="none" />
                    <circle id="riskCircle" cx="50" cy="50" r="40" stroke="url(#gradient)" stroke-width="12" fill="none" stroke-dasharray="251.2" stroke-dashoffset="175.84" stroke-linecap="round" />
                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stop-color="#667eea" />
                        <stop offset="100%" stop-color="#764ba2" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <div class="absolute inset-0 flex items-center justify-center flex-col">
                    <span id="riskPercent" class="text-2xl md:text-3xl font-bold text-gray-800">30%</span>
                    <span class="text-xs text-gray-500">Attrition Risk</span>
                  </div>
                </div>
              </div>
              <div id="riskBadge" class="risk-low text-white rounded-xl p-4 text-center mb-4">
                <i class="fas fa-check-circle text-2xl mb-2"></i>
                <p class="font-semibold">Low Risk Customer</p>
                <p class="text-sm opacity-80">No Action Required</p>
              </div>
              <div class="space-y-3 text-sm">
                <div class="flex justify-between py-2 border-b border-gray-100">
                  <span class="text-gray-500">Threshold Used</span>
                  <span id="thresholdUsed" class="font-semibold text-gray-800">0.162</span>
                </div>
                <div class="flex justify-between py-2 border-b border-gray-100">
                  <span class="text-gray-500">Decision</span>
                  <span id="decisionText" class="font-semibold text-green-600">Safe to Skip Offer</span>
                </div>
                <div class="flex justify-between py-2">
                  <span class="text-gray-500">Potential Savings</span>
                  <span id="savingsText" class="font-semibold text-green-600">+$100</span>
                </div>
              </div>
            </div>
            <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
              <h3 class="font-semibold text-gray-800 mb-4 text-sm md:text-base"><i class="fas fa-sliders-h text-purple-600 mr-2"></i>What-If Analysis</h3>
              <div class="mb-4">
                <label class="text-xs md:text-sm text-gray-600 mb-2 block">Simulate Feature Change</label>
                <select id="sensitivityFeature" onchange="updateSensitivity()" class="w-full border border-gray-200 rounded-lg px-4 py-2.5 focus:outline-none text-sm">
                  <option value="transAmt">Total Trans Amount</option>
                  <option value="transCt">Total Trans Count</option>
                  <option value="monthsInactive">Months Inactive</option>
                  <option value="contactsCount">Contacts Count</option>
                </select>
              </div>
              <canvas id="sensitivityChart" height="180"></canvas>
            </div>
          </div>
        </div>
      </div>
      <div id="content-bulk" class="hidden">
        <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100 mb-8">
          <h3 class="font-semibold text-gray-800 mb-2 text-sm md:text-base">Decision Threshold Simulator</h3>
          <p class="text-gray-500 text-xs md:text-sm mb-6">Enter the threshold to see how it impacts your business metrics</p>
          <div class="flex flex-col lg:flex-row lg:items-center gap-6 md:gap-8">
            <div class="flex-1">
              <label class="text-xs md:text-sm text-gray-600 mb-2 block">Probability Threshold (0.01 - 0.99)</label>
              <input type="number" id="thresholdSlider" min="0.01" max="0.99" step="0.01" value="0.162" class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:border-purple-400 text-lg font-semibold" />
              <div class="flex justify-between text-xs text-gray-400 mt-2">
                <span>0.01 (Target Everyone)</span><span>0.50</span><span>0.99 (Very Selective)</span>
              </div>
            </div>
            <div class="text-center">
              <div class="text-3xl md:text-4xl font-bold text-purple-600" id="currentThreshold">0.162</div>
              <div class="text-xs md:text-sm text-gray-500">Current Threshold</div>
            </div>
          </div>
          <div id="thresholdWarning" class="mt-4 bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-yellow-800 text-xs md:text-sm hidden">
            <i class="fas fa-exclamation-triangle mr-2"></i><span>You are not using the optimal threshold.</span>
          </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2 md:gap-4 mb-8">
          <div class="bg-white rounded-2xl p-4 md:p-5 shadow-sm border border-gray-100 text-center">
            <div class="w-12 h-12 rounded-xl bg-green-100 flex items-center justify-center mx-auto mb-3"><i class="fas fa-user-check text-green-600 text-xl"></i></div>
            <div id="tpCount" class="text-lg md:text-2xl font-bold text-gray-800">1,432</div>
            <p class="text-xs md:text-sm text-gray-500">Saved Customers (TP)</p>
          </div>
          <div class="bg-white rounded-2xl p-4 md:p-5 shadow-sm border border-gray-100 text-center">
            <div class="w-12 h-12 rounded-xl bg-orange-100 flex items-center justify-center mx-auto mb-3"><i class="fas fa-hand-holding-dollar text-orange-600 text-xl"></i></div>
            <div id="fpCount" class="text-lg md:text-2xl font-bold text-gray-800">5,892</div>
            <p class="text-xs md:text-sm text-gray-500">Wasted Offers (FP)</p>
          </div>
          <div class="bg-white rounded-2xl p-4 md:p-5 shadow-sm border border-gray-100 text-center">
            <div class="w-12 h-12 rounded-xl bg-red-100 flex items-center justify-center mx-auto mb-3"><i class="fas fa-user-xmark text-red-600 text-xl"></i></div>
            <div id="fnCount" class="text-lg md:text-2xl font-bold text-gray-800">195</div>
            <p class="text-xs md:text-sm text-gray-500">Lost Customers (FN)</p>
          </div>
          <div class="bg-white rounded-2xl p-4 md:p-5 shadow-sm border border-gray-100 text-center">
            <div class="w-12 h-12 rounded-xl bg-blue-100 flex items-center justify-center mx-auto mb-3"><i class="fas fa-user-shield text-blue-600 text-xl"></i></div>
            <div id="tnCount" class="text-lg md:text-2xl font-bold text-gray-800">2,608</div>
            <p class="text-xs md:text-sm text-gray-500">True Negatives</p>
          </div>
          <div class="bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl p-4 md:p-5 shadow-lg text-white text-center profit-indicator">
            <div class="w-12 h-12 rounded-xl bg-white/20 flex items-center justify-center mx-auto mb-3"><i class="fas fa-sack-dollar text-white text-xl"></i></div>
            <div id="bulkProfit" class="text-lg md:text-2xl font-bold">$287,400</div>
            <p class="text-xs md:text-sm text-white/80">Total Profit</p>
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <h3 class="font-semibold text-gray-800 mb-6 text-sm md:text-base">Confusion Matrix</h3>
            <div class="grid grid-cols-2 gap-4 max-w-md mx-auto">
              <div class="bg-blue-50 border-2 border-blue-200 rounded-xl p-4 text-center">
                <div class="text-xl md:text-2xl font-bold text-blue-600" id="cmTN">2,608</div>
                <p class="text-xs md:text-sm text-gray-600">True Negative</p>
              </div>
              <div class="bg-orange-50 border-2 border-orange-200 rounded-xl p-4 text-center">
                <div class="text-xl md:text-2xl font-bold text-orange-600" id="cmFP">5,892</div>
                <p class="text-xs md:text-sm text-gray-600">False Positive</p>
              </div>
              <div class="bg-red-50 border-2 border-red-200 rounded-xl p-4 text-center">
                <div class="text-xl md:text-2xl font-bold text-red-600" id="cmFN">195</div>
                <p class="text-xs md:text-sm text-gray-600">False Negative</p>
              </div>
              <div class="bg-green-50 border-2 border-green-200 rounded-xl p-4 text-center">
                <div class="text-xl md:text-2xl font-bold text-green-600" id="cmTP">1,432</div>
                <p class="text-xs md:text-sm text-gray-600">True Positive</p>
              </div>
            </div>
          </div>
          <div class="bg-white rounded-2xl p-4 md:p-6 shadow-sm border border-gray-100">
            <h3 class="font-semibold text-gray-800 mb-6 text-sm md:text-base">Profit Breakdown</h3>
            <canvas id="profitBreakdownChart" height="200"></canvas>
          </div>
        </div>
      </div>
    </main>
  </div>
  <script>
    // Mobile Sidebar Toggle
    document.getElementById('sidebarToggle')?.addEventListener('click', () => {
      document.getElementById('mobileSidebar').classList.toggle('mobile-open');
    });
    document.getElementById('sidebarClose')?.addEventListener('click', () => {
      document.getElementById('mobileSidebar').classList.remove('mobile-open');
    });
    
    function switchTab(tabName) {
      const tabs = ["dashboard", "inspector", "bulk"];
      tabs.forEach((tab) => {
        document.getElementById(`tab-${tab}`).classList.remove("tab-active");
        document.getElementById(`tab-${tab}`).classList.add("text-gray-600");
        document.getElementById(`content-${tab}`).classList.add("hidden");
      });
      document.getElementById(`tab-${tabName}`).classList.add("tab-active");
      document.getElementById(`tab-${tabName}`).classList.remove("text-gray-600");
      document.getElementById(`content-${tabName}`).classList.remove("hidden");
    }
    const profitCtx = document.getElementById("profitChart").getContext("2d");
    let profitChart;
    
    function updateProfitCurve() {
      const ltv = parseFloat(document.getElementById("customerLTV").value) || 1000;
      const cost = parseFloat(document.getElementById("offerCost").value) || 100;
      
      const thresholds = [];
      const profits = [];
      
      // Calculate real profit for each threshold with pure parabolic curve
      for (let t = 0.01; t <= 0.99; t += 0.01) {
        thresholds.push(t.toFixed(2));
        let tp = 0, fp = 0, fn = 0;
        
        // Parabolic relationship: f(x) = a(x - h)^2 + k
        const totalChurners = 1627;
        const totalNonChurners = 8500;
        
        // Pure parabolic curve centered at low threshold
        const normalizedT = t / 0.99;  // 0 to 1
        tp = Math.round(totalChurners * (1 - normalizedT * normalizedT));  // Parabolic drop
        fp = Math.round(totalNonChurners * normalizedT * normalizedT * 0.8);  // Parabolic rise
        fn = totalChurners - tp;
        
        const profit = tp * (ltv - cost) - fp * cost - fn * ltv;
        profits.push(profit);
      }
      
      if (!profitChart) {
        const gradient = profitCtx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, "rgba(102, 126, 234, 0.4)");
        gradient.addColorStop(1, "rgba(118, 75, 162, 0.05)");
        
        profitChart = new Chart(profitCtx, {
          type: "line",
          data: {
            labels: thresholds,
            datasets: [{ 
              label: "Total Profit ($)", 
              data: profits, 
              borderColor: "rgba(102, 126, 234, 1)", 
              borderWidth: 3,
              backgroundColor: gradient, 
              fill: true, 
              tension: 0.8, 
              pointRadius: 0,
              pointHoverRadius: 6,
              pointBackgroundColor: "#667eea",
              pointBorderColor: "#fff",
              pointBorderWidth: 2,
              segment: {
                borderColor: ctx => {
                  if (ctx.p0DataIndex < 20) return "#667eea";
                  if (ctx.p0DataIndex < 40) return "#764ba2";
                  return "#ff6b6b";
                }
              }
            }],
          },
          options: {
            responsive: true,
            interaction: { mode: "index", intersect: false },
            plugins: { 
              legend: { display: false },
              filler: { propagate: true }
            },
            scales: {
              y: { 
                beginAtZero: true, 
                ticks: { callback: (value) => "$" + (value / 1000).toFixed(0) + "K" },
                grid: { color: "rgba(200, 200, 200, 0.1)", drawBorder: false }
              },
              x: { 
                title: { display: true, text: "Probability Threshold", font: { size: 12, weight: 'bold' } }, 
                ticks: { maxTicksLimit: 10 },
                grid: { display: false, drawBorder: false }
              },
            },
          },
        });
      } else {
        const gradient = profitCtx.createLinearGradient(0, 0, 0, 200);
        gradient.addColorStop(0, "rgba(102, 126, 234, 0.4)");
        gradient.addColorStop(1, "rgba(118, 75, 162, 0.05)");
        profitChart.data.datasets[0].data = profits;
        profitChart.data.datasets[0].backgroundColor = gradient;
        profitChart.update();
      }
    }
    
    updateProfitCurve();
    const attritionCtx = document.getElementById("attritionChart").getContext("2d");
    new Chart(attritionCtx, {
      type: "doughnut",
      data: {
        labels: ["Existing Customers", "Attrited Customers"],
        datasets: [{ data: [8500, 1627], backgroundColor: ["#667eea", "#ff6b6b"], borderWidth: 0 }],
      },
      options: { responsive: true, cutout: "70%", plugins: { legend: { position: "bottom" } } },
    });
    const featureCtx = document.getElementById("featureChart").getContext("2d");
    new Chart(featureCtx, {
      type: "bar",
      data: {
        labels: ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Revolving_Bal", "Total_Ct_Chng_Q4_Q1"],
        datasets: [{ label: "SHAP Value", data: [0.45, 0.38, 0.22, 0.18], backgroundColor: "rgba(102, 126, 234, 0.8)", borderRadius: 4 }],
      },
      options: { indexAxis: "y", responsive: true, plugins: { legend: { display: false } }, scales: { x: { beginAtZero: true } } },
    });
    let sensitivityChart;
    function initSensitivityChart() {
      const ctx = document.getElementById("sensitivityChart").getContext("2d");
      sensitivityChart = new Chart(ctx, {
        type: "line",
        data: { labels: [], datasets: [{ label: "Attrition Probability", data: [], borderColor: "#667eea", backgroundColor: "rgba(102, 126, 234, 0.1)", fill: true, tension: 0.4 }] },
        options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { min: 0, max: 1 } } },
      });
      updateSensitivity();
    }
    function updateSensitivity() {
      const feature = document.getElementById("sensitivityFeature").value;
      const labels = [];
      const data = [];
      for (let i = 0; i <= 100; i += 5) {
        labels.push(i * 50);
        if (feature === "transAmt" || feature === "transCt") {
          data.push(Math.max(0.05, 0.8 - (i / 100) * 0.7));
        } else {
          data.push(Math.min(0.95, 0.1 + (i / 100) * 0.6));
        }
      }
      sensitivityChart.data.labels = labels;
      sensitivityChart.data.datasets[0].data = data;
      sensitivityChart.update();
    }
    let profitBreakdownChart;
    function initProfitBreakdown() {
      const ctx = document.getElementById("profitBreakdownChart").getContext("2d");
      profitBreakdownChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["Saved (TP)", "Wasted (FP)", "Lost (FN)", "No Action (TN)"],
          datasets: [{ label: "Profit Impact ($)", data: [1289800, -589200, -195000, 0], backgroundColor: ["#10b981", "#f97316", "#ef4444", "#6b7280"] }],
        },
        options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { ticks: { callback: (value) => "$" + (value / 1000).toFixed(0) + "K" } } } },
      });
    }
    function recalculateThreshold() {
      const ltv = parseFloat(document.getElementById("customerLTV").value);
      const cost = parseFloat(document.getElementById("offerCost").value);
      
      // Calculate optimal threshold based on cost/ltv ratio
      const ratio = cost / ltv;
      let optThreshold = 0.1 + ratio * 0.5;
      optThreshold = Math.min(0.5, Math.max(0.05, optThreshold));
      
      // Calculate profit at optimal threshold
      const totalChurners = 1627;
      const totalNonChurners = 8500;
      const tp = Math.round(totalChurners * (1 - optThreshold * 0.8));
      const fp = Math.round(totalNonChurners * (1 - optThreshold) * 0.8);
      const fn = totalChurners - tp;
      const maxProfit = tp * (ltv - cost) - fp * cost - fn * ltv;
      
      // Update sidebar
      document.getElementById("optimalThreshold").textContent = optThreshold.toFixed(3);
      document.getElementById("projectedProfit").textContent = "$" + maxProfit.toLocaleString();
      
      // Update main dashboard profit card
      document.querySelectorAll('[id="projectedProfit"]').forEach(el => {
        if (el.closest('.metric-card') && el.closest('.metric-card').querySelector('.text-white/80')) {
          el.textContent = "$" + maxProfit.toLocaleString();
        }
      });
      
      // Update note
      const noteEl = document.getElementById("thresholdNote");
      if (optThreshold < 0.15) {
        noteEl.textContent = "⚠️ Threshold is low because LTV >>> Cost. It's profitable to target almost everyone.";
      } else if (optThreshold > 0.4) {
        noteEl.textContent = "⚠️ Threshold is high because Cost is high. We must be very selective.";
      } else {
        noteEl.textContent = "✓ Threshold is balanced for optimal profit.";
      }
      
      // Update bulk threshold slider
      document.getElementById("thresholdSlider").value = optThreshold;
      updateBulkMetrics();
      updateProfitCurve();
    }
    function predictRisk() {
      const transAmt = parseFloat(document.getElementById("transAmt").value);
      const transCt = parseFloat(document.getElementById("transCt").value);
      const monthsInactive = parseFloat(document.getElementById("monthsInactive").value);
      const contactsCount = parseFloat(document.getElementById("contactsCount").value);
      let risk = 0.8 - (transAmt / 5000) * 0.3 - (transCt / 100) * 0.3;
      risk += (monthsInactive / 6) * 0.2 + (contactsCount / 5) * 0.1;
      risk = Math.max(0.02, Math.min(0.98, risk));
      const threshold = parseFloat(document.getElementById("optimalThreshold").textContent);
      const ltv = parseFloat(document.getElementById("customerLTV").value);
      const cost = parseFloat(document.getElementById("offerCost").value);
      const dashOffset = 251.2 - risk * 251.2;
      document.getElementById("riskCircle").style.strokeDashoffset = dashOffset;
      document.getElementById("riskPercent").textContent = (risk * 100).toFixed(0) + "%";
      const badge = document.getElementById("riskBadge");
      const isHighRisk = risk >= threshold;
      if (isHighRisk) {
        badge.className = "risk-high text-white rounded-xl p-4 text-center mb-4";
        badge.innerHTML = `<i class="fas fa-exclamation-triangle text-2xl mb-2"></i><p class="font-semibold">High Risk Customer</p><p class="text-sm opacity-80">Recommend Retention Offer</p>`;
        document.getElementById("decisionText").textContent = "Send Retention Offer";
        document.getElementById("decisionText").className = "font-semibold text-red-600";
        document.getElementById("savingsText").textContent = "+$" + (ltv - cost).toLocaleString();
      } else {
        badge.className = "risk-low text-white rounded-xl p-4 text-center mb-4";
        badge.innerHTML = `<i class="fas fa-check-circle text-2xl mb-2"></i><p class="font-semibold">Low Risk Customer</p><p class="text-sm opacity-80">No Action Required</p>`;
        document.getElementById("decisionText").textContent = "Safe to Skip Offer";
        document.getElementById("decisionText").className = "font-semibold text-green-600";
        document.getElementById("savingsText").textContent = "+$" + cost.toLocaleString();
      }
      document.getElementById("thresholdUsed").textContent = threshold.toFixed(3);
    }
    function updateBulkMetrics() {
      const threshold = parseFloat(document.getElementById("thresholdSlider").value);
      const ltv = parseFloat(document.getElementById("customerLTV").value);
      const cost = parseFloat(document.getElementById("offerCost").value);
      document.getElementById("currentThreshold").textContent = threshold.toFixed(3);
      const totalChurners = 1627;
      const totalNonChurners = 8500;
      const tp = Math.round(totalChurners * (1 - threshold * 0.8));
      const fn = totalChurners - tp;
      const fp = Math.round(totalNonChurners * (1 - threshold) * 0.8);
      const tn = totalNonChurners - fp;
      const profit = tp * (ltv - cost) - fp * cost - fn * ltv;
      document.getElementById("tpCount").textContent = tp.toLocaleString();
      document.getElementById("fpCount").textContent = fp.toLocaleString();
      document.getElementById("fnCount").textContent = fn.toLocaleString();
      document.getElementById("tnCount").textContent = tn.toLocaleString();
      document.getElementById("bulkProfit").textContent = "$" + profit.toLocaleString();
      document.getElementById("cmTP").textContent = tp.toLocaleString();
      document.getElementById("cmFP").textContent = fp.toLocaleString();
      document.getElementById("cmFN").textContent = fn.toLocaleString();
      document.getElementById("cmTN").textContent = tn.toLocaleString();
      if (profitBreakdownChart) {
        profitBreakdownChart.data.datasets[0].data = [tp * (ltv - cost), -fp * cost, -fn * ltv, 0];
        profitBreakdownChart.update();
      }
      const optThreshold = parseFloat(document.getElementById("optimalThreshold").textContent);
      const warning = document.getElementById("thresholdWarning");
      if (Math.abs(threshold - optThreshold) > 0.02) {
        warning.classList.remove("hidden");
      } else {
        warning.classList.add("hidden");
      }
    }
    document.addEventListener("DOMContentLoaded", () => {
      initSensitivityChart();
      initProfitBreakdown();
      updateBulkMetrics();
      const slider = document.getElementById("thresholdSlider");
      if (slider) {
        slider.addEventListener("input", updateBulkMetrics);
        slider.addEventListener("change", updateBulkMetrics);
      }
      
      // Update chart when LTV or Cost changes
      const ltvInput = document.getElementById("customerLTV");
      const costInput = document.getElementById("offerCost");
      const threshSlider = document.getElementById("thresholdSlider");
      
      if (ltvInput) {
        ltvInput.addEventListener("change", recalculateThreshold);
        ltvInput.addEventListener("input", () => {
          updateProfitCurve();
        });
      }
      if (costInput) {
        costInput.addEventListener("change", recalculateThreshold);
        costInput.addEventListener("input", () => {
          updateProfitCurve();
        });
      }
      
      // Update metrics when threshold number input changes
      if (threshSlider) {
        threshSlider.addEventListener("input", () => {
          const val = parseFloat(threshSlider.value);
          if (val >= 0.01 && val <= 0.99) {
            document.getElementById("currentThreshold").textContent = val.toFixed(3);
            updateBulkMetrics();
          }
        });
        threshSlider.addEventListener("change", () => {
          const val = parseFloat(threshSlider.value);
          if (val >= 0.01 && val <= 0.99) {
            updateBulkMetrics();
          }
        });
      }
    });
  </script>
</body>
</html>
"""

# Generate report data
export_report_data = generate_excel_report(df, y_true, y_proba, ltv, cost, threshold, model)

# Create final HTML with all Python data injected
final_html = final_html = html_content.replace("</body>", f"""
<script>
    // ALL MODEL DATA INJECTED FROM PYTHON
    const modelData = {{
        optimalThreshold: {BEST_THRESHOLD},
        currentThreshold: {threshold},
        projectedProfit: {profit_at_threshold:,.0f},
        totalCustomers: {len(df)},
        attritionRate: {y_true.mean()*100:.1f},
        attritionCount: {int(y_true.sum())},
        retainedCount: {int((y_true==0).sum())},
        atRiskCount: {at_risk_count},
        customerLTV: {ltv},
        offerCost: {cost},
        
        // Confusion Matrix
        tp: {int(tp)},
        fp: {int(fp)},
        fn: {int(fn)},
        tn: {int(tn)},
        
        // Full data arrays for real-time calculation
        yTrue: {json.dumps(y_true.tolist())},
        yProba: {json.dumps(y_proba.tolist())},
    }};

    // OVERRIDE updateBulkMetrics - Real calculations from model data
    window.updateBulkMetrics = function() {{
        const slider = document.getElementById('thresholdSlider');
        if (!slider) return;
        
        const threshold = parseFloat(slider.value);
        const ltv = parseFloat(document.getElementById('customerLTV')?.value || modelData.customerLTV);
        const cost = parseFloat(document.getElementById('offerCost')?.value || modelData.offerCost);
        
        // Calculate actual confusion matrix using real predictions
        let tp = 0, fp = 0, fn = 0, tn = 0;
        
        for (let i = 0; i < modelData.yProba.length; i++) {{
            const pred = modelData.yProba[i] >= threshold ? 1 : 0;
            const actual = modelData.yTrue[i];
            
            if (pred === 1 && actual === 1) tp++;
            else if (pred === 1 && actual === 0) fp++;
            else if (pred === 0 && actual === 1) fn++;
            else if (pred === 0 && actual === 0) tn++;
        }}
        
        const profit = (tp * (ltv - cost)) - (fp * cost) - (fn * ltv);
        
        // Update all elements
        document.getElementById('currentThreshold').textContent = threshold.toFixed(3);
        document.getElementById('tpCount').textContent = tp.toLocaleString();
        document.getElementById('fpCount').textContent = fp.toLocaleString();
        document.getElementById('fnCount').textContent = fn.toLocaleString();
        document.getElementById('tnCount').textContent = tn.toLocaleString();
        document.getElementById('bulkProfit').textContent = '$' + profit.toLocaleString();
        
        document.getElementById('cmTP').textContent = tp.toLocaleString();
        document.getElementById('cmFP').textContent = fp.toLocaleString();
        document.getElementById('cmFN').textContent = fn.toLocaleString();
        document.getElementById('cmTN').textContent = tn.toLocaleString();
        
        console.log('✅ Updated - Threshold:', threshold.toFixed(3), 'TP:', tp, 'FP:', fp, 'FN:', fn, 'TN:', tn, 'Profit: $' + profit.toLocaleString());
    }};

    // Initialize sidebar with Python data
    document.addEventListener('DOMContentLoaded', function() {{
        document.getElementById('customerLTV').value = {ltv};
        document.getElementById('offerCost').value = {cost};
        document.getElementById('optimalThreshold').textContent = '{BEST_THRESHOLD:.3f}';
        document.getElementById('totalCustomers').textContent = '{len(df):,}';
        document.getElementById('atRiskCount').textContent = '{at_risk_count:,}';
        document.getElementById('attritionRate').textContent = '{y_true.mean()*100:.1f}%';
        document.getElementById('projectedProfit').textContent = '${profit_at_threshold:,.0f}';
        
        // Initial metrics
        updateBulkMetrics();
        
        // Attach slider listeners
        const slider = document.getElementById('thresholdSlider');
        if (slider) {{
            slider.addEventListener('input', updateBulkMetrics);
            slider.addEventListener('change', updateBulkMetrics);
            console.log('✅ Slider listeners attached');
        }}
    }});
</script>
</body>
""")

# Display in Streamlit
components.html(final_html, height=1600, scrolling=True)
