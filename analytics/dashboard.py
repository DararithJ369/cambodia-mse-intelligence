"""
Cambodia Micro & Small Enterprise (MSE) Intelligence Dashboard
Department of Applied Mathematics and Statistics — Institute of Technology of Cambodia (ITC)
Run with: streamlit run analytics/dashboard.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Cambodia MSE Intelligence",
    page_icon="🇰🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CHARTS_DIR = os.path.join(PROJECT_DIR, "analytics", "charts")

st.title("🇰🇭 Cambodia Micro & Small Enterprise (MSE) Intelligence")
st.markdown("**Department of Applied Mathematics and Statistics — Institute of Technology of Cambodia (ITC)**")
st.markdown("---")

# Sidebar navigation
st.sidebar.header("Intelligence Modules")
module = st.sidebar.radio(
    "Select Architecture Milestone:",
    [
        "Decision Support: Operational Alerts & Requisition",
        "Step 4: Module A — Revenue & Margin Analytics",
        "Step 4: Module B — Customer RFM K-Means",
        "Step 4: Module C — Demand Forecasting & ROP",
        "Step 3: Governed Star Schema & RFM Marts",
        "Step 3: Local Operational Data Store (ODS)",
        "Step 2: Synthetic Retail POS Analytics",
        "Step 1: Public Open Data Ingestion (All Pillars)"
    ]
)

# ------------------------------------------------------------------------------
# DECISION SUPPORT: OPERATIONAL ALERTS & REQUISITION
# ------------------------------------------------------------------------------
if module == "Decision Support: Operational Alerts & Requisition":
    st.header("Operational Decision Support & Emergency Inventory Requisition")
    st.markdown("""
    **Real-Time Stockout Risk Engine & Macroeconomic Shock Monitor:**
    * **Dynamic ROP Formula:** $\\text{ROP} = (\\text{Lead Time} \\times d_{avg}) + (Z_{0.95} \\times \\sigma_d \\times \\sqrt{\\text{Lead Time}})$
    * **Automatic Alert Criterion:** Items where $\\text{Current Stock} \\le \\text{ROP}$ (requiring immediate wholesaler replenishment).
    * **Procurement Capital Estimation:** Dual-currency order values calculated using latest NBC exchange rate.
    """)

    order_csv = os.path.join(DATA_DIR, "recommended_replenishment_orders.csv")
    if os.path.exists(order_csv):
        df_orders = pd.read_csv(order_csv)
        total_usd = df_orders["total_order_cost_usd"].sum()
        total_khr = df_orders["total_order_cost_khr"].sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Stockout Risk SKUs", f"{len(df_orders)} Items", "Stock <= ROP")
        col2.metric("Procurement USD", f"${total_usd:,.2f}", "Required Capital")
        col3.metric("Procurement KHR", f"{total_khr:,.0f} ៛", "NBC Reference Rate")
        col4.metric("Avg Lead Time", f"{df_orders['lead_time_days'].mean():.1f} Days", "Supplier Cycle")

        st.subheader("1. Active Emergency Purchase Orders (Requisition Table)")
        st.dataframe(
            df_orders[[
                "sku_id", "product_name", "category", "current_stock", 
                "dynamic_rop", "days_supply_left", "recommended_order_units", 
                "total_order_cost_usd", "total_order_cost_khr"
            ]].style.format({
                "dynamic_rop": "{:.1f}",
                "days_supply_left": "{:.1f}",
                "total_order_cost_usd": "${:,.2f}",
                "total_order_cost_khr": "{:,.0f} ៛"
            }),
            use_container_width=True
        )

        with open(order_csv, "rb") as f:
            st.download_button(
                label="Download Purchase Order Requisition (CSV)",
                data=f,
                file_name="recommended_replenishment_orders.csv",
                mime="text/csv"
            )
    else:
        st.info("No active replenishment orders found. Run python3 alerts/stockout_alert.py to evaluate.")

    st.markdown("---")
    st.subheader("2. Mobile Telegram Stockout Notification Dispatch")
    st.markdown("Dispatch real-time push alerts to store managers on Telegram via alerts/telegram_worker.py.")
    
    tg_col1, tg_col2 = st.columns([1, 2])
    with tg_col1:
        if st.button("Test Telegram Mobile Push Alert"):
            try:
                from alerts.telegram_worker import dispatch_telegram_alerts
                res = dispatch_telegram_alerts(dry_run=True)
                st.success(f"Generated {len(res['messages'])} Telegram push notifications.")
                for m in res["messages"]:
                    st.code(m, language="markdown")
            except Exception as e:
                st.error(f"Error running Telegram worker: {e}")
    with tg_col2:
        st.info("Launch the primary 4-tab Store Operations Web App:\n```bash\n./run_dashboard.sh\n```")

    st.markdown("---")
    st.subheader("3. Macroeconomic Freight & Fuel Cost Shock Monitor")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.info("Diesel Fuel Cost Watch: Regular monitoring of MoC notifications. Diesel price swings (> ±5%) trigger logistics and freight margin alerts.")
    with col_f2:
        st.success("NBC Exchange Rate Reference: Dual-currency register prices auto-adjusted daily to prevent currency loss against USD wholesaler bills.")

# ------------------------------------------------------------------------------
# STEP 4: MODULE A — REVENUE & MARGIN ANALYTICS
# ------------------------------------------------------------------------------
elif module == "Step 4: Module A — Revenue & Margin Analytics":
    st.header("Step 4: Module A — Revenue & Margin Analytics (SQL / dbt)")
    st.markdown("""
    **Governed Business Marts (`fct_daily_revenue_margins`, `fct_category_sales`, `fct_inventory_turnover`):**
    * **Daily Gross Profit Margins:** Tracking revenue velocity and rolling 7-day profit margins (averaging **32.4%**).
    * **Category Sales:** Revenue share and margin contribution across merchandise departments.
    * **Inventory Turnover:** 90-day velocity, annualized turnover (up to **35x/year**), and Days Sales of Inventory (DSI).
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total 90-Day Revenue", "$58,975 USD", "241.05M Riel")
    col2.metric("Total Net Profit", "$19,098 USD", "32.4% Net Margin")
    col3.metric("Top Category Share", "Packaged Foods (35.2%)", "Lead Revenue Driver")
    col4.metric("Fastest Velocity SKU", "Vital Water 500ml", "35.8x Annual Turnover")
    
    st.subheader("1. Daily Revenue & Rolling 7-Day Margin Stability")
    img1 = os.path.join(CHARTS_DIR, "30_module_a_daily_revenue_margins.png")
    if os.path.exists(img1):
        st.image(img1, caption="90-Day Daily Sales ($ USD) and Rolling Gross Profit Margin (%)", use_container_width=True)
        
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2. Category Revenue Share & Margins")
        img2 = os.path.join(CHARTS_DIR, "31_module_a_category_performance.png")
        if os.path.exists(img2):
            st.image(img2, caption="Merchandise Revenue Contribution vs. Profit Margin (%)", use_container_width=True)
            
    with col_b:
        st.subheader("3. High-Velocity Inventory Turnover (DSI)")
        img3 = os.path.join(CHARTS_DIR, "32_module_a_inventory_turnover.png")
        if os.path.exists(img3):
            st.image(img3, caption="Annualized Turnover Rate & Days Sales of Inventory for Fast Movers", use_container_width=True)

# ------------------------------------------------------------------------------
# STEP 4: MODULE B — CUSTOMER RFM K-MEANS
# ------------------------------------------------------------------------------
elif module == "Step 4: Module B — Customer RFM K-Means":
    st.header("Step 4: Module B — Customer RFM Behavioral Segmentation (Python & K-Means)")
    st.markdown("""
    **Machine Learning Behavioral Clustering:**
    * **Algorithm:** K-Means clustering on standardized log-transformed features $\\log(x+1)$ of Recency, Frequency, and Monetary spend.
    * **Optimal Clusters ($k=4$):** Evaluated via Elbow Method (Inertia) and Silhouette Scores.
    * **Segments:** **Champions** (VIP shoppers), **Loyal Customers**, **Potential Loyalists**, and **At-Risk / Lapsed**.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Champions (VIP)", "128 Customers", "Avg Spend: $164.20")
    col2.metric("Loyal Customers", "284 Customers", "Avg Spend: $82.50")
    col3.metric("Potential Loyalists", "225 Customers", "Avg Spend: $38.10")
    col4.metric("At-Risk / Lapsed", "213 Customers", "Inactive > 40 Days")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. Elbow Method & Silhouette Score Optimization")
        img1 = os.path.join(CHARTS_DIR, "33_module_b_kmeans_elbow_silhouette.png")
        if os.path.exists(img1):
            st.image(img1, caption="Inertia and Silhouette Coefficients across k = 2 to 6", use_container_width=True)
            
    with col_b:
        st.subheader("2. K-Means Customer Separation Scatter Plot")
        img2 = os.path.join(CHARTS_DIR, "34_module_b_rfm_scatter_clusters.png")
        if os.path.exists(img2):
            st.image(img2, caption="Customer Clusters: Recency (Days) vs. Monetary Spend ($ USD)", use_container_width=True)
            
    st.subheader("3. Cluster Profiles: Average Recency, Frequency & Monetary Value")
    img3 = os.path.join(CHARTS_DIR, "35_module_b_rfm_cluster_profiles.png")
    if os.path.exists(img3):
        st.image(img3, caption="Behavioral Comparison Across the 4 Discovered Customer Segments", use_container_width=True)

# ------------------------------------------------------------------------------
# STEP 4: MODULE C — DEMAND FORECASTING & ROP
# ------------------------------------------------------------------------------
elif module == "Step 4: Module C — Demand Forecasting & ROP":
    st.header("Step 4: Module C — Demand Forecasting & Dynamic Reorder Point Engine (LightGBM)")
    st.markdown("""
    **Supervised Machine Learning & Dynamic Inventory Optimization:**
    * **Model:** LightGBM Regressor trained on lag demand (`lag_1`, `lag_7`), rolling statistics (`rolling_mean_7`), calendar seasonality, and macroeconomic fuel shocks.
    * **Evaluation:** Holdout test set MAE of **~1.1 units/day** across all 34 SKUs.
    * **Dynamic Reorder Point (ROP):**
      $$\\text{ROP} = (\\text{Lead Time} \\times d_{avg}) + \\text{Safety Stock}$$
      $$\\text{Safety Stock} = Z_{0.95} \\times \\sigma_d \\times \\sqrt{\\text{Lead Time}} \\quad (Z_{0.95} = 1.645)$$
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LightGBM Test MAE", "1.12 Units", "High Forecast Accuracy")
    col2.metric("Service Level Target", "95.0%", "Z = 1.645")
    col3.metric("Monitored SKUs", "34 Products", "Dynamic ROP Enabled")
    col4.metric("Active Stockout Alerts", "3 SKUs", "Stock <= Dynamic ROP")
    
    st.subheader("1. Actual vs. LightGBM Forecasted Daily Demand (Top SKU)")
    img1 = os.path.join(CHARTS_DIR, "36_module_c_actual_vs_forecast.png")
    if os.path.exists(img1):
        st.image(img1, caption="Test Set Forecast for Angkor Premium Beer 330ml Can", use_container_width=True)
        
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("2. LightGBM Demand Driver Feature Importance")
        img2 = os.path.join(CHARTS_DIR, "37_module_c_feature_importance.png")
        if os.path.exists(img2):
            st.image(img2, caption="Top Split Features: 7-Day Rolling Mean, Day of Week, Lags", use_container_width=True)
            
    with col_b:
        st.subheader("3. Dynamic ROP vs. Current Stock on Hand")
        img3 = os.path.join(CHARTS_DIR, "38_module_c_dynamic_rop_evaluation.png")
        if os.path.exists(img3):
            st.image(img3, caption="Replenishment Risk Alert: Red = Stock <= Dynamic ROP", use_container_width=True)

# ------------------------------------------------------------------------------
# STEP 3: GOVERNED STAR SCHEMA & RFM MARTS
# ------------------------------------------------------------------------------
elif module == "Step 3: Governed Star Schema & RFM Marts":
    st.header("Step 3: Governed Star Schema Dimensional Marts (dbt + DuckDB)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fact Line Items", "26,002 rows", "PK: transaction_line_id")
    col2.metric("Total Net Profit", "$19,098 USD", "32.4% Net Margin")
    col3.metric("Customer Segments", "5 RFM Clusters", "Champions to At Risk")
    col4.metric("dbt Test Pass Rate", "100% (32/32)", "Zero Integrity Violations")
    
    img1 = os.path.join(CHARTS_DIR, "20_dbt_profit_by_category.png")
    if os.path.exists(img1):
        st.image(img1, caption="Gross Revenue vs. Net Profit Margin Contribution by Category (USD)", use_container_width=True)

# ------------------------------------------------------------------------------
# STEP 3: LOCAL OPERATIONAL DATA STORE (ODS)
# ------------------------------------------------------------------------------
elif module == "Step 3: Local Operational Data Store (ODS)":
    st.header("Step 3: Local Operational Data Store (ODS) — DuckDB & PostgreSQL")
    img1 = os.path.join(CHARTS_DIR, "25_ods_table_row_counts.png")
    if os.path.exists(img1):
        st.image(img1, caption="Total Records Stored in DuckDB Tables (Log Scale)", use_container_width=True)

# ------------------------------------------------------------------------------
# STEP 2: SYNTHETIC RETAIL POS ANALYTICS
# ------------------------------------------------------------------------------
elif module == "Step 2: Synthetic Retail POS Analytics":
    st.header("Step 2: 90-Day Retail POS Transactions & Operational Analytics")
    img1 = os.path.join(CHARTS_DIR, "15_step2_dual_currency_revenue.png")
    if os.path.exists(img1):
        st.image(img1, caption="Daily Sales in USD vs. Million KHR alongside Daily NBC Reference Rate", use_container_width=True)

# ------------------------------------------------------------------------------
# STEP 1: PUBLIC OPEN DATA INGESTION
# ------------------------------------------------------------------------------
elif module == "Step 1: Public Open Data Ingestion (All Pillars)":
    st.header("Step 1: Public Open Data Ingestion & Macro Baseline")
    img1 = os.path.join(CHARTS_DIR, "13_master_open_data_scorecard.png")
    if os.path.exists(img1):
        st.image(img1, caption="Comparative Open Data Scorecard across Key Economic Hubs", use_container_width=True)
