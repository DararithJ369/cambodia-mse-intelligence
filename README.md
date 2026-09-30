# 🇰🇭 Cambodia Micro & Small Enterprise (MSE) Intelligence

**Independent Research & Applied Engineering Project**  
**Institute of Technology of Cambodia (ITC)**  
*Department of Applied Mathematics and Statistics*

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Process_OLAP-FFF000.svg)](https://duckdb.org/)
[![dbt](https://img.shields.io/badge/dbt-1.12.5-FF694B.svg)](https://www.getdbt.com/)
[![LightGBM](https://img.shields.io/badge/Machine_Learning-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/dbt_Tests-32%2F32_Passed-success.svg)](#)

---

## 📋 Executive Overview

In Cambodia, Micro and Small Enterprises (MSEs) comprise over **89.8% of all commercial establishments** (World Bank Catalog ID 8224), anchoring employment and neighborhood retail distribution. However, local MSEs face distinct structural friction:
1. **Dual-Currency Friction:** Transactions occur simultaneously in **US Dollars (USD)** and **Khmer Riel (KHR)** with daily exchange rate fluctuations from the National Bank of Cambodia (NBC).
2. **The Cashless Leapfrog:** Digital QR payments (**ABA KHQR & Bakong**) have surged to over **58% of checkout volume**, rapidly eclipsing physical cash.
3. **Macroeconomic Fuel & Supply Shocks:** Diesel fuel volatility (swinging from 4,000 to 8,200 KHR/L) squeezes freight margins for inter-provincial merchandise delivery.
4. **Informal Financing Barriers:** Over 95% of informal MSEs lack access to formal commercial bank credit (World Bank Catalog ID 6414), making transaction-backed inventory optimization essential to prevent working capital lockup.

**Cambodia MSE Intelligence** is an end-to-end analytics and machine learning platform that ingests public macroeconomic data, synthesizes local retail transaction logs, structures data into a governed **dbt Star Schema**, and trains machine learning models to optimize pricing margins, behavioral customer segmentation, and dynamic inventory replenishment.

---

## 🏛️ End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               1. PUBLIC OPEN DATA INGESTION                            │
│  - Retail Fuel Prices (ODC)            - Subnational Census 2019 (NIS)                 │
│  - Enterprise Survey (WB ID 8224/6414) - Commercial POIs & Highways (OpenStreetMap)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                         2. REALISTIC SYNTHETIC POS GENERATOR                           │
│  - 90-Day Retail POS Transactions     - Dual-Currency Ledger (USD & KHR)               │
│  - ABA KHQR / Bakong / Cash Splits    - 34 Product SKUs & 851 Customer Profiles        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                    3. OPERATIONAL DATA STORE & dbt STAR SCHEMA                         │
│  - Engine: DuckDB / PostgreSQL        - Star Schema Marts:                             │
│  - 32/32 dbt Tests Passing              * fct_sales_transactions, dim_products         │
│                                         * dim_customers, dim_dates_macro               │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                       4. CORE ANALYTICAL & ML MODULES                                  │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │       MODULE A        │  │        MODULE B        │  │         MODULE C          │  │
│  │   Revenue & Margins   │  │   RFM K-Means Clusters │  │  LightGBM Demand & Dynamic│  │
│  │  (SQL / dbt Marts)    │  │   (scikit-learn, k=4)  │  │  Reorder Point (ROP) Engine│ │
│  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                       5. DECISION SUPPORT & APPLICATION LAYER                          │
│  - Streamlit BI Application (analytics/dashboard.py)                                   │
│  - Automated Cost Shock & Stockout Alerts (alerts/)                                    │
│  - 11 Governed Jupyter Notebooks (analytics/notebooks/)                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
cambodia-mse-intelligence/
├── README.md                               # Project documentation & execution guide
├── requirements.txt                        # Python dependencies
├── run_pipeline.sh                         # 1-Click master pipeline execution script
├── .gitignore                              # Git exclusion rules
│
├── data/                                   # Data layer
│   ├── cambodia_mse.duckdb                 # Analytical DuckDB database (dbt warehouse)
│   ├── sme_cambodia.duckdb                 # Operational Data Store (ODS DuckDB)
│   ├── synthetic_pos/                      # Raw synthesized POS parquet & CSV logs
│   ├── sku_demand_rop_forecast.csv         # Module C demand forecasts & dynamic ROP
│   ├── customer_rfm_kmeans_clustered.csv   # Module B customer clusters
│   └── recommended_replenishment_orders.csv# Emergency procurement requisition
│
├── ingestion/                              # Data generators & ODS loaders
│   ├── load_raw_data.py                    # Open data validator
│   ├── generate_synthetic_pos.py           # 90-day POS dual-currency generator
│   ├── setup_ods_database.py               # DuckDB ODS table DDL & loader
│   └── init_ods_postgres.sql               # PostgreSQL enterprise DDL script
│
├── models_dbt/                             # Governed dbt dimensional modeling
│   ├── dbt_project.yml                     # dbt project configuration
│   ├── profiles.yml                        # DuckDB connection profile
│   └── models/                             # Staging, dimensional entities & data marts
│       ├── staging/                        # stg_sales, stg_products, stg_customers
│       ├── marts/                          # fct_sales_transactions, dim_products, etc.
│       └── schema.yml                      # 32 data integrity and constraint tests
│
├── app_streamlit/                          # User-friendly Store Operations Web App (Step 5)
│   └── main.py                             # 4-Tab Executive & Store Manager Application
│
├── analytics/                              # Presentation & reporting layer
│   ├── dashboard.py                        # Architectural Deep-Dive Streamlit Dashboard
│   ├── explore_data.py                     # CLI quick inspection tool
│   ├── charts/                             # 38 High-resolution exported chart PNGs
│   └── notebooks/                          # 11 Governed Jupyter Notebooks (01 - 11)
│
├── alerts/                                 # Operational business decision alerts
│   ├── cost_shock_alert.py                 # Macro fuel price surge monitor
│   ├── stockout_alert.py                   # Dynamic ROP stockout replenishment engine
│   └── telegram_worker.py                  # Real-time mobile push notifications for store managers
│
└── scripts/                                # Maintenance & build utilities
    └── rebuild_all_notebooks.py            # Master notebook builder & evaluator
```

---

## 📓 Governed Jupyter Notebook Catalog

All notebooks in [`analytics/notebooks/`](analytics/notebooks/) have been formatted with pre-rendered visual previews and verified to run top-to-bottom with **0 errors**:

| # | Notebook Name | Milestone | Core Analysis & Visualizations |
| :---: | :--- | :---: | :--- |
| **01** | [`01_retail_fuel_prices_odc.ipynb`](analytics/notebooks/01_retail_fuel_prices_odc.ipynb) | **Step 1.1** | ODC retail fuel history, diesel price spread, MoC notification step-change shocks. |
| **02** | [`02_subnational_population_density_nis.ipynb`](analytics/notebooks/02_subnational_population_density_nis.ipynb) | **Step 1.2** | 2019 Census population sizing, 5km retail catchment analysis, density log scatter. |
| **03** | [`03_cambodia_enterprise_firm_data_wb.ipynb`](analytics/notebooks/03_cambodia_enterprise_firm_data_wb.ipynb) | **Step 1.3** | World Bank FAT 8224 firm size tiers, tech adoption funnel, WBES 6414 informal finance. |
| **04** | [`04_geospatial_osm_points_of_interest.ipynb`](analytics/notebooks/04_geospatial_osm_points_of_interest.ipynb) | **Step 1.4** | OpenStreetMap commercial POI distributions, economic corridors, highway classifications. |
| **05** | [`05_step1_open_data_master_synthesis.ipynb`](analytics/notebooks/05_step1_open_data_master_synthesis.ipynb) | **Step 1.5** | Cross-synthesis across 4 urban hubs (Phnom Penh, Siem Reap, Battambang, Sihanoukville). |
| **06** | [`06_step2_synthetic_pos_retail_analytics.ipynb`](analytics/notebooks/06_step2_synthetic_pos_retail_analytics.ipynb) | **Step 2** | 90-day dual-currency sales, KHQR/Bakong payment share (58.2%), intraday rush hours. |
| **07** | [`07_dbt_star_schema_governed_analytics.ipynb`](analytics/notebooks/07_dbt_star_schema_governed_analytics.ipynb) | **Step 3** | dbt Star Schema marts, 32/32 tests, category gross margin waterline, day-of-week trends. |
| **08** | [`08_ods_operational_data_store.ipynb`](analytics/notebooks/08_ods_operational_data_store.ipynb) | **Step 3** | DuckDB ODS record volume check, USD vs. KHR rounding consistency, channel preferences. |
| **09** | [`09_module_a_revenue_margin_analytics.ipynb`](analytics/notebooks/09_module_a_revenue_margin_analytics.ipynb) | **Step 4** | Daily margin stability (32.4%), category revenue vs. margin, inventory turnover (DSI). |
| **10** | [`10_module_b_customer_rfm_kmeans.ipynb`](analytics/notebooks/10_module_b_customer_rfm_kmeans.ipynb) | **Step 4** | K-Means clustering ($k=4$, silhouette score = 0.41), Champions, Loyal, At-Risk profiling. |
| **11** | [`11_module_c_demand_forecasting_rop.ipynb`](analytics/notebooks/11_module_c_demand_forecasting_rop.ipynb) | **Step 4** | LightGBM SKU demand forecast (MAE: 1.1 units), dynamic ROP with 95% safety stock. |

---

## ⚡ Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies in Python 3.11+:

```bash
git clone <repo-url>
cd cambodia-mse-intelligence

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline (1-Click)
Execute the master automated bash script to ingest data, generate POS logs, compile dbt models, train ML models, rebuild all notebooks, and evaluate alerts:

```bash
./run_pipeline.sh
```

### 3. Launch the Interactive Streamlit Applications

#### Option A: Dedicated 4-Tab Store Manager Web App (Step 5 Recommended)
Designed specifically for retail supervisors and store owners to monitor real-time cash flow, prevent stockouts, simulate batch orders, and retain VIP shoppers:
```bash
python3 -m streamlit run app_streamlit/main.py
# (or if streamlit is in your PATH: streamlit run app_streamlit/main.py)
```

#### Option B: Full Architectural Milestone Dashboard
Covers deep-dive technical lineage across Step 1 to Step 4:
```bash
python3 -m streamlit run analytics/dashboard.py
```
*Access the dashboard at `http://localhost:8501`.*

### 4. Trigger Operational Business Alerts & Mobile Notifications
Test the decision support alerting engines directly from the terminal:

```bash
# 1. Evaluate Ministry of Commerce fuel price shocks
python3 alerts/cost_shock_alert.py

# 2. Evaluate inventory stockout risk & generate replenishment purchase orders
python3 alerts/stockout_alert.py

# 3. Dispatch real-time Telegram mobile push notifications to store managers
python3 alerts/telegram_worker.py
# For live dispatch: TELEGRAM_BOT_TOKEN="xxx" TELEGRAM_CHAT_ID="yyy" python3 alerts/telegram_worker.py
```

---

## 📊 Star Schema Data Dictionary

The dimensional warehouse inside [`data/cambodia_mse.duckdb`](data/cambodia_mse.duckdb) is governed by dbt:

```text
                  +--------------------------+
                  |       dim_products       |
                  +--------------------------+
                  | PK: sku_id               |
                  |     product_name_khmer   |
                  |     category_id          |
                  |     unit_cost_usd        |
                  |     current_stock_on_hand|
                  |     reorder_point_units  |
                  +------------+-------------+
                               |
                               | 1:N
                               v
+-----------------------+     +--------------------------+     +-----------------------+
|     dim_customers     |     |  fct_sales_transactions  |     |    dim_dates_macro    |
+-----------------------+     +--------------------------+     +-----------------------+
| PK: customer_id       |<----+ FK: transaction_id       +---->| PK: date_key          |
|     customer_name     | 1:N |     line_item_id         | N:1 |     date              |
|     customer_segment  |     | FK: sku_id               |     |     nbc_official_rate |
|     recency_days      |     | FK: customer_id          |     |     fuel_price_diesel |
|     frequency_count   |     | FK: date_key             |     |     is_weekend_flag   |
|     monetary_total    |     |     quantity_sold        |     |     holiday_event_flag|
|     rfm_segment       |     |     gross_revenue_usd    |     +-----------------------+
+-----------------------+     |     cogs_usd             |
                              |     net_profit_margin_usd|
                              |     payment_method       |
                              +--------------------------+
```

* **Data Quality Testing:** Evaluated using 32 dbt schema tests (`unique`, `not_null`, `relationships`, `accepted_values`), passing with 100% compliance.

---

## 💡 Key Business Takeaways for Cambodian MSEs

1. **Dual-Currency Margin Preservation:** Over 90% of inventory purchases are settled in USD, while retail cash inflow is partially in KHR. Establishing dynamic daily NBC FX reference pricing shields retailers from currency drag.
2. **KHQR Payment Dominance:** With **58.2% of transactions** conducted via ABA KHQR and Bakong QR, retail POS checkout speed depends on instant QR generation and dual standees during the **12:00 lunch** and **18:00 evening rush periods**.
3. **Dynamic Reorder Point Optimization:** Replacing static supplier reorder thresholds with **LightGBM dynamic ROP ($\text{Lead Time} \times d_{\text{avg}} + Z_{0.95}\sigma_d\sqrt{\text{Lead Time}}$)** reduced capital tied up in slow-moving items by **~22%** while preventing stockouts of staple SKUs.
4. **Behavioral Customer Retention:** K-Means clustering revealed that **Champions and Loyal Customers (45% of customer base)** contribute **71% of total gross profit**, providing a clear business case for Telegram VIP bundling promotions.

---

## 👥 Author & Institutional Context
* **Author / Developer:** Dararith ([@DararithJ369](https://github.com/DararithJ369))
* **Institution:** Institute of Technology of Cambodia (ITC)
* **Department:** Department of Applied Mathematics and Statistics
* **Project Type:** Independent Applied Data Science & Business Intelligence Project
* **Year:** 2026
