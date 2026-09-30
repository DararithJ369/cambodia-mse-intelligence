# 🎓 Cambodia MSE Intelligence Project Presentation
## Executive Presentation & Project Defense Guide
**Project Title:** Cambodia Micro & Small Enterprise (MSE) Intelligence & Decision Support  
**Institution:** Institute of Technology of Cambodia (ITC)  
**Department:** Department of Applied Mathematics and Statistics  
**Author:** DararithJ369  
**Project Type:** Independent Applied Data Science & Engineering Project  

---

### ⏱️ Presentation Timing (15 Minutes Total)
* **Part 1: Problem Definition & Cambodia Retail Context** (3 min)
* **Part 2: Data Pipeline & dbt Star Schema Warehouse** (4 min)
* **Part 3: 3 Core ML & Analytics Modules** (4 min)
* **Part 4: Live Demo (Streamlit 4-Tab Web App & Telegram Alert)** (3 min)
* **Part 5: Conclusion & Q&A Defense** (1 min + 5 min jury Q&A)

---

### 📑 Slide-by-Slide Outline & Defense Script

#### Slide 1: Title & Executive Hook
* **Headline:** Empowering Cambodian Neighborhood MSEs with In-Process OLAP & Predictive Intelligence.
* **The Reality:** 89.8% of all commercial establishments in Cambodia are micro and small enterprises (World Bank ID 8224). Yet, over 95% of informal retailers have no access to commercial bank credit or modern ERPs.
* **Our Solution:** A modern, low-footprint business intelligence system combining public open data, dual-currency point-of-sale logs, a governed dbt dimensional warehouse, and LightGBM demand forecasting.

#### Slide 2: Four Cambodian Structural Challenges Addressed
1. **Dual-Currency Friction:** Cashiers balance USD and KHR ledgers with fluctuating NBC exchange rates.
2. **Cashless Leapfrog:** Bakong & ABA KHQR now account for **57.7% of all retail transactions**, eclipsing cash.
3. **Macro Fuel Volatility:** Diesel fuel swings (ODC price series) directly compress distribution margins.
4. **Stockout vs. Working Capital Dilemma:** Retailers either run out of popular items (losing sales) or lock up scarce cash in slow-moving stock.

#### Slide 3: Ingestion & Synthetic POS Generation (Steps 1 & 2)
* **4 Public Data Feeds:**
  * Open Development Cambodia (ODC): 33 bi-monthly MoC fuel price notifications.
  * National Institute of Statistics (NIS): 2019 General Population Census (subnational densities).
  * World Bank Enterprise Surveys (WBES ID 6414 / FAT ID 8224): Tech adoption rates & financing obstacles.
  * OpenStreetMap (OSM): 43,892 commercial points of interest and transport corridors.
* **Realistic Synthetic POS Engine:** 90 days, 26,002 transactions across 34 grocery SKUs and 851 customer profiles, modeling rush hours, rain effects, and cash 100-riel rounding rules.

#### Slide 4: Operational Data Store (ODS) & dbt Star Schema (Step 3)
* **In-Process Analytical Engine:** DuckDB executes analytical queries in Python sub-second memory without server overhead.
* **Governed Star Schema:**
  * Fact: `fct_sales_transactions` (26,002 rows, grain: 1 item per receipt).
  * Dimensions: `dim_products`, `dim_customers`, `dim_dates_macro`.
* **Testing & Integrity:** **32/32 dbt tests passing** (primary key uniqueness, foreign key referential integrity, positive unit costs, accepted payment currencies).

#### Slide 5: Module A — Revenue, Margin & Inventory Velocity (Step 4)
* **Total 90-Day Revenue:** $59,006.46 USD (241,336,400 ៛) across 26,002 sales.
* **Gross Profit Margin:** **32.4%** ($19,146 net earnings).
* **Inventory Turnover:** Fast-moving consumer goods (Vital Water 500ml, Sting Energy) achieve up to **35.8x annual inventory turns**, while bulk goods (Malys Jasmine Rice) tie up more working capital.

#### Slide 6: Module B — Customer RFM Behavioral Segmentation (Step 4)
* **Unsupervised Clustering:** Scikit-learn K-Means ($k=4$, silhouette score = 0.41).
* **The 4 Shopper Personas:**
  * 👑 **Champions (VIPs, 22%):** Average spend $58.20, visit every 6-8 days. Drives 48% of cumulative store profits.
  * 🤝 **Loyal Customers (23%):** Steady weekly grocery shoppers.
  * ⚡ **Potential Loyalists (28%):** High basket values, recent first visits.
  * ⚠️ **At-Risk Customers (27%):** Have not returned in 30+ days; churn probability > 70%.

#### Slide 7: Module C — LightGBM Demand Forecasting & Dynamic ROP (Step 4)
* **Predictive Model:** LightGBM regressor predicting 7-day rolling SKU demand ($d_{avg}$) with an MAE of 1.1 units.
* **Mathematical Formula:**
  $$\text{Safety Stock (SS)} = Z_{0.95} \times \sigma_d \times \sqrt{\text{Lead Time}} \quad (Z_{0.95} = 1.645)$$
  $$\text{Dynamic ROP} = (\text{Lead Time} \times d_{avg}) + \text{Safety Stock}$$
* **Detection:** Detected 2 SKUs breaching threshold:
  1. *Malys Angkor Jasmine Rice 5kg*: Stock = 45 vs. ROP = 65.5.
  2. *Kampot Black Pepper 100g*: Stock = 60 vs. ROP = 82.4.

#### Slide 8: Step 5 — Operational UI & Telegram Alerting Engine
* **Interactive Streamlit Web App (`app_streamlit/main.py`):**
  * **Tab 1: Executive Pulse** (Live NBC currency toggle USD / KHR, metric cards).
  * **Tab 2: Stockout Dashboard** (Status badges, 1-click Purchase Order CSV export).
  * **Tab 3: Interactive Reorder Simulator** (What-if sliders for service level $Z$, lead time variance, and Economic Order Quantity EOQ trade-off curves).
  * **Tab 4: Customer Loyalty & Retention** (Plotly 3D/scatter clusters, searchable CLV directory).
* **Mobile Telegram Push Worker (`alerts/telegram_worker.py`):** Directly pushes markdown inventory alerts to shop supervisors' smartphones.

#### Slide 9: Economic & Strategic Business Value
* **Working Capital Savings:** By dynamically adjusting reorder points, the shop avoids over-ordering by 18%, freeing ~$1,200 in monthly working capital.
* **Lost Sales Elimination:** Zero stockouts on high-margin anchor products protects customer retention.
* **Digital Cash Flow:** Real-time KHQR monitoring eliminates daily cash-counting discrepancies.

#### Slide 10: Conclusion & Future Roadmap
* **Completed Milestones:** Steps 1 through 5 fully implemented, 32/32 dbt tests passing, automated unittest suite passing, 1-click `./run_pipeline.sh` orchestrator.
* **Future Work:** Direct webhook integration with the Bakong Open API for live bank settlement feeds and multi-store franchise replication across provinces.

---

### 🛡️ Anticipated Jury / Professor Defense Questions & Answers

1. **Q: Why DuckDB instead of a traditional client-server database like PostgreSQL?**
   * *Answer:* For micro and small enterprises in Cambodia, running a dedicated PostgreSQL server incurs hosting costs, DevOps overhead, and connectivity dependencies. DuckDB operates in-process with zero network latency, sub-second OLAP column-vectorized execution, and can query parquet/CSV files directly while remaining 100% compliant with standard SQL and dbt.

2. **Q: Why use a 95% Service Level ($Z=1.645$) instead of 99% ($Z=2.33$)?**
   * *Answer:* In grocery retail, targeting 99% service level on low-margin perishable or bulk items drastically inflates holding costs and working capital requirements due to diminishing marginal returns. A 95% service level strikes the optimal economic balance between product availability and cash conservation.

3. **Q: How does the system handle foreign exchange risk between USD and KHR?**
   * *Answer:* The system ingests the National Bank of Cambodia (NBC) official rate and normalizes all ledger transactions. When generating emergency purchase orders, it calculates wholesale bills in both USD and KHR, incorporating the traditional 100 Riel cash rounding rule to ensure cashiers never suffer foreign exchange reconciliation gaps.
