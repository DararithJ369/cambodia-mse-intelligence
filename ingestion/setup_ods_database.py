"""
Cambodia MSE Intelligence - Step 3: Local Operational Data Store (ODS) Setup
Generates the 4 standardized ODS CSV files, creates DDL tables, ingests into DuckDB,
and executes post-ingestion data integrity checks.
Run with: python3 ingestion/setup_ods_database.py
"""

import os
import sqlite3
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SYNTHETIC_DIR = os.path.join(DATA_DIR, "synthetic_pos")
DUCKDB_PATH = os.path.join(DATA_DIR, "sme_cambodia.duckdb")
POSTGRES_SQL_PATH = os.path.join(PROJECT_DIR, "ingestion", "init_ods_postgres.sql")

# Category to numeric category_id mapping
CATEGORY_MAP = {
    "Beverages": 1,
    "Packaged Foods": 2,
    "Snacks": 3,
    "Personal Care": 4,
    "Household": 5,
    "Dairy & Cold": 6
}

# City to ISO 3166-2 province code mapping
PROVINCE_CODE_MAP = {
    "Phnom Penh": "KH-12",
    "Siem Reap": "KH-17",
    "Battambang": "KH-02",
    "Sihanoukville": "KH-18",
    "Kandal": "KH-08"
}

def generate_ods_csvs():
    print("="*65)
    print("STEP 3: GENERATING STANDARDIZED ODS CSV FILES")
    print("="*65)
    
    # 1. INVENTORY CSV (synthetic_inventory.csv)
    print("1. Preparing data/synthetic_inventory.csv...")
    df_prod = pd.read_csv(os.path.join(SYNTHETIC_DIR, "dim_products.csv"))
    inventory_records = []
    for _, row in df_prod.iterrows():
        cat_id = CATEGORY_MAP.get(row["category"], 99)
        cost_usd = round(float(row["cost_price_usd"]), 4)
        cost_khr = round(cost_usd * 4085.0 / 100) * 100.0
        stock = int(row["current_stock_level"])
        reorder = int(row["reorder_point"])
        safety = int(round(reorder * 0.5))
        lead_time = int(row["lead_time_days"])
        shelf_life = 14 if cat_id == 6 else (180 if cat_id in [1, 2, 3] else 730)
        
        inventory_records.append({
            "sku_id": row["sku"],
            "product_name_khmer": row["product_name"],
            "category_id": cat_id,
            "unit_cost_usd": cost_usd,
            "unit_cost_khr": cost_khr,
            "current_stock_on_hand": stock,
            "allocated_stock": int(round(stock * 0.1)),
            "reorder_point_units": reorder,
            "safety_stock_level": safety,
            "lead_time_days": lead_time,
            "lead_time_variance_days": 0.50,
            "holding_cost_per_unit_annual": 1.20,
            "stockout_event_count_30d": 0 if stock > reorder else 1,
            "shelf_life_days": shelf_life
        })
    df_inv_ods = pd.DataFrame(inventory_records)
    inv_csv_path = os.path.join(DATA_DIR, "synthetic_inventory.csv")
    df_inv_ods.to_csv(inv_csv_path, index=False)
    print(f"  -> Saved {len(df_inv_ods)} records to {inv_csv_path}")

    # 2. CUSTOMERS CSV (synthetic_customers.csv)
    print("2. Preparing data/synthetic_customers.csv...")
    df_cust = pd.read_csv(os.path.join(SYNTHETIC_DIR, "dim_customers.csv"))
    df_txns = pd.read_parquet(os.path.join(SYNTHETIC_DIR, "pos_transactions_90d.parquet"))
    
    # Calculate empirical customer aggregates
    cust_agg = df_txns.groupby("customer_id").agg({
        "transaction_id": "nunique",
        "total_amount_usd": "sum",
        "date": "max",
        "payment_method": lambda x: x.mode()[0] if not x.empty else "KHQR_ABA"
    }).reset_index()
    cust_agg.columns = ["customer_id", "frequency", "monetary", "last_date", "top_payment"]
    max_date = pd.to_datetime(df_txns["date"].max())
    cust_agg["recency"] = (max_date - pd.to_datetime(cust_agg["last_date"])).dt.days
    cust_agg_dict = cust_agg.set_index("customer_id").to_dict(orient="index")
    
    customer_records = []
    for _, row in df_cust.iterrows():
        c_id = row["customer_id"]
        p_code = PROVINCE_CODE_MAP.get(row["primary_city"], "KH-12")
        agg = cust_agg_dict.get(c_id, {"frequency": 1, "monetary": 15.0, "recency": 10, "top_payment": "KHQR_Bakong"})
        
        freq = int(agg["frequency"])
        monetary = round(float(agg["monetary"]), 2)
        recency = int(agg["recency"])
        avg_basket = round(monetary / max(1, freq), 2)
        
        # RFM Cluster integer (1: Champions, 2: Loyal, 3: Potential, 4: At Risk, 5: Hibernating)
        if recency <= 14 and freq >= 15: rfm_int = 1
        elif recency <= 30 and freq >= 10: rfm_int = 2
        elif recency <= 30 and freq < 10: rfm_int = 3
        elif recency > 30 and freq >= 8: rfm_int = 4
        else: rfm_int = 5
        
        churn_prob = round(min(1.0, max(0.0, recency / 90.0)), 2)
        clv = round(monetary * 1.5, 2)
        pref_channel = "KHQR_BAKONG" if "Bakong" in agg["top_payment"] else ("KHQR_ABA" if "ABA" in agg["top_payment"] else "CASH")

        customer_records.append({
            "customer_id": c_id,
            "registration_date": f"{row['account_created_date']} 08:00:00",
            "province_code": p_code,
            "recency_days": recency,
            "frequency_count_180d": freq,
            "monetary_total_usd": monetary,
            "preferred_payment_channel": pref_channel,
            "rfm_segment_cluster": rfm_int,
            "churn_probability_score": churn_prob,
            "historical_clv_usd": monetary,
            "predicted_clv_usd": clv,
            "avg_basket_size_usd": avg_basket
        })
    df_cust_ods = pd.DataFrame(customer_records)
    cust_csv_path = os.path.join(DATA_DIR, "synthetic_customers.csv")
    df_cust_ods.to_csv(cust_csv_path, index=False)
    print(f"  -> Saved {len(df_cust_ods)} records to {cust_csv_path}")

    # 3. MACRO EXTERNAL CSV (odc_fuel_nbc_combined.csv)
    print("3. Preparing data/odc_fuel_nbc_combined.csv...")
    df_fx = pd.read_csv(os.path.join(SYNTHETIC_DIR, "dim_fx_rates.csv"))
    df_gas = pd.read_csv(os.path.join(DATA_DIR, "Gasoline price - Gas_EN.csv"))
    df_gas["date"] = pd.to_datetime(df_gas["im_date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
    gas_dict = df_gas.set_index("date").to_dict(orient="index")
    
    macro_records = []
    for _, row in df_fx.iterrows():
        d_str = row["rate_date"]
        g_info = gas_dict.get(d_str, {"regu_gas": 4800, "diesel_gas": 5450})
        # Check Cambodian national holiday events (e.g. late September Pchum Ben / Constitution Day)
        is_holiday = d_str in ['2026-09-24', '2026-09-27', '2026-09-28', '2026-09-29']
        
        macro_records.append({
            "date_key": d_str,
            "fuel_price_regular_khr": float(g_info["regu_gas"]),
            "fuel_price_diesel_khr": float(g_info["diesel_gas"]),
            "nbc_official_rate": round(float(row["nbc_usd_khr_rate"]), 2),
            "provincial_pop_density": 3136.00, # Phnom Penh anchor benchmark
            "holiday_event_flag": is_holiday,
            "rainfall_mm_local": round(float(np.random.uniform(5.0, 30.0)), 2)
        })
    df_macro_ods = pd.DataFrame(macro_records)
    macro_csv_path = os.path.join(DATA_DIR, "odc_fuel_nbc_combined.csv")
    df_macro_ods.to_csv(macro_csv_path, index=False)
    print(f"  -> Saved {len(df_macro_ods)} records to {macro_csv_path}")

    # 4. SALES TRANSACTIONS CSV (synthetic_sales.csv)
    print("4. Preparing data/synthetic_sales.csv...")
    sales_records = []
    prod_cost_dict = dict(zip(df_inv_ods["sku_id"], df_inv_ods["unit_cost_usd"]))
    
    for _, row in df_txns.iterrows():
        t_line_id = f"{row['transaction_id']}_{row['item_line_no']}"
        sku = row["sku"]
        unit_cost = prod_cost_dict.get(sku, 0.50)
        qty = int(row["quantity"])
        unit_price_usd = round(float(row["unit_price_usd"]), 2)
        applied_fx = round(float(row["nbc_exchange_rate"]), 2)
        unit_price_khr = round((unit_price_usd * applied_fx) / 100) * 100.0
        
        # Payment Currency convention
        pay_method = row["payment_method"]
        pay_curr = "KHR" if pay_method in ["Cash_KHR", "KHQR_Bakong"] else "USD"
        
        gross_rev = round(qty * unit_price_usd, 2)
        cogs = round(qty * unit_cost, 2)
        profit = round(gross_rev - cogs, 2)
        
        sales_records.append({
            "transaction_line_id": t_line_id,
            "transaction_id": row["transaction_id"],
            "transaction_timestamp": row["timestamp"],
            "sku_id": sku,
            "customer_id": row["customer_id"],
            "quantity_sold": qty,
            "unit_selling_price_usd": unit_price_usd,
            "unit_selling_price_khr": unit_price_khr,
            "applied_exchange_rate": applied_fx,
            "payment_currency": pay_curr,
            "discount_amount_usd": 0.00,
            "gross_revenue_usd": gross_rev,
            "cogs_usd": cogs,
            "net_profit_margin_usd": profit,
            "khqr_merchant_mcc": "5411",
            "payment_settlement_status": "SETTLED"
        })
    df_sales_ods = pd.DataFrame(sales_records)
    sales_csv_path = os.path.join(DATA_DIR, "synthetic_sales.csv")
    df_sales_ods.to_csv(sales_csv_path, index=False)
    print(f"  -> Saved {len(df_sales_ods)} records to {sales_csv_path}")

    return inv_csv_path, cust_csv_path, macro_csv_path, sales_csv_path

def setup_duckdb_ods(inv_csv, cust_csv, macro_csv, sales_csv):
    print("\n" + "="*65)
    print("SETTING UP DuckDB OPERATIONAL DATA STORE (ODS)")
    print(f"Database: {DUCKDB_PATH}")
    print("="*65)
    
    con = duckdb.connect(DUCKDB_PATH)
    
    # Drop existing ODS tables if present
    con.execute("""
        DROP TABLE IF EXISTS sales_transactions;
        DROP TABLE IF EXISTS inventory_stock;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS macro_external_data;
    """)
    
    # 1. Inventory & SKU Table
    print("Creating table: inventory_stock...")
    con.execute("""
        CREATE TABLE inventory_stock (
            sku_id                       VARCHAR(64) PRIMARY KEY,
            product_name_khmer           VARCHAR(255) NOT NULL,
            category_id                  INT NOT NULL,
            unit_cost_usd                DECIMAL(10,4) NOT NULL CHECK (unit_cost_usd >= 0),
            unit_cost_khr                DECIMAL(12,2) NOT NULL CHECK (unit_cost_khr >= 0),
            current_stock_on_hand        INT NOT NULL DEFAULT 0 CHECK (current_stock_on_hand >= 0),
            allocated_stock              INT DEFAULT 0 CHECK (allocated_stock >= 0),
            reorder_point_units          INT DEFAULT 10,
            safety_stock_level           INT DEFAULT 5,
            lead_time_days               INT DEFAULT 3,
            lead_time_variance_days      DECIMAL(5,2) DEFAULT 0.5,
            holding_cost_per_unit_annual DECIMAL(8,2) DEFAULT 1.00,
            stockout_event_count_30d     INT DEFAULT 0,
            shelf_life_days              INT NULL
        );
    """)

    # 2. Customers Table
    print("Creating table: customers...")
    con.execute("""
        CREATE TABLE customers (
            customer_id                  VARCHAR(64) PRIMARY KEY,
            registration_date            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            province_code                VARCHAR(10) DEFAULT 'KH-12',
            recency_days                 INT DEFAULT 0,
            frequency_count_180d         INT DEFAULT 1,
            monetary_total_usd           DECIMAL(12,2) DEFAULT 0.00,
            preferred_payment_channel    VARCHAR(32) DEFAULT 'KHQR_BAKONG',
            rfm_segment_cluster          INT DEFAULT 0,
            churn_probability_score      DECIMAL(3,2) DEFAULT 0.00,
            historical_clv_usd           DECIMAL(10,2) DEFAULT 0.00,
            predicted_clv_usd            DECIMAL(10,2) DEFAULT 0.00,
            avg_basket_size_usd          DECIMAL(8,2) DEFAULT 0.00
        );
    """)

    # 3. Macro External Data Table
    print("Creating table: macro_external_data...")
    con.execute("""
        CREATE TABLE macro_external_data (
            date_key                     DATE PRIMARY KEY,
            fuel_price_regular_khr       DECIMAL(8,2) NULL,
            fuel_price_diesel_khr        DECIMAL(8,2) NULL,
            nbc_official_rate            DECIMAL(8,2) NOT NULL DEFAULT 4050.00,
            provincial_pop_density       DECIMAL(10,2) NULL,
            holiday_event_flag           BOOLEAN DEFAULT FALSE,
            rainfall_mm_local            DECIMAL(6,2) DEFAULT 0.0
        );
    """)

    # 4. Sales Transactions Table
    print("Creating table: sales_transactions...")
    con.execute("""
        CREATE TABLE sales_transactions (
            transaction_line_id          VARCHAR(64) PRIMARY KEY,
            transaction_id               VARCHAR(64) NOT NULL,
            transaction_timestamp        TIMESTAMP NOT NULL,
            sku_id                       VARCHAR(64) NOT NULL REFERENCES inventory_stock(sku_id),
            customer_id                  VARCHAR(64) NULL REFERENCES customers(customer_id),
            quantity_sold                INT NOT NULL CHECK (quantity_sold > 0),
            unit_selling_price_usd       DECIMAL(10,2) NOT NULL CHECK (unit_selling_price_usd >= 0),
            unit_selling_price_khr       DECIMAL(12,2) NOT NULL CHECK (unit_selling_price_khr >= 0),
            applied_exchange_rate        DECIMAL(8,2) NOT NULL DEFAULT 4050.00,
            payment_currency             VARCHAR(3) NOT NULL CHECK (payment_currency IN ('USD', 'KHR')),
            discount_amount_usd          DECIMAL(8,2) DEFAULT 0.00,
            gross_revenue_usd            DECIMAL(12,2) NOT NULL,
            cogs_usd                     DECIMAL(12,2) NOT NULL,
            net_profit_margin_usd        DECIMAL(12,2) NOT NULL,
            khqr_merchant_mcc            VARCHAR(4) DEFAULT '5999',
            payment_settlement_status    VARCHAR(20) DEFAULT 'SETTLED'
        );
    """)

    # Fast Bulk Copy / Insertion from CSVs
    print("\nLoading CSVs into DuckDB tables...")
    con.execute(f"COPY inventory_stock FROM '{inv_csv}' (HEADER TRUE);")
    con.execute(f"COPY customers FROM '{cust_csv}' (HEADER TRUE);")
    con.execute(f"COPY macro_external_data FROM '{macro_csv}' (HEADER TRUE);")
    con.execute(f"COPY sales_transactions FROM '{sales_csv}' (HEADER TRUE);")
    print("  -> Data loaded successfully into DuckDB!")

    # Post-Ingestion Data Integrity Checks
    print("\n" + "-"*65)
    print("POST-INGESTION DATA INTEGRITY CHECKS")
    print("-" * 65)
    
    # 1. Total row count check
    counts_df = con.execute("""
        SELECT 'sales_transactions' AS table_name, COUNT(*) AS row_count FROM sales_transactions
        UNION ALL
        SELECT 'inventory_stock', COUNT(*) FROM inventory_stock
        UNION ALL
        SELECT 'customers', COUNT(*) FROM customers
        UNION ALL
        SELECT 'macro_external_data', COUNT(*) FROM macro_external_data;
    """).fetchdf()
    print("1. Table Row Counts:\n", counts_df.to_string(index=False))

    # 2. Dual-currency ledger consistency verification
    fx_check_df = con.execute("""
        SELECT 
            transaction_id,
            payment_currency,
            unit_selling_price_usd,
            unit_selling_price_khr,
            applied_exchange_rate,
            ROUND(unit_selling_price_khr / applied_exchange_rate, 2) AS calculated_usd,
            ROUND(ABS(unit_selling_price_usd - (unit_selling_price_khr / applied_exchange_rate)), 4) AS delta_usd
        FROM sales_transactions
        LIMIT 5;
    """).fetchdf()
    print("\n2. Dual-Currency Ledger Consistency Sample:\n", fx_check_df.to_string(index=False))

    con.close()
    return counts_df, fx_check_df

def generate_postgres_script():
    print(f"\nGenerating PostgreSQL DDL & Ingestion script: {POSTGRES_SQL_PATH}...")
    sql_content = """-- =============================================================
-- Cambodia MSE Intelligence - PostgreSQL ODS Setup Script
-- Run via psql:
-- psql -U <username> -d <dbname> -f ingestion/init_ods_postgres.sql
-- =============================================================

-- Drop tables in reverse dependency order
DROP TABLE IF EXISTS sales_transactions CASCADE;
DROP TABLE IF EXISTS inventory_stock CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS macro_external_data CASCADE;

-- 1. Inventory & SKU Management Table
CREATE TABLE inventory_stock (
    sku_id                       VARCHAR(64) PRIMARY KEY,
    product_name_khmer           VARCHAR(255) NOT NULL,
    category_id                  INT NOT NULL,
    unit_cost_usd                DECIMAL(10,4) NOT NULL CHECK (unit_cost_usd >= 0),
    unit_cost_khr                DECIMAL(12,2) NOT NULL CHECK (unit_cost_khr >= 0),
    current_stock_on_hand        INT NOT NULL DEFAULT 0 CHECK (current_stock_on_hand >= 0),
    allocated_stock              INT DEFAULT 0 CHECK (allocated_stock >= 0),
    reorder_point_units          INT DEFAULT 10,
    safety_stock_level           INT DEFAULT 5,
    lead_time_days               INT DEFAULT 3,
    lead_time_variance_days      DECIMAL(5,2) DEFAULT 0.5,
    holding_cost_per_unit_annual DECIMAL(8,2) DEFAULT 1.00,
    stockout_event_count_30d     INT DEFAULT 0,
    shelf_life_days              INT NULL
);

-- 2. Customer Behavioral & Profile Table
CREATE TABLE customers (
    customer_id                  VARCHAR(64) PRIMARY KEY,
    registration_date            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    province_code                VARCHAR(10) DEFAULT 'KH-12',
    recency_days                 INT DEFAULT 0,
    frequency_count_180d         INT DEFAULT 1,
    monetary_total_usd           DECIMAL(12,2) DEFAULT 0.00,
    preferred_payment_channel    VARCHAR(32) DEFAULT 'KHQR_BAKONG',
    rfm_segment_cluster          INT DEFAULT 0,
    churn_probability_score      DECIMAL(3,2) DEFAULT 0.00,
    historical_clv_usd           DECIMAL(10,2) DEFAULT 0.00,
    predicted_clv_usd            DECIMAL(10,2) DEFAULT 0.00,
    avg_basket_size_usd          DECIMAL(8,2) DEFAULT 0.00
);

-- 3. External Macroeconomic Reference Table
CREATE TABLE macro_external_data (
    date_key                     DATE PRIMARY KEY,
    fuel_price_regular_khr       DECIMAL(8,2) NULL,
    fuel_price_diesel_khr        DECIMAL(8,2) NULL,
    nbc_official_rate            DECIMAL(8,2) NOT NULL DEFAULT 4050.00,
    provincial_pop_density       DECIMAL(10,2) NULL,
    holiday_event_flag           BOOLEAN DEFAULT FALSE,
    rainfall_mm_local            DECIMAL(6,2) DEFAULT 0.0
);

-- 4. POS Sales Transactions Table
CREATE TABLE sales_transactions (
    transaction_line_id          VARCHAR(64) PRIMARY KEY,
    transaction_id               VARCHAR(64) NOT NULL,
    transaction_timestamp        TIMESTAMP NOT NULL,
    sku_id                       VARCHAR(64) NOT NULL REFERENCES inventory_stock(sku_id),
    customer_id                  VARCHAR(64) NULL REFERENCES customers(customer_id),
    quantity_sold                INT NOT NULL CHECK (quantity_sold > 0),
    unit_selling_price_usd       DECIMAL(10,2) NOT NULL CHECK (unit_selling_price_usd >= 0),
    unit_selling_price_khr       DECIMAL(12,2) NOT NULL CHECK (unit_selling_price_khr >= 0),
    applied_exchange_rate        DECIMAL(8,2) NOT NULL DEFAULT 4050.00,
    payment_currency             VARCHAR(3) NOT NULL CHECK (payment_currency IN ('USD', 'KHR')),
    discount_amount_usd          DECIMAL(8,2) DEFAULT 0.00,
    gross_revenue_usd            DECIMAL(12,2) NOT NULL,
    cogs_usd                     DECIMAL(12,2) NOT NULL,
    net_profit_margin_usd        DECIMAL(12,2) NOT NULL,
    khqr_merchant_mcc            VARCHAR(4) DEFAULT '5999',
    payment_settlement_status    VARCHAR(20) DEFAULT 'SETTLED'
);

-- Bulk Copy from CSVs
\\copy inventory_stock FROM 'data/synthetic_inventory.csv' WITH (FORMAT csv, HEADER true);
\\copy customers FROM 'data/synthetic_customers.csv' WITH (FORMAT csv, HEADER true);
\\copy macro_external_data FROM 'data/odc_fuel_nbc_combined.csv' WITH (FORMAT csv, HEADER true);
\\copy sales_transactions FROM 'data/synthetic_sales.csv' WITH (FORMAT csv, HEADER true);

-- Post-Ingestion Verification
SELECT 'sales_transactions' AS table_name, COUNT(*) AS row_count FROM sales_transactions
UNION ALL
SELECT 'inventory_stock', COUNT(*) FROM inventory_stock
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'macro_external_data', COUNT(*) FROM macro_external_data;
"""
    with open(POSTGRES_SQL_PATH, 'w', encoding='utf-8') as f:
        f.write(sql_content)
    print("  -> PostgreSQL script successfully written.")

if __name__ == "__main__":
    inv, cust, macro, sales = generate_ods_csvs()
    setup_duckdb_ods(inv, cust, macro, sales)
    generate_postgres_script()
    print("\n" + "="*65)
    print("STEP 3 ODS SETUP COMPLETE!")
    print("="*65)
