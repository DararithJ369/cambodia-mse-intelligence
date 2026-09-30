-- =============================================================
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
\copy inventory_stock FROM 'data/synthetic_inventory.csv' WITH (FORMAT csv, HEADER true);
\copy customers FROM 'data/synthetic_customers.csv' WITH (FORMAT csv, HEADER true);
\copy macro_external_data FROM 'data/odc_fuel_nbc_combined.csv' WITH (FORMAT csv, HEADER true);
\copy sales_transactions FROM 'data/synthetic_sales.csv' WITH (FORMAT csv, HEADER true);

-- Post-Ingestion Verification
SELECT 'sales_transactions' AS table_name, COUNT(*) AS row_count FROM sales_transactions
UNION ALL
SELECT 'inventory_stock', COUNT(*) FROM inventory_stock
UNION ALL
SELECT 'customers', COUNT(*) FROM customers
UNION ALL
SELECT 'macro_external_data', COUNT(*) FROM macro_external_data;
