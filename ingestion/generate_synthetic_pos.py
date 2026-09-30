"""
Cambodia MSE Intelligence - Step 2: Realistic Synthetic Retail POS Generator
Generates a 90-day retail Point-of-Sale (POS) transaction dataset reflecting
local operational realities in Cambodia (Dual-Currency USD/KHR, ABA KHQR/Bakong,
inventory SKUs, lead times, and customer IDs).
Run with: python3 ingestion/generate_synthetic_pos.py
"""

import os
import sqlite3
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from faker import Faker

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "synthetic_pos")
DB_PATH = os.path.join(DATA_DIR, "cambodia_mse.db")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
fake = Faker(['en_US'])

# ------------------------------------------------------------------------------
# 1. GENERATE NBC DAILY FX RATES (DUAL-CURRENCY LEDGER)
# ------------------------------------------------------------------------------
def generate_fx_rates(start_date, num_days=90):
    print("Generating 90-Day NBC Foreign Exchange Rates (USD/KHR)...")
    records = []
    base_rate = 4085.0  # Typical National Bank of Cambodia reference rate
    
    current_date = start_date
    for i in range(num_days):
        # Realistic daily random walk with mean reversion
        drift = -0.05 * (base_rate - 4090.0)
        shock = np.random.normal(0, 3.5)
        base_rate += drift + shock
        rounded_rate = round(base_rate, 2)
        
        records.append({
            "rate_date": current_date.strftime("%Y-%m-%d"),
            "nbc_usd_khr_rate": rounded_rate,
            "bank_buy_rate": round(rounded_rate - 5.0, 2),
            "bank_sell_rate": round(rounded_rate + 7.0, 2)
        })
        current_date += timedelta(days=1)
        
    df_fx = pd.DataFrame(records)
    df_fx.to_csv(os.path.join(OUTPUT_DIR, "dim_fx_rates.csv"), index=False)
    print(f"  -> Generated {len(df_fx)} daily FX rate records. Mean: {df_fx['nbc_usd_khr_rate'].mean():.1f} KHR/USD")
    return df_fx

# ------------------------------------------------------------------------------
# 2. GENERATE PRODUCT SKUs & INVENTORY DIMENSION
# ------------------------------------------------------------------------------
def generate_products():
    print("Generating Product SKUs, Cost Prices & Inventory Parameters...")
    
    # Authentic Cambodian convenience store / retail mart catalog
    catalog = [
        # Beverages
        ("SKU-BEV-001", "Angkor Premium Beer 330ml Can", "Beverages", 0.65, 0.90, 150, 40, 2, "Cambrew Ltd"),
        ("SKU-BEV-002", "Cambodia Beer 330ml Can", "Beverages", 0.62, 0.85, 140, 35, 2, "Khmer Beverages"),
        ("SKU-BEV-003", "Hanuman Premium Lager 330ml", "Beverages", 0.70, 1.00, 100, 30, 3, "Hanuman Beverages"),
        ("SKU-BEV-004", "Sting Energy Drink 250ml Can", "Beverages", 0.40, 0.65, 200, 50, 2, "PepsiCo Cambodia"),
        ("SKU-BEV-005", "Carabao Energy Drink 250ml", "Beverages", 0.38, 0.60, 180, 45, 2, "Carabao Tawandang"),
        ("SKU-BEV-006", "Vital Premium Water 500ml", "Beverages", 0.18, 0.35, 300, 80, 1, "N.V.C Corporation"),
        ("SKU-BEV-007", "Yeo's Soy Bean Milk 300ml Can", "Beverages", 0.42, 0.70, 120, 30, 2, "Yeo Hiap Seng"),
        ("SKU-BEV-008", "Canned Iced Coffee Milk 240ml", "Beverages", 0.48, 0.80, 160, 40, 2, "Local Dairy Wholesale"),
        ("SKU-BEV-009", "Oishi Green Tea Honey Lemon 500ml", "Beverages", 0.55, 0.90, 110, 30, 3, "Oishi Group"),
        ("SKU-BEV-010", "Coca-Cola Original 330ml Can", "Beverages", 0.45, 0.75, 220, 60, 2, "Cambodia Beverage Co"),
        
        # Packaged Foods & Staples
        ("SKU-FOD-001", "Mama Instant Noodles Pork 55g (Pack of 5)", "Packaged Foods", 1.10, 1.60, 130, 35, 3, "Thai President Foods"),
        ("SKU-FOD-002", "Wai Wai Noodles Tom Yum 60g", "Packaged Foods", 0.25, 0.40, 250, 60, 2, "Thai Preserved Food"),
        ("SKU-FOD-003", "Malys Angkor Jasmine Rice 5kg", "Packaged Foods", 4.20, 5.75, 45, 15, 4, "Battambang Rice Mill"),
        ("SKU-FOD-004", "Phsar Thom Quality Fish Sauce 500ml", "Packaged Foods", 0.85, 1.30, 80, 25, 3, "Kampot Fish Sauce Co"),
        ("SKU-FOD-005", "Kampot Black Pepper Sealed 100g", "Packaged Foods", 1.80, 2.75, 60, 15, 5, "Kampot Pepper Assn"),
        ("SKU-FOD-006", "Traditional Palm Sugar Block 500g", "Packaged Foods", 1.10, 1.70, 70, 20, 4, "Kampong Speu Agri Coop"),
        ("SKU-FOD-007", "Teuk Doh Koh Sweet Condensed Milk 380g", "Packaged Foods", 0.75, 1.15, 140, 40, 2, "Indochina Dairy"),
        ("SKU-FOD-008", "Aroy-D Coconut Milk 400ml Can", "Packaged Foods", 0.95, 1.45, 90, 25, 3, "Thai Agri Foods"),
        ("SKU-FOD-009", "Knorr Chicken Broth Powder 400g", "Packaged Foods", 1.50, 2.20, 75, 20, 3, "Unilever Cambodia"),
        ("SKU-FOD-010", "Roza Canned Sardines in Tomato Sauce", "Packaged Foods", 0.60, 0.95, 160, 40, 2, "Hi-Q Food Products"),

        # Snacks & Confectionery
        ("SKU-SNK-001", "Lay's Classic Potato Chips 50g", "Snacks", 0.70, 1.10, 110, 30, 3, "PepsiCo Foods"),
        ("SKU-SNK-002", "Pocky Chocolate Biscuit Sticks 47g", "Snacks", 0.65, 1.00, 130, 35, 3, "Glico Thailand"),
        ("SKU-SNK-003", "Cambodian Dried Mango Slices 150g", "Snacks", 1.40, 2.25, 80, 20, 4, "Kirirom Food Prod"),
        ("SKU-SNK-004", "Roasted Garlic Chili Peanuts 120g", "Snacks", 0.50, 0.85, 150, 40, 2, "Local Food Artisans"),
        ("SKU-SNK-005", "Hanami Prawn Crackers 60g", "Snacks", 0.55, 0.90, 120, 30, 3, "Hanami Co"),

        # Personal Care & Household
        ("SKU-HOU-001", "Sunsilk Co-Creations Shampoo 320ml", "Personal Care", 2.20, 3.20, 50, 15, 3, "Unilever Cambodia"),
        ("SKU-HOU-002", "Colgate Total Toothpaste 150g", "Personal Care", 1.25, 1.85, 75, 20, 3, "Colgate-Palmolive"),
        ("SKU-HOU-003", "Sunlight Lime Dishwashing Liquid 750ml", "Household", 1.10, 1.65, 90, 25, 2, "Unilever Cambodia"),
        ("SKU-HOU-004", "Dettol Anti-Bacterial Bar Soap 100g", "Personal Care", 0.60, 0.95, 140, 35, 2, "Reckitt Benckiser"),
        ("SKU-HOU-005", "Premier 3-Ply Facial Tissue (Pack of 4)", "Household", 1.50, 2.20, 65, 20, 3, "NTPM Paper"),
        ("SKU-HOU-006", "Attack Easy Detergent Powder 800g", "Household", 1.45, 2.10, 70, 20, 3, "Kao Commercial"),
        
        # Dairy & Fresh Essentials
        ("SKU-DAI-001", "Dutch Mill UHT Fresh Milk 180ml (x4)", "Dairy & Cold", 1.40, 2.00, 85, 25, 2, "Dutch Mill Co"),
        ("SKU-DAI-002", "Yakult Probiotic Fermented Milk (Pack of 5)", "Dairy & Cold", 1.55, 2.25, 60, 20, 2, "Yakult Cambodia"),
        ("SKU-DAI-003", "Fresh Local Eggs (Tray of 10)", "Dairy & Cold", 1.15, 1.65, 100, 30, 1, "Takeo Poultry Farm")
    ]
    
    df_products = pd.DataFrame(catalog, columns=[
        "sku", "product_name", "category", "cost_price_usd", "selling_price_usd",
        "current_stock_level", "reorder_point", "lead_time_days", "supplier_name"
    ])
    df_products["gross_margin_pct"] = ((df_products["selling_price_usd"] - df_products["cost_price_usd"]) / df_products["selling_price_usd"]) * 100
    df_products.to_csv(os.path.join(OUTPUT_DIR, "dim_products.csv"), index=False)
    print(f"  -> Generated {len(df_products)} SKUs across {df_products['category'].nunique()} categories.")
    return df_products

# ------------------------------------------------------------------------------
# 3. GENERATE CUSTOMER DIMENSION
# ------------------------------------------------------------------------------
def generate_customers(num_customers=850):
    print("Generating Customer Profiles with Cambodian Names & Phone Prefixes...")
    
    khmer_first_names = [
        "Sokha", "Bopha", "Chantha", "Veasna", "Dara", "Piseth", "Kalyan", "Rithy",
        "Sophal", "Mony", "Sreymom", "Theary", "Kosal", "Vannak", "Sovann", "Phalla",
        "Boramey", "Socheat", "Chhay", "Makara", "Samnang", "Chenda", "Rathana", "Visal"
    ]
    khmer_last_names = [
        "Meas", "Kim", "Ouk", "Seng", "Chea", "Heng", "Pich", "Keo",
        "Ros", "Noun", "Sok", "Khun", "Tep", "Ung", "Mao", "Prom",
        "Sin", "Yin", "Nhem", "Chhum", "Chou", "Chhorn", "Lim", "Vong"
    ]
    
    phone_prefixes = ["012", "017", "077", "089", "092", "070", "081", "086", "093", "098", "097", "088"]
    cities = ["Phnom Penh", "Siem Reap", "Battambang", "Sihanoukville", "Kandal"]
    city_weights = [0.55, 0.18, 0.15, 0.07, 0.05]
    
    customers = []
    # Anonymous walk-in customer profile
    customers.append({
        "customer_id": "CUST-ANON-0000",
        "customer_name": "Walk-in Guest",
        "phone_number": "N/A",
        "loyalty_tier": "None",
        "primary_city": "Phnom Penh",
        "account_created_date": "2026-01-01"
    })
    
    for i in range(1, num_customers + 1):
        c_id = f"CUST-{i:04d}"
        name = f"{random.choice(khmer_last_names)} {random.choice(khmer_first_names)}"
        prefix = random.choice(phone_prefixes)
        phone = f"+855 {prefix} {random.randint(100, 999)} {random.randint(100, 999)}"
        tier = np.random.choice(["Regular", "Silver", "Gold", "Platinum"], p=[0.70, 0.18, 0.09, 0.03])
        city = np.random.choice(cities, p=city_weights)
        created = (datetime(2026, 7, 1) - timedelta(days=random.randint(10, 500))).strftime("%Y-%m-%d")
        
        customers.append({
            "customer_id": c_id,
            "customer_name": name,
            "phone_number": phone,
            "loyalty_tier": tier,
            "primary_city": city,
            "account_created_date": created
        })
        
    df_customers = pd.DataFrame(customers)
    df_customers.to_csv(os.path.join(OUTPUT_DIR, "dim_customers.csv"), index=False)
    print(f"  -> Generated {len(df_customers)} customer records (including anonymous walk-in).")
    return df_customers

# ------------------------------------------------------------------------------
# 4. GENERATE 90-DAY POS TRANSACTIONS (DUAL-CURRENCY & REALISTIC PAYMENTS)
# ------------------------------------------------------------------------------
def generate_pos_transactions(df_products, df_customers, df_fx, start_date, num_days=90):
    print(f"Generating 90-Day Retail POS Transactions ({num_days} days)...")
    
    fx_dict = dict(zip(df_fx["rate_date"], df_fx["nbc_usd_khr_rate"]))
    skus_list = df_products["sku"].tolist()
    prod_dict = df_products.set_index("sku").to_dict(orient="index")
    
    registered_cust_ids = df_customers[df_customers["customer_id"] != "CUST-ANON-0000"]["customer_id"].tolist()
    
    # Store Locations across Cambodia
    stores = [
        ("STORE-PP-01", "Tuol Kork Flagship", "Phnom Penh"),
        ("STORE-PP-02", "BKK1 Convenience Mart", "Phnom Penh"),
        ("STORE-SR-01", "Old Market Branch", "Siem Reap"),
        ("STORE-BTB-01", "Battambang Central Mart", "Battambang")
    ]
    store_weights = [0.42, 0.28, 0.18, 0.12]
    
    # Payment Method distribution matching local operational reality:
    # 60-70% via ABA KHQR / Bakong QR codes and cash
    payment_methods = ["KHQR_ABA", "KHQR_Bakong", "Cash_KHR", "Cash_USD", "Debit_Credit_Card"]
    # ABA KHQR: 36%, Bakong QR: 22%, Cash KHR: 20%, Cash USD: 14%, Card: 8%
    # -> Digital QR = 58%, Total QR + Cash = 92%
    payment_weights = [0.36, 0.22, 0.20, 0.14, 0.08]
    
    all_transactions = []
    txn_counter = 1
    
    current_date = start_date
    for day_idx in range(num_days):
        date_str = current_date.strftime("%Y-%m-%d")
        daily_fx = fx_dict.get(date_str, 4085.0)
        
        # Day of week seasonality: Fri/Sat/Sun +25% volume
        is_weekend = current_date.weekday() in [4, 5, 6]
        base_daily_txns = 140 if not is_weekend else 180
        daily_txns_count = int(np.random.normal(base_daily_txns, 12))
        
        for _ in range(daily_txns_count):
            txn_id = f"TXN-{current_date.strftime('%Y%m%d')}-{txn_counter:05d}"
            txn_counter += 1
            
            # Realistic hour distribution: lunch rush (11-13) and evening rush (17-21)
            hour_prob = [
                0.01, 0.00, 0.00, 0.00, 0.00, 0.01, # 0-5
                0.03, 0.06, 0.07, 0.06, 0.07, 0.10, # 6-11
                0.11, 0.06, 0.05, 0.06, 0.08, 0.11, # 12-17
                0.12, 0.09, 0.05, 0.02, 0.01, 0.00  # 18-23
            ]
            hour_prob = np.array(hour_prob) / np.sum(hour_prob)
            hour = np.random.choice(range(24), p=hour_prob)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            txn_timestamp = f"{date_str} {hour:02d}:{minute:02d}:{second:02d}"
            
            # Select store
            store_idx = np.random.choice(len(stores), p=store_weights)
            store_id, store_name, store_city = stores[store_idx]
            
            # Customer: 65% repeat tracked customer, 35% anonymous walk-in
            if random.random() < 0.65:
                cust_id = random.choice(registered_cust_ids)
            else:
                cust_id = "CUST-ANON-0000"
                
            # Payment Method
            payment = np.random.choice(payment_methods, p=payment_weights)
            
            # Basket size: 1 to 4 items per transaction
            basket_size = np.random.choice([1, 2, 3, 4], p=[0.45, 0.32, 0.16, 0.07])
            selected_skus = random.sample(skus_list, basket_size)
            
            for item_idx, sku in enumerate(selected_skus, 1):
                p_info = prod_dict[sku]
                # Quantity distribution
                qty = np.random.choice([1, 2, 3, 4, 6], p=[0.68, 0.20, 0.07, 0.03, 0.02])
                unit_cost_usd = p_info["cost_price_usd"]
                unit_price_usd = p_info["selling_price_usd"]
                
                # Dual Currency calculations
                # In Cambodia, retail KHR is traditionally rounded to the nearest 100 Riel
                unit_price_khr = round((unit_price_usd * daily_fx) / 100) * 100
                
                line_total_usd = round(qty * unit_price_usd, 2)
                line_total_khr = int(qty * unit_price_khr)
                line_cost_usd = round(qty * unit_cost_usd, 2)
                line_profit_usd = round(line_total_usd - line_cost_usd, 2)
                
                all_transactions.append({
                    "transaction_id": txn_id,
                    "item_line_no": item_idx,
                    "timestamp": txn_timestamp,
                    "date": date_str,
                    "day_of_week": current_date.strftime("%A"),
                    "hour": hour,
                    "store_id": store_id,
                    "store_name": store_name,
                    "store_city": store_city,
                    "customer_id": cust_id,
                    "sku": sku,
                    "product_name": p_info["product_name"],
                    "category": p_info["category"],
                    "quantity": qty,
                    "unit_cost_usd": unit_cost_usd,
                    "unit_price_usd": unit_price_usd,
                    "nbc_exchange_rate": daily_fx,
                    "unit_price_khr": unit_price_khr,
                    "total_amount_usd": line_total_usd,
                    "total_amount_khr": line_total_khr,
                    "gross_profit_usd": line_profit_usd,
                    "payment_method": payment
                })
                
        current_date += timedelta(days=1)
        
    df_txns = pd.DataFrame(all_transactions)
    
    # Save CSV and Parquet formats
    csv_file = os.path.join(OUTPUT_DIR, "fct_pos_transactions.csv")
    parquet_file = os.path.join(OUTPUT_DIR, "pos_transactions_90d.parquet")
    
    df_txns.to_csv(csv_file, index=False)
    df_txns.to_parquet(parquet_file, index=False)
    
    print(f"  -> Generated {len(df_txns):,} item transactions across {df_txns['transaction_id'].nunique():,} unique checkout baskets.")
    print(f"  -> Total Revenue: ${df_txns['total_amount_usd'].sum():,.2f} USD ({df_txns['total_amount_khr'].sum():,.0f} KHR)")
    return df_txns

# ------------------------------------------------------------------------------
# 5. PERSIST INTO SQLite DATA WAREHOUSE
# ------------------------------------------------------------------------------
def persist_to_database(df_products, df_customers, df_fx, df_txns):
    print("Persisting synthetic retail dataset into SQLite warehouse...")
    conn = sqlite3.connect(DB_PATH)
    
    df_products.to_sql("dim_products", conn, if_exists="replace", index=False)
    df_customers.to_sql("dim_customers", conn, if_exists="replace", index=False)
    df_fx.to_sql("dim_fx_rates", conn, if_exists="replace", index=False)
    df_txns.to_sql("fct_pos_transactions", conn, if_exists="replace", index=False)
    
    # Create indexes for performant analytical SQL queries
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON fct_pos_transactions (date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_sku ON fct_pos_transactions (sku);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_cust ON fct_pos_transactions (customer_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_pay ON fct_pos_transactions (payment_method);")
    conn.commit()
    conn.close()
    print(f"  -> Successfully updated warehouse tables in {DB_PATH}")

if __name__ == "__main__":
    print("="*65)
    print("STEP 2: GENERATING REALISTIC 90-DAY SYNTHETIC RETAIL POS DATASET")
    print("="*65)
    
    # 90-day time window
    start_date = datetime(2026, 7, 1)
    
    df_fx = generate_fx_rates(start_date=start_date, num_days=90)
    df_products = generate_products()
    df_customers = generate_customers(num_customers=850)
    df_txns = generate_pos_transactions(df_products, df_customers, df_fx, start_date=start_date, num_days=90)
    persist_to_database(df_products, df_customers, df_fx, df_txns)
    
    print("="*65)
    print("Step 2 Dataset Generation Complete!")
    print(f"Files saved in: {OUTPUT_DIR}")
    print("="*65)
