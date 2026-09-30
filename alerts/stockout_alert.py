"""
Cambodia MSE Intelligence - Dynamic Stockout Risk Alert Engine
Connects to Module C Demand Forecast & Dynamic Reorder Point (ROP) Mart.
Flags items with Current Stock <= Dynamic ROP and generates emergency replenishment orders.

Formula:
    Safety Stock = Z (1.645) * sigma_demand * sqrt(Lead Time)
    Dynamic ROP  = (Lead Time * d_avg_forecast) + Safety Stock
    Alert Condition: Current Stock <= Dynamic ROP

Run with: python3 alerts/stockout_alert.py
"""

import os
import math
import duckdb
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "cambodia_mse.duckdb")
OUTPUT_CSV = os.path.join(DATA_DIR, "recommended_replenishment_orders.csv")

def evaluate_stockout_alerts():
    print("=" * 80)
    print("CAMBODIA MSE INTELLIGENCE — OPERATIONAL STOCKOUT ALERT ENGINE (MODULE C)")
    print("=" * 80)

    forecast_csv = os.path.join(DATA_DIR, "sku_demand_rop_forecast.csv")
    if not os.path.exists(forecast_csv):
        print("[ERROR] sku_demand_rop_forecast.csv not found. Run Module C or scripts/rebuild_all_notebooks.py first.")
        return

    df_rop = pd.read_csv(forecast_csv)
    
    # Query wholesale acquisition costs and currency rates from DuckDB
    con = duckdb.connect(DB_PATH, read_only=True)
    costs_df = con.execute("""
        SELECT sku_id, unit_cost_usd, unit_cost_khr 
        FROM dim_products;
    """).fetchdf()
    
    fx_rate = float(con.execute("""
        SELECT nbc_official_rate FROM dim_dates_macro ORDER BY date_key DESC LIMIT 1;
    """).fetchone()[0])
    con.close()

    df_merged = df_rop.merge(costs_df, on="sku_id", how="left")
    
    # Filter for stockout risk items (Current Stock <= Dynamic ROP)
    at_risk = df_merged[df_merged["stockout_risk"] == True].copy()
    
    if len(at_risk) == 0:
        print("✅ ALL INVENTORY LEVELS HEALTHY: No SKUs breached their dynamic ROP threshold.")
        print("=" * 80)
        return

    # Calculate Days of Supply Remaining and Recommended Purchase Order Quantity
    # Target stock buffer = 2.5 * dynamic_rop (to cover lead time + safety buffer + review period)
    at_risk["days_supply_left"] = (at_risk["current_stock"] / at_risk["d_avg_7d"]).round(1)
    at_risk["target_stock"] = (at_risk["dynamic_rop"] * 2.2).apply(math.ceil)
    at_risk["recommended_order_units"] = (at_risk["target_stock"] - at_risk["current_stock"]).clip(lower=10)
    at_risk["total_order_cost_usd"] = (at_risk["recommended_order_units"] * at_risk["unit_cost_usd"]).round(2)
    at_risk["total_order_cost_khr"] = (at_risk["total_order_cost_usd"] * fx_rate).round(-2)

    # Sort by urgency (least days of supply left)
    at_risk = at_risk.sort_values(by="days_supply_left", ascending=True).reset_index(drop=True)

    print(f"🚨 ALERT: {len(at_risk)} SKUs CRITICALLY BELOW DYNAMIC REORDER POINT (ROP)!")
    print(f"Current FX Reference Rate: 1 USD = {fx_rate:,.0f} KHR\n")
    print("-" * 80)
    print(f"{'SKU ID':<13} {'Product Name':<28} {'Stock':<7} {'ROP':<7} {'DSI (d)':<8} {'Order':<7} {'Cost (USD)':<10}")
    print("-" * 80)

    for _, row in at_risk.iterrows():
        print(f"{row['sku_id']:<13} {row['product_name'][:26]:<28} {int(row['current_stock']):<7} {row['dynamic_rop']:<7.1f} {row['days_supply_left']:<8.1f} {int(row['recommended_order_units']):<7} ${row['total_order_cost_usd']:<9.2f}")

    total_procurement_usd = at_risk["total_order_cost_usd"].sum()
    total_procurement_khr = at_risk["total_order_cost_khr"].sum()
    print("-" * 80)
    print(f"TOTAL PROCUREMENT WORKING CAPITAL REQUIRED:")
    print(f"  -> $ {total_procurement_usd:,.2f} USD")
    print(f"  -> {total_procurement_khr:,.0f} KHR")
    print("=" * 80)

    # Export Purchase Order requisition for store managers
    export_cols = [
        "sku_id", "product_name", "category", "lead_time_days",
        "current_stock", "d_avg_7d", "dynamic_rop", "days_supply_left",
        "recommended_order_units", "unit_cost_usd", "total_order_cost_usd", "total_order_cost_khr"
    ]
    at_risk[export_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"📄 Purchase Order requisition exported: {OUTPUT_CSV}")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_stockout_alerts()
