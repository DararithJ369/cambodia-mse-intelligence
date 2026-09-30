"""
Cambodia MSE Intelligence - Telegram Bot Mobile Alerting Engine
Department of Applied Mathematics and Statistics — Institute of Technology of Cambodia (ITC)

Queries inventory stock in DuckDB, evaluates stockout risks against dynamic ROP
(Module C Demand Forecast: current_stock <= dynamic_rop), formats Telegram-compatible
Markdown notifications, and dispatches push alerts via the official Telegram Bot API
(https://api.telegram.org/bot<TOKEN>/sendMessage).

If no Bot Token or Chat ID is provided, it runs in a high-fidelity Simulation Mode,
printing the exact formatted messages and payloads for verification.

Usage:
  python3 alerts/telegram_worker.py
  python3 alerts/telegram_worker.py --dry-run
  TELEGRAM_BOT_TOKEN="xxx" TELEGRAM_CHAT_ID="yyy" python3 alerts/telegram_worker.py
"""

import os
import sys
import argparse
import requests
import duckdb
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PRIMARY = os.path.join(DATA_DIR, "cambodia_mse.duckdb")
DB_FALLBACK = os.path.join(DATA_DIR, "sme_cambodia.duckdb")
FORECAST_CSV = os.path.join(DATA_DIR, "sku_demand_rop_forecast.csv")

def query_low_stock_items():
    """
    Queries SKUs where current stock breaches the Reorder Point.
    Uses dynamic ROP from Module C (Demand Forecast) if available,
    joined with master catalog metadata from DuckDB.
    """
    con = None
    if os.path.exists(DB_PRIMARY):
        con = duckdb.connect(DB_PRIMARY, read_only=True)
        table_name = "dim_products"
    elif os.path.exists(DB_FALLBACK):
        con = duckdb.connect(DB_FALLBACK, read_only=True)
        table_name = "inventory_stock"
    else:
        raise FileNotFoundError("DuckDB database file not found in data directory.")

    db_items = con.execute(f"""
        SELECT 
            sku_id,
            product_name_khmer,
            category_id,
            current_stock_on_hand,
            reorder_point_units AS static_rop,
            safety_stock_level,
            lead_time_days,
            unit_cost_usd,
            unit_cost_khr
        FROM {table_name};
    """).fetchdf()
    con.close()

    # Check for Dynamic ROP from Module C Forecast
    if os.path.exists(FORECAST_CSV):
        df_forecast = pd.read_csv(FORECAST_CSV)
        merged = db_items.merge(
            df_forecast[["sku_id", "product_name", "d_avg_7d", "safety_stock", "dynamic_rop", "stockout_risk"]],
            on="sku_id",
            how="left"
        )
        # Use dynamic ROP where available, fallback to static ROP
        merged["reorder_point_units"] = merged["dynamic_rop"].fillna(merged["static_rop"]).round(1)
        merged["safety_stock_level"] = merged["safety_stock"].fillna(merged["safety_stock_level"]).round(1)
        # Flag items where current stock <= ROP
        at_risk = merged[(merged["current_stock_on_hand"] <= merged["reorder_point_units"]) | (merged["stockout_risk"] == True)].copy()
    else:
        # Fallback to static ROP
        merged = db_items.copy()
        merged["reorder_point_units"] = merged["static_rop"]
        at_risk = merged[merged["current_stock_on_hand"] <= merged["reorder_point_units"]].copy()

    if not at_risk.empty:
        # Recommended order: buffer to 2x ROP or minimum 20 units
        at_risk["recommended_reorder_qty"] = (at_risk["reorder_point_units"] * 2.0 - at_risk["current_stock_on_hand"]).apply(lambda x: max(20, round(x)))
        at_risk["stock_ratio"] = at_risk["current_stock_on_hand"] / at_risk["reorder_point_units"].replace(0, 1)
        at_risk = at_risk.sort_values(by="stock_ratio", ascending=True).reset_index(drop=True)

    return at_risk

def format_telegram_alert(row, location="Phnom Penh Central Branch"):
    """
    Formats the inventory stockout alert message following the mobile-first standard.
    """
    prod_name = row.get("product_name") or row.get("product_name_khmer", "Unknown Product")
    khmer_name = row.get("product_name_khmer", "")
    display_title = f"{prod_name} ({khmer_name})" if khmer_name and khmer_name != prod_name else prod_name
    
    qty = int(row["recommended_reorder_qty"])
    cost_usd = float(row["unit_cost_usd"]) * qty
    cost_khr = float(row["unit_cost_khr"]) * qty

    msg = (
        f"[INVENTORY WARNING] Low Stock Alert\n"
        f"Location: {location}\n"
        f"SKU: {row['sku_id']} ({display_title})\n"
        f"Category: {row['category_id']}\n"
        f"-----------------------------------\n"
        f"Current Stock: {int(row['current_stock_on_hand'])} units\n"
        f"Reorder Point (ROP): {row['reorder_point_units']} units\n"
        f"Safety Stock Buffer: {row['safety_stock_level']} units\n"
        f"Supplier Lead Time: {int(row['lead_time_days'])} days\n"
        f"-----------------------------------\n"
        f"Recommended Reorder: {qty} units\n"
        f"Estimated Cost: ${cost_usd:,.2f} USD ({cost_khr:,.0f} ៛)\n"
        f"Action: Submit purchase order to wholesale distributor."
    )
    return msg

def dispatch_telegram_alerts(bot_token=None, chat_id=None, location="Phnom Penh Central Branch", dry_run=False):
    """
    Dispatches alerts to Telegram API or outputs simulation payloads.
    Returns a dictionary summarizing execution results.
    """
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    loc = location or os.environ.get("TELEGRAM_LOCATION", "Phnom Penh Central Branch")

    df_alerts = query_low_stock_items()
    
    results = {
        "total_at_risk": len(df_alerts),
        "dispatched_count": 0,
        "failed_count": 0,
        "mode": "Live API" if (token and chat and not dry_run) else "Simulation (Dry-Run)",
        "messages": [],
        "details": []
    }

    if df_alerts.empty:
        return results

    is_live = bool(token and chat and not dry_run)

    for _, row in df_alerts.iterrows():
        msg_text = format_telegram_alert(row, location=loc)
        results["messages"].append(msg_text)

        if is_live:
            api_url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {
                "chat_id": chat,
                "text": msg_text,
                "parse_mode": "Markdown"
            }
            try:
                resp = requests.post(api_url, json=payload, timeout=10)
                if resp.status_code == 200:
                    results["dispatched_count"] += 1
                    results["details"].append({"sku_id": row["sku_id"], "status": "SUCCESS", "response": resp.json()})
                else:
                    results["failed_count"] += 1
                    results["details"].append({"sku_id": row["sku_id"], "status": f"HTTP {resp.status_code}", "response": resp.text})
            except Exception as e:
                results["failed_count"] += 1
                results["details"].append({"sku_id": row["sku_id"], "status": "ERROR", "error": str(e)})
        else:
            results["dispatched_count"] += 1
            results["details"].append({"sku_id": row["sku_id"], "status": "SIMULATED", "text": msg_text})

    return results

def main():
    parser = argparse.ArgumentParser(description="Cambodia MSE Telegram Stockout Alert Worker")
    parser.add_argument("--token", help="Telegram Bot Token (from @BotFather)")
    parser.add_argument("--chat-id", help="Target Telegram Chat ID")
    parser.add_argument("--location", default="Phnom Penh Central Branch", help="Store location name")
    parser.add_argument("--dry-run", action="store_true", help="Simulate alert dispatch without calling API")
    args = parser.parse_args()

    print("=" * 75)
    print("🇰🇭 CAMBODIA MSE INTELLIGENCE — TELEGRAM MOBILE ALERT WORKER")
    print("=" * 75)

    res = dispatch_telegram_alerts(
        bot_token=args.token,
        chat_id=args.chat_id,
        location=args.location,
        dry_run=args.dry_run
    )

    print(f"Operational Mode: {res['mode']}")
    print(f"Store Location:   {args.location}")
    print(f"Low-Stock SKUs:   {res['total_at_risk']}")
    print("-" * 75)

    for i, msg in enumerate(res["messages"], 1):
        print(f"\n[Telegram Message #{i}]")
        print(msg)
        print("-" * 50)

    if res["mode"] == "Simulation (Dry-Run)":
        print("\n💡 Tip: To send real push alerts to your smartphone:")
        print("  1. Create a bot with @BotFather on Telegram to get a BOT_TOKEN.")
        print("  2. Get your CHAT_ID by messaging @userinfobot.")
        print("  3. Run: python3 alerts/telegram_worker.py --token <TOKEN> --chat-id <CHAT_ID>")
    else:
        print(f"\nDispatched {res['dispatched_count']}/{res['total_at_risk']} alerts to Telegram Chat ID: {args.chat_id or os.environ.get('TELEGRAM_CHAT_ID')}")

    print("=" * 75)

if __name__ == "__main__":
    main()
