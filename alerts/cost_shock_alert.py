"""
Cambodia MSE Intelligence - Fuel Cost Shock Alert System
Monitors Ministry of Commerce fuel price notifications and triggers business alerts.
Run with: python3 alerts/cost_shock_alert.py
"""

import os
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data", "Gasoline price - Gas_EN.csv")

def evaluate_fuel_alerts(threshold_pct=5.0):
    print("="*60)
    print("CAMBODIA MSE INTELLIGENCE - COST SHOCK ALERT SYSTEM")
    print("="*60)
    
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["im_date"], format="%d-%m-%Y")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Identify unique Ministry of Commerce notifications
    notifs = df.drop_duplicates(subset=["moc_no"]).sort_values("date").reset_index(drop=True)
    
    latest = notifs.iloc[-1]
    prev = notifs.iloc[-2]
    
    reg_change = ((latest["regu_gas"] - prev["regu_gas"]) / prev["regu_gas"]) * 100
    die_change = ((latest["diesel_gas"] - prev["diesel_gas"]) / prev["diesel_gas"]) * 100
    
    print(f"Latest Notification: MoC No. {latest['moc_no']} ({latest['date'].strftime('%Y-%m-%d')})")
    print(f"Previous Notification: MoC No. {prev['moc_no']} ({prev['date'].strftime('%Y-%m-%d')})")
    print(f"Regular Gasoline: {prev['regu_gas']:,.0f} -> {latest['regu_gas']:,.0f} KHR ({reg_change:+.2f}%)")
    print(f"Diesel Fuel:      {prev['diesel_gas']:,.0f} -> {latest['diesel_gas']:,.0f} KHR ({die_change:+.2f}%)")
    print("-" * 60)
    
    triggered = False
    if abs(reg_change) >= threshold_pct:
        triggered = True
        status = "PRICE SURGE" if reg_change > 0 else "PRICE DROP"
        print(f"⚠️ [ALERT] REGULAR GASOLINE {status}: {reg_change:+.2f}% change exceeds ±{threshold_pct}% threshold!")
        print("  -> Impact: Higher last-mile delivery and tuk-tuk distribution fees for urban MSEs.")
        
    if abs(die_change) >= threshold_pct:
        triggered = True
        status = "PRICE SURGE" if die_change > 0 else "PRICE DROP"
        print(f"🚨 [ALERT] DIESEL FUEL {status}: {die_change:+.2f}% change exceeds ±{threshold_pct}% threshold!")
        print("  -> Impact: Heavy transport surcharge, inter-provincial shipping hikes, and increased generator backup costs.")
        
    if not triggered:
        print(f"✅ Fuel prices are stable (both below ±{threshold_pct}% threshold).")
        
    print("="*60)

if __name__ == "__main__":
    evaluate_fuel_alerts()
