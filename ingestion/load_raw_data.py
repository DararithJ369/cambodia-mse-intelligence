"""
Cambodia MSE Intelligence - Data Ingestion & Transformation Pipeline
Cleans and standardizes raw datasets into SQLite / Parquet for downstream dbt & analytics.
Run with: python3 ingestion/load_raw_data.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "cambodia_mse.db")

def ingest_population(conn):
    print("Ingesting 2019 Population Census...")
    path = os.path.join(DATA_DIR, "Cambodia Population 2019.xlsx")
    df = pd.read_excel(path, sheet_name="Data", header=3).dropna(subset=["Province"])
    df = df[df["Province"] != "Total"].copy()
    df.columns = ["province_name", "population_total", "population_density_per_sqkm"]
    df["population_total"] = pd.to_numeric(df["population_total"])
    df["population_density_per_sqkm"] = pd.to_numeric(df["population_density_per_sqkm"])
    df.to_sql("stg_provinces", conn, if_exists="replace", index=False)
    print(f"  -> Ingested {len(df)} province records into 'stg_provinces'.")

def ingest_fuel_prices(conn):
    print("Ingesting Ministry of Commerce Fuel Prices...")
    path = os.path.join(DATA_DIR, "Gasoline price - Gas_EN.csv")
    df = pd.read_csv(path)
    df["record_date"] = pd.to_datetime(df["im_date"], format="%d-%m-%Y").dt.strftime("%Y-%m-%d")
    df = df.rename(columns={
        "regu_gas": "price_regular_gas_khr",
        "diesel_gas": "price_diesel_khr",
        "moc_no": "moc_notification_number",
        "reference": "document_reference"
    })
    df = df[["record_date", "price_regular_gas_khr", "price_diesel_khr", "moc_notification_number", "document_reference"]]
    df.to_sql("stg_fuel_prices", conn, if_exists="replace", index=False)
    print(f"  -> Ingested {len(df)} fuel price records into 'stg_fuel_prices'.")

def ingest_informal_wbes(conn):
    print("Ingesting WBES Informal Enterprise Indicators...")
    path = os.path.join(DATA_DIR, "CustomQuery-InformalSectorWBES-Sep-29-2026.xlsx")
    df = pd.read_excel(path, sheet_name="Custom Query")
    cambodia_cols = [c for c in df.columns if "Cambodia" in str(c)]
    sub_df = df[["All indicators\n  \n  Indicator*"] + cambodia_cols].dropna(subset=["All indicators\n  \n  Indicator*"]).copy()
    sub_df.columns = ["indicator_name", "val_battambang", "val_phnom_penh", "val_siem_reap", "val_sihanoukville"]
    sub_df.to_sql("stg_informal_wbes", conn, if_exists="replace", index=False)
    print(f"  -> Ingested {len(sub_df)} indicator rows into 'stg_informal_wbes'.")

def ingest_fat_survey(conn):
    print("Ingesting FAT Firm Technology Survey Microdata...")
    path = os.path.join(DATA_DIR, "fat_cambodia_disclosure.dta")
    df = pd.read_stata(path)
    # Extract core feature subset for performant SQL querying
    cols_to_keep = {
        "s7": "num_employees",
        "a3b": "foreign_ownership_pct",
        "a4e": "is_female_owner",
        "a5a": "industry_sector",
        "a6": "year_started",
        "a7a": "manager_experience_years",
        "a7b": "is_female_top_manager",
        "a7c": "manager_highest_education",
        "b1a": "has_electrical_connection",
        "b1b": "power_outages_per_year",
        "b2a": "owns_or_shares_generator",
        "b4a": "count_computers",
        "b4b": "count_smartphones",
        "b5a": "has_internet",
        "b5f": "has_website",
        "b5g": "uses_social_media",
        "b5h": "uses_cloud_services",
        "b13b1": "sales_premises",
        "b13b3": "sales_social_media",
        "b13b4": "sales_online_platforms",
        "s1c": "sampling_region"
    }
    subset = df[list(cols_to_keep.keys())].rename(columns=cols_to_keep)
    subset.to_sql("stg_fat_enterprises", conn, if_exists="replace", index=False)
    print(f"  -> Ingested {len(subset)} firm survey records into 'stg_fat_enterprises'.")

def ingest_osm_pois(conn):
    print("Ingesting OpenStreetMap POIs & Spatial Features...")
    csv_path = os.path.join(DATA_DIR, "osm_cambodia_pois.csv")
    if not os.path.exists(csv_path):
        print("  Generating osm_cambodia_pois.csv from cambodia-260927.osm.pbf...")
        import osmium
        class POIHandler(osmium.SimpleHandler):
            def __init__(self):
                super().__init__()
                self.pois = []
            def node(self, n):
                amenity = n.tags.get('amenity')
                shop = n.tags.get('shop')
                tourism = n.tags.get('tourism')
                category = 'retail_shop' if shop else ('amenity_commercial' if amenity in ['marketplace', 'fuel', 'bank', 'atm', 'restaurant', 'cafe', 'pharmacy', 'fast_food'] else ('hospitality' if tourism in ['hotel', 'guest_house'] else None))
                if category and n.location.valid():
                    self.pois.append({
                        'osm_id': n.id, 'lat': n.location.lat, 'lon': n.location.lon,
                        'category': category, 'poi_type': shop or amenity or tourism,
                        'name': n.tags.get('name', n.tags.get('name:en', 'Unnamed'))
                    })
        handler = POIHandler()
        handler.apply_file(os.path.join(DATA_DIR, "cambodia-260927.osm.pbf"), locations=True)
        df_pois = pd.DataFrame(handler.pois)
        df_pois.to_csv(csv_path, index=False)
    else:
        df_pois = pd.read_csv(csv_path)
    df_pois.to_sql("stg_osm_pois", conn, if_exists="replace", index=False)
    print(f"  -> Ingested {len(df_pois)} geospatial POI records into 'stg_osm_pois'.")

if __name__ == "__main__":
    print("="*60)
    print("INGESTING RAW DATA INTO SQLite DATA WAREHOUSE")
    print("="*60)
    conn = sqlite3.connect(DB_PATH)
    ingest_population(conn)
    ingest_fuel_prices(conn)
    ingest_informal_wbes(conn)
    ingest_fat_survey(conn)
    ingest_osm_pois(conn)
    conn.close()
    print("="*60)
    print(f"Database successfully updated: {DB_PATH}")
    print("="*60)
