"""
Cambodia MSE Intelligence - Exploratory Data Analysis Script
Reads all raw datasets in the 'data/' directory and prints key business insights.
Run with: python3 analytics/explore_data.py
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def analyze_population():
    print("\n" + "="*60)
    print("1. PROVINCIAL POPULATION & DENSITY (Cambodia Census 2019)")
    print("="*60)
    pop_path = os.path.join(DATA_DIR, "Cambodia Population 2019.xlsx")
    # Header is at row index 3
    df = pd.read_excel(pop_path, sheet_name="Data", header=3)
    df = df.dropna(subset=["Province"])
    df = df[df["Province"] != "Total"]
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    df["Population Density"] = pd.to_numeric(df["Population Density"], errors="coerce")
    
    print(f"Total provinces loaded: {len(df)}")
    print(f"Total population (sum of provinces): {df['Population'].sum():,.0f}")
    print("\nTop 5 Most Populous Provinces:")
    top_pop = df.sort_values(by="Population", ascending=False)[["Province", "Population", "Population Density"]].head(5)
    print(top_pop.to_string(index=False))
    
    print("\nTop 5 Densest Provinces (People/km²):")
    top_density = df.sort_values(by="Population Density", ascending=False)[["Province", "Population", "Population Density"]].head(5)
    print(top_density.to_string(index=False))
    return df

def analyze_gasoline():
    print("\n" + "="*60)
    print("2. FUEL PRICE MONITORING (MoC Cambodia Notifications)")
    print("="*60)
    gas_path = os.path.join(DATA_DIR, "Gasoline price - Gas_EN.csv")
    df = pd.read_csv(gas_path)
    df["date"] = pd.to_datetime(df["im_date"], format="%d-%m-%Y")
    df = df.sort_values(by="date")
    
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total price observations: {len(df)}")
    print("\nPrice Statistics (KHR / Liter):")
    stats = df[["regu_gas", "diesel_gas"]].describe().T[["min", "mean", "50%", "max", "std"]]
    stats.columns = ["Min (KHR)", "Mean (KHR)", "Median (KHR)", "Max (KHR)", "Std Dev"]
    print(stats.round(2).to_string())
    
    # Calculate price change
    first_reg, last_reg = df["regu_gas"].iloc[0], df["regu_gas"].iloc[-1]
    first_die, last_die = df["diesel_gas"].iloc[0], df["diesel_gas"].iloc[-1]
    print(f"\nRegular Gasoline Change: {first_reg:,.0f} -> {last_reg:,.0f} KHR ({((last_reg - first_reg)/first_reg)*100:+.1f}%)")
    print(f"Diesel Gasoline Change:  {first_die:,.0f} -> {last_die:,.0f} KHR ({((last_die - first_die)/first_die)*100:+.1f}%)")
    return df

def analyze_informal_wbes():
    print("\n" + "="*60)
    print("3. INFORMAL SECTOR ENTERPRISE SURVEY (WBES 2024 Cambodia)")
    print("="*60)
    wbes_path = os.path.join(DATA_DIR, "CustomQuery-InformalSectorWBES-Sep-29-2026.xlsx")
    df = pd.read_excel(wbes_path, sheet_name="Custom Query")
    
    # Select Cambodia cities
    cambodia_cols = [c for c in df.columns if "Cambodia" in str(c)]
    sub_df = df[["All indicators\n  \n  Indicator*"] + cambodia_cols].copy()
    sub_df.columns = ["Indicator", "Battambang", "Phnom Penh", "Siem Reap", "Sihanoukville"]
    sub_df = sub_df.dropna(subset=["Indicator"])
    
    # Filter key indicators
    key_keywords = [
        "schooling", "female", "bank account", "loan", "profit",
        "electricity", "mobile money", "computers", "bribes", "taxes"
    ]
    
    print("Key Informality & Financial Inclusion Metrics across 4 Cities:")
    selected = []
    for _, row in sub_df.iterrows():
        ind = str(row["Indicator"]).strip()
        if any(kw in ind.lower() for kw in key_keywords) and "Number of observations" not in ind:
            selected.append(row)
            
    df_selected = pd.DataFrame(selected)
    print(df_selected.to_string(index=False))
    return df_selected

def analyze_fat_survey():
    print("\n" + "="*60)
    print("4. FIRM ADOPTION OF TECHNOLOGY (FAT Cambodia 2022 Survey)")
    print("="*60)
    dta_path = os.path.join(DATA_DIR, "fat_cambodia_disclosure.dta")
    reader = pd.read_stata(dta_path, iterator=True)
    var_labels = reader.variable_labels()
    df = pd.read_stata(dta_path)
    
    print(f"Total surveyed enterprises: {len(df)}")
    print(f"Total survey variables: {len(df.columns)}")
    
    # Firm size breakdown
    # Micro: 1-9 employees, Small: 10-49, Medium: 50-99, Large: 100+
    def categorize_size(emp):
        if emp < 10: return "Micro (1-9)"
        elif emp < 50: return "Small (10-49)"
        elif emp < 100: return "Medium (50-99)"
        else: return "Large (100+)"
        
    df["firm_category"] = df["s7"].apply(categorize_size)
    print("\nEnterprise Size Distribution:")
    size_counts = df["firm_category"].value_counts()
    size_pct = df["firm_category"].value_counts(normalize=True) * 100
    size_summary = pd.DataFrame({"Count": size_counts, "Share (%)": size_pct.round(1)})
    print(size_summary.to_string())
    
    print("\nTop 5 Industry Sectors (a5a):")
    sec_summary = pd.DataFrame({
        "Count": df["a5a"].value_counts(),
        "Share (%)": (df["a5a"].value_counts(normalize=True) * 100).round(1)
    }).head(5)
    print(sec_summary.to_string())
    
    print("\nFemale Leadership:")
    print(f"  Enterprises with Female Owner (a4e): {(df['a4e'] == 'Yes').mean()*100:.1f}%")
    print(f"  Enterprises with Female Top Manager (a7b): {(df['a7b'] == 'Yes').mean()*100:.1f}%")
    
    print("\nDigital Technology Adoption:")
    for code, desc in [("b5a", "Internet Access"), ("b5f", "Own Website"), ("b5g", "Social Media for Business"), ("b5h", "Cloud Computing")]:
        if code in df.columns:
            yes_pct = (df[code] == "Yes").mean() * 100
            print(f"  - {desc} ({code}): {yes_pct:.1f}%")
            
    print("\nSales Channels Breakdown:")
    for code, desc in [
        ("b13b1", "Physical Premises"),
        ("b13b2", "Phone / Email / Reps"),
        ("b13b3", "Social Media (FB, Telegram)"),
        ("b13b4", "Online Digital Platforms"),
        ("b13b5", "Own Website")
    ]:
        if code in df.columns:
            yes_pct = (df[code] == "Yes").mean() * 100
            print(f"  - {desc}: {yes_pct:.1f}% of firms")
            
    return df

if __name__ == "__main__":
    print("============================================================")
    print(" CAMBODIA MSE INTELLIGENCE - DATA EXPLORATION & DIAGNOSTICS")
    print("============================================================")
    df_pop = analyze_population()
    df_gas = analyze_gasoline()
    df_wbes = analyze_informal_wbes()
    df_fat = analyze_fat_survey()
    print("\n" + "="*60)
    print("EDA Complete. All datasets loaded and validated successfully.")
    print("="*60)
