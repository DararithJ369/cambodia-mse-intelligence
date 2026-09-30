#!/usr/bin/env bash
# ==============================================================================
# Cambodia MSE Intelligence - Master Pipeline Execution Script
# Institute of Technology of Cambodia (ITC) - Department of Applied Mathematics and Statistics
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "🇰🇭 CAMBODIA MSE INTELLIGENCE — END-TO-END PIPELINE"
echo "Department of Applied Mathematics and Statistics — ITC"
echo "========================================================================"

# Step 0: Check Environment
echo -e "\n[STEP 0/5] Checking Python Environment & Dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH."
    exit 1
fi
python3 -c "
import duckdb, pandas, numpy, lightgbm, sklearn, matplotlib, seaborn, streamlit
print('  ✅ All core dependencies available (DuckDB, Pandas, LightGBM, Streamlit, etc.)')
"

# Step 1 & 2: Data Ingestion & Synthetic POS Generation
echo -e "\n[STEP 1/5] Ingesting Public Open Data Feeds..."
python3 ingestion/load_raw_data.py

echo -e "\n[STEP 2/5] Generating 90-Day Dual-Currency Synthetic POS Dataset..."
python3 ingestion/generate_synthetic_pos.py

# Step 3: Local Operational Data Store (ODS) & dbt Transformations
echo -e "\n[STEP 3/5] Setting up Local Operational Data Store (sme_cambodia.duckdb)..."
python3 ingestion/setup_ods_database.py

echo -e "\n[STEP 3.1/5] Executing dbt Star Schema Transformations & Data Integrity Tests..."
python3 -c "
import os
from dbt.cli.main import dbtRunner

runner = dbtRunner()
cli_args = ['run', '--project-dir', 'models_dbt', '--profiles-dir', 'models_dbt']
res = runner.invoke(cli_args)
if not res.success:
    print('❌ dbt run encountered errors.')
    exit(1)
print('  ✅ All 15 dbt models compiled and materialized into DuckDB.')

test_res = runner.invoke(['test', '--project-dir', 'models_dbt', '--profiles-dir', 'models_dbt'])
if not test_res.success:
    print('❌ dbt tests failed.')
    exit(1)
print('  ✅ All 32/32 dbt tests passed (PK Uniqueness, Non-Null, Referential Integrity, Accepted Values).')
"

# Step 4: Analytical & ML Modules + Notebook Rebuilding
echo -e "\n[STEP 4/5] Executing ML Modules & Rebuilding All 11 Governed Notebooks..."
python3 scripts/rebuild_all_notebooks.py

# Step 5: Operational Decision Support & Alerting Engine
echo -e "\n[STEP 5/5] Running Operational Decision Support & Alert Systems..."
python3 alerts/cost_shock_alert.py
echo ""
python3 alerts/stockout_alert.py
echo ""
python3 alerts/telegram_worker.py --dry-run

echo -e "\n========================================================================"
echo "🎉 PIPELINE EXECUTED SUCCESSFULLY WITH ZERO ERRORS!"
echo "========================================================================"
echo "To explore the interactive Business Intelligence application, run:"
echo "    python3 -m streamlit run app_streamlit/main.py"
echo "    (or: streamlit run app_streamlit/main.py)"
echo "========================================================================"
