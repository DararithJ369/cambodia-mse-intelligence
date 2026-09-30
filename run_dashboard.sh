#!/usr/bin/env bash
# ==============================================================================
# Cambodia MSE Intelligence - Quick Dashboard Launcher
# Runs the user-friendly Store Operations Web App (Step 5)
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "========================================================================"
echo "🇰🇭 LAUNCHING CAMBODIA MSE INTELLIGENCE DASHBOARD"
echo "========================================================================"
echo "Starting Streamlit web server at http://localhost:8501 ..."
echo "Press Ctrl+C to terminate."
echo "========================================================================"

# Try uv run if uv is installed and virtualenv exists, else fallback to python3 -m streamlit
if command -v uv &> /dev/null && [ -d ".venv" ]; then
    uv run streamlit run app_streamlit/main.py
elif command -v python3 &> /dev/null; then
    python3 -m streamlit run app_streamlit/main.py
elif command -v streamlit &> /dev/null; then
    streamlit run app_streamlit/main.py
else
    echo "❌ Neither uv, python3, nor streamlit was found."
    exit 1
fi
