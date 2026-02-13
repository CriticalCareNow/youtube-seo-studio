#!/usr/bin/env bash
set -euo pipefail
cd '/Users/haneymallemat/Documents/New project'
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
