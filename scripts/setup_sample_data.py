"""
setup_sample_data.py
Run this after cloning the repo to set up sample data so the dashboard boots immediately.
For the full 3.7GB dataset see the Google Drive link in README.md.

Usage:
    conda activate retail-ai
    python scripts/setup_sample_data.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

copies = [
    ("data/sample/processed/replenishment_policy_inputs_demandsense.csv",
     "data/processed/training/replenishment_policy_inputs_demandsense.csv"),
    ("data/sample/processed/demandSense_v2_predictions.csv",
     "data/processed/training/demandSense_v2_predictions.csv"),
    ("data/sample/processed/backtest_daily_demandsense.csv",
     "data/processed/training/backtest_daily_demandsense.csv"),
    ("data/sample/processed/network_master_alerts.csv",
     "data/processed/nexus/allstore/network_master_alerts.csv"),
    ("data/sample/processed/weekly_alerts.csv",
     "data/processed/training/weekly_alerts.csv"),
    ("data/sample/raw/sales_transactions.csv",
     "data/raw/output/csv/sales_transactions.csv"),
    ("data/sample/raw/demand_forecasts.csv",
     "data/raw/output/csv/demand_forecasts.csv"),
    ("data/sample/raw/inventory_snapshots.csv",
     "data/raw/output/csv/inventory_snapshots.csv"),
    ("data/sample/raw/replenishment_logs.csv",
     "data/raw/output/csv/replenishment_logs.csv"),
    ("data/sample/raw/stockout_events.csv",
     "data/raw/output/csv/stockout_events.csv"),
    ("data/sample/raw/store_layout.csv",
     "data/raw/output/csv/store_layout.csv"),
]

print("Setting up sample data...")
for src_rel, dst_rel in copies:
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if not src.exists():
        print(f"  SKIP (not found): {src_rel}")
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied to {dst_rel}")

print("\nDone. Run the dashboard:")
print("  python -m streamlit run streamlit_app/app.py --server.port 8511")
print("  python -m streamlit run streamlit_app/llm_chat.py --server.port 8501")
