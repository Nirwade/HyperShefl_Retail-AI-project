"""
prepare_for_github.py
Run once from your retail-ai-project root:
    conda activate retail-ai
    python prepare_for_github.py

This script will:
  1. Write the real ss_ai.py to streamlit_app/
  2. Create 500-row sample CSVs in data/sample/ (mirrors full data structure)
  3. Create scripts/setup_sample_data.py (for people who clone your repo)
  4. Delete all fix_*.py and patch_*.py junk files
  5. Create .gitignore
  6. Create README.md
"""

import os, sys, shutil, textwrap
from pathlib import Path

ROOT = Path(__file__).parent
print(f"\n{'='*60}")
print("HyperShelf DemandSense v2 — GitHub Preparation Script")
print(f"{'='*60}\n")

# ── 1. WRITE REAL ss_ai.py ─────────────────────────────────────
print("Step 1: Writing real ss_ai.py to streamlit_app/...")
ss_ai_content = '''"""
ss_ai.py — Safety Stock AI module
Python computes exact data from the filtered repl DataFrame.
LLaMA 3.2 explains the result in plain English.
Returns (text: str, fig: go.Figure | None)
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

C_BG="#0A1628";C_CARD="#0F2040";C_TEAL="#0D9488";C_TEAL2="#14B8A6"
C_RED="#EF4444";C_AMBER="#F59E0B";C_GREEN="#10B981";C_WHITE="#F0F9FF";C_GRAY="#64748B"

def _dl(title="",height=280):
    return dict(
        title=dict(text=title,font=dict(color=C_WHITE,size=13)),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_WHITE,family="IBM Plex Sans",size=11),height=height,
        margin=dict(l=48,r=20,t=40,b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)",zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)",zerolinecolor="rgba(255,255,255,0.1)"),
    )

def _llm(system,user,temperature=0.1):
    try:
        resp=requests.post("http://localhost:11434/api/generate",json={
            "model":"llama3.2:3b",
            "prompt":f"<|system|>\\n{system}\\n<|user|>\\n{user}\\n<|assistant|>",
            "stream":False,"options":{"temperature":temperature,"num_predict":280}},timeout=25)
        return resp.json().get("response","").strip()
    except Exception as e:
        return f"[LLaMA unavailable: {e}]"

def _has(q,*words):
    q=q.lower();return any(w in q for w in words)

def _handle_understocked(df,store_id):
    if "reorder_point" not in df.columns or "target" not in df.columns:
        return "Cannot compute understocked SKUs — reorder_point or target column missing.",None
    df=df.copy();df["units_short"]=(df["reorder_point"]-df["target"]).clip(lower=0)
    below=df[df["units_short"]>0].copy()
    if below.empty:
        return f"All SKUs at {store_id} are currently above their reorder point. Store is well stocked.",None
    below=below.sort_values("units_short",ascending=False);n=len(below)
    if "mu_daily" in df.columns:
        below["days_oos"]=(below["target"]/below["mu_daily"].clip(lower=0.1)).round(1)
        urgent=int((below["days_oos"]<3).sum())
    else:
        below["days_oos"]=None;urgent=None
    top=below.head(10)
    sku_col="sku_id" if "sku_id" in top.columns else None
    labels=top[sku_col].astype(str).tolist() if sku_col else [str(i) for i in top.index]
    vals=top["units_short"].tolist()
    cats=top["category"].tolist() if "category" in top.columns else [""]*len(labels)
    fig=go.Figure(go.Bar(x=vals,y=labels,orientation="h",
        marker_color=[C_RED if v>20 else C_AMBER for v in vals],
        text=[f"{v:.0f}u short" for v in vals],textposition="outside",
        textfont=dict(color=C_WHITE,size=10),customdata=cats,
        hovertemplate="%{y}<br>%{customdata}<br>Units short: %{x:.0f}<extra></extra>"))
    layout=_dl(f"Top 10 understocked SKUs — {store_id}",height=300)
    layout["yaxis"]["autorange"]="reversed";fig.update_layout(**layout)
    top5=below.head(5);lines=[]
    for _,row in top5.iterrows():
        lines.append(f"  {row.get(\'sku_id\',\'\')} ({row.get(\'category\',\'\')}): {row[\'units_short\']:.0f} units short, ~{row.get(\'days_oos\',\'?\'):.1f} days until OOS" if row.get("days_oos") else f"  {row.get(\'sku_id\',\'\')} ({row.get(\'category\',\'\')}): {row[\'units_short\']:.0f} units short")
    summary=(f"Store: {store_id}\\nTotal SKUs below ROP: {n}\\n"
        +(f"SKUs with less than 3 days until OOS: {urgent}\\n" if urgent else "")
        +"Top 5:\\n"+("\\n".join(lines)))
    system="You are a retail supply chain analyst. Answer ONLY using the data provided. Do not calculate or estimate. Be direct. Max 4 sentences."
    return _llm(system,f"Data:\\n{summary}\\n\\nQuestion: Which SKUs are most understocked right now?"),fig

def _handle_overstock_capital(df,store_id):
    if not all(c in df.columns for c in ["target","reorder_point","safety_stock_units"]):
        return "Cannot compute overstock — target, reorder_point, or safety_stock_units missing.",None
    df=df.copy();threshold=df["reorder_point"]+df["safety_stock_units"]*2
    over=df[df["target"]>threshold].copy();over["excess_units"]=(over["target"]-threshold).round(0)
    if over.empty:
        return f"No overstocked SKUs at {store_id}. Inventory within safety stock bounds.",None
    if "unit_price_actual" in over.columns:
        over["capital_locked"]=over["excess_units"]*over["unit_price_actual"]
        total_capital=over["capital_locked"].sum();capital_str=f"${total_capital:,.0f}"
    else:
        total_capital=None;capital_str="price data not available"
    n=len(over);fig=None
    if "category" in over.columns and total_capital:
        cat_g=over.groupby("category")["capital_locked"].sum().reset_index().sort_values("capital_locked",ascending=False)
        fig=go.Figure(go.Bar(x=cat_g["capital_locked"],y=cat_g["category"],orientation="h",
            marker_color=C_AMBER,text=[f"${v:,.0f}" for v in cat_g["capital_locked"]],
            textposition="outside",textfont=dict(color=C_WHITE,size=10)))
        layout=_dl(f"Capital locked by category — {store_id}",height=280)
        layout["yaxis"]["autorange"]="reversed";fig.update_layout(**layout)
    top=over.sort_values("excess_units",ascending=False).head(5);lines=[]
    for _,row in top.iterrows():
        cap=f"${row[\'capital_locked\']:,.0f}" if "capital_locked" in row else f"{row[\'excess_units\']:.0f} excess units"
        lines.append(f"  {row.get(\'sku_id\',\'\')} ({row.get(\'category\',\'\')}): {row[\'excess_units\']:.0f} excess units — {cap}")
    summary=f"Store: {store_id}\\nOverstocked SKUs: {n}\\nCapital locked: {capital_str}\\nTop 5:\\n"+("\\n".join(lines))
    system="You are a retail inventory analyst. Answer ONLY using the data provided. Focus on actionable insight. Max 4 sentences."
    return _llm(system,f"Data:\\n{summary}\\n\\nQuestion: How much capital is locked in overstock?"),fig

def _handle_below_rop(df,store_id):
    if not all(c in df.columns for c in ["reorder_point","target","mu_daily"]):
        return "Cannot compute — reorder_point, target, or mu_daily missing.",None
    df=df.copy();df["units_short"]=(df["reorder_point"]-df["target"]).clip(lower=0)
    df["days_until_oos"]=(df["target"]/df["mu_daily"].clip(lower=0.1)).round(1)
    below=df[df["units_short"]>0].sort_values("days_until_oos")
    if below.empty:
        return f"All SKUs at {store_id} are above ROP. No immediate reorders needed.",None
    critical=below[below["days_until_oos"]<3];warning=below[(below["days_until_oos"]>=3)&(below["days_until_oos"]<7)]
    colors=below["days_until_oos"].apply(lambda d:C_RED if d<3 else(C_AMBER if d<7 else C_TEAL2))
    sku_labels=below["sku_id"].astype(str) if "sku_id" in below.columns else below.index.astype(str)
    cat_labels=below["category"].astype(str) if "category" in below.columns else pd.Series([""]*len(below))
    fig=go.Figure();fig.add_trace(go.Scatter(x=below["days_until_oos"],y=below["units_short"],
        mode="markers",marker=dict(color=colors,size=7,opacity=0.85),
        text=sku_labels,customdata=cat_labels,
        hovertemplate="%{text}<br>%{customdata}<br>Days until OOS: %{x:.1f}<br>Units short: %{y:.0f}<extra></extra>"))
    layout=_dl(f"SKUs below ROP — {store_id} (red=<3 days)",height=280)
    fig.update_layout(**layout);fig.add_vline(x=3,line_dash="dash",line_color=C_RED,opacity=0.5)
    fig.add_vline(x=7,line_dash="dash",line_color=C_AMBER,opacity=0.5)
    fig.update_xaxes(title_text="Days until stockout");fig.update_yaxes(title_text="Units short of ROP")
    top5=below.head(5);lines=[]
    for _,row in top5.iterrows():
        lines.append(f"  {row.get(\'sku_id\',\'\')} ({row.get(\'category\',\'\')}): {row[\'days_until_oos\']:.1f} days, {row[\'units_short\']:.0f} units short")
    summary=(f"Store: {store_id}\\nBelow ROP: {len(below)}\\nCritical (<3 days): {len(critical)}\\nWarning (3-7 days): {len(warning)}\\nTop 5:\\n"+("\\n".join(lines)))
    system="You are a retail operations analyst. Answer ONLY using the data. Prioritise urgency. Max 4 sentences."
    return _llm(system,f"Data:\\n{summary}\\n\\nQuestion: Show SKUs below reorder point with days until stockout."),fig

def _handle_simulate(df,store_id,pct=-10):
    from scipy import stats as sc
    if not all(c in df.columns for c in ["safety_stock_units","sigma_daily","lead_time_days_avg","mu_daily"]):
        return "Cannot simulate — safety_stock_units, sigma_daily, lead_time_days_avg, or mu_daily missing.",None
    df=df.copy();df["ss_new"]=(df["safety_stock_units"]*(1+pct/100)).round(0)
    df["rop_new"]=(df["mu_daily"]*df["lead_time_days_avg"]+df["ss_new"]).round(0)
    denom=(df["sigma_daily"]*np.sqrt(df["lead_time_days_avg"])).clip(lower=0.01)
    df["z_new"]=(df["ss_new"]/denom).clip(upper=3.5)
    df["sl_new"]=df["z_new"].apply(lambda z:sc.norm.cdf(z)*100).round(1)
    avg_ss_old=df["safety_stock_units"].mean();avg_ss_new=df["ss_new"].mean()
    avg_rop_old=(df["mu_daily"]*df["lead_time_days_avg"]+df["safety_stock_units"]).mean()
    avg_rop_new=df["rop_new"].mean();avg_sl_new=df["sl_new"].mean();carry_delta=(avg_ss_new-avg_ss_old)*30
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Current",x=["Safety Stock","Avg ROP"],y=[avg_ss_old,avg_rop_old],
        marker_color=C_GRAY,text=[f"{avg_ss_old:.0f}u",f"{avg_rop_old:.0f}u"],
        textposition="outside",textfont=dict(color=C_WHITE,size=10)))
    fig.add_trace(go.Bar(name=f"Simulated ({pct:+d}%)",x=["Safety Stock","Avg ROP"],y=[avg_ss_new,avg_rop_new],
        marker_color=C_RED if pct<0 else C_GREEN,text=[f"{avg_ss_new:.0f}u",f"{avg_rop_new:.0f}u"],
        textposition="outside",textfont=dict(color=C_WHITE,size=10)))
    layout=_dl(f"Safety stock simulation ({pct:+d}%) — {store_id}",height=260);layout["barmode"]="group"
    fig.update_layout(**layout)
    summary=(f"Store: {store_id}\\nSimulation: {pct:+d}% SS change\\n"
        f"Current avg SS: {avg_ss_old:.1f}u\\nNew avg SS: {avg_ss_new:.1f}u (delta: {avg_ss_new-avg_ss_old:+.1f}u)\\n"
        f"New service level: {avg_sl_new:.1f}%\\nNew avg ROP: {avg_rop_new:.1f}u\\n"
        f"Carrying cost change/SKU/month: ${carry_delta:+,.0f}")
    direction="reduced" if pct<0 else "increased"
    system="You are a retail supply chain strategist. Answer ONLY using the simulation data. Give a clear recommendation. Max 4 sentences."
    return _llm(system,f"Data:\\n{summary}\\n\\nQuestion: What happens if we {direction} SS by {abs(pct)}%?",0.15),fig

def answer(question,df,store_id):
    """Main router. Returns (text, fig)."""
    q=question.strip().lower()
    if df is None or df.empty:
        return f"No data loaded for {store_id}. Select a store with replenishment data.",None
    if _has(q,"simulat","what if","what would","reduce","increase","decrease","lower","raise","change"):
        pct=-10;m=re.search(r\'([+-]?\\d+)\\s*%\',question)
        if m: pct=int(m.group(1))
        elif _has(q,"increase","raise"): pct=10
        return _handle_simulate(df,store_id,pct)
    if _has(q,"capital","locked","overstock","overstocked","excess","too much"):
        return _handle_overstock_capital(df,store_id)
    if _has(q,"below rop","below reorder","days until","reorder point"):
        return _handle_below_rop(df,store_id)
    if _has(q,"understock","understocked","which sku","sku","short","shortage"):
        return _handle_understocked(df,store_id)
    n_total=len(df)
    n_below=int((df["target"]<df["reorder_point"]).sum()) if all(c in df.columns for c in ["target","reorder_point"]) else "unknown"
    avg_ss=f"{df[\'safety_stock_units\'].mean():.1f}u" if "safety_stock_units" in df.columns else "unknown"
    summary=f"Store: {store_id}\\nTotal SKUs: {n_total}\\nSKUs below ROP: {n_below}\\nAvg safety stock: {avg_ss}"
    system="You are a retail supply chain analyst. Answer using only the store data provided. Max 4 sentences."
    return _llm(system,f"Data:\\n{summary}\\n\\nQuestion: {question}",0.15),None
'''

ss_path = ROOT / "streamlit_app" / "ss_ai.py"
ss_path.write_text(ss_ai_content)
print(f"  Written: {ss_path}")

# ── 2. CREATE SAMPLE CSVs ──────────────────────────────────────
import pandas as pd
import numpy as np

SAMPLE_ROWS = 500

# Map: original path → sample path (mirrors structure under data/sample/)
large_files = {
    "data/raw/output/csv/sales_transactions.csv":               "data/sample/raw/sales_transactions.csv",
    "data/raw/output/csv/demand_forecasts.csv":                 "data/sample/raw/demand_forecasts.csv",
    "data/raw/output/csv/inventory_snapshots.csv":              "data/sample/raw/inventory_snapshots.csv",
    "data/raw/output/csv/replenishment_logs.csv":               "data/sample/raw/replenishment_logs.csv",
    "data/raw/output/csv/stockout_events.csv":                  "data/sample/raw/stockout_events.csv",
    "data/raw/output/csv/store_layout.csv":                     "data/sample/raw/store_layout.csv",
    "data/processed/training/replenishment_policy_inputs_demandsense.csv": "data/sample/processed/replenishment_policy_inputs_demandsense.csv",
    "data/processed/training/demandSense_v2_predictions.csv":   "data/sample/processed/demandSense_v2_predictions.csv",
    "data/processed/training/backtest_daily_demandsense.csv":   "data/sample/processed/backtest_daily_demandsense.csv",
    "data/processed/nexus/allstore/network_master_alerts.csv":  "data/sample/processed/network_master_alerts.csv",
    "data/processed/training/weekly_alerts.csv":                "data/sample/processed/weekly_alerts.csv",
}

print(f"\nStep 2: Creating {SAMPLE_ROWS}-row sample CSVs in data/sample/...")
created = 0
skipped = 0
for src_rel, dst_rel in large_files.items():
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if not src.exists():
        print(f"  SKIP (not found): {src_rel}")
        skipped += 1
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        df = pd.read_csv(src, nrows=SAMPLE_ROWS)
        df.to_csv(dst, index=False)
        size_kb = dst.stat().st_size // 1024
        print(f"  OK  {dst_rel}  ({len(df)} rows, {size_kb}KB)")
        created += 1
    except Exception as e:
        print(f"  ERR {src_rel}: {e}")
        skipped += 1

print(f"\n  Created: {created} sample files   Skipped: {skipped}")

# ── 3. CREATE scripts/setup_sample_data.py ───────────────────
print("\nStep 3: Creating scripts/setup_sample_data.py...")
scripts_dir = ROOT / "scripts"
scripts_dir.mkdir(exist_ok=True)
setup_script = '''\
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

print("\\nDone. Run the dashboard:")
print("  python -m streamlit run streamlit_app/app.py --server.port 8511")
print("  python -m streamlit run streamlit_app/llm_chat.py --server.port 8501")
'''
(scripts_dir / "setup_sample_data.py").write_text(setup_script)
print("  Written: scripts/setup_sample_data.py")

# ── 4. DELETE fix_*.py and patch_*.py ────────────────────────
print("\nStep 4: Deleting fix_*.py and patch_*.py junk files...")
junk_patterns = ["fix_*.py", "patch_*.py", "diagnose.py"]
deleted = 0
for pattern in junk_patterns:
    for f in ROOT.glob(pattern):
        f.unlink()
        print(f"  Deleted: {f.name}")
        deleted += 1

# Delete root-level duplicates
for dup in ["store_comparison_ai.py", "whatif_ai.py"]:
    f = ROOT / dup
    if f.exists():
        f.unlink()
        print(f"  Deleted root duplicate: {dup}")
        deleted += 1

print(f"  Total deleted: {deleted} files")

# ── 5. CREATE .gitignore ──────────────────────────────────────
print("\nStep 5: Creating .gitignore...")
gitignore = """\
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/

# Environments
.env
.venv
env/
venv/

# Large data files — download from Google Drive link in README
data/raw/
data/processed/
models/*.pkl
models/*.joblib
models/*.bin

# Notebooks output (generated, not source)
notebooks/*.csv
notebooks/*.html
notebooks/nexus_alert_outputs/
notebooks/nexus_forecast_accuracy_outputs/

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Logs and outputs
logs/
outputs/
*.log

# Large model binaries
*.pkl
*.joblib
"""
(ROOT / ".gitignore").write_text(gitignore)
print("  Written: .gitignore")

# ── 6. CREATE README.md ───────────────────────────────────────
print("\nStep 6: Creating README.md...")
readme = """\
# HyperShelf DemandSense v2

**End-to-end retail AI pipeline — LightGBM demand forecasting, Random Forest phantom inventory detection, and LLaMA 3.2 explaining it all in plain English.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![LLaMA](https://img.shields.io/badge/LLaMA-3.2%203B-teal)](https://ollama.com)

> 478 stores · 85 suppliers · 1,265 SKUs · 3.27M predictions · $275.6M revenue recovered

---

## What this is

HyperShelf DemandSense v2 is a production-grade retail intelligence system built for a Globant AI Studios consulting engagement at the University at Buffalo. It replaces a 30-day moving average with probabilistic LightGBM forecasting, detects phantom inventory with a Random Forest classifier, and serves both through a locally-hosted LLaMA 3.2 chat interface.

| Metric | Baseline | DemandSense v2 |
|--------|----------|----------------|
| WAPE | 28.89% | **24.06%** |
| P90 Coverage | 32.7% | **89.75%** |
| Safety Stock | μ only | **Dynamic SS = Z × √(LT·σD² + D²·σLT²) × SF** |
| Phantom detection | None | **14,041 candidates flagged** |
| Revenue recovered | — | **$275.6M vs old policy** |

---

## Quick start (5 commands)

### Prerequisites
- macOS or Linux
- [Anaconda](https://www.anaconda.com/) or Miniconda
- [Homebrew](https://brew.sh/) (macOS)

### 1 — Clone and install
```bash
git clone https://github.com/Nirwade/retail-ai-project
cd retail-ai-project
conda create -n retail-ai python=3.10
conda activate retail-ai
pip install -r requirements.txt
```

### 2 — Set up sample data
```bash
python scripts/setup_sample_data.py
```
This copies 500-row sample CSVs to the correct paths so the dashboard boots immediately.
For the full 3.7GB dataset see the [Google Drive link below](#full-dataset).

### 3 — Install and start Ollama (local LLM)
```bash
brew install ollama
ollama pull llama3.2:3b
ollama serve
```
Leave this running in a terminal tab.

### 4 — Launch the dashboard (10 pages with AI)
```bash
conda activate retail-ai
python -m streamlit run streamlit_app/app.py --server.port 8511
```
Open: [http://localhost:8511](http://localhost:8511)

### 5 — Launch the standalone AI chat
```bash
conda activate retail-ai
python -m streamlit run streamlit_app/llm_chat.py --server.port 8501
```
Open: [http://localhost:8501](http://localhost:8501)

---

## Full dataset

The full dataset is 3.7GB and cannot be hosted on GitHub.

**[Download full data from Google Drive →](ADD_YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading, extract into the `data/` folder so the structure matches what
`scripts/setup_sample_data.py` expects.

---

## Project structure

```
retail-ai-project/
├── streamlit_app/
│   ├── app.py                 # 10-page Streamlit dashboard
│   ├── llm_chat.py            # Standalone conversational AI
│   ├── ss_ai.py               # Safety stock AI module
│   ├── executive_ai.py        # Executive overview AI
│   ├── store_ai.py            # Store deep dive AI
│   ├── forecast_ai.py         # Forecast & model AI
│   ├── phantom_ai.py          # Phantom inventory AI
│   ├── supplier_ai.py         # Supplier performance AI
│   └── ...                    # Other page AI modules
├── data/
│   ├── sample/                # 500-row sample CSVs (committed)
│   ├── raw/                   # Full raw data (gitignored — use Google Drive)
│   └── processed/             # Full processed data (gitignored — use Google Drive)
├── models/
│   ├── stores_features.csv    # Store metadata
│   └── products_features.csv  # Product metadata
├── notebooks/                 # Jupyter analysis notebooks
├── src/                       # Pipeline source code
├── mlops/                     # MLOps and monitoring scripts
├── assets/                    # Images and GIF demos
└── scripts/
    └── setup_sample_data.py   # Copies samples to correct paths
```

---

## Architecture

```
Raw CSVs (4.58GB, 18.3M rows)
    ↓ Feature engineering (22 features)
LightGBM → P10/P50/P90 demand forecasts (3.27M rows)
Random Forest → Phantom inventory candidates (14,041 flagged)
    ↓
Python router (600-line answer() function, exact CSV reads)
    ↓ structured text summary only (200 words max)
LLaMA 3.2 3B (local, via Ollama, temp 0.1/0.2)
    ↓
Streamlit dashboard (port 8511) + Standalone AI chat (port 8501)
```

**Core principle:** Python computes exact facts from CSVs. LLaMA explains them in plain English. The LLM never sees raw data — it cannot hallucinate numbers that are not already in its prompt.

---

## Models

| Component | Algorithm | Key metric |
|-----------|-----------|------------|
| Demand forecasting | LightGBM | WAPE 24.06% · P90 89.75% |
| Phantom detection | Random Forest | F2-optimised threshold · 14,041 candidates |
| Language interface | LLaMA 3.2 3B | Temperature 0.1 (facts) · 0.2 (explanations) |
| Safety stock | Stochastic formula | 8.33% stockout reduction in backtest |

---

## Dashboard — 10 pages

| Page | What the AI does |
|------|-----------------|
| Executive Overview | Network KPI summary · ROI simulation · regional breakdown |
| Store Deep Dive | Days-of-supply gauges · daily revenue loss · supplier directive |
| What-If Simulator | Live SS/ROP/SL recalculation · scenario comparison |
| Store Comparison | 5-factor priority breakdown · supplier scorecard |
| Safety Stock & ROP | SS% adjustment · below-ROP table · old vs AI policy |
| Phantom Inventory | Aisle check list · capital locked · confidence scoring |
| Localization | Category mismatch vs region · revenue uplift estimate |
| Supplier Performance | Fill rate vs reliability · risk tier assignment |
| Forecast & Model | WAPE spike detection · P10/P90 band · category accuracy |
| Export Center | Role-based downloads · Excel with risk tier color coding |

---

## Built by

University at Buffalo + Globant AI Studios · Nexus Team 2 · May 2026

Portfolio: [nirwade.github.io](https://nirwade.github.io)
"""
(ROOT / "README.md").write_text(readme)
print("  Written: README.md")

# ── SUMMARY ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DONE. Your repo is ready for GitHub.")
print(f"{'='*60}")
print("""
Next steps (run these in your terminal, one at a time):

  git init
  git add .
  git commit -m "Initial commit: HyperShelf DemandSense v2"
  git branch -M main
  git remote add origin https://github.com/Nirwade/retail-ai-project.git
  git push -u origin main

Before pushing:
  1. Create the repo at github.com/Nirwade → New repository
     Name: retail-ai-project   (make it Public)
     Do NOT add README or .gitignore (we already created them)

  2. Upload data/ to Google Drive and paste the share link into README.md
     where it says ADD_YOUR_GOOGLE_DRIVE_LINK_HERE
""")