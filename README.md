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
