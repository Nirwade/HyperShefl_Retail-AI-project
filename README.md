# HyperShelf DemandSense v2

**End-to-end retail AI pipeline — LightGBM demand forecasting, Random Forest phantom inventory detection, and LLaMA 3.2 explaining every decision in plain English.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)](https://streamlit.io)
[![LLaMA](https://img.shields.io/badge/LLaMA-3.2%203B-teal)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> 478 stores · 85 suppliers · 1,265 SKUs · 3.27M predictions · $275.6M revenue recovered

**Portfolio website:** [nirwade.github.io](https://nirwade.github.io)

---

## What this is

HyperShelf DemandSense v2 is a production-grade retail intelligence system built for a Globant AI Studios consulting engagement at the University at Buffalo. It replaces a 30-day moving average with probabilistic LightGBM forecasting, detects phantom inventory with a Random Forest classifier, and serves both through a locally-hosted LLaMA 3.2 chat interface that explains every number without hallucinating.

| Metric | Baseline (MovingAvg30) | DemandSense v2 |
|--------|----------------------|----------------|
| WAPE | 28.89% | **24.06%** |
| P90 Coverage | 32.7% | **89.75%** |
| Safety Stock | Target = μ (no buffer) | **SS = Z × √(LT·σD² + D²·σLT²) × SF** |
| Phantom detection | None | **14,041 candidates flagged** |
| Revenue recovered | — | **$275.6M vs old policy** |
| Stockout events prevented | — | **188,143** |

---

## Quick start

### Prerequisites
- macOS or Linux
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [Homebrew](https://brew.sh/) (macOS only — for Ollama)

### Step 1 — Clone and install
```bash
git clone https://github.com/Nirwade/HyperShelf_Retail-AI-project
cd HyperShelf_Retail-AI-project
conda create -n retail-ai python=3.10
conda activate retail-ai
pip install -r requirements.txt
```

### Step 2 — Set up sample data
```bash
python scripts/setup_sample_data.py
```
This copies 500-row sample CSVs to the correct paths so the dashboard boots immediately with real-looking data. For the full 3.7GB dataset see [Full Dataset](https://drive.google.com/drive/folders/1jFjS_lV4rXaEa_0GM0_-3jfDOr?usp=sharing....c/ss)

### Step 3 — Install Ollama and pull LLaMA 3.2
```bash
brew install ollama
ollama pull llama3.2:3b
ollama serve
```
Leave `ollama serve` running in this terminal tab. It serves the local LLM on port 11434.

### Step 4 — Launch the 10-page AI dashboard
```bash
conda activate retail-ai
python -m streamlit run streamlit_app/app.py --server.port 8511
```
Open: [http://localhost:8511](http://localhost:8511)

### Step 5 — Launch the standalone AI chat
```bash
conda activate retail-ai
python -m streamlit run streamlit_app/llm_chat.py --server.port 8501
```
Open: [http://localhost:8501](http://localhost:8501)

---

## Full dataset

The full dataset is 3.7GB and cannot be hosted on GitHub. The sample data (Step 2) is enough to run and explore the full dashboard.

**[Download full dataset from Google Drive →](ADD_YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading, extract into the `data/` folder preserving the directory structure.

---

## Architecture

```
Raw CSVs (4.58GB · 18.3M rows · 7 source files)
    ↓ Feature engineering — 22 features across 5 groups
LightGBM → P10/P50/P90 demand forecasts (3.27M prediction rows)
Random Forest → Phantom inventory candidates (14,041 flagged)
    ↓
Python router — 600-line answer() function
  Reads exact values from CSVs · computes facts · passes 200-word summary
    ↓
LLaMA 3.2 3B (local via Ollama · temp 0.1 facts · 0.2 explanations)
    ↓
Streamlit dashboard (port 8511) · Standalone AI chat (port 8501)
```

**Core principle:** Python computes exact facts from CSVs. LLaMA explains them in plain English. The LLM never sees raw data and cannot hallucinate numbers that are not already in its prompt.

---

## Project structure

```
HyperShelf_Retail-AI-project/
├── streamlit_app/
│   ├── app.py                    # 10-page Streamlit dashboard
│   ├── llm_chat.py               # Standalone conversational AI
│   ├── ss_ai.py                  # Safety stock AI module
│   ├── executive_ai.py           # Executive overview AI
│   ├── store_ai.py               # Store deep dive AI
│   ├── forecast_ai.py            # Forecast & model AI
│   ├── phantom_ai.py             # Phantom inventory AI
│   ├── supplier_ai.py            # Supplier performance AI
│   ├── loc_ai.py                 # Localization AI
│   ├── store_comparison_ai.py    # Store comparison AI
│   ├── whatif_ai.py              # What-if simulator AI
│   └── export_ai.py              # Export center AI
├── src/
│   ├── pipeline.py               # Full ML training pipeline
│   ├── analytics.py              # Nexus analytics layer
│   ├── mlops.py                  # MLOps and monitoring
│   ├── model_registry.py         # Champion/Challenger registry
│   └── tools.py                  # LLM tool dispatch functions
├── data/
│   ├── sample/                   # 500-row sample CSVs (committed)
│   ├── raw/                      # Full raw data (gitignored)
│   └── processed/                # Full processed data (gitignored)
├── models/
│   ├── stores_features.csv       # Store metadata
│   └── products_features.csv     # Product metadata
├── notebooks/                    # Jupyter analysis notebooks
│   ├── Forecast.ipynb            # LightGBM training and evaluation
│   ├── nexus_complete_final.ipynb
│   ├── nexus_forecast_accuracy.ipynb
│   └── ...
├── assets/                       # Dashboard screenshots and demos
├── mlops/                        # MLOps scripts and monitoring
├── scripts/
│   └── setup_sample_data.py      # Copy samples to correct paths
├── index.html                    # Portfolio website (also at nirwade.github.io)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Models

| Component | Algorithm | Key metrics |
|-----------|-----------|------------|
| Demand forecasting | LightGBM | WAPE 24.06% · P90 89.75% · Bias -0.148 |
| Phantom detection | Random Forest | F2-optimised · 14,041 candidates |
| Language interface | LLaMA 3.2 3B | Temp 0.1 facts · 0.2 explanations · sub-2s |
| Safety stock | Stochastic formula | 8.33% stockout reduction in backtest |

---

## Dashboard — 10 pages

| # | Page | What the AI module does |
|---|------|------------------------|
| 01 | Executive Overview | Network KPI summary · ROI simulation · top stores by urgency |
| 02 | Store Deep Dive | Days-of-supply gauges · daily revenue loss · supplier directive |
| 03 | What-If Simulator | Live SS/ROP/SL recalculation · 3-scenario comparison |
| 04 | Store Comparison | 5-factor priority breakdown · supplier scorecard |
| 05 | Safety Stock & ROP | SS% adjustment · below-ROP table · old vs AI policy |
| 06 | Phantom Inventory | Aisle check list · capital locked · confidence scoring |
| 07 | Localization | Category mismatch vs region · revenue uplift estimate |
| 08 | Supplier Performance | Fill rate vs reliability · risk tier assignment |
| 09 | Forecast & Model | WAPE spike detection · P10/P90 band · category accuracy |
| 10 | Export Center | Role-based downloads · Excel with risk tier color coding |

---

## MLOps governance

- **2-week** scheduled retraining cadence (Continuous Training)
- **12-week** rolling validation window
- **Champion/Challenger** deployment — all 3 gates must pass simultaneously:
  - WAPE must improve
  - P90 must stay within 88–92%
  - Zero regression in Beverages or Snacks categories
- **Emergency triggers:** WAPE worsens >1.5pts · P90 exits band · Bias >0.5u

---

## Tech stack

| Technology | Role |
|-----------|------|
| LightGBM | Gradient boosting demand forecasting |
| scikit-learn | Random Forest phantom classifier |
| LLaMA 3.2 3B | Local LLM via Ollama |
| Streamlit | Dashboard and chat interface |
| Plotly | All visualisations including in-chat charts |
| Pandas + scipy | Data computation and statistical functions |
| Python 3.10 | Everything |

---

## Built by

University at Buffalo · Globant AI Studios · Nexus Team 2 · May 2026

Portfolio: [nirwade.github.io](https://nirwade.github.io) · Code: [github.com/Nirwade/HyperShelf_Retail-AI-project](https://github.com/Nirwade/HyperShelf_Retail-AI-project)
