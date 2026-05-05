# ─────────────────────────────────────────────────────────────────
# src/model_registry.py
# Loads all saved .joblib files once at startup.
# All tool functions import from here — models load only once.
# ─────────────────────────────────────────────────────────────────
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
PROCESSED    = PROJECT_ROOT / "data/processed/training"
CSV_PATH     = PROJECT_ROOT / "data/raw/output/csv"

class ModelRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        print("[ModelRegistry] Loading all model artifacts...")

        # ── Forecast models ───────────────────────────────────────
        self.p50_high = joblib.load(MODELS_DIR / "demandsense_p50_high.joblib")
        self.p50_low  = joblib.load(MODELS_DIR / "demandsense_p50_low.joblib")
        self.p05      = joblib.load(MODELS_DIR / "demandsense_p05.joblib")
        self.p95      = joblib.load(MODELS_DIR / "demandsense_p95.joblib")

        # ── Phantom model ─────────────────────────────────────────
        self.rf_phantom      = joblib.load(MODELS_DIR / "rf_phantom.joblib")
        self.phantom_threshold = joblib.load(MODELS_DIR / "phantom_threshold.joblib")
        self.phantom_features  = joblib.load(MODELS_DIR / "phantom_features.joblib")

        # ── Feature lists and encoders ────────────────────────────
        self.forecast_features = joblib.load(MODELS_DIR / "forecast_features.joblib")
        self.cat_cols          = joblib.load(MODELS_DIR / "cat_cols.joblib")
        self.hi_tiers          = joblib.load(MODELS_DIR / "hi_tiers.joblib")
        self.lo_tiers          = joblib.load(MODELS_DIR / "lo_tiers.joblib")
        self.csl_by_tier       = joblib.load(MODELS_DIR / "csl_by_tier.joblib")
        self.le_region         = joblib.load(MODELS_DIR / "le_region.joblib")
        self.le_format         = joblib.load(MODELS_DIR / "le_format.joblib")
        self.le_tier           = joblib.load(MODELS_DIR / "le_tier.joblib")
        self.le_cat            = joblib.load(MODELS_DIR / "le_cat.joblib")

        # ── Reference tables ──────────────────────────────────────
        self.stores_fe   = pd.read_parquet(MODELS_DIR / "stores_features.parquet")
        self.products_fe = pd.read_parquet(MODELS_DIR / "products_features.parquet")

        # ── Pre-computed outputs (fast lookup — no inference needed)
        print("[ModelRegistry] Loading pre-computed outputs...")
        self.predictions  = pd.read_csv(
            PROCESSED / "demandSense_v2_predictions.csv",
            parse_dates=["date"]
        )
        self.repl_inputs  = pd.read_csv(
            PROCESSED / "replenishment_policy_inputs_demandsense.csv",
            parse_dates=["date"]
        )
        self.weekly_monitor = pd.read_csv(
            PROCESSED / "weekly_monitor_demandsense.csv"
        )
        self.model_summary = pd.read_csv(
            PROCESSED / "demandSense_model_summary.csv"
        )
        self.category_impact = pd.read_csv(
            PROCESSED / "category_impact_scores.csv"
        )
        self.stores   = pd.read_csv(CSV_PATH / "stores.csv")
        self.products = pd.read_csv(CSV_PATH / "products.csv")
        self.suppliers = pd.read_csv(CSV_PATH / "suppliers.csv")

        print(f"[ModelRegistry] Ready — "
              f"{len(self.predictions):,} prediction rows · "
              f"{len(self.repl_inputs):,} replenishment rows")


# Singleton — import this anywhere
registry = ModelRegistry()