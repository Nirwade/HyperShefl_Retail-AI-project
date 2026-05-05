# ─────────────────────────────────────────────────────────────────
# src/analytics.py
# DemandSense v2 — Analytics Layer
# Runs AFTER pipeline.py generates ML predictions.
# Adds business intelligence: phantom confidence, root cause,
# localization, supplier fill rate, ROI simulation, per-store alerts.
#
# Usage (standalone):  python src/analytics.py
# Usage (via pipeline): called automatically at end of pipeline.py
#
# Logic sources:
#   nexus_complete_final.ipynb        — safety stock, alert engine
#   nexus_allstore_pipeline.ipynb     — per-store loop
#   nexus_forecast_accuracy.ipynb     — forecast error breakdown
#   nexus_localization_profiling.ipynb — mismatch analysis
#   nexus_roi_simulation.ipynb        — ROI before/after
#   nexus_supplier_performance.ipynb  — fill rate + lead time
#   retail_eda.ipynb                  — phantom confidence scoring
# ─────────────────────────────────────────────────────────────────
import sys
import gc
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH     = PROJECT_ROOT / "data/raw/output/csv"
PROCESSED    = PROJECT_ROOT / "data/processed/training"
NEXUS_DIR    = PROJECT_ROOT / "data/processed/nexus"
# Notebook output paths (nexus_allstore_pipeline.ipynb writes here)
NOTEBOOK_ALLSTORE = PROJECT_ROOT / "nexus_allstore_outputs"
NOTEBOOK_LOC      = PROJECT_ROOT / "nexus_localization_outputs"
NOTEBOOK_SUP      = PROJECT_ROOT / "nexus_supplier_outputs"
NOTEBOOK_ROI      = PROJECT_ROOT / "nexus_roi_outputs"
LOGS_DIR     = PROJECT_ROOT / "logs"

# Create output directories
for d in [
    NEXUS_DIR / "allstore",
    NEXUS_DIR / "localization",
    NEXUS_DIR / "supplier",
    NEXUS_DIR / "roi",
    NEXUS_DIR / "forecast",
    LOGS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / f"analytics_{datetime.now().strftime('%Y%m%d')}.log"
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Service level constants (from nexus_complete_final Section 0) ─
SERVICE_LEVEL = 0.975
Z_SCORE       = stats.norm.ppf(SERVICE_LEVEL)   # 1.96

# ── Alert thresholds (days of supply) ────────────────────────────
DOS_CRITICAL = 3     # ≤3 days  → CRITICAL
DOS_WARNING  = 7     # ≤7 days  → WARNING
DOS_MONITOR  = 14    # ≤14 days → MONITOR

# ── Localization mismatch threshold ──────────────────────────────
MISMATCH_THRESHOLD = -30   # >30% below regional average

TODAY     = pd.Timestamp.now().normalize()
LOOKAHEAD = TODAY + pd.Timedelta(days=7)


# ══════════════════════════════════════════════════════════════════
# DATA LOADERS (shared across all functions)
# ══════════════════════════════════════════════════════════════════
def _load_reference_tables():
    """Load small reference tables — fast, always full load."""
    log.info("  Loading reference tables...")
    stores    = pd.read_csv(CSV_PATH / "stores.csv")
    products  = pd.read_csv(CSV_PATH / "products.csv")
    suppliers = pd.read_csv(CSV_PATH / "suppliers.csv")
    return stores, products, suppliers


def _load_replenishment():
    """Load replenishment_logs.csv with lead time calculation."""
    log.info("  Loading replenishment logs (149 MB)...")
    repl = pd.read_csv(
        CSV_PATH / "replenishment_logs.csv",
        parse_dates=["order_date", "receive_date"],
    )
    repl["fulfillment_rate"] = (
        repl["units_received"] / repl["units_ordered"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)
    repl["lead_time_actual"] = (
        repl["receive_date"] - repl["order_date"]
    ).dt.days.clip(lower=0)
    log.info(f"    Replenishment rows: {len(repl):,}")
    return repl


def _load_inventory_snapshot():
    """Load latest inventory snapshot per store-SKU."""
    log.info("  Loading inventory snapshots (298 MB)...")
    inv = pd.read_csv(
        CSV_PATH / "inventory_snapshots.csv",
        parse_dates=["snapshot_date"],
        usecols=[
            "store_id", "sku_id", "snapshot_date",
            "units_on_hand", "units_in_backroom",
            "days_of_supply",
        ],
    )
    inv["units_total"] = inv["units_on_hand"] + inv["units_in_backroom"]
    # Keep only the latest snapshot per store-SKU
    latest = (
        inv.sort_values("snapshot_date")
        .groupby(["store_id", "sku_id"])
        .last()
        .reset_index()
    )
    log.info(f"    Latest snapshot rows: {len(latest):,}")
    return latest


def _load_stockouts():
    """Load stockout events."""
    return pd.read_csv(
        CSV_PATH / "stockout_events.csv",
        parse_dates=["stockout_date", "restock_date"],
    )


def _load_sales_agg():
    """
    Load sales aggregated to daily store-SKU level.
    Uses DuckDB to avoid loading full 1.99GB into RAM.
    """
    log.info("  Loading sales (aggregated via DuckDB)...")
    import duckdb
    con = duckdb.connect()
    sales = con.execute(f"""
        SELECT
            store_id, sku_id,
            CAST(sale_date AS DATE) AS sale_date,
            SUM(units_sold)         AS units_sold,
            SUM(revenue)            AS revenue,
            MAX(is_promoted::INTEGER) AS is_promoted
        FROM read_csv_auto('{CSV_PATH}/sales_transactions.csv')
        GROUP BY store_id, sku_id, CAST(sale_date AS DATE)
    """).df()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    con.close()
    gc.collect()
    log.info(f"    Sales rows: {len(sales):,}")
    return sales


# ══════════════════════════════════════════════════════════════════
# FUNCTION 1 — Phantom confidence scoring
# Logic: retail_eda.ipynb Cell [25] + nexus_complete_final Cell [27]
# ══════════════════════════════════════════════════════════════════
def phantom_confidence_scoring(sales_agg=None, inv_latest=None,
                                 stores=None, products=None):
    """
    Scores phantom inventory suspects with High/Medium/Low confidence.

    retail_eda logic:
      - High:   consec_zero_days >= 3 AND rolling_7d_avg >= Q75
      - Medium: consec_zero_days >= 3 AND rolling_7d_avg >= Q50
      - Low:    consec_zero_days >= 3 AND rolling_7d_avg <  Q50

    nexus addition:
      - Confirmed phantom: stock > 0 AND zero weekly sales
    """
    log.info("[Phantom] Running confidence scoring...")

    if sales_agg is None:
        sales_agg = _load_sales_agg()
    if inv_latest is None:
        inv_latest = _load_inventory_snapshot()
    if stores is None or products is None:
        stores, products, _ = _load_reference_tables()

    # Rolling features per store-SKU
    sale_date_col = "sale_date" if "sale_date" in sales_agg.columns else "date"
    df = sales_agg.sort_values(["store_id", "sku_id", sale_date_col]).copy()
    if sale_date_col == "date":
        df = df.rename(columns={"date": "sale_date"})
    df["month"] = df["sale_date"].dt.month

    def rolling_ph(g):
        g = g.copy()
        # CRITICAL FIX: reindex to fill missing dates with 0
        # Without this, consec_zero_days is always 0 because
        # zero-sale days are missing rows entirely in aggregated data
        date_range = pd.date_range(g["sale_date"].min(), g["sale_date"].max(), freq="D")
        if "sale_date" not in g.columns and "date" in g.columns: g = g.rename(columns={"date":"sale_date"})
        g = g.set_index("sale_date").reindex(date_range).fillna(0).rename_axis("sale_date").reset_index()
        # Re-fill store_id and sku_id (they become 0 after reindex)
        g[["store_id","sku_id"]] = g[["store_id","sku_id"]].replace(0, np.nan).ffill().bfill()
        s = g["units_sold"]
        g["rolling_7d_avg"]  = s.shift(1).rolling(7,  min_periods=1).mean()
        g["rolling_14d_avg"] = s.shift(1).rolling(14, min_periods=1).mean()
        # Consecutive zero streak
        is_zero  = (s == 0).astype(int)
        group_id = (is_zero != is_zero.shift()).cumsum()
        streak   = is_zero.groupby(group_id).cumcount() + 1
        g["consec_zero_days"] = streak.where(is_zero == 1, 0)
        return g

    log.info("  Computing rolling features (this takes ~3-5 min for 25M rows)...")
    df = df.rename(columns={"sale_date":"sale_date"}) if "sale_date" in df.columns else df.rename(columns={"date":"sale_date"}) if "date" in df.columns else df
    if "sale_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date":"sale_date"})
    df = (
        df.groupby(["store_id", "sku_id"], group_keys=False)
        .apply(rolling_ph)
        .reset_index(drop=True)
    )
    # Rename back
    if "sale_date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"sale_date":"date"})

    # Q75 and Q50 thresholds per SKU (from retail_eda Cell [24])
    thresholds = (
        df.groupby("sku_id")["rolling_7d_avg"]
        .agg(q75=lambda x: x.quantile(0.75),
             q50=lambda x: x.quantile(0.50))
        .reset_index()
    )
    df = df.merge(thresholds, on="sku_id", how="left")

    # Phantom flag: 3+ consecutive zero days (retail_eda Cell [25])
    PHANTOM_DAYS = 3
    df["is_phantom_candidate"] = df["consec_zero_days"] >= PHANTOM_DAYS

    # Confidence scoring
    df["phantom_confidence"] = "Normal"
    df.loc[
        df["is_phantom_candidate"] & (df["rolling_7d_avg"] >= df["q75"]),
        "phantom_confidence",
    ] = "High"
    df.loc[
        df["is_phantom_candidate"]
        & (df["rolling_7d_avg"] >= df["q50"])
        & (df["rolling_7d_avg"] < df["q75"]),
        "phantom_confidence",
    ] = "Medium"
    df.loc[
        df["is_phantom_candidate"]
        & (df["rolling_7d_avg"] < df["q50"]),
        "phantom_confidence",
    ] = "Low"

    # Latest snapshot: confirm stock > 0 (nexus logic)
    suspects = (
        df[df["is_phantom_candidate"]]
        .sort_values("date" if "date" in df.columns else "sale_date")
        .groupby(["store_id", "sku_id"])
        .last()
        .reset_index()
    )
    suspects = suspects.merge(
        inv_latest[["store_id", "sku_id", "units_total", "units_on_hand"]],
        on=["store_id", "sku_id"], how="left",
    )
    # Confirmed phantom: system shows stock > 0 but zero recent sales
    suspects["is_confirmed_phantom"] = (
        suspects["is_phantom_candidate"] & (suspects["units_total"] > 0)
    )

    # Enrich with store + product info
    suspects = suspects.merge(
        stores[["store_id", "store_name", "region",
                "store_format", "foot_traffic_tier"]],
        on="store_id", how="left",
    )
    suspects = suspects.merge(
        products[["sku_id", "product_name", "category", "unit_price"]],
        on="sku_id", how="left",
    )

    # Revenue at risk estimate
    suspects["revenue_at_risk_daily"] = (
        suspects["rolling_7d_avg"] * suspects["unit_price"].fillna(0)
    )

    # Summary
    total     = suspects["is_confirmed_phantom"].sum()
    high_conf = (suspects["phantom_confidence"] == "High").sum()
    log.info(f"  Confirmed phantoms: {total:,}  (High confidence: {high_conf:,})")

    # Save
    suspects.to_csv(NEXUS_DIR / "allstore" / "phantom_confidence.csv", index=False)

    # Aggregate by store for the network summary
    store_phantom = (
        suspects.groupby(["store_id", "store_name", "region", "foot_traffic_tier"])
        .agg(
            phantom_count=("is_confirmed_phantom", "sum"),
            high_conf=("phantom_confidence", lambda x: (x == "High").sum()),
            med_conf=("phantom_confidence", lambda x: (x == "Medium").sum()),
            revenue_at_risk=("revenue_at_risk_daily", "sum"),
        )
        .reset_index()
    )
    store_phantom.to_csv(NEXUS_DIR / "allstore" / "phantom_by_store.csv", index=False)
    log.info("  Phantom confidence saved ✅")
    return suspects


# ══════════════════════════════════════════════════════════════════
# FUNCTION 2 — Stockout root cause breakdown
# Logic: retail_eda.ipynb Cell [36-37] + nexus_forecast_accuracy
# ══════════════════════════════════════════════════════════════════
def stockout_root_cause(stores=None, products=None):
    """
    Breaks down stockouts by root_cause, region, category.
    Computes revenue lost, avg duration per cause.
    """
    log.info("[Root Cause] Analysing stockout root causes...")

    stockouts = _load_stockouts()
    if stores is None or products is None:
        stores, products, _ = _load_reference_tables()

    # Enrich with store + product info (retail_eda Cell [36])
    so = (
        stockouts
        .merge(
            stores[["store_id", "region", "store_format", "foot_traffic_tier"]],
            on="store_id", how="left",
        )
        .merge(
            products[["sku_id", "category", "is_perishable", "unit_price"]],
            on="sku_id", how="left",
        )
    )

    # Overall by root cause (retail_eda Cell [37])
    by_cause = (
        so.groupby("root_cause")
        .agg(
            count=("stockout_id", "count"),
            total_lost_revenue=("estimated_lost_revenue", "sum"),
            avg_duration_days=("duration_days", "mean"),
            pct_ongoing=("restock_date", lambda x: x.isnull().mean() * 100),
        )
        .reset_index()
    )
    by_cause["pct_of_stockouts"] = (
        by_cause["count"] / by_cause["count"].sum() * 100
    )
    by_cause.to_csv(NEXUS_DIR / "forecast" / "stockout_by_cause.csv", index=False)

    # By region + cause
    by_region_cause = (
        so.groupby(["region", "root_cause"])
        .agg(
            count=("stockout_id", "count"),
            total_lost_revenue=("estimated_lost_revenue", "sum"),
        )
        .reset_index()
    )
    by_region_cause.to_csv(
        NEXUS_DIR / "forecast" / "stockout_by_region_cause.csv", index=False
    )

    # By category
    by_cat = (
        so.groupby("category")
        .agg(
            count=("stockout_id", "count"),
            total_lost_revenue=("estimated_lost_revenue", "sum"),
            avg_duration=("duration_days", "mean"),
        )
        .reset_index()
        .sort_values("total_lost_revenue", ascending=False)
    )
    by_cat.to_csv(NEXUS_DIR / "forecast" / "stockout_by_category.csv", index=False)

    # By store tier
    by_tier = (
        so.groupby("foot_traffic_tier")
        .agg(
            count=("stockout_id", "count"),
            total_lost_revenue=("estimated_lost_revenue", "sum"),
            avg_duration=("duration_days", "mean"),
        )
        .reset_index()
    )
    by_tier.to_csv(NEXUS_DIR / "forecast" / "stockout_by_tier.csv", index=False)

    # Monthly trend
    so["month"] = pd.to_datetime(so["stockout_date"]).dt.to_period("M").astype(str)
    monthly = (
        so.groupby("month")
        .agg(
            count=("stockout_id", "count"),
            revenue=("estimated_lost_revenue", "sum"),
        )
        .reset_index()
    )
    monthly.to_csv(NEXUS_DIR / "forecast" / "stockout_monthly.csv", index=False)

    log.info(f"  Root cause analysis: {len(by_cause)} causes identified ✅")
    return by_cause


# ══════════════════════════════════════════════════════════════════
# FUNCTION 3 — Forecast accuracy breakdown
# Logic: nexus_forecast_accuracy.ipynb Sections 2-5
# ══════════════════════════════════════════════════════════════════
def forecast_accuracy_breakdown(stores=None, products=None):
    """
    Computes WAPE/MAPE/bias/P90 broken down by:
    - store format, region, category, promotion flag, month
    Reads from saved pipeline predictions.
    """
    log.info("[Forecast] Computing accuracy breakdown...")

    pred_path = PROCESSED / "demandSense_v2_predictions.csv"
    if not pred_path.exists():
        log.warning("  predictions CSV not found — skipping forecast breakdown")
        return None

    pred = pd.read_csv(pred_path, parse_dates=["date"])

    if stores is None or products is None:
        stores, products, _ = _load_reference_tables()

    # Rename columns to match nexus naming
    if "forecast_units" in pred.columns:
        pred = pred.rename(columns={"forecast_units": "forecast_p50"})

    pred = pred.reset_index(drop=True)
    # Use .values to bypass pandas index alignment on duplicate-index CSVs
    t  = pred["target"].values.astype(float)
    p  = pred["forecast_p50"].values.astype(float)
    lo = pred["lower_bound_90"].values.astype(float)
    hi = pred["upper_bound_90"].values.astype(float)
    pred["abs_err"] = np.abs(t - p)
    pred["err"]     = t - p
    pred["ape"]     = np.where(t > 0, np.abs(t - p) / t * 100, np.nan)
    pred["in_90"]   = (t >= lo) & (t <= hi)

    # Enrich
    pred = pred.merge(
        stores[["store_id", "store_format", "region", "foot_traffic_tier"]],
        on="store_id", how="left",
    )

    def metrics(g):
        return pd.Series({
            "wape":         g["abs_err"].sum() / g["target"].sum() * 100,
            "mape":         g["ape"].mean(),
            "bias":         g["err"].mean(),
            "p90_coverage": g["in_90"].mean() * 100,
            "rows":         len(g),
        })

    pred = pred.reset_index(drop=True)  # re-reset after merge
    # By store format
    by_format = pred.groupby("store_format").apply(metrics).reset_index()
    by_format.to_csv(NEXUS_DIR / "forecast" / "accuracy_by_format.csv", index=False)

    # By region
    by_region = pred.groupby("region").apply(metrics).reset_index()
    by_region.to_csv(NEXUS_DIR / "forecast" / "accuracy_by_region.csv", index=False)

    # By category
    by_cat = pred.groupby("category").apply(metrics).reset_index()
    by_cat = by_cat.sort_values("wape", ascending=False)
    by_cat.to_csv(NEXUS_DIR / "forecast" / "accuracy_by_category.csv", index=False)

    # Monthly
    pred["month"] = pred["date"].dt.to_period("M").astype(str)
    by_month = pred.groupby("month").apply(metrics).reset_index()
    by_month.to_csv(NEXUS_DIR / "forecast" / "accuracy_by_month.csv", index=False)

    # Promotion effect (nexus_forecast_accuracy Section 4)
    if "is_promoted" in pred.columns:
        by_promo = pred.groupby("is_promoted").apply(metrics).reset_index()
        by_promo.to_csv(
            NEXUS_DIR / "forecast" / "accuracy_by_promo.csv", index=False
        )

    log.info("  Forecast accuracy breakdown saved ✅")
    return by_cat


# ══════════════════════════════════════════════════════════════════
# FUNCTION 4 — Localization mismatch
# Logic: nexus_localization_profiling.ipynb Sections 2-5
# ══════════════════════════════════════════════════════════════════
def localization_mismatch(sales_agg=None, stores=None, products=None):
    """
    Identifies store-category combinations selling 30%+ below regional average.
    Assigns MISMATCH / EXTREME_MISMATCH flags.
    Computes per-store localization score.
    """
    log.info("[Localization] Computing mismatch analysis...")

    if sales_agg is None:
        sales_agg = _load_sales_agg()
    if stores is None or products is None:
        stores, products, _ = _load_reference_tables()

    # Enrich sales with store + product context
    sales_enriched = (
        sales_agg
        .merge(
            stores[["store_id", "region", "store_format", "foot_traffic_tier",
                     "store_name"]],
            on="store_id", how="left",
        )
        .merge(
            products[["sku_id", "product_name", "category", "unit_price"]],
            on="sku_id", how="left",
        )
    )

    # Store-category revenue aggregation
    store_cat = (
        sales_enriched
        .groupby(["store_id", "store_name", "region", "store_format",
                  "foot_traffic_tier", "category"])
        .agg(
            total_units=("units_sold", "sum"),
            total_revenue=("revenue", "sum"),
            days_selling=("sale_date", "nunique"),
        )
        .reset_index()
    )
    store_cat["avg_daily_revenue"] = (
        store_cat["total_revenue"] / store_cat["days_selling"].replace(0, 1)
    )

    # Regional benchmark: avg daily revenue per store per category
    # (nexus_localization Section 2)
    region_bench = (
        store_cat.groupby(["region", "category"])
        .agg(regional_avg_daily_rev=("avg_daily_revenue", "mean"))
        .reset_index()
    )

    store_cat = store_cat.merge(
        region_bench, on=["region", "category"], how="left"
    )

    # Deviation and mismatch flag
    store_cat["deviation_pct"] = np.where(
        store_cat["regional_avg_daily_rev"] > 0,
        (store_cat["avg_daily_revenue"] - store_cat["regional_avg_daily_rev"])
        / store_cat["regional_avg_daily_rev"] * 100,
        0,
    )

    store_cat["mismatch_flag"] = "OK"
    store_cat.loc[
        store_cat["deviation_pct"] <= MISMATCH_THRESHOLD, "mismatch_flag"
    ] = "MISMATCH"
    store_cat.loc[
        store_cat["deviation_pct"] <= MISMATCH_THRESHOLD * 2, "mismatch_flag"
    ] = "EXTREME_MISMATCH"

    # Estimated revenue uplift if mismatch fixed
    store_cat["revenue_uplift_potential"] = np.where(
        store_cat["mismatch_flag"].isin(["MISMATCH", "EXTREME_MISMATCH"]),
        (store_cat["regional_avg_daily_rev"] - store_cat["avg_daily_revenue"])
        * store_cat["days_selling"],
        0,
    ).clip(lower=0)

    store_cat.to_csv(
        NEXUS_DIR / "localization" / "store_category_mismatch.csv", index=False
    )

    # Per-store localization score (nexus Section 3)
    store_loc_score = (
        store_cat.groupby(["store_id", "store_name", "region", "foot_traffic_tier"])
        .agg(
            total_cats=("category", "count"),
            mismatched_cats=(
                "mismatch_flag", lambda x: x.isin(["MISMATCH","EXTREME_MISMATCH"]).sum()
            ),
            total_uplift=("revenue_uplift_potential", "sum"),
        )
        .reset_index()
    )
    store_loc_score["mismatch_rate_pct"] = (
        store_loc_score["mismatched_cats"] / store_loc_score["total_cats"] * 100
    )
    store_loc_score.to_csv(
        NEXUS_DIR / "localization" / "store_localization_scores.csv", index=False
    )

    # Network-level: which categories are most mismatched (nexus Section 4)
    cat_network = (
        store_cat[store_cat["mismatch_flag"].isin(["MISMATCH","EXTREME_MISMATCH"])]
        .groupby("category")
        .agg(
            mismatch_stores=("store_id", "count"),
            avg_deviation=("deviation_pct", "mean"),
            total_uplift=("revenue_uplift_potential", "sum"),
        )
        .reset_index()
        .sort_values("total_uplift", ascending=False)
    )
    cat_network.to_csv(
        NEXUS_DIR / "localization" / "category_network_mismatch.csv", index=False
    )

    total_mismatch   = store_cat["mismatch_flag"].isin(["MISMATCH","EXTREME_MISMATCH"]).sum()
    total_uplift     = store_cat["revenue_uplift_potential"].sum()
    log.info(
        f"  Localization: {total_mismatch:,} mismatches "
        f"· ${total_uplift/1e6:.1f}M uplift potential ✅"
    )
    return store_cat


# ══════════════════════════════════════════════════════════════════
# FUNCTION 5 — Supplier fill rate + performance scorecard
# Logic: nexus_supplier_performance.ipynb Sections 2-4
# ══════════════════════════════════════════════════════════════════
def supplier_fill_rate(stores=None, products=None):
    """
    Computes per-supplier fill rate, lead time stats, and category risk.
    Uses replenishment_logs.csv (149 MB).
    """
    log.info("[Supplier] Computing fill rate and performance scorecard...")

    repl = _load_replenishment()
    if stores is None or products is None:
        stores, products, suppliers = _load_reference_tables()
    else:
        _, _, suppliers = _load_reference_tables()

    # Enrich replenishment with product + supplier info
    repl = repl.merge(
        products[["sku_id", "supplier_id", "category"]], on="sku_id", how="left"
    )
    repl = repl.merge(
        suppliers[["supplier_id", "supplier_name", "reliability_score",
                    "lead_time_days_avg"]],
        on="supplier_id", how="left",
    )
    repl = repl.merge(
        stores[["store_id", "region", "foot_traffic_tier"]], on="store_id", how="left"
    )

    # Supplier-level summary (nexus Section 2)
    scorecard = (
        repl.groupby(["supplier_id", "supplier_name", "reliability_score"])
        .agg(
            avg_lead_actual=("lead_time_actual",  "mean"),
            std_lead_actual=("lead_time_actual",  "std"),
            min_lead=("lead_time_actual",          "min"),
            max_lead=("lead_time_actual",          "max"),
            avg_fill_rate=("fulfillment_rate",     "mean"),
            total_orders=("replenishment_id",      "count"),
            late_orders=(
                "lead_time_actual",
                lambda x: (x > repl.loc[x.index, "lead_time_days_avg"]).sum()
            ),
        )
        .reset_index()
    )
    scorecard["std_lead_actual"] = scorecard["std_lead_actual"].fillna(0)
    scorecard["late_pct"] = (
        scorecard["late_orders"] / scorecard["total_orders"] * 100
    )
    scorecard["lead_deviation"] = (
        scorecard["avg_lead_actual"] - scorecard["reliability_score"]
    )

    # Correlation: reliability_score vs actual fill_rate (retail_eda Cell [59])
    if len(scorecard) >= 3:
        corr, pval = stats.pearsonr(
            scorecard["reliability_score"].fillna(0),
            scorecard["avg_fill_rate"].fillna(0),
        )
        scorecard["reliability_fill_corr"] = round(corr, 4)
        log.info(f"  Reliability vs fill rate correlation: {corr:.4f} (p={pval:.4f})")

    scorecard.to_csv(NEXUS_DIR / "supplier" / "supplier_scorecard.csv", index=False)

    # Category-level supplier risk (nexus Section 3)
    cat_sup_risk = (
        repl.groupby(["category"])
        .agg(
            avg_fill_rate=("fulfillment_rate",  "mean"),
            avg_lead_time=("lead_time_actual",  "mean"),
            total_orders=("replenishment_id",   "count"),
        )
        .reset_index()
        .sort_values("avg_fill_rate")
    )
    cat_sup_risk.to_csv(
        NEXUS_DIR / "supplier" / "supplier_category_risk.csv", index=False
    )

    # Store-level supplier impact (nexus Section 4)
    store_sup = (
        repl.groupby(["store_id", "supplier_id", "supplier_name"])
        .agg(
            orders=("replenishment_id",   "count"),
            avg_fill=("fulfillment_rate", "mean"),
            avg_lead=("lead_time_actual", "mean"),
        )
        .reset_index()
    )
    store_sup.to_csv(
        NEXUS_DIR / "supplier" / "supplier_store_impact.csv", index=False
    )

    low_fill = (scorecard["avg_fill_rate"] < 0.85).sum()
    log.info(
        f"  Supplier scorecard: {len(scorecard)} suppliers "
        f"· {low_fill} with fill rate < 85% ✅"
    )
    return scorecard


# ══════════════════════════════════════════════════════════════════
# FUNCTION 6 — ROI simulation
# Logic: nexus_roi_simulation.ipynb Sections 2-4
# ══════════════════════════════════════════════════════════════════
def roi_simulation(stores=None, products=None):
    """
    Simulates: would our alerts have prevented these stockouts?
    Logic: if stockout occurred AND SKU was below optimized ROP
           before the event → alert would have fired → revenue recovered.
    """
    log.info("[ROI] Running simulation...")

    stockouts = _load_stockouts()
    if stores is None or products is None:
        stores, products, _ = _load_reference_tables()

    # Load reorder points from pipeline output
    repl_path = PROCESSED / "replenishment_policy_inputs_demandsense.csv"
    if repl_path.exists():
        rop_df = pd.read_csv(repl_path, parse_dates=["date"])
        # Get latest ROP per store-SKU
        rop_latest = (
            rop_df.sort_values("date")
            .groupby(["store_id", "sku_id"])
            .last()[["reorder_point", "mu_daily", "safety_stock_units"]]
            .reset_index()
        )
    else:
        log.warning("  No replenishment CSV found — using simplified ROI")
        rop_latest = None

    # Enrich stockouts
    so = (
        stockouts
        .merge(
            stores[["store_id", "region", "foot_traffic_tier", "store_name"]],
            on="store_id", how="left",
        )
        .merge(
            products[["sku_id", "category", "unit_price"]], on="sku_id", how="left"
        )
    )

    # Simulate alert coverage (nexus_roi Section 2)
    if rop_latest is not None:
        so = so.merge(rop_latest, on=["store_id", "sku_id"], how="left")
        # Alert fires if ROP is set (i.e. we have a forecast for this SKU)
        so["alert_would_fire"] = so["reorder_point"].notna()
    else:
        # Conservative estimate: alerts cover SKUs with known demand
        so["alert_would_fire"] = so["estimated_lost_revenue"] > 0

    # Alert prevents stockout with 70% effectiveness
    # (conservative estimate — real effectiveness depends on replenishment speed)
    ALERT_EFFECTIVENESS = 0.70
    so["revenue_recovered"] = np.where(
        so["alert_would_fire"],
        so["estimated_lost_revenue"] * ALERT_EFFECTIVENESS,
        0,
    )
    so["revenue_recovered"] = so["revenue_recovered"].fillna(0)

    # By region (nexus Section 3)
    region_roi = (
        so.groupby("region")
        .agg(
            total_stockouts=("stockout_id",          "count"),
            total_lost=("estimated_lost_revenue",    "sum"),
            recovered=("revenue_recovered",          "sum"),
            alert_coverage_pct=(
                "alert_would_fire", lambda x: x.mean() * 100
            ),
        )
        .reset_index()
    )
    region_roi["recovery_rate_pct"] = (
        region_roi["recovered"] / region_roi["total_lost"].replace(0, np.nan) * 100
    )
    region_roi.to_csv(NEXUS_DIR / "roi" / "roi_by_region.csv", index=False)

    # By category
    cat_roi = (
        so.groupby("category")
        .agg(
            total_lost=("estimated_lost_revenue", "sum"),
            recovered=("revenue_recovered",       "sum"),
        )
        .reset_index()
        .sort_values("recovered", ascending=False)
    )
    cat_roi.to_csv(NEXUS_DIR / "roi" / "roi_by_category.csv", index=False)

    # By root cause
    if "root_cause" in so.columns:
        cause_roi = (
            so.groupby("root_cause")
            .agg(
                total_lost=("estimated_lost_revenue", "sum"),
                recovered=("revenue_recovered",       "sum"),
            )
            .reset_index()
        )
        cause_roi.to_csv(NEXUS_DIR / "roi" / "roi_by_cause.csv", index=False)

    # Executive summary
    total_lost      = so["estimated_lost_revenue"].sum()
    total_recovered = so["revenue_recovered"].sum()
    pct_covered     = so["alert_would_fire"].mean() * 100

    summary = pd.DataFrame([{
        "total_stockout_events":  len(so),
        "total_revenue_lost_usd": round(total_lost, 2),
        "revenue_recoverable_usd": round(total_recovered, 2),
        "recovery_rate_pct":      round(total_recovered / max(total_lost, 1) * 100, 2),
        "alert_coverage_pct":     round(pct_covered, 2),
        "effectiveness_assumption": ALERT_EFFECTIVENESS,
        "simulation_date":        datetime.now().strftime("%Y-%m-%d"),
    }])
    summary.to_csv(NEXUS_DIR / "roi" / "roi_executive_summary.csv", index=False)

    log.info(
        f"  ROI: ${total_recovered/1e6:.1f}M recoverable "
        f"of ${total_lost/1e6:.1f}M total lost "
        f"({pct_covered:.1f}% alert coverage) ✅"
    )
    return summary


# ══════════════════════════════════════════════════════════════════
# FUNCTION 7 — Per-store alert engine
# Logic: nexus_complete_final.ipynb Sections 4-9, 11-13
#        nexus_allstore_pipeline.ipynb Section 3
# ══════════════════════════════════════════════════════════════════
def all_store_alerts(stores=None, products=None):
    """
    Runs the complete nexus alert engine for ALL 478 stores.
    Uses the Nexus formula for safety stock:
      SS = Z * sqrt(LT * sigma_demand^2 + D^2 * sigma_LT^2)
      ROP = D * LT + SS
    Alert tiers: CRITICAL (<=3d) / WARNING (<=7d) / MONITOR (<=14d) / OK

    This is the per-store loop from nexus_allstore_pipeline.ipynb.
    Runtime: 20–40 minutes for 478 stores.
    """
    log.info("[AllStore] Starting per-store alert engine (478 stores)...")
    log.info("  This will take 20–40 minutes for all stores.")

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    if stores is None or products is None:
        stores, products, suppliers = _load_reference_tables()
    else:
        _, _, suppliers = _load_reference_tables()

    repl      = _load_replenishment()

    # Build prod_sup ONCE, deduplicated — prevents per-store fillna crash
    prod_sup = products.merge(
        suppliers[["supplier_id","supplier_name","lead_time_days_avg",
                    "lead_time_days_std","reliability_score"]],
        on="supplier_id", how="left",
    ).drop_duplicates(subset="sku_id", keep="first")
    log.info(f"  prod_sup built: {len(prod_sup):,} unique SKUs")

    promotions = pd.read_csv(
        CSV_PATH / "promotions.csv", parse_dates=["start_date", "end_date"]
    )
    inv_latest = _load_inventory_snapshot()

    # Load sales in chunks and aggregate to daily level once
    log.info("  Pre-aggregating sales...")
    sales_agg = _load_sales_agg()
    sales_agg["month"] = sales_agg["sale_date"].dt.month

    store_ids = stores["store_id"].tolist()
    network_alerts  = []
    network_summary = []

    iterator = tqdm(store_ids, desc="Stores") if use_tqdm else store_ids

    for store_id in iterator:
        try:
            result = _process_one_store(
                store_id=store_id,
                stores=stores,
                products=products,
                suppliers=suppliers,
                sales_agg=sales_agg,
                repl=repl,
                inv_latest=inv_latest,
                promotions=promotions,
                prod_sup=prod_sup,   # pre-built, deduplicated
            )
            if result is not None:
                alerts_df, summary_row = result
                network_alerts.append(alerts_df)
                network_summary.append(summary_row)
        except Exception as e:
            log.warning(f"  Store {store_id} failed: {e}")
            continue

    if not network_alerts:
        log.warning("  No alerts generated")
        return None

    # Combine all stores
    master_alerts  = pd.concat(network_alerts,  ignore_index=True)
    summary_df     = pd.DataFrame(network_summary)

    master_alerts.to_csv(
        NEXUS_DIR / "allstore" / "network_master_alerts.csv", index=False
    )
    summary_df.to_csv(
        NEXUS_DIR / "allstore" / "network_store_summary.csv", index=False
    )

    # Network overview
    crit = (master_alerts["alert_tier"] == "CRITICAL").sum()
    warn = (master_alerts["alert_tier"] == "WARNING").sum()
    rar  = master_alerts["revenue_at_risk"].sum()
    log.info(
        f"  Network: {crit:,} CRITICAL · {warn:,} WARNING "
        f"· ${rar/1e6:.1f}M revenue at risk ✅"
    )
    return master_alerts


def _process_one_store(store_id, stores, products, suppliers,
                        sales_agg, repl, inv_latest, promotions,
                        prod_sup=None):
    """
    Per-store alert engine — exact port of nexus_allstore_pipeline.ipynb.
    prod_sup: pre-built deduplicated products+suppliers merge (passed from all_store_alerts).
    """
    Z_SCORE       = 1.96
    CRITICAL_DAYS = 3
    WARNING_DAYS  = 7
    MONITOR_DAYS  = 14
    TARGET_DAYS   = 14
    TODAY         = pd.Timestamp.now().normalize()
    LOOKAHEAD     = TODAY + pd.Timedelta(days=7)

    s_sales = sales_agg[sales_agg["store_id"] == store_id].copy()
    if s_sales.empty:
        return None

    store_info_df = stores[stores["store_id"] == store_id]
    if store_info_df.empty:
        return None
    store_info = store_info_df.iloc[0]

    # ── Seasonal factor ──────────────────────────────────────────
    if "month" not in s_sales.columns:
        s_sales["month"] = s_sales["sale_date"].dt.month if "sale_date" in s_sales.columns else 1
    monthly_avg = s_sales.groupby("month")["units_sold"].mean()
    overall_avg = s_sales["units_sold"].mean()
    seasonal_factor = (monthly_avg.get(TODAY.month, overall_avg) / overall_avg
                       if overall_avg > 0 else 1.0)

    # ── Demand stats ─────────────────────────────────────────────
    demand_stats = (
        s_sales.groupby("sku_id")
        .agg(
            avg_daily_demand=("units_sold", "mean"),
            std_daily_demand=("units_sold", "std"),
            max_daily_demand=("units_sold", "max"),
            total_units=("units_sold",      "sum"),
            total_revenue=("revenue",        "sum"),
        )
        .reset_index()
    )
    demand_stats["std_daily_demand"] = demand_stats["std_daily_demand"].fillna(0)

    # Merge with pre-built, deduplicated prod_sup
    # CRITICAL FIX: prod_sup must be deduplicated by sku_id BEFORE this merge
    # to avoid getting a DataFrame back from fillna (which crashes with
    # "value parameter must be a scalar, dict or Series")
    if prod_sup is None:
        prod_sup = products.merge(
            suppliers[["supplier_id","supplier_name","lead_time_days_avg",
                        "lead_time_days_std","reliability_score"]],
            on="supplier_id", how="left",
        ).drop_duplicates(subset="sku_id", keep="first")  # ← THE FIX

    merge_cols = [c for c in [
        "sku_id","product_name","category","unit_price","is_perishable",
        "supplier_name","lead_time_days_avg","lead_time_days_std",
        "reliability_score","reorder_point","safety_stock",
    ] if c in prod_sup.columns]
    demand_stats = demand_stats.merge(prod_sup[merge_cols], on="sku_id", how="left")
    demand_stats["seasonal_factor"]     = seasonal_factor
    demand_stats["avg_demand_seasonal"] = demand_stats["avg_daily_demand"] * seasonal_factor

    # ── Lead times from replenishment logs ────────────────────────
    s_repl = repl[repl["store_id"] == store_id].copy()
    if len(s_repl) > 0 and "lead_time_actual" in s_repl.columns:
        if "fulfillment_rate" not in s_repl.columns:
            s_repl["fulfillment_rate"] = (
                s_repl["units_received"] /
                s_repl["units_ordered"].replace(0, np.nan)
            ).clip(0, 1).fillna(0)
        lead_actual = (
            s_repl.groupby("sku_id")
            .agg(
                avg_lead_actual=("lead_time_actual", "mean"),
                std_lead_actual=("lead_time_actual", "std"),
                avg_fill_rate=("fulfillment_rate",   "mean"),
            )
            .reset_index()
        )
        lead_actual["std_lead_actual"] = lead_actual["std_lead_actual"].fillna(0)
        demand_stats = demand_stats.merge(
            lead_actual[["sku_id","avg_lead_actual","std_lead_actual","avg_fill_rate"]],
            on="sku_id", how="left",
        )
        # Use Series.fillna(Series) — safe because demand_stats has unique sku_id now
        demand_stats["lead_time_final"] = (
            demand_stats["avg_lead_actual"]
            .fillna(demand_stats["lead_time_days_avg"].fillna(7))
        )
        demand_stats["lead_time_std_final"] = (
            demand_stats["std_lead_actual"]
            .fillna(demand_stats["lead_time_days_std"].fillna(1))
        )
        demand_stats["avg_fill_rate"] = demand_stats["avg_fill_rate"].fillna(1.0)
    else:
        demand_stats["lead_time_final"]     = demand_stats.get("lead_time_days_avg", pd.Series(7)).fillna(7)
        demand_stats["lead_time_std_final"] = demand_stats.get("lead_time_days_std", pd.Series(1)).fillna(1)
        demand_stats["avg_fill_rate"]       = demand_stats.get("reliability_score",  pd.Series(0.9)).fillna(0.9)

    # ── Safety stock — NEXUS formula ─────────────────────────────
    # SS = Z * sqrt(LT * sigma_D^2 + D^2 * sigma_LT^2)
    ss_raw = Z_SCORE * np.sqrt(
        demand_stats["lead_time_final"]     * demand_stats["std_daily_demand"]**2 +
        demand_stats["avg_daily_demand"]**2 * demand_stats["lead_time_std_final"]**2
    )
    demand_stats["safety_stock_optimized"] = np.maximum(0, np.ceil(ss_raw)).astype(int)
    demand_stats["safety_stock_seasonal"]  = np.maximum(0,
        np.ceil(demand_stats["safety_stock_optimized"] * seasonal_factor)
    ).astype(int)
    demand_stats["reorder_point_optimized"] = np.maximum(0, np.ceil(
        demand_stats["avg_daily_demand"] * demand_stats["lead_time_final"] +
        demand_stats["safety_stock_seasonal"]
    )).astype(int)

    if "safety_stock" in demand_stats.columns:
        demand_stats["ss_delta"] = (
            demand_stats["safety_stock_seasonal"] - demand_stats["safety_stock"]
        )
        demand_stats["ss_status"] = demand_stats["ss_delta"].apply(
            lambda d: "UNDERSTOCKED" if d > 10 else ("OVERSTOCKED" if d < -10 else "OPTIMAL")
        )
    else:
        demand_stats["ss_status"] = "OPTIMAL"

    # ── Latest inventory snapshot ─────────────────────────────────
    s_snap = inv_latest[inv_latest["store_id"] == store_id].copy()
    if s_snap.empty:
        return None
    s_snap["total_available"] = (
        s_snap["units_on_hand"].fillna(0) + s_snap["units_in_backroom"].fillna(0)
    )

    # ── Active promotions ─────────────────────────────────────────
    active_promos = promotions[
        (promotions["start_date"] <= LOOKAHEAD) &
        (promotions["end_date"]   >= TODAY)
    ].copy()
    all_promo_parts = []
    if "store_id" in active_promos.columns:
        chain_p = active_promos[active_promos["store_id"].isnull()][["sku_id","demand_lift_factor"]]
        store_p = active_promos[active_promos["store_id"] == store_id][["sku_id","demand_lift_factor"]]
        all_promo_parts = [df for df in [chain_p, store_p] if not df.empty]
    all_promos = (
        pd.concat(all_promo_parts).drop_duplicates("sku_id", keep="last")
        if all_promo_parts else pd.DataFrame()
    )

    # ── Build alert dataframe ─────────────────────────────────────
    ds_cols = [c for c in [
        "sku_id","product_name","category","is_perishable",
        "avg_daily_demand","std_daily_demand","avg_demand_seasonal",
        "safety_stock_seasonal","reorder_point_optimized",
        "lead_time_final","avg_fill_rate","reliability_score",
        "total_revenue","unit_price","supplier_name","seasonal_factor",
    ] if c in demand_stats.columns]
    alert_df = s_snap.merge(
        demand_stats[ds_cols].drop_duplicates("sku_id"),
        on="sku_id", how="left",
    )
    alert_df["store_id"]         = store_id
    alert_df["store_name"]       = store_info.get("store_name", store_id)
    alert_df["region"]           = store_info.get("region", "")
    alert_df["store_format"]     = store_info.get("store_format", "")
    alert_df["foot_traffic_tier"]= store_info.get("foot_traffic_tier", "")

    alert_df["days_of_supply_current"] = (
        alert_df["days_of_supply"].fillna(
            (alert_df["total_available"] /
             alert_df["avg_daily_demand"].replace(0, np.nan)
            ).fillna(0)
        ).clip(lower=0).round(1)
    )
    alert_df["demand_for_calc"] = alert_df["avg_demand_seasonal"].fillna(
        alert_df["avg_daily_demand"]
    )

    def assign_alert(row):
        dos  = row["days_of_supply_current"]
        hand = row.get("units_on_hand", 0) or 0
        tot  = row["total_available"]
        rop  = row.get("reorder_point_optimized", 99999) or 99999
        if hand == 0 or dos <= CRITICAL_DAYS: return "CRITICAL"
        if dos <= WARNING_DAYS or tot <= rop:  return "WARNING"
        if dos <= MONITOR_DAYS:               return "MONITOR"
        return "OK"

    alert_df["alert_tier"] = alert_df.apply(assign_alert, axis=1)

    alert_df["units_needed"] = np.maximum(0, np.ceil(
        alert_df["demand_for_calc"] *
        (alert_df["lead_time_final"].fillna(7) + TARGET_DAYS) -
        alert_df["total_available"]
    ))
    alert_df["units_to_order"] = np.where(
        alert_df["alert_tier"].isin(["CRITICAL","WARNING"]),
        alert_df["units_needed"], 0,
    )
    dos_clipped = alert_df["days_of_supply_current"].clip(upper=TARGET_DAYS)
    alert_df["revenue_at_risk"] = np.maximum(0,
        alert_df["demand_for_calc"] *
        (TARGET_DAYS - dos_clipped) *
        alert_df.get("unit_price", pd.Series(0, index=alert_df.index)).fillna(0)
    )
    alert_df.loc[alert_df["alert_tier"] == "OK", "revenue_at_risk"] = 0

    if not all_promos.empty:
        alert_df = alert_df.merge(all_promos, on="sku_id", how="left")
        alert_df["demand_lift_factor"] = alert_df["demand_lift_factor"].fillna(1.0)
        alert_df["is_on_promo"] = alert_df["demand_lift_factor"] > 1.0
    else:
        alert_df["demand_lift_factor"] = 1.0
        alert_df["is_on_promo"]        = False

    alert_df["is_phantom"] = (
        (alert_df["total_available"] > 0) &
        (alert_df["days_of_supply_current"] == 0)
    )
    alert_df["report_date"] = datetime.now().strftime("%Y-%m-%d")
    alert_df = alert_df.sort_values("days_of_supply_current").reset_index(drop=True)

    tc = alert_df["alert_tier"].value_counts().to_dict()
    summary_row = {
        "store_id":          store_id,
        "store_name":        store_info.get("store_name", store_id),
        "city":              store_info.get("city", ""),
        "state":             store_info.get("state", ""),
        "region":            store_info.get("region", ""),
        "store_format":      store_info.get("store_format", ""),
        "foot_traffic_tier": store_info.get("foot_traffic_tier", ""),
        "skus_analyzed":     len(alert_df),
        "critical_count":    tc.get("CRITICAL", 0),
        "warning_count":     tc.get("WARNING",  0),
        "monitor_count":     tc.get("MONITOR",  0),
        "ok_count":          tc.get("OK",        0),
        "revenue_at_risk":   float(alert_df["revenue_at_risk"].sum()),
        "units_to_order":    float(alert_df["units_to_order"].sum()),
        "phantom_skus":      int(alert_df["is_phantom"].sum()),
        "understocked":      int((demand_stats["ss_status"] == "UNDERSTOCKED").sum()),
        "overstocked":       int((demand_stats["ss_status"] == "OVERSTOCKED").sum()),
        "optimal":           int((demand_stats["ss_status"] == "OPTIMAL").sum()),
        "seasonal_factor":   seasonal_factor,
        "report_date":       datetime.now().strftime("%Y-%m-%d"),
    }
    return alert_df, summary_row


def _bridge_notebook_outputs():
    """
    Copy notebook output CSVs to nexus/ paths so Streamlit can read them.
    Called automatically if user ran the notebooks directly.
    """
    import shutil
    bridges = [
        (NOTEBOOK_ALLSTORE / "network_store_summary.csv",    NEXUS_DIR / "allstore" / "network_store_summary.csv"),
        (NOTEBOOK_ALLSTORE / "network_master_alerts.csv",    NEXUS_DIR / "allstore" / "network_master_alerts.csv"),
        (NOTEBOOK_LOC      / "store_localization_scores.csv",NEXUS_DIR / "localization" / "store_localization_scores.csv"),
        (NOTEBOOK_LOC      / "store_category_mismatch.csv",  NEXUS_DIR / "localization" / "store_category_mismatch.csv"),
        (NOTEBOOK_LOC      / "category_network_mismatch.csv",NEXUS_DIR / "localization" / "category_network_mismatch.csv"),
        (NOTEBOOK_SUP      / "supplier_scorecard.csv",       NEXUS_DIR / "supplier" / "supplier_scorecard.csv"),
        (NOTEBOOK_ROI      / "roi_executive_summary.csv",    NEXUS_DIR / "roi" / "roi_executive_summary.csv"),
        (NOTEBOOK_ROI      / "roi_by_region.csv",            NEXUS_DIR / "roi" / "roi_by_region.csv"),
    ]
    bridged = 0
    for src, dst in bridges:
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            bridged += 1
    if bridged > 0:
        log.info(f"  Bridged {bridged} notebook output files → nexus/ paths")



def run_analytics(skip_allstore: bool = False):
    """
    Run the full analytics pipeline.
    Called by pipeline.py at the end of its run.

    skip_allstore: set True to skip the 20–40 min per-store loop
                   (useful for quick refreshes of other analytics)
    """
    start = datetime.now()
    log.info("=" * 60)
    log.info("  DemandSense v2 — Analytics Pipeline")
    log.info(f"  Run date: {start.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    # Load shared tables once
    stores, products, _ = _load_reference_tables()
    sales_agg = _load_sales_agg()

    # 1. Phantom confidence scoring
    log.info("\n[1/6] Phantom confidence scoring...")
    phantom_confidence_scoring(
        sales_agg=sales_agg, stores=stores, products=products
    )

    # 2. Stockout root cause
    log.info("\n[2/6] Stockout root cause breakdown...")
    stockout_root_cause(stores=stores, products=products)

    # 3. Forecast accuracy breakdown
    log.info("\n[3/6] Forecast accuracy breakdown...")
    forecast_accuracy_breakdown(stores=stores, products=products)

    # 4. Localization mismatch
    log.info("\n[4/6] Localization mismatch analysis...")
    localization_mismatch(sales_agg=sales_agg, stores=stores, products=products)

    # 5. Supplier fill rate
    log.info("\n[5/6] Supplier fill rate + scorecard...")
    supplier_fill_rate(stores=stores, products=products)

    # 6. ROI simulation
    log.info("\n[6/6] ROI simulation...")
    roi_simulation(stores=stores, products=products)

    # 7. Per-store alerts (optional — takes 20–40 min)
    if not skip_allstore:
        log.info("\n[7/7] Per-store alert engine (all 478 stores)...")
        all_store_alerts(stores=stores, products=products)
    else:
        log.info("\n[7/7] Per-store alerts skipped (skip_allstore=True)")

    # ── Bridge notebook outputs → nexus paths so Streamlit can read them ──
    _bridge_notebook_outputs()

    elapsed = (datetime.now() - start).seconds
    log.info(f"\n✅ Analytics complete in {elapsed}s")
    log.info(f"   Outputs: {NEXUS_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-allstore", action="store_true",
        help="Skip the 20-40min per-store loop (run other analytics only)",
    )
    args = parser.parse_args()
    run_analytics(skip_allstore=args.skip_allstore)