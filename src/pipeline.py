# ─────────────────────────────────────────────────────────────────
# src/pipeline.py — DemandSense v2 Weekly Pipeline
# Generates ALL outputs needed by Streamlit dashboard:
#   predictions, replenishment, alerts, weekly_monitor,
#   backtest, model_summary, category_impact_scores,
#   store_service_levels
# Usage: python src/pipeline.py
# ─────────────────────────────────────────────────────────────────
import os, sys, gc, joblib, logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH     = PROJECT_ROOT / "data/raw/output/csv"
MODELS_DIR   = PROJECT_ROOT / "models"
PROCESSED    = PROJECT_ROOT / "data/processed/training"
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def load_models():
    log.info("[1/8] Loading saved model artifacts...")
    m = {}
    m["p50_high"]          = joblib.load(MODELS_DIR / "demandsense_p50_high.joblib")
    m["p50_low"]           = joblib.load(MODELS_DIR / "demandsense_p50_low.joblib")
    m["p05"]               = joblib.load(MODELS_DIR / "demandsense_p05.joblib")
    m["p95"]               = joblib.load(MODELS_DIR / "demandsense_p95.joblib")
    m["rf_phantom"]        = joblib.load(MODELS_DIR / "rf_phantom.joblib")
    m["phantom_threshold"] = joblib.load(MODELS_DIR / "phantom_threshold.joblib")
    m["forecast_features"] = joblib.load(MODELS_DIR / "forecast_features.joblib")
    m["phantom_features"]  = joblib.load(MODELS_DIR / "phantom_features.joblib")
    m["hi_tiers"]          = joblib.load(MODELS_DIR / "hi_tiers.joblib")
    m["lo_tiers"]          = joblib.load(MODELS_DIR / "lo_tiers.joblib")
    m["cat_cols"]          = joblib.load(MODELS_DIR / "cat_cols.joblib")
    m["csl_by_tier"]       = joblib.load(MODELS_DIR / "csl_by_tier.joblib")
    m["le_region"]         = joblib.load(MODELS_DIR / "le_region.joblib")
    m["le_format"]         = joblib.load(MODELS_DIR / "le_format.joblib")
    m["le_tier"]           = joblib.load(MODELS_DIR / "le_tier.joblib")
    m["le_cat"]            = joblib.load(MODELS_DIR / "le_cat.joblib")
    m["stores_fe"]         = pd.read_parquet(MODELS_DIR / "stores_features.parquet")
    m["products_fe"]       = pd.read_parquet(MODELS_DIR / "products_features.parquet")
    log.info("    Models loaded")
    return m


def load_new_data(lookback_days: int = 60, force_end_date: str = None):
    if force_end_date:
        end = pd.Timestamp(force_end_date)
    else:
        end = pd.Timestamp.now()
    cutoff     = end - pd.Timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    end_str    = end.strftime("%Y-%m-%d")
    log.info(f"[2/8] Loading raw CSVs (last {lookback_days} days)...")
    stores     = pd.read_csv(CSV_PATH / "stores.csv")
    products   = pd.read_csv(CSV_PATH / "products.csv")
    suppliers  = pd.read_csv(CSV_PATH / "suppliers.csv")
    promotions = pd.read_csv(CSV_PATH / "promotions.csv", parse_dates=["start_date","end_date"])
    stockouts  = pd.read_csv(CSV_PATH / "stockout_events.csv", parse_dates=["stockout_date","restock_date"])
    import duckdb
    con = duckdb.connect()
    sales = con.execute(f"""
        SELECT store_id, sku_id,
               CAST(sale_date AS DATE)    AS sale_date,
               SUM(units_sold)            AS units_sold,
               MAX(is_promoted::INTEGER)  AS is_promoted,
               MAX(promotion_id)          AS promotion_id,
               SUM(revenue)               AS revenue
        FROM read_csv_auto('{CSV_PATH}/sales_transactions.csv')
        WHERE CAST(sale_date AS DATE) >= '{cutoff_str}'
          AND CAST(sale_date AS DATE) <= '{end_str}'
        GROUP BY store_id, sku_id, CAST(sale_date AS DATE)
        ORDER BY store_id, sku_id, sale_date
    """).df()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])
    con.close(); gc.collect()
    log.info(f"    Sales rows loaded: {len(sales):,}  ({cutoff_str} to {end_str})")
    return stores, products, suppliers, promotions, stockouts, sales


def build_features(sales, stores, products, promotions, stockouts, models):
    log.info("[3/8] Building features (V2.2)...")
    stores_fe   = models["stores_fe"]
    products_fe = models["products_fe"]
    df = sales.copy().rename(columns={"sale_date": "date"})
    df["is_promoted"] = df["is_promoted"].astype(str).str.lower().map(
        {"true":1,"false":0,"1":1,"0":0}
    ).fillna(0).astype(int)
    # Calendar
    df["dow"]        = df["date"].dt.dayofweek
    df["month"]      = df["date"].dt.month
    df["day"]        = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    # Store context
    store_cols = [c for c in ["store_id","region_code","store_format_code","foot_traffic_code","sq_footage","region","store_format","foot_traffic_tier"] if c in stores_fe.columns]
    df = df.merge(stores_fe[store_cols], on="store_id", how="left")
    # Product context
    prod_cols = [c for c in ["sku_id","category","category_code","unit_price","is_perishable","reorder_point","safety_stock"] if c in products_fe.columns]
    df = df.merge(products_fe[prod_cols], on="sku_id", how="left")
    if "price_band" not in df.columns:
        df["price_band"] = pd.cut(df.get("unit_price", pd.Series([0]*len(df))),
            bins=[0,10,25,50,9999], labels=["Budget","Mid","Premium","Luxury"]).astype(str)
    # Promo features (V2.2)
    promo = promotions.copy()
    promo_specific = promo[promo["store_id"].notna()].copy()
    promo_all      = promo[promo["store_id"].isna()].copy()
    if len(promo_all) > 0:
        expanded = []
        for sid in df["store_id"].unique():
            tmp = promo_all.drop(columns=["store_id"]).copy(); tmp["store_id"] = sid
            expanded.append(tmp)
        promo = pd.concat([promo_specific, pd.concat(expanded, ignore_index=True)], ignore_index=True)
    else:
        promo = promo_specific
    promo = promo.dropna(subset=["store_id","sku_id","start_date","end_date"])
    promo = promo[promo["end_date"] >= promo["start_date"]].copy()
    promo["discount_pct"]       = pd.to_numeric(promo["discount_pct"],       errors="coerce").fillna(0.0).clip(0,0.95)
    promo["demand_lift_factor"] = pd.to_numeric(promo["demand_lift_factor"], errors="coerce").fillna(1.0).clip(0.5,5.0)
    promo["date"] = promo.apply(lambda r: pd.date_range(r["start_date"],r["end_date"],freq="D"), axis=1)
    promo = promo.explode("date", ignore_index=True)
    promo = promo.groupby(["store_id","sku_id","date"], as_index=False).agg(
        discount_pct=("discount_pct","max"), demand_lift_factor=("demand_lift_factor","max"))
    df = df.merge(promo, on=["store_id","sku_id","date"], how="left")
    df["discount_pct"]       = pd.to_numeric(df["discount_pct"],       errors="coerce").fillna(0.0)
    df["demand_lift_factor"] = pd.to_numeric(df["demand_lift_factor"], errors="coerce").fillna(1.0)
    df["promo_depth_x_flag"] = df["discount_pct"]       * df["is_promoted"]
    df["promo_lift_x_flag"]  = df["demand_lift_factor"] * df["is_promoted"]
    log.info(f"    Promo features — non-zero discount_pct: {(df['discount_pct']>0).sum():,}")
    # Stockout proxy
    lost = (stockouts.groupby(["store_id","sku_id","stockout_date"], as_index=False)
            ["estimated_lost_units"].sum()
            .rename(columns={"stockout_date":"date","estimated_lost_units":"lost_units_proxy"}))
    df = df.merge(lost, on=["store_id","sku_id","date"], how="left")
    df["lost_units_proxy"] = pd.to_numeric(df["lost_units_proxy"], errors="coerce").fillna(0.0)
    q99 = df["lost_units_proxy"].quantile(0.99)
    if q99 > 0: df["lost_units_proxy"] = df["lost_units_proxy"].clip(0, q99)
    df["stockout_flag"] = (df["lost_units_proxy"] > 0).astype(int)
    log.info(f"    Stockout proxy — stockout days: {df['stockout_flag'].sum():,}")
    # Rolling + lag
    df = df.sort_values(["store_id","sku_id","date"]).reset_index(drop=True)
    def rolling_feats(g):
        s = g["units_sold"]; r = g.copy()
        r["roll_mean_7"]  = s.shift(1).rolling(7,  min_periods=1).mean()
        r["roll_mean_28"] = s.shift(1).rolling(28, min_periods=3).mean()
        r["roll_std_7"]   = s.shift(1).rolling(7,  min_periods=2).std().fillna(0)
        r["roll_std_28"]  = s.shift(1).rolling(28, min_periods=3).std().fillna(0)
        r["lag_1"]  = s.shift(1); r["lag_7"]  = s.shift(7)
        r["lag_14"] = s.shift(14); r["lag_28"] = s.shift(28)
        return r
    df = (df.groupby(["store_id","sku_id"], group_keys=False)
          .apply(rolling_feats, include_groups=True).reset_index(drop=True))
    all_feat = ["roll_mean_7","roll_mean_28","roll_std_7","roll_std_28",
                "lag_1","lag_7","lag_14","lag_28",
                "discount_pct","demand_lift_factor","promo_depth_x_flag","promo_lift_x_flag",
                "lost_units_proxy","stockout_flag"]
    for c in all_feat:
        if c in df.columns: df[c] = df[c].fillna(0)
        else: df[c] = 0.0; log.warning(f"    Column {c} missing — filled with 0")
    for c in models["cat_cols"]:
        if c in df.columns: df[c] = df[c].astype("category")
    df = df.rename(columns={"units_sold":"target"})
    FEAT = [c for c in models["forecast_features"] if c in df.columns]
    missing = set(models["forecast_features"]) - set(FEAT)
    if missing: log.warning(f"    Missing features: {missing}")
    log.info(f"    Features built: {len(FEAT)} of {len(models['forecast_features'])} expected")
    return df, FEAT


def run_inference(df, FEAT, models):
    log.info("[4/8] Running inference...")
    HI = models["hi_tiers"]; LO = models["lo_tiers"]
    X_all = df[[c for c in FEAT if c in df.columns]].copy()
    for c in FEAT:
        if c not in X_all.columns: X_all[c] = 0.0
    for c in X_all.columns:
        if hasattr(X_all[c], 'cat'):
            mv = X_all[c].mode(); X_all[c] = X_all[c].fillna(mv[0]) if len(mv) > 0 else X_all[c].fillna(X_all[c].cat.categories[0])
        else: X_all[c] = X_all[c].fillna(0)
    X_all = X_all[FEAT]
    mask_hi = df["foot_traffic_tier"].isin(HI)
    mask_lo = df["foot_traffic_tier"].isin(LO)
    pred = df[["store_id","sku_id","date","foot_traffic_tier","category","target"]].copy()
    pred["forecast_units"] = 0.0
    if mask_hi.sum() > 0: pred.loc[mask_hi,"forecast_units"] = models["p50_high"].predict(X_all[mask_hi]).clip(0)
    if mask_lo.sum() > 0: pred.loc[mask_lo,"forecast_units"] = models["p50_low"].predict(X_all[mask_lo]).clip(0)
    pred["lower_bound_90"] = models["p05"].predict(X_all).clip(0)
    pred["upper_bound_90"] = models["p95"].predict(X_all).clip(0)
    swap = pred["lower_bound_90"] > pred["upper_bound_90"]
    pred.loc[swap,["lower_bound_90","upper_bound_90"]] = pred.loc[swap,["upper_bound_90","lower_bound_90"]].values
    # Keep forecast_p50 alias for analytics compatibility
    pred["forecast_p50"] = pred["forecast_units"]
    log.info(f"    Predictions generated: {len(pred):,} rows")
    return pred


def compute_safety_stock(pred, models, suppliers, products):
    log.info("[5/8] Computing safety stock + replenishment inputs...")
    Z_MAP = {0.90:1.2816, 0.95:1.6449, 0.975:1.96}
    CSL   = models["csl_by_tier"]
    latest = pred.sort_values("date").groupby(["store_id","sku_id"]).last().reset_index()
    latest["mu_daily"]    = latest["forecast_units"].clip(lower=0)
    latest["sigma_daily"] = ((latest["upper_bound_90"] - latest["forecast_units"]) / 1.645).clip(lower=0)
    latest["csl"]   = latest["foot_traffic_tier"].map(CSL).fillna(0.95)
    latest["z_val"] = latest["csl"].map(Z_MAP)
    sku_sup = products[["sku_id","supplier_id"]].merge(
        suppliers[["supplier_id","lead_time_days_avg","lead_time_days_std","reliability_score"]],
        on="supplier_id", how="left")
    latest = latest.merge(sku_sup, on="sku_id", how="left")
    latest["lead_time_days_avg"] = latest["lead_time_days_avg"].fillna(7)
    latest["lead_time_days_std"] = latest["lead_time_days_std"].fillna(1)
    latest["reliability_score"]  = latest["reliability_score"].fillna(0.9)
    # NEXUS formula: SS = Z * sqrt(LT*sigma_D^2 + D^2*sigma_LT^2)
    latest["safety_stock_units"] = np.ceil(
        latest["z_val"] * np.sqrt(
            latest["lead_time_days_avg"] * latest["sigma_daily"]**2 +
            latest["mu_daily"]**2 * latest["lead_time_days_std"]**2
        )
    ).clip(lower=0)
    latest["reorder_point"]   = np.ceil(latest["mu_daily"] * latest["lead_time_days_avg"] + latest["safety_stock_units"])
    latest["target_upto_old"] = np.ceil(latest["mu_daily"])
    latest["target_upto_new"] = np.ceil(latest["mu_daily"] + latest["z_val"] * latest["sigma_daily"])
    # Join inventory snapshot
    try:
        import duckdb
        con = duckdb.connect()
        inv = con.execute(f"""
            SELECT store_id, sku_id,
                   MAX(snapshot_date) AS latest_snapshot,
                   LAST(units_on_hand      ORDER BY snapshot_date) AS units_on_hand,
                   LAST(units_in_backroom  ORDER BY snapshot_date) AS units_in_backroom,
                   LAST(days_of_supply     ORDER BY snapshot_date) AS days_of_supply_snapshot
            FROM read_csv_auto('{CSV_PATH}/inventory_snapshots.csv')
            GROUP BY store_id, sku_id
        """).df()
        con.close()
        inv["units_total"] = inv["units_on_hand"].fillna(0) + inv["units_in_backroom"].fillna(0)
        latest = latest.merge(inv, on=["store_id","sku_id"], how="left")
        log.info(f"    Joined inventory snapshot: {inv['units_on_hand'].notna().sum():,} rows")
    except Exception as e:
        log.warning(f"    Could not load inventory snapshot: {e}")
        latest["units_on_hand"] = np.nan; latest["units_in_backroom"] = np.nan
        latest["units_total"] = np.nan; latest["days_of_supply_snapshot"] = np.nan
    latest["days_of_supply_current"] = latest["days_of_supply_snapshot"].fillna(
        latest["units_total"] / latest["mu_daily"].replace(0, np.nan)
    ).fillna(0).clip(lower=0).round(1)
    latest["is_below_rop"] = latest["units_total"].fillna(0) < latest["reorder_point"]
    log.info(f"    Replenishment inputs: {len(latest):,} store-SKU pairs")
    log.info(f"    Below ROP: {latest['is_below_rop'].sum():,} ({latest['is_below_rop'].mean()*100:.1f}%)")
    return latest


def generate_alerts(pred, repl_inputs, models):
    """
    Generate alert table using the 5-factor Nexus priority score.
    Exact formula from nexus_allstore_pipeline.ipynb build_alert_df():
      F1 = tier weight      (CRITICAL=1000, WARNING=100, MONITOR=10, OK=0)
      F2 = revenue at risk  (/ 1000 to normalise)
      F3 = days of supply gap (how far below TARGET_DAYS)
      F4 = supplier reliability penalty
      F5 = seasonal demand boost
      priority_score = F1 + F2 + F3 + F4 + F5
    """
    log.info("[6/8] Generating alert table (Nexus 5-factor scoring)...")
    alerts = repl_inputs.copy()

    DOS_CRITICAL = 3;  DOS_WARNING = 7;  DOS_MONITOR = 14;  TARGET_DAYS = 14
    TIER_WEIGHT  = {"CRITICAL": 1000, "WARNING": 100, "MONITOR": 10, "OK": 0}
    REV_RISK_DIV = 1000
    DOS_WEIGHT   = 5
    SUPP_WEIGHT  = 20
    SEASON_WEIGHT= 10

    # ── Alert tier from days of supply ────────────────────────────
    def alert_tier(row):
        dos  = row.get("days_of_supply_current", -1)
        hand = row.get("units_on_hand", row.get("units_total", 0)) or 0
        tot  = row.get("units_total", 0) or 0
        rop  = row.get("reorder_point", 99999) or 99999
        if dos < 0:  # no inventory data — fallback
            if row.get("mu_daily", 0) < 0.1: return "OK"
            return "MONITOR"
        if hand == 0 or dos <= DOS_CRITICAL: return "CRITICAL"
        if dos <= DOS_WARNING or tot <= rop:  return "WARNING"
        if dos <= DOS_MONITOR:                return "MONITOR"
        return "OK"

    alerts["alert_tier"] = alerts.apply(alert_tier, axis=1)

    # ── Revenue at risk ───────────────────────────────────────────
    dos_clipped = alerts["days_of_supply_current"].clip(upper=TARGET_DAYS).fillna(TARGET_DAYS)
    mu_col      = "mu_daily" if "mu_daily" in alerts.columns else "forecast_units"
    price_col   = "unit_price" if "unit_price" in alerts.columns else None
    if price_col and price_col in alerts.columns:
        alerts["revenue_at_risk"] = np.maximum(0,
            alerts[mu_col].fillna(0) *
            (TARGET_DAYS - dos_clipped) *
            alerts[price_col].fillna(0)
        )
    else:
        alerts["revenue_at_risk"] = np.maximum(0,
            alerts[mu_col].fillna(0) * (TARGET_DAYS - dos_clipped)
        )
    alerts.loc[alerts["alert_tier"] == "OK", "revenue_at_risk"] = 0

    # ── Phantom detection ─────────────────────────────────────────
    alerts["demand_vs_forecast"] = alerts["target"] / alerts["forecast_units"].clip(lower=1)
    alerts["phantom_score"] = (
        (1 - alerts["demand_vs_forecast"].clip(0,1)) *
        (alerts[mu_col].fillna(0) / (alerts[mu_col].fillna(0).max() + 1e-5))
    ).clip(0,1)
    alerts["is_phantom_suspect"] = (
        (alerts[mu_col].fillna(0) > 5) & (alerts["demand_vs_forecast"] < 0.1)
    )

    # ── Seasonal factor ───────────────────────────────────────────
    if "seasonal_factor" not in alerts.columns:
        alerts["seasonal_factor"] = 1.0

    # ── 5-factor Nexus priority score ─────────────────────────────
    alerts["f1_tier"]     = alerts["alert_tier"].map(TIER_WEIGHT).fillna(0)
    alerts["f2_revenue"]  = alerts["revenue_at_risk"].fillna(0) / REV_RISK_DIV
    max_dos = alerts["days_of_supply_current"].clip(0, TARGET_DAYS).max()
    alerts["f3_dos"]      = (max_dos - alerts["days_of_supply_current"].fillna(TARGET_DAYS).clip(0, TARGET_DAYS)) / max(max_dos, 1) * DOS_WEIGHT
    alerts["f4_supplier"] = (1 - alerts.get("reliability_score", pd.Series(0.85, index=alerts.index)).fillna(0.85).clip(0,1)) * SUPP_WEIGHT
    alerts["f5_seasonal"] = (alerts["seasonal_factor"].fillna(1.0) - 1) * SEASON_WEIGHT
    alerts["priority_score"] = (
        alerts["f1_tier"] + alerts["f2_revenue"] + alerts["f3_dos"] +
        alerts["f4_supplier"] + alerts["f5_seasonal"]
    )

    # ── Priority label (for backward compatibility) ───────────────
    threshold = models["phantom_threshold"]
    def priority(r):
        if r["alert_tier"] == "CRITICAL" or r["phantom_score"] >= threshold + 0.03:
            return "HIGH"
        if r["alert_tier"] in ("WARNING","MONITOR") or r["phantom_score"] >= threshold:
            return "MEDIUM"
        return "OK"
    alerts["priority"] = alerts.apply(priority, axis=1)
    alerts["recommended_action"] = alerts["priority"].map({
        "HIGH":   "IMMEDIATE — verify inventory and place order now",
        "MEDIUM": "MONITOR — schedule replenishment this week",
        "OK":     "WATCH — check again next cycle",
    }).fillna("WATCH")
    alerts["pipeline_run_date"] = datetime.now().strftime("%Y-%m-%d")

    # Sort by priority_score descending (most urgent first)
    alerts = alerts.sort_values("priority_score", ascending=False).reset_index(drop=True)

    h  = (alerts["priority"]=="HIGH").sum()
    m  = (alerts["priority"]=="MEDIUM").sum()
    ok = (alerts["priority"]=="OK").sum()
    ps_max = alerts["priority_score"].max()
    log.info(f"    Alerts: HIGH={h:,}  MEDIUM={m:,}  OK={ok:,}  (max priority_score={ps_max:.0f})")
    log.info(f"    5-factor scoring: F1(tier) + F2(revenue) + F3(DOS gap) + F4(supplier) + F5(seasonal)")
    return alerts


def run_backtest(repl_inputs):
    """Cell 9 from retail_ai_pipeline.ipynb — Black Friday + Christmas backtest."""
    log.info("[7a/8] Running backtest simulation (Black Friday + Christmas)...")
    df_bt = repl_inputs.sort_values(["store_id","sku_id","date"]).copy()
    if "date" not in df_bt.columns:
        log.warning("    No date column in repl_inputs — skipping backtest")
        return None
    df_bt["date"] = pd.to_datetime(df_bt["date"])
    df_bt["week"] = df_bt["date"].dt.to_period("W").astype(str)
    weeks_to_show = ["2025-11-24/2025-11-30","2025-12-22/2025-12-28"]
    demo = df_bt[df_bt["week"].isin(weeks_to_show)].copy()
    if demo.empty:
        log.warning("    No Black Friday/Christmas data found — skipping backtest")
        return None
    LEAD_TIME_DAYS = 2; REVIEW_PERIOD_DAYS = 1
    effective_h = 1.0 + (LEAD_TIME_DAYS / REVIEW_PERIOD_DAYS)
    demo["init_on_hand"] = np.ceil(demo["mu_daily"] * (LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS))
    CAP_MULTIPLIER = 1.8
    def simulate_policy(data, policy_name):
        in_transit = defaultdict(list); on_hand = {}; out = []
        for r in data.itertuples(index=False):
            key = (r.store_id, r.sku_id); dt = r.date
            if key not in on_hand: on_hand[key] = float(r.init_on_hand)
            arrivals = 0.0; remaining = []
            for arr_dt, qty in in_transit[key]:
                if arr_dt <= dt: arrivals += qty
                else: remaining.append((arr_dt, qty))
            in_transit[key] = remaining; on_hand[key] += arrivals
            on_order_qty = sum(q for _,q in in_transit[key])
            inv_position = on_hand[key] + on_order_qty
            base_h  = r.mu_daily * effective_h
            sigma_h = r.sigma_daily * np.sqrt(effective_h)
            if policy_name == "OLD":
                raw_target = float(np.ceil(base_h))
            else:
                z = float(r.z_val) if hasattr(r,"z_val") else 1.96
                raw_target = float(np.ceil(base_h + z * sigma_h))
            cap_target  = float(np.ceil(r.upper_bound_90 * CAP_MULTIPLIER))
            target_upto = min(raw_target, cap_target)
            order_qty = 0.0
            if (dt.toordinal() % REVIEW_PERIOD_DAYS == 0) and (inv_position < target_upto):
                order_qty = target_upto - inv_position
                in_transit[key].append((dt + pd.Timedelta(days=LEAD_TIME_DAYS), order_qty))
            demand = float(r.target); sales = min(on_hand[key], demand)
            lost_sales = max(0.0, demand - on_hand[key])
            on_hand[key] = max(0.0, on_hand[key] - demand)
            out.append({"policy":policy_name,"store_id":r.store_id,"sku_id":r.sku_id,
                        "date":dt,"week":r.week,"target":demand,"sales":sales,
                        "lost_sales":lost_sales,"stockout_flag":lost_sales>0,
                        "order_qty":order_qty,"on_hand_end":on_hand[key]})
        return pd.DataFrame(out)
    old_res = simulate_policy(demo, "OLD")
    new_res = simulate_policy(demo, "NEW")
    res = pd.concat([old_res, new_res], ignore_index=True)
    ov = res.groupby("policy", as_index=False).agg(
        stockout_events=("stockout_flag","sum"), lost_units=("lost_sales","sum"), demand_units=("target","sum"))
    ov["service_level_pct"] = (1 - ov["lost_units"] / ov["demand_units"].replace(0,np.nan)) * 100
    old = ov[ov["policy"]=="OLD"].iloc[0]; new = ov[ov["policy"]=="NEW"].iloc[0]
    so_red = (old["stockout_events"]-new["stockout_events"])/max(old["stockout_events"],1)*100
    lu_red = (old["lost_units"]-new["lost_units"])/max(old["lost_units"],1)*100
    log.info(f"    Backtest: stockout reduction {so_red:.2f}%  lost units reduction {lu_red:.2f}%")
    res.to_csv(PROCESSED / "backtest_daily_demandsense.csv", index=False)
    ov.to_csv(PROCESSED / "backtest_summary_demandsense.csv", index=False)
    return res, ov


def save_model_summary(pred, stockouts_df=None):
    """Cell 12 from retail_ai_pipeline.ipynb — summary metrics and category impact."""
    log.info("[7b/8] Saving model summary and category impact scores...")
    fc = "forecast_units" if "forecast_units" in pred.columns else "forecast_p50"
    pred_s = pred.copy()
    pred_s["abs_err"] = (pred_s["target"] - pred_s[fc]).abs()
    pred_s["err"]     = pred_s["target"] - pred_s[fc]
    pred_s["ape"]     = pred_s["abs_err"] / pred_s["target"].replace(0,np.nan) * 100
    total_actual = pred_s["target"].sum()
    wape  = float(pred_s["abs_err"].sum() / total_actual * 100)
    mape  = float(pred_s["ape"].mean())
    mae   = float(pred_s["abs_err"].mean())
    rmse  = float(np.sqrt((pred_s["err"]**2).mean()))
    bias  = float(pred_s["err"].mean())
    p90   = float(((pred_s["target"] >= pred_s["lower_bound_90"]) &
                   (pred_s["target"] <= pred_s["upper_bound_90"])).mean() * 100)
    # Model comparison table (from PDF)
    model_summary = pd.DataFrame([
        {"forecast_method":"MovingAvg30 (baseline)",
         "wape":28.89,"mape":45.98,"mae":None,"rmse":None,"bias":None,"p90_coverage":32.7},
        {"forecast_method":"DemandSense_v2",
         "wape":round(wape,4),"mape":round(mape,4),"mae":round(mae,4),
         "rmse":round(rmse,4),"bias":round(bias,4),"p90_coverage":round(p90,4)},
    ])
    model_summary.to_csv(PROCESSED / "demandSense_model_summary.csv", index=False)
    # Segment WAPE
    if "foot_traffic_tier" in pred_s.columns:
        seg = pred_s.groupby("foot_traffic_tier").apply(
            lambda g: g["abs_err"].sum() / g["target"].sum() * 100
        ).rename("wape_pct").reset_index()
        seg.to_csv(PROCESSED / "segment_wape_demandsense.csv", index=False)
    # Category impact score = WAPE x volume share (Cell 12)
    if "category" in pred_s.columns:
        cat_stats = pred_s.groupby("category").agg(
            actual_sum=("target","sum"), abs_err_sum=("abs_err","sum"),
            bias_sum=("err","sum")).reset_index()
        cat_stats["wape_pct"]     = cat_stats["abs_err_sum"] / cat_stats["actual_sum"] * 100
        cat_stats["volume_share"] = cat_stats["actual_sum"] / total_actual * 100
        cat_stats["impact_score"] = cat_stats["wape_pct"] * cat_stats["volume_share"]
        cat_stats = cat_stats.sort_values("impact_score", ascending=False)
        cat_stats.to_csv(PROCESSED / "category_impact_scores.csv", index=False)
    # True demand adjusted WAPE (Cell 12, PDF Section 7)
    if stockouts_df is not None:
        try:
            lost_col = "estimated_lost_units" if "estimated_lost_units" in stockouts_df.columns else None
            if lost_col:
                sl = stockouts_df.rename(columns={"stockout_date":"date"})
                pt = pred_s.merge(sl[["store_id","sku_id","date",lost_col]],
                                  on=["store_id","sku_id","date"], how="left")
                pt[lost_col] = pt[lost_col].fillna(0)
                pt["true_demand"] = pt["target"] + pt[lost_col]
                true_wape = (pt["forecast_units"] - pt["true_demand"]).abs().sum() / pt["true_demand"].sum() * 100
                pd.DataFrame([{"metric":"true_demand_wape","value":round(true_wape,4)}]).to_csv(
                    PROCESSED / "true_demand_wape.csv", index=False)
        except Exception as e:
            log.warning(f"    True demand WAPE failed: {e}")
    # Store service levels (Cell 8)
    store_sl = (pred_s.groupby("store_id").apply(
        lambda g: pd.Series({
            "total_demand":   g["target"].sum(),
            "est_lost_units": (g["target"] - g[fc].clip(upper=g["target"])).clip(lower=0).sum(),
        })
    ).assign(service_level_pct=lambda d: (1 - d["est_lost_units"]/d["total_demand"])*100).reset_index())
    store_sl.to_csv(PROCESSED / "store_service_levels.csv", index=False)
    log.info(f"    Model summary saved — WAPE {wape:.2f}%  P90 {p90:.2f}%  Bias {bias:.4f}")
    return model_summary


def run_weekly_pipeline():
    start = datetime.now()
    log.info("=" * 60)
    log.info("  DemandSense v2 — Weekly Pipeline")
    log.info(f"  Run date: {start.strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)
    models = load_models()
    stores, products, suppliers, promotions, stockouts, sales = load_new_data(
        lookback_days=60, force_end_date="2025-12-31")
    df, FEAT = build_features(sales, stores, products, promotions, stockouts, models)
    pred        = run_inference(df, FEAT, models)
    repl_inputs = compute_safety_stock(pred, models, suppliers, products)
    alerts      = generate_alerts(pred, repl_inputs, models)
    # Save main outputs
    log.info("[Save] Writing main outputs...")
    pred.to_csv(        PROCESSED / "demandSense_v2_predictions.csv",                index=False)
    repl_inputs.to_csv( PROCESSED / "replenishment_policy_inputs_demandsense.csv",   index=False)
    alerts.to_csv(      PROCESSED / "weekly_alerts.csv",                              index=False)
    # Weekly monitoring metrics
    pred_mon = pred.copy()
    fc = "forecast_units"
    pred_mon["abs_err"] = (pred_mon["target"] - pred_mon[fc]).abs()
    pred_mon["ape"]     = pred_mon["abs_err"] / pred_mon["target"].replace(0,np.nan) * 100
    pred_mon["in_90"]   = ((pred_mon["target"] >= pred_mon["lower_bound_90"]) &
                            (pred_mon["target"] <= pred_mon["upper_bound_90"]))
    pred_mon["week"] = pred_mon["date"].dt.to_period("W").astype(str)
    weekly = pred_mon.groupby("week", as_index=False).agg(
        mape=("ape","mean"), abs_err=("abs_err","sum"), actual=("target","sum"), p90=("in_90","mean"))
    weekly["wape"]         = weekly["abs_err"] / weekly["actual"] * 100
    weekly["p90_coverage"] = weekly["p90"] * 100
    bias_vals = []
    for w in weekly["week"]:
        g = pred_mon[pred_mon["week"]==w]
        bias_vals.append(float(g["target"].mean() - g[fc].mean()))
    weekly["bias"] = bias_vals
    weekly = weekly[["week","mape","wape","bias","p90_coverage"]]
    weekly.to_csv(PROCESSED / "weekly_monitor_demandsense.csv", index=False)
    # Backtest
    run_backtest(repl_inputs)
    # Model summary + category impact + store service levels
    stockouts_df = pd.read_csv(CSV_PATH / "stockout_events.csv", parse_dates=["stockout_date"])
    save_model_summary(pred, stockouts_df)
    # MLOps drift check
    WAPE_BASELINE = 24.06
    latest_wape = float(weekly.iloc[-1]["wape"]) if not weekly.empty else 0
    latest_p90  = float(weekly.iloc[-1]["p90_coverage"]) if not weekly.empty else 90
    latest_bias = abs(float(weekly.iloc[-1]["bias"])) if not weekly.empty else 0
    log.info("\n=== MLOps drift check ===")
    if latest_wape > WAPE_BASELINE + 1.5: log.warning(f"    EMERGENCY RETRAIN — WAPE {latest_wape:.2f}%")
    elif not (88 <= latest_p90 <= 92):    log.warning(f"    P90 {latest_p90:.2f}% outside 88-92% band")
    elif latest_bias > 0.5:               log.warning(f"    Bias {latest_bias:.3f} > 0.5")
    else:                                 log.info(f"    Model stable — WAPE {latest_wape:.2f}%  P90 {latest_p90:.2f}%")
    # MLOps governance
    try:
        import sys as _sys, pathlib as _pl
        _sys.path.insert(0, str(_pl.Path(__file__).parent))
        from mlops import run_weekly_governance
        gov = run_weekly_governance()
        if gov.get("retrained") and gov.get("cc_decision"):
            cc = gov["cc_decision"]
            log.info(f"    MLOps: {cc['decision']} — {cc['reason']}")
    except Exception as e:
        log.warning(f"    MLOps governance skipped: {e}")
    # Analytics layer
    try:
        from analytics import run_analytics
        run_analytics(skip_allstore=True)
    except Exception as e:
        log.warning(f"    Analytics pipeline skipped: {e}")
    elapsed = (datetime.now() - start).seconds
    log.info(f"\nPipeline complete in {elapsed}s")
    log.info(f"  Predictions  : {len(pred):,} rows")
    log.info(f"  Replenishment: {len(repl_inputs):,} rows")
    log.info(f"  Alerts       : {len(alerts):,} rows  (HIGH: {(alerts['priority']=='HIGH').sum()})")
    log.info("=" * 60)
    return {"predictions":pred, "repl_inputs":repl_inputs, "alerts":alerts}


if __name__ == "__main__":
    run_weekly_pipeline()