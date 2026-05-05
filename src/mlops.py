# ─────────────────────────────────────────────────────────────────
# src/mlops.py
# MLOps: Champion/Challenger governance — exact from Forecast PDF
#
# Summary checklist (from PDF):
#   Every week  → check last 12 weeks, validate WAPE/P90/bias
#   If YES      → trigger emergency retrain
#   Otherwise   → stick to 2-week scheduled retrain
#   After retrain → compare Champion vs Challenger
#   Only promote  → better accuracy + good P90 + NO category damage
# ─────────────────────────────────────────────────────────────────
import sys
import gc
import json
import joblib
import logging
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Allow importing pipeline.py from the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
PROCESSED    = PROJECT_ROOT / "data/processed/training"
MLOPS_DIR    = PROJECT_ROOT / "mlops"
LOGS_DIR     = PROJECT_ROOT / "logs"

MLOPS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

log = logging.getLogger(__name__)

# ── Thresholds — exact from Forecast Accuracy PDF ─────────────────
WAPE_BASELINE      = 24.06   # DemandSense_v2 established baseline
WAPE_RETRAIN_DELTA = 1.5     # emergency retrain if WAPE worsens by this
P90_LOWER          = 88.0    # P90 band lower bound
P90_UPPER          = 92.0    # P90 band upper bound
BIAS_THRESHOLD     = 0.5     # bias alert for high-volume items
MONITOR_WEEKS      = 12      # rolling window for drift detection
SCHEDULED_DAYS     = 14      # scheduled retrain every 2 weeks
GUARD_CATEGORIES   = ["Beverages", "Snacks"]  # cannot regress


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Weekly drift check
# ══════════════════════════════════════════════════════════════════
def check_drift() -> dict:
    """
    Check last 12 weeks of WAPE, P90, bias.
    Uses 4-week rolling average for WAPE to avoid holiday false alarms.
    """
    monitor_path = PROCESSED / "weekly_monitor_demandsense.csv"
    if not monitor_path.exists():
        log.warning("[MLOps] No weekly monitor file found — skipping drift check")
        return {"action": "skip", "reason": "no monitor data"}

    df     = pd.read_csv(monitor_path)
    recent = df.tail(MONITOR_WEEKS).copy()

    if recent.empty:
        return {"action": "skip", "reason": "insufficient history"}

    latest      = recent.iloc[-1]
    latest_wape = float(latest["wape"])
    latest_p90  = float(latest["p90_coverage"])
    latest_bias = abs(float(latest["bias"]))

    # Use 4-week rolling average to avoid single holiday-spike false alarms
    avg_wape_4w  = float(recent["wape"].tail(4).mean())
    rolling_wape = float(recent["wape"].mean())

    flags     = []
    emergency = False

    # Flag 1: 4-week avg WAPE worsened by > 1.5 pts from baseline
    if avg_wape_4w > WAPE_BASELINE + WAPE_RETRAIN_DELTA:
        flags.append(
            f"4-week avg WAPE {avg_wape_4w:.2f}% exceeds "
            f"baseline {WAPE_BASELINE}% + {WAPE_RETRAIN_DELTA} pts"
        )
        emergency = True

    # Flag 2: P90 coverage outside 88–92% band
    if not (P90_LOWER <= latest_p90 <= P90_UPPER):
        flags.append(
            f"P90 coverage {latest_p90:.2f}% outside "
            f"{P90_LOWER}–{P90_UPPER}% band"
        )
        emergency = True

    # Flag 3: Systematic bias > 0.5 units
    if latest_bias > BIAS_THRESHOLD:
        flags.append(
            f"Bias {latest_bias:.3f} > {BIAS_THRESHOLD} — systematic drift"
        )
        emergency = True

    # Flag 4: WAPE trending up over last 4 weeks (warning only — not emergency)
    if len(recent) >= 4:
        last4 = recent["wape"].tail(4).values
        if last4[-1] > last4[0] + 0.8:
            flags.append(
                f"WAPE trending up +{last4[-1] - last4[0]:.2f} pts "
                f"over last 4 weeks (informational)"
            )

    # Scheduled retrain check
    state_path         = MLOPS_DIR / "mlops_state.json"
    state              = _load_state(state_path)
    days_since_retrain = (
        datetime.now() -
        datetime.fromisoformat(state.get("last_retrain_date", "2025-01-01"))
    ).days
    scheduled_due = days_since_retrain >= SCHEDULED_DAYS

    action = (
        "emergency_retrain" if emergency else
        "scheduled_retrain" if scheduled_due else
        "monitor"
    )

    result = {
        "timestamp":          datetime.now().isoformat(),
        "action":             action,
        "emergency":          emergency,
        "scheduled_due":      scheduled_due,
        "days_since_retrain": days_since_retrain,
        "latest_wape":        latest_wape,
        "avg_wape_4w":        avg_wape_4w,
        "latest_p90":         latest_p90,
        "latest_bias":        latest_bias,
        "rolling_wape_12w":   rolling_wape,
        "flags":              flags,
        "weeks_monitored":    len(recent),
    }

    log.info("\n" + "=" * 55)
    log.info("  MLOps Weekly Drift Check")
    log.info("=" * 55)
    log.info(f"  Latest WAPE      : {latest_wape:.2f}%")
    log.info(f"  4-week avg WAPE  : {avg_wape_4w:.2f}%  (threshold: >{WAPE_BASELINE + WAPE_RETRAIN_DELTA:.2f}%)")
    log.info(f"  Latest P90       : {latest_p90:.2f}%   (target: {P90_LOWER}–{P90_UPPER}%)")
    log.info(f"  Latest bias      : {latest_bias:.3f}    (threshold: >{BIAS_THRESHOLD})")
    log.info(f"  Days since retrain: {days_since_retrain}  (scheduled: every {SCHEDULED_DAYS})")

    if flags:
        log.warning("\n  ⚠️  Flags triggered:")
        for f in flags:
            log.warning(f"    • {f}")
    else:
        log.info("\n  ✅ No flags — model stable")

    log.info(f"\n  → Action: {action.upper()}")
    log.info("=" * 55)

    _append_history(result)
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Retrain challenger model
# ══════════════════════════════════════════════════════════════════
def retrain_challenger(trigger: str = "scheduled") -> dict:
    """
    Retrain on latest data window and save as challenger.
    Does NOT replace champion — that happens in champion_challenger_decision().
    """
    from lightgbm import LGBMRegressor
    import lightgbm as lgb

    # FIX: import from pipeline (same directory), not src.pipeline
    from pipeline import load_new_data, build_features, load_models

    log.info(f"\n[MLOps] Starting challenger retrain — trigger: {trigger}")
    start = datetime.now()

    champ_dir = MLOPS_DIR / "champion"
    chall_dir = MLOPS_DIR / "challenger"
    champ_dir.mkdir(exist_ok=True)
    chall_dir.mkdir(exist_ok=True)

    # Back up current champion
    for f in MODELS_DIR.glob("demandsense_*.joblib"):
        shutil.copy2(f, champ_dir / f.name)
    log.info(f"  Champion backed up to {champ_dir}")

    # Load data — FIX: force_end_date because dataset ends 2025-12-31
    models = load_models()
    stores, products, suppliers, promotions, stockouts, sales = load_new_data(
        lookback_days=90,
        force_end_date="2025-12-31",
    )
    df, FEAT = build_features(sales, stores, products, promotions, stockouts, models)
    del sales
    gc.collect()

    HI  = models["hi_tiers"]
    LO  = models["lo_tiers"]
    CAT = models["cat_cols"]

    TRAIN_END  = "2025-06-30"
    VALID_END  = "2025-09-30"

    train = df[df["date"] <= TRAIN_END].copy()
    valid = df[(df["date"] > TRAIN_END) & (df["date"] <= VALID_END)].copy()

    if train.empty or valid.empty:
        log.warning("  Insufficient data for fixed split — using 80/20 split")
        split = int(len(df) * 0.8)
        train = df.iloc[:split].copy()
        valid = df.iloc[split:].copy()

    for c in CAT:
        if c in train.columns:
            train[c] = train[c].astype("category")
            valid[c] = valid[c].astype("category")

    def fit_model(tr, va, objective, alpha=None, tag=""):
        params = dict(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=300, subsample=0.8, colsample_bytree=0.8,
            force_col_wise=True, n_jobs=-1, random_state=42, verbose=-1,
        )
        if objective == "quantile":
            params.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})
        else:
            params.update({"objective": "regression", "metric": "l1"})

        m = LGBMRegressor(**params)
        for c in CAT:
            if c in tr.columns:
                tr[c] = tr[c].astype("category")
            if c in va.columns:
                va[c] = va[c].astype("category")

        m.fit(
            tr[FEAT], tr["target"],
            eval_set=[(va[FEAT], va["target"])],
            eval_metric=params["metric"],
            categorical_feature=CAT,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(200),
            ],
        )
        log.info(f"    {tag} — best iter: {m.best_iteration_}")
        return m

    tr_hi = train[train["foot_traffic_tier"].isin(HI)].copy()
    va_hi = valid[valid["foot_traffic_tier"].isin(HI)].copy()
    tr_lo = train[train["foot_traffic_tier"].isin(LO)].copy()
    va_lo = valid[valid["foot_traffic_tier"].isin(LO)].copy()

    log.info("  Training challenger P50 High/Premium...")
    p50_hi = fit_model(tr_hi, va_hi, "regression", tag="P50 Hi/Premium")
    log.info("  Training challenger P50 Low/Medium...")
    p50_lo = fit_model(tr_lo, va_lo, "regression", tag="P50 Lo/Medium")
    log.info("  Training challenger P05...")
    p05    = fit_model(train.copy(), valid.copy(), "quantile", alpha=0.05, tag="P05")
    log.info("  Training challenger P95...")
    p95    = fit_model(train.copy(), valid.copy(), "quantile", alpha=0.95, tag="P95")

    joblib.dump(p50_hi, chall_dir / "demandsense_p50_high.joblib")
    joblib.dump(p50_lo, chall_dir / "demandsense_p50_low.joblib")
    joblib.dump(p05,    chall_dir / "demandsense_p05.joblib")
    joblib.dump(p95,    chall_dir / "demandsense_p95.joblib")

    # Evaluate challenger on test period (Oct–Dec 2025)
    test = df[df["date"] > VALID_END].copy()
    if test.empty:
        log.warning("  No test data beyond VALID_END — using validation set")
        test = valid.copy()

    for c in CAT:
        if c in test.columns:
            test[c] = test[c].astype("category")

    X_test = test[FEAT].copy()
    for c in X_test.columns:
        if hasattr(X_test[c], 'cat'):
            mode_val = X_test[c].mode()
            if len(mode_val) > 0:
                X_test[c] = X_test[c].fillna(mode_val[0])
        else:
            X_test[c] = X_test[c].fillna(0)
    pred    = test[["store_id", "sku_id", "date", "target",
                    "category", "foot_traffic_tier"]].copy()
    mask_hi = test["foot_traffic_tier"].isin(HI)
    mask_lo = test["foot_traffic_tier"].isin(LO)

    pred["forecast_p50"] = 0.0
    if mask_hi.sum() > 0:
        pred.loc[mask_hi, "forecast_p50"] = p50_hi.predict(X_test[mask_hi]).clip(0)
    if mask_lo.sum() > 0:
        pred.loc[mask_lo, "forecast_p50"] = p50_lo.predict(X_test[mask_lo]).clip(0)

    pred["lower_bound_90"] = p05.predict(X_test).clip(0)
    pred["upper_bound_90"] = p95.predict(X_test).clip(0)

    swap = pred["lower_bound_90"] > pred["upper_bound_90"]
    pred.loc[swap, ["lower_bound_90", "upper_bound_90"]] = \
        pred.loc[swap, ["upper_bound_90", "lower_bound_90"]].values

    challenger_metrics = _compute_metrics(pred, "challenger")
    challenger_metrics["retrain_trigger"] = trigger
    challenger_metrics["retrain_date"]    = datetime.now().isoformat()
    challenger_metrics["train_rows"]      = len(train)

    with open(chall_dir / "challenger_metrics.json", "w") as fh:
        json.dump(challenger_metrics, fh, indent=2)

    log.info(f"\n  Challenger metrics:")
    log.info(f"    WAPE : {challenger_metrics['wape']:.2f}%")
    log.info(f"    P90  : {challenger_metrics['p90_coverage']:.2f}%")
    log.info(f"    Bias : {challenger_metrics['bias']:.4f}")
    log.info(f"  Retrain complete in {(datetime.now() - start).seconds}s")

    gc.collect()
    return challenger_metrics


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Champion/Challenger decision
# ══════════════════════════════════════════════════════════════════
def champion_challenger_decision() -> dict:
    """
    3-scenario decision from PDF (pages 25–26):
      Scenario A — clear winner  : WAPE improved + P90 in band + no category damage → PROMOTE
      Scenario B — broken bounds : better WAPE but P90 outside 88–92% → REJECT
      Scenario C — category damage: globally better but Beverages/Snacks worse → REJECT
    """
    chall_dir          = MLOPS_DIR / "challenger"
    champ_metrics_path = MLOPS_DIR / "champion_metrics.json"
    chall_metrics_path = chall_dir / "challenger_metrics.json"

    if not chall_metrics_path.exists():
        log.warning("[MLOps] No challenger metrics found — run retrain_challenger() first")
        return {"decision": "no_challenger", "promoted": False}

    champ = (
        json.load(open(champ_metrics_path))
        if champ_metrics_path.exists()
        else {
            "forecast_method": "DemandSense_v2 (champion)",
            "wape": WAPE_BASELINE,
            "p90_coverage": 89.75,
            "bias": -0.148,
            "category_wape": {},
        }
    )
    chall = json.load(open(chall_metrics_path))

    log.info("\n" + "=" * 55)
    log.info("  Champion / Challenger Decision")
    log.info("=" * 55)
    log.info(f"  {'Metric':<22} {'Champion':>10} {'Challenger':>12}  Pass/Fail")
    log.info("  " + "-" * 54)

    wape_pass = chall["wape"] < champ["wape"]
    log.info(
        f"  {'WAPE (lower better)':<22} {champ['wape']:>9.2f}%"
        f" {chall['wape']:>11.2f}%  {'✅ PASS' if wape_pass else '❌ FAIL'}"
    )

    p90_pass = P90_LOWER <= chall["p90_coverage"] <= P90_UPPER
    log.info(
        f"  {'P90 coverage':<22} {champ['p90_coverage']:>9.2f}%"
        f" {chall['p90_coverage']:>11.2f}%  {'✅ PASS' if p90_pass else '❌ FAIL (broken bounds)'}"
    )

    cat_pass   = True
    cat_reason = ""
    chall_cat  = chall.get("category_wape", {})
    champ_cat  = champ.get("category_wape", {})

    for cat in GUARD_CATEGORIES:
        champ_w = champ_cat.get(cat, champ["wape"])
        chall_w = chall_cat.get(cat, chall["wape"])
        if chall_w > champ_w + 0.5:
            cat_pass   = False
            cat_reason = f"{cat} WAPE worsened: {champ_w:.2f}% → {chall_w:.2f}%"
            log.info(
                f"  {cat:<22} {champ_w:>9.2f}%"
                f" {chall_w:>11.2f}%  ❌ FAIL (category damage)"
            )
            break
        else:
            log.info(
                f"  {cat:<22} {champ_w:>9.2f}%"
                f" {chall_w:>11.2f}%  ✅ PASS"
            )

    log.info("  " + "-" * 54)

    if wape_pass and p90_pass and cat_pass:
        scenario = "A"
        decision = "PROMOTE"
        promoted = True
        reason   = (
            f"Challenger wins — WAPE {chall['wape']:.2f}% < {champ['wape']:.2f}%, "
            f"P90 in band, no category damage"
        )
    elif wape_pass and not p90_pass:
        scenario = "B"
        decision = "REJECT"
        promoted = False
        reason   = (
            f"Broken bounds — P90 {chall['p90_coverage']:.2f}% "
            f"outside {P90_LOWER}–{P90_UPPER}% band"
        )
    elif not cat_pass:
        scenario = "C"
        decision = "REJECT"
        promoted = False
        reason   = f"Category damage — {cat_reason}"
    else:
        scenario = "D"
        decision = "REJECT"
        promoted = False
        reason   = f"WAPE did not improve: {chall['wape']:.2f}% vs {champ['wape']:.2f}%"

    log.info(f"\n  Scenario {scenario}: {decision}")
    log.info(f"  Reason: {reason}")

    if promoted:
        log.info("\n  Promoting challenger to production...")
        for f in chall_dir.glob("demandsense_*.joblib"):
            shutil.copy2(f, MODELS_DIR / f.name)
            log.info(f"    Copied {f.name}")
        chall["promoted_at"]         = datetime.now().isoformat()
        chall["prior_champion_wape"] = champ["wape"]
        with open(MLOPS_DIR / "champion_metrics.json", "w") as fh:
            json.dump(chall, fh, indent=2)
        log.info("  ✅ Champion updated — new model is live")
    else:
        log.info("\n  ⚠️  Challenger rejected — champion remains in production")

    result = {
        "scenario":   scenario,
        "decision":   decision,
        "promoted":   promoted,
        "reason":     reason,
        "champion":   champ,
        "challenger": chall,
        "timestamp":  datetime.now().isoformat(),
    }
    _append_history(result)
    log.info("=" * 55)
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 4 — Full weekly governance run
# ══════════════════════════════════════════════════════════════════
def run_weekly_governance() -> dict:
    """
    PDF checklist:
      1. Check last 12 weeks
      2. Emergency → retrain now
      3. Scheduled due → retrain now
      4. After retrain → Champion/Challenger decision
      5. Log everything
    """
    log.info("\n[MLOps] === Weekly governance run ===")
    drift     = check_drift()
    action    = drift["action"]
    retrained = False
    cc_result = None

    if action in ("emergency_retrain", "scheduled_retrain"):
        log.info(f"\n[MLOps] Retraining triggered: {action}")
        retrain_challenger(trigger=action)
        retrained = True
        cc_result = champion_challenger_decision()
    else:
        log.info("\n[MLOps] No retrain needed this week — model stable")

    state = _load_state(MLOPS_DIR / "mlops_state.json")
    state["last_check_date"] = datetime.now().isoformat()
    if retrained:
        state["last_retrain_date"] = datetime.now().isoformat()
    _save_state(MLOPS_DIR / "mlops_state.json", state)

    return {
        "drift_check": drift,
        "retrained":   retrained,
        "cc_decision": cc_result,
        "run_date":    datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def _compute_metrics(pred: pd.DataFrame, label: str) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    y       = pred["target"]
    f       = pred["forecast_p50"]
    err     = y - f
    abs_err = err.abs()
    ape     = abs_err / y.replace(0, np.nan) * 100
    cov     = (
        (y >= pred["lower_bound_90"]) & (y <= pred["upper_bound_90"])
    ).mean() * 100

    cat_wape = {}
    if "category" in pred.columns:
        for cat, grp in pred.groupby("category"):
            cat_wape[cat] = float(
                (grp["target"] - grp["forecast_p50"]).abs().sum() /
                grp["target"].sum() * 100
            )

    return {
        "forecast_method": label,
        "mape":            float(np.nanmean(ape)),
        "wape":            float(abs_err.sum() / y.sum() * 100),
        "mae":             float(mean_absolute_error(y, f)),
        "rmse":            float(np.sqrt(mean_squared_error(y, f))),
        "bias":            float(err.mean()),
        "p90_coverage":    float(cov),
        "category_wape":   cat_wape,
        "rows_evaluated":  len(pred),
    }


def _load_state(path: Path) -> dict:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {"last_retrain_date": "2025-06-30", "last_check_date": "2025-06-30"}


def _save_state(path: Path, state: dict):
    with open(path, "w") as fh:
        json.dump(state, fh, indent=2)


def _append_history(entry: dict):
    path = MLOPS_DIR / "governance_history.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )
    run_weekly_governance()
