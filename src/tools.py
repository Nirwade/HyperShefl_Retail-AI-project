# ─────────────────────────────────────────────────────────────────
# src/tools.py  — v2.0
# Upgraded tool functions with BLUF (Bottom Line Up Front) format.
# Every tool returns:
#   directive    — 1 sentence: exact action + dollar value
#   current_state — 1 sentence: the problem
#   action       — 1 sentence: what to do physically/in system
#   impact       — 1 sentence: financial/operational result
# ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.model_registry import registry

TODAY  = datetime.today()
Z_MAP  = {0.90: 1.2816, 0.95: 1.6449, 0.975: 1.96, 0.999: 3.09}


# ─────────────────────────────────────────────────────────────────
# HELPER — get supplier info for a SKU
# ─────────────────────────────────────────────────────────────────
def _get_supplier(sku_id):
    prod = registry.products[registry.products["sku_id"] == sku_id]
    if prod.empty:
        return {"supplier_name": "Unknown", "lead_time_days_avg": 7,
                "lead_time_days_std": 1.5, "reliability_score": 0.85,
                "contact_email": "N/A"}
    sup_id = prod["supplier_id"].iloc[0]
    sup    = registry.suppliers[registry.suppliers["supplier_id"] == sup_id]
    if sup.empty:
        return {"supplier_name": "Unknown", "lead_time_days_avg": 7,
                "lead_time_days_std": 1.5, "reliability_score": 0.85,
                "contact_email": "N/A"}
    r = sup.iloc[0]
    return {
        "supplier_id":        str(r.get("supplier_id", "")),
        "supplier_name":      str(r.get("supplier_name", "Unknown")),
        "lead_time_days_avg": float(r.get("lead_time_days_avg", 7)),
        "lead_time_days_std": float(r.get("lead_time_days_std", 1.5)),
        "reliability_score":  float(r.get("reliability_score", 0.85)),
        "contact_email":      str(r.get("contact_email", "N/A")),
        "country":            str(r.get("country", "N/A")),
    }


def _get_store_info(store_id):
    s = registry.stores[registry.stores["store_id"] == store_id]
    if s.empty:
        return {}
    r = s.iloc[0]
    return {k: str(r[k]) for k in r.index if k in
            ["store_id","store_name","city","state","region",
             "store_format","foot_traffic_tier","address"]}


def _get_product_info(sku_id):
    p = registry.products[registry.products["sku_id"] == sku_id]
    if p.empty:
        return {}
    r = p.iloc[0]
    return {k: r[k] for k in r.index if k in
            ["sku_id","product_name","category","subcategory",
             "unit_price","unit_cost","pack_size","aisle_location",
             "shelf_section","is_perishable","shelf_life_days"]}


# ─────────────────────────────────────────────────────────────────
# TOOL 1 — Weekly replenishment plan for a store-SKU
# ─────────────────────────────────────────────────────────────────
def get_replenishment_action(store_id: str, sku_id: str) -> dict:
    """
    Full replenishment recommendation: order qty, supplier, order date,
    expected arrival date, aisle placement, and BLUF directive.
    """
    repl = registry.repl_inputs
    mask = (repl["store_id"] == store_id) & (repl["sku_id"] == sku_id)
    rows = repl[mask].sort_values("date")

    if rows.empty:
        return {"status": "not_found",
                "message": f"No replenishment data for {sku_id} at {store_id}"}

    latest     = rows.iloc[-1]
    mu         = float(latest["mu_daily"])
    sigma      = float(latest["sigma_daily"])
    z          = float(latest["z_val"])
    csl        = float(latest["csl"])
    ss         = float(latest["safety_stock_units"])
    rop        = float(latest["reorder_point"])
    target_new = float(latest["target_upto_new"])
    target_old = float(latest["target_upto_old"])
    lead_time  = float(latest["lead_time_days_avg"])
    category   = str(latest.get("category", "Unknown"))
    tier       = str(latest.get("foot_traffic_tier", "Unknown"))

    # Product and supplier
    prod_info  = _get_product_info(sku_id)
    store_info = _get_store_info(store_id)
    sup_info   = _get_supplier(sku_id)

    product_name  = prod_info.get("product_name", sku_id)
    unit_price    = float(prod_info.get("unit_price", 0))
    unit_cost     = float(prod_info.get("unit_cost", 0))
    pack_size     = int(prod_info.get("pack_size", 1)) or 1
    aisle         = str(prod_info.get("aisle_location", "Check planogram"))
    shelf_section = str(prod_info.get("shelf_section", "N/A"))

    # Order quantity rounded to pack size
    order_qty   = int(np.ceil(target_new / pack_size) * pack_size)
    order_value = order_qty * unit_price
    requires_approval = order_value > 5000 or sigma > mu * 0.8

    # Timing
    order_deadline = TODAY.strftime("%A, %B %d")
    order_by_time  = "3:00 PM" if TODAY.weekday() < 5 else "9:00 AM Monday"
    lead_days      = int(np.ceil(lead_time))
    arrival_date   = (TODAY + timedelta(days=lead_days)).strftime("%A, %B %d")

    # Days of supply remaining
    dos = float(latest.get("days_of_supply_current",
                target_new / max(mu, 0.01)))
    stockout_date = (TODAY + timedelta(days=dos)).strftime("%A, %B %d")
    daily_loss    = mu * unit_price

    # Weekend flag
    days_to_fri = (4 - TODAY.weekday()) % 7
    weekend_demand = mu * 1.35  # saturdays avg 35% higher from analysis
    weekend_flag   = dos <= days_to_fri + 2

    # BLUF
    directive = (
        f"Order {order_qty} units of {product_name} from "
        f"{sup_info['supplier_name']} before {order_by_time} today "
        f"to prevent a stockout by {stockout_date} "
        f"and protect ${daily_loss * dos:,.0f} in revenue."
    )
    current_state = (
        f"{product_name} at {store_info.get('store_name', store_id)} "
        f"has {dos:.1f} days of supply remaining at current demand "
        f"of {mu:.1f} units/day — hitting reorder point of {rop:.0f} units."
    )
    action = (
        f"Place PO for {order_qty} units to {sup_info['supplier_name']} "
        f"({sup_info['contact_email']}); "
        f"stock to {aisle} {shelf_section} upon arrival on {arrival_date}."
    )
    impact = (
        f"Maintains {csl*100:.1f}% service level "
        f"and protects ${mu * 7 * unit_price:,.0f} in weekly category revenue."
    )

    return {
        "status": "ok",
        # BLUF
        "directive":     directive,
        "current_state": current_state,
        "action":        action,
        "impact":        impact,
        # Store + Product
        "store_id":           store_id,
        "store_name":         store_info.get("store_name", store_id),
        "city":               store_info.get("city", ""),
        "state":              store_info.get("state", ""),
        "sku_id":             sku_id,
        "product_name":       product_name,
        "category":           category,
        "aisle_location":     aisle,
        "shelf_section":      shelf_section,
        # Order details
        "recommended_order_qty": order_qty,
        "order_value_usd":    round(order_value, 2),
        "requires_approval":  requires_approval,
        "order_by":           f"{order_deadline} {order_by_time}",
        # Supplier
        "supplier_name":       sup_info["supplier_name"],
        "supplier_id":         sup_info.get("supplier_id", ""),
        "supplier_contact":    sup_info.get("contact_email", "N/A"),
        "supplier_reliability": round(sup_info["reliability_score"] * 100, 1),
        "lead_time_days":      lead_days,
        "expected_arrival":    arrival_date,
        # Inventory
        "days_of_supply_current": round(dos, 1),
        "projected_stockout_date": stockout_date,
        "safety_stock_units":  round(ss, 1),
        "reorder_point":       round(rop, 1),
        "mu_daily":            round(mu, 2),
        "sigma_daily":         round(sigma, 2),
        # Policy
        "old_policy_qty":      int(round(target_old, 0)),
        "new_policy_qty":      order_qty,
        "ai_buffer_units":     round(target_new - target_old, 1),
        "service_level_pct":   round(csl * 100, 1),
        "formula":             f"Target = {mu:.1f} + {z} × {sigma:.1f} = {target_new:.1f} units",
        # Weekend risk
        "weekend_risk":        weekend_flag,
        "weekend_demand_est":  round(weekend_demand, 1),
    }


# ─────────────────────────────────────────────────────────────────
# TOOL 2 — Full weekly demand forecast for a store-SKU
# ─────────────────────────────────────────────────────────────────
def query_forecast(store_id: str, sku_id: str, days: int = 7) -> dict:
    """
    Daily P50/P05/P95 forecast with weekend spike flag, peak day,
    and BLUF summary for the coming week.
    """
    pred = registry.predictions
    mask = (pred["store_id"] == store_id) & (pred["sku_id"] == sku_id)
    rows = pred[mask].sort_values("date").tail(days)

    if rows.empty:
        return {"status": "not_found", "store_id": store_id,
                "sku_id": sku_id,
                "message": f"No forecast data for {sku_id} at {store_id}"}

    prod_info  = _get_product_info(sku_id)
    store_info = _get_store_info(store_id)
    product_name = prod_info.get("product_name", sku_id)
    unit_price   = float(prod_info.get("unit_price", 0))

    daily = []
    for _, r in rows.iterrows():
        p50 = round(float(r["forecast_units"]), 1)
        lo  = round(float(r.get("lower_bound_90", p50 * 0.8)), 1)
        hi  = round(float(r.get("upper_bound_90", p50 * 1.2)), 1)
        act = round(float(r["target"]), 1)
        d   = pd.to_datetime(r["date"])
        daily.append({
            "date":        d.strftime("%Y-%m-%d"),
            "day_name":    d.strftime("%A"),
            "forecast_p50": p50,
            "lower_p05":   lo,
            "upper_p95":   hi,
            "actual_units": act,
            "is_weekend":  d.dayofweek >= 5,
            "revenue_est":  round(p50 * unit_price, 2),
        })

    avg_forecast   = round(float(rows["forecast_units"].mean()), 1)
    avg_actual     = round(float(rows["target"].mean()), 1)
    peak_day       = max(daily, key=lambda x: x["forecast_p50"])
    weekend_avg    = round(sum(d["forecast_p50"] for d in daily if d["is_weekend"]) /
                     max(sum(1 for d in daily if d["is_weekend"]), 1), 1)
    weekday_avg    = round(sum(d["forecast_p50"] for d in daily if not d["is_weekend"]) /
                     max(sum(1 for d in daily if not d["is_weekend"]), 1), 1)
    weekend_spike  = weekend_avg > weekday_avg * 1.1
    total_week_rev = round(sum(d["revenue_est"] for d in daily), 2)
    trend          = ("increasing" if rows["forecast_units"].iloc[-1] > rows["forecast_units"].iloc[0]
                      else "decreasing" if rows["forecast_units"].iloc[-1] < rows["forecast_units"].iloc[0]
                      else "stable")

    directive = (
        f"{'Stock up before the weekend — demand spikes to ' + str(weekend_avg) + ' units/day Saturday vs ' + str(weekday_avg) + ' weekdays.' if weekend_spike else 'Demand is stable at ' + str(avg_forecast) + ' units/day.'} "
        f"Peak day is {peak_day['day_name']} at {peak_day['forecast_p50']} units "
        f"worth ${peak_day['forecast_p50'] * unit_price:,.0f}."
    )

    return {
        "status":        "ok",
        "directive":     directive,
        "store_id":      store_id,
        "store_name":    store_info.get("store_name", store_id),
        "sku_id":        sku_id,
        "product_name":  product_name,
        "category":      prod_info.get("category", ""),
        "aisle_location": prod_info.get("aisle_location", "Check planogram"),
        "period_days":   days,
        "avg_forecast":  avg_forecast,
        "avg_actual":    avg_actual,
        "trend":         trend,
        "peak_day":      peak_day["day_name"],
        "peak_forecast": peak_day["forecast_p50"],
        "weekend_avg_demand":  weekend_avg,
        "weekday_avg_demand":  weekday_avg,
        "weekend_spike":       weekend_spike,
        "total_week_revenue_est": total_week_rev,
        "daily_forecast":     daily,
    }


# ─────────────────────────────────────────────────────────────────
# TOOL 3 — Stockout risk with exact date and daily cost
# ─────────────────────────────────────────────────────────────────
def get_stockout_risk(store_id: str, sku_id: str) -> dict:
    """
    Risk score, exact projected stockout date, daily revenue loss,
    and BLUF directive for immediate action.
    """
    repl = registry.repl_inputs
    mask = (repl["store_id"] == store_id) & (repl["sku_id"] == sku_id)
    rows = repl[mask].sort_values("date")

    if rows.empty:
        return {"status": "not_found", "store_id": store_id,
                "sku_id": sku_id,
                "message": f"No data for {sku_id} at {store_id}"}

    latest    = rows.iloc[-1]
    mu        = float(latest["mu_daily"])
    sigma     = float(latest["sigma_daily"])
    ss        = float(latest["safety_stock_units"])
    rop       = float(latest["reorder_point"])
    target_new= float(latest["target_upto_new"])
    target_old= float(latest["target_upto_old"])
    lead_time = float(latest["lead_time_days_avg"])
    reliability = float(latest.get("reliability_score", 0.9))

    coverage_ratio = target_new / max(mu * (lead_time + 1), 0.1)
    base_risk = max(0.0, min(1.0, 1.0 - (coverage_ratio - 1.0) / 2.0))
    rel_penalty = max(0.0, (0.95 - reliability) * 0.5)
    risk_score = min(1.0, round(base_risk + rel_penalty, 3))

    if risk_score >= 0.70:   tier = "HIGH";   action_str = "IMMEDIATE — place order now"
    elif risk_score >= 0.40: tier = "MEDIUM"; action_str = "MONITOR — order this week"
    else:                    tier = "LOW";    action_str = "WATCH — check next cycle"

    prod_info  = _get_product_info(sku_id)
    store_info = _get_store_info(store_id)
    sup_info   = _get_supplier(sku_id)

    product_name = prod_info.get("product_name", sku_id)
    unit_price   = float(prod_info.get("unit_price", 0))
    aisle        = str(prod_info.get("aisle_location", "Check planogram"))

    dos  = float(latest.get("days_of_supply_current", target_new / max(mu, 0.01)))
    stockout_date = (TODAY + timedelta(days=max(dos, 0))).strftime("%A, %B %d")
    daily_loss    = round(mu * unit_price, 2)
    weekly_loss   = round(daily_loss * 7, 2)

    recent_7  = rows.tail(7)["target"].mean()  if len(rows) >= 7  else mu
    recent_14 = rows.tail(14)["target"].mean() if len(rows) >= 14 else mu
    demand_trend = ("rising"  if recent_7 > recent_14 * 1.05 else
                    "falling" if recent_7 < recent_14 * 0.95 else "stable")

    directive = (
        f"{'URGENT: ' if tier == 'HIGH' else ''}"
        f"{product_name} at {store_info.get('store_name', store_id)} "
        f"will stock out by {stockout_date} — "
        f"order from {sup_info['supplier_name']} today "
        f"to prevent ${daily_loss:,.0f}/day in lost revenue."
    )
    current_state = (
        f"{dos:.1f} days of supply remaining at {mu:.1f} units/day demand "
        f"({demand_trend}) — supplier reliability is "
        f"{reliability*100:.0f}% with {lead_time:.0f}-day lead time."
    )
    action_text = (
        f"Place PO to {sup_info['supplier_name']} for replenishment to "
        f"{aisle} before stock hits zero on {stockout_date}."
    )
    impact_text = (
        f"Each day of stockout costs ${daily_loss:,.0f} in lost sales "
        f"(${weekly_loss:,.0f}/week at current demand)."
    )

    return {
        "status":              "ok",
        "directive":           directive,
        "current_state":       current_state,
        "action":              action_text,
        "impact":              impact_text,
        "store_id":            store_id,
        "store_name":          store_info.get("store_name", store_id),
        "sku_id":              sku_id,
        "product_name":        product_name,
        "category":            prod_info.get("category", ""),
        "aisle_location":      aisle,
        "risk_score":          risk_score,
        "risk_tier":           tier,
        "recommended_action":  action_str,
        "days_of_supply":      round(dos, 1),
        "projected_stockout_date": stockout_date,
        "daily_revenue_at_risk":   daily_loss,
        "weekly_revenue_at_risk":  weekly_loss,
        "mu_daily":            round(mu, 2),
        "sigma_daily":         round(sigma, 2),
        "demand_trend":        demand_trend,
        "safety_stock_units":  round(ss, 1),
        "reorder_point":       round(rop, 1),
        "lead_time_days":      round(lead_time, 1),
        "supplier_name":       sup_info["supplier_name"],
        "supplier_reliability_pct": round(reliability * 100, 1),
    }


# ─────────────────────────────────────────────────────────────────
# TOOL 4 — Phantom inventory alerts
# ─────────────────────────────────────────────────────────────────
def get_phantom_alerts(store_id: str = None, top_n: int = 10) -> dict:
    """
    Phantom inventory suspects read directly from phantom_confidence.csv
    which has pre-computed is_phantom_candidate and phantom_confidence scores.
    """
    import pandas as pd
    from pathlib import Path as _P
    _nexus = _P(__file__).parent.parent / "data/processed/nexus"
    _ph_path = _nexus / "allstore" / "phantom_confidence.csv"

    try:
        ph = pd.read_csv(_ph_path)
    except Exception as e:
        return {"status": "error", "message": f"Could not load phantom_confidence.csv: {e}"}

    # Filter to confirmed phantom candidates
    ph = ph[ph["is_phantom_candidate"] == True].copy()
    if store_id:
        ph = ph[ph["store_id"] == store_id]

    # Sort: High confidence first, then most zero-sales days
    conf_order = {"High": 0, "Medium": 1, "Low": 2}
    ph["conf_rank"] = ph["phantom_confidence"].map(conf_order).fillna(3)
    ph["daily_rev"] = ph["rolling_14d_avg"].fillna(0) * ph["unit_price"].fillna(0)
    suspects = ph.sort_values(["conf_rank", "consec_zero_days"],
                               ascending=[True, False]).head(top_n)

    threshold = 0.5
    def confidence(s):
        return s  # already "High"/"Medium"/"Low" from CSV
        return "Low"

    alerts = []
    total_capital_locked = 0.0
    for _, r in suspects.iterrows():
        conf       = str(r.get("phantom_confidence", "Low"))
        daily_rev  = float(r.get("daily_rev", 0))
        capital    = daily_rev * 14  # 14-day capital estimate
        total_capital_locked += capital
        zero_days  = int(r.get("consec_zero_days", 0))
        pname      = str(r.get("product_name", r["sku_id"]))
        category   = str(r.get("category", ""))

        alerts.append({
            "store_id":             r["store_id"],
            "store_name":           str(r.get("store_name", r["store_id"])),
            "sku_id":               r["sku_id"],
            "product_name":         pname,
            "category":             category,
            "aisle_location":       "Check planogram",
            "phantom_score":        conf,
            "confidence":           conf,
            "zero_sales_days":      zero_days,
            "est_daily_rev_at_risk": round(daily_rev, 2),
            "capital_locked":       round(capital, 2),
            "directive": (
                f"Send associate to verify {pname} ({category}) is on shelf — "
                f"{zero_days} consecutive zero-sales days. "
                f"${daily_rev:,.0f}/day revenue at risk."
            ),
            "recommended_action": (
                "IMMEDIATE — verify shelf now" if conf == "High"
                else "AUDIT — flag for next cycle count"
            ),
        })

    store_name_disp = (registry.stores[registry.stores["store_id"] == store_id]
                       ["store_name"].iloc[0]
                       if store_id and not registry.stores[
                           registry.stores["store_id"] == store_id].empty
                       else "All stores")

    network_directive = (
        f"{'Audit ' + store_name_disp + ': ' + str(len(suspects)) + ' phantom SKUs with $' + f'{total_capital_locked:,.0f}' + ' capital locked — send cycle count team.' if len(suspects) > 0 else 'No phantom inventory detected at ' + store_name_disp + '.'}"
    )

    return {
        "status":               "ok",
        "directive":            network_directive,
        "store_id":             store_id or "all",
        "store_name":           store_name_disp,
        "total_suspects":       len(suspects),
        "total_capital_locked": round(total_capital_locked, 2),
        "alerts":               alerts,
    }


# ─────────────────────────────────────────────────────────────────
# TOOL 5 — Network status with top stores and region breakdown
# ─────────────────────────────────────────────────────────────────
def get_network_status(region: str = None) -> dict:
    """
    Full network KPI summary with BLUF morning briefing directive.
    """
    import pandas as pd
    from pathlib import Path

    PROJECT = Path(__file__).resolve().parent.parent
    NEXUS   = PROJECT / "data/processed/nexus/allstore"
    PROC    = PROJECT / "data/processed/training"

    def rd(p):
        try: return pd.read_csv(p)
        except: return pd.DataFrame()

    ns = rd(NEXUS / "network_store_summary.csv")
    ns = ns.rename(columns={"critical": "critical_count",
                             "warning":  "warning_count",
                             "phantom_count": "phantom_skus",
                             "monitor":  "monitor_count"})

    stores = rd(PROJECT / "data/raw/output/csv/stores.csv")
    if not ns.empty and not stores.empty:
        for col in ["store_name","city","state","region",
                    "store_format","foot_traffic_tier"]:
            if col not in ns.columns and col in stores.columns:
                ns = ns.merge(stores[["store_id", col]],
                              on="store_id", how="left")

    if ns.empty:
        return {"status": "error",
                "message": "Network summary not found. Run analytics.py first."}

    if region and "region" in ns.columns:
        ns = ns[ns["region"].str.lower() == region.lower()]

    def gi(c): return int(ns[c].sum())   if c in ns.columns else 0
    def gf(c): return float(ns[c].sum()) if c in ns.columns else 0.0

    rar  = gf("revenue_at_risk")
    crit = gi("critical_count")
    warn = gi("warning_count")
    ph   = gi("phantom_skus")
    uto  = gi("units_to_order")

    sort_c = "urgency_score" if "urgency_score" in ns.columns else "revenue_at_risk"
    top5   = []
    for _, r in ns.nlargest(5, sort_c).iterrows():
        top5.append({
            "store_id":       r["store_id"],
            "store_name":     r.get("store_name", ""),
            "city":           r.get("city", ""),
            "region":         r.get("region", ""),
            "critical_count": int(r.get("critical_count", 0)),
            "revenue_at_risk": float(r.get("revenue_at_risk", 0)),
            "units_to_order": int(r.get("units_to_order", 0)),
            "phantom_skus":   int(r.get("phantom_skus", 0)),
            "directive": (
                f"{r.get('store_name',r['store_id'])} ({r.get('city','')}) — "
                f"order {int(r.get('units_to_order',0)):,} units across "
                f"{int(r.get('critical_count',0))} CRITICAL SKUs "
                f"to recover ${float(r.get('revenue_at_risk',0)):,.0f}."
            ),
        })

    reg_rar = {}
    reg_units = {}
    if "region" in ns.columns:
        for rr, vv in ns.groupby("region")["revenue_at_risk"].sum().sort_values(ascending=False).items():
            reg_rar[rr] = round(float(vv), 2)
        for rr, vv in ns.groupby("region")["units_to_order"].sum().sort_values(ascending=False).items():
            reg_units[rr] = int(vv)

    ms = rd(PROC / "demandSense_model_summary.csv")
    model_info = {}
    if not ms.empty:
        v2 = ms[ms["forecast_method"].str.contains("DemandSense_v2", na=False)]
        if not v2.empty:
            model_info = {
                "model":   "DemandSense v2 — PROMOTED",
                "wape":    round(float(v2.iloc[0]["wape"]), 2),
                "p90":     round(float(v2.iloc[0].get("p90_coverage", 0)), 2),
                "bias":    round(float(v2.iloc[0].get("bias", 0)), 4),
            }

    morning_directive = (
        f"MORNING BRIEFING: Action required at "
        f"{top5[0]['store_name'] if top5 else 'top stores'} "
        f"and {len(top5)-1} others — "
        f"{crit:,} CRITICAL SKUs across the network risk "
        f"${rar/1e6:.1f}M in revenue if no orders placed today."
    )

    return {
        "status":             "ok",
        "directive":          morning_directive,
        "scope":              region or "full_network",
        "stores_monitored":   len(ns),
        "revenue_at_risk_usd": round(rar, 2),
        "revenue_at_risk_m":  round(rar / 1e6, 2),
        "critical_skus":      crit,
        "warning_skus":       warn,
        "phantom_skus":       ph,
        "units_to_order":     uto,
        "top_5_urgent_stores": top5,
        "revenue_by_region":  reg_rar,
        "units_by_region":    reg_units,
        "model_performance":  model_info,
        "summary": (
            f"Network: ${rar/1e6:.1f}M at risk, {crit:,} CRITICAL, "
            f"{warn:,} WARNING, {ph} phantoms, "
            f"{uto:,} units to order across {len(ns)} stores."
        ),
    }


# ─────────────────────────────────────────────────────────────────
# TOOL 6 — Morning store briefing (all critical SKUs for one store)
# ─────────────────────────────────────────────────────────────────
def get_store_briefing(store_id: str) -> dict:
    """
    Complete morning briefing for one store:
    all CRITICAL and WARNING SKUs, order list, supplier contacts,
    expected arrival dates, and aisle placement guide.
    """
    repl = registry.repl_inputs
    s_all = repl[repl["store_id"] == store_id].copy()

    if s_all.empty:
        return {"status": "not_found",
                "message": f"No data for store {store_id}"}

    # Deduplicate — keep latest row per SKU only
    s = (s_all.sort_values("date")
         .groupby("sku_id").last().reset_index())

    # Override with network_master_alerts if available (more accurate)
    try:
        pass  # pandas already imported as pd
        _na_p = Path(__file__).parent.parent / "data/processed/nexus/allstore/network_master_alerts.csv"
        _na = pd.read_csv(_na_p)
        _na_s = _na[_na["store_id"]==store_id].copy()
        if not _na_s.empty:
            _na_s["dos"] = _na_s["days_of_supply_current"]
            _na_s["mu_daily"] = _na_s["avg_daily_demand"]
            _na_s["target_upto_new"] = _na_s["units_to_order"]
            s = _na_s
    except: pass


    store_info = _get_store_info(store_id)

    # Build alert tiers
    if "alert_tier" in s.columns:
        critical = s[s["alert_tier"]=="CRITICAL"].copy()
        warning  = s[s["alert_tier"]=="WARNING"].copy()
        monitor  = s[s["alert_tier"]=="MONITOR"].copy()
    else:
        s["dos"] = s["target_upto_new"] / s["mu_daily"].clip(lower=0.1)
        critical = s[s["dos"] <= 3].copy()
        warning  = s[(s["dos"] > 3) & (s["dos"] <= 7)].copy()
        monitor  = s[(s["dos"] > 7) & (s["dos"] <= 14)].copy()

    def build_order_list(df, tier):
        orders = []
        for _, r in df.iterrows():
            prod_info = _get_product_info(r["sku_id"])
            sup_info  = _get_supplier(r["sku_id"])
            mu        = float(r["mu_daily"])
            target    = float(r["target_upto_new"])
            pack_size = int(prod_info.get("pack_size", 1)) or 1
            order_qty = int(np.ceil(target / pack_size) * pack_size)
            unit_price = float(prod_info.get("unit_price", 0))
            lead_days = int(np.ceil(float(r.get("lead_time_days_avg",
                            sup_info["lead_time_days_avg"]))))
            arrival   = (TODAY + timedelta(days=lead_days)).strftime("%b %d")
            orders.append({
                "tier":           tier,
                "sku_id":         r["sku_id"],
                "product_name":   prod_info.get("product_name", r["sku_id"]),
                "category":       prod_info.get("category", ""),
                "aisle":          prod_info.get("aisle_location", "TBD"),
                "shelf_section":  prod_info.get("shelf_section", "TBD"),
                "order_qty":      order_qty,
                "days_of_supply": round(float(r["dos"]), 1),
                "revenue_at_risk": round(mu * 7 * unit_price, 2),
                "supplier":       sup_info["supplier_name"],
                "supplier_contact": sup_info.get("contact_email", "N/A"),
                "lead_time_days": lead_days,
                "arrival_date":   arrival,
                "order_value":    round(order_qty * unit_price, 2),
            })
        return sorted(orders, key=lambda x: -x["revenue_at_risk"])

    crit_orders = build_order_list(critical, "CRITICAL")
    warn_orders = build_order_list(warning, "WARNING")

    total_order_value = sum(o["order_value"] for o in crit_orders + warn_orders)
    total_rar         = sum(o["revenue_at_risk"] for o in crit_orders + warn_orders)

    directive = (
        f"MORNING BRIEFING for {store_info.get('store_name', store_id)} "
        f"({store_info.get('city', '')}): "
        f"Place {len(crit_orders)} CRITICAL orders today "
        f"({len(warn_orders)} WARNING) worth ${total_order_value:,.0f} "
        f"to protect ${total_rar:,.0f} in weekly revenue."
    )

    return {
        "status":             "ok",
        "directive":          directive,
        "store_id":           store_id,
        "store_name":         store_info.get("store_name", store_id),
        "city":               store_info.get("city", ""),
        "state":              store_info.get("state", ""),
        "region":             store_info.get("region", ""),
        "store_format":       store_info.get("store_format", ""),
        "as_of_date":         TODAY.strftime("%A, %B %d, %Y"),
        "critical_count":     len(critical),
        "warning_count":      len(warning),
        "monitor_count":      len(monitor),
        "total_order_value":  round(total_order_value, 2),
        "total_revenue_at_risk": round(total_rar, 2),
        "critical_orders":    crit_orders[:20],
        "warning_orders":     warn_orders[:20],
        "summary": (
            f"{store_info.get('store_name', store_id)}: "
            f"{len(crit_orders)} CRITICAL, {len(warn_orders)} WARNING. "
            f"Total orders: ${total_order_value:,.0f}. "
            f"Revenue at risk: ${total_rar:,.0f}."
        ),
    }

# ─────────────────────────────────────────────────────────────────
# TOOL 7 — What-If CSL Simulation
# ─────────────────────────────────────────────────────────────────
def simulate_csl_what_if(target_csl_pct: float, foot_traffic_tier: str = "All") -> dict:
    """
    Simulates the financial and inventory impact of changing the Service Level (CSL).
    """
    repl = registry.repl_inputs.copy()
    # Use latest row per store-SKU to avoid counting all dates
    repl = repl.sort_values('date').groupby(['store_id','sku_id']).last().reset_index()

    if foot_traffic_tier != "All":
        repl = repl[repl['foot_traffic_tier'].str.lower() == foot_traffic_tier.lower()]
        
    if repl.empty:
        return {"status": "error", "message": f"No data found for tier: {foot_traffic_tier}"}

    import scipy.stats as stats
    new_z = stats.norm.ppf(target_csl_pct / 100.0)
    
    # Current state totals
    current_ss = repl['safety_stock_units'].sum()
    
    # Simulated state totals (re-running the Nexus SS formula)
    simulated_ss = np.ceil(
        new_z * np.sqrt(
            repl["lead_time_days_avg"] * repl["sigma_daily"]**2 +
            repl["mu_daily"]**2 * repl["lead_time_days_std"]**2
        )
    ).clip(lower=0).sum()
    
    unit_diff = simulated_ss - current_ss
    pct_change = (unit_diff / current_ss * 100) if current_ss > 0 else 0
    
    directive = (
        f"Increasing {foot_traffic_tier} stores to {target_csl_pct}% CSL "
        f"would require holding {simulated_ss:,.0f} total safety stock units "
        f"({'+' if unit_diff > 0 else ''}{pct_change:.1f}% change)."
    )
    
    return {
        "status": "ok",
        "directive": directive,
        "scenario": f"{target_csl_pct}% CSL for {foot_traffic_tier} stores",
        "current_safety_stock_units": round(current_ss, 0),
        "simulated_safety_stock_units": round(simulated_ss, 0),
        "net_unit_change": round(unit_diff, 0),
        "impact": f"This policy change adds {unit_diff:,.0f} units to holding inventory across the network."
    }

# ─────────────────────────────────────────────────────────────────
# TOOL 8 — Supplier Risk Detection
# ─────────────────────────────────────────────────────────────────
def get_supplier_risk(requested_tier: str = 'HIGH_RISK') -> dict:
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / 'data/processed/nexus/supplier/supplier_scorecard.csv'
    if not p.exists():
        return {'status': 'error', 'message': 'Supplier scorecard not found.'}
    sup = pd.read_csv(p)
    req = requested_tier.upper().replace(' ','_')

    high   = sup[sup['risk_tier']=='HIGH_RISK'].sort_values('risk_score',ascending=False)
    medium = sup[sup['risk_tier']=='MEDIUM_RISK'].sort_values('risk_score',ascending=False)
    low    = sup[sup['risk_tier']=='LOW_RISK'].sort_values('avg_fulfillment_rate',ascending=False)

    tier_summary = {
        'HIGH_RISK':   len(high),
        'MEDIUM_RISK': len(medium),
        'LOW_RISK':    len(low),
        'total':       len(sup),
    }

    def make_alert(r):
        return {
            'supplier_name':    r['supplier_name'],
            'risk_tier':        r['risk_tier'],
            'risk_score':       round(float(r['risk_score']),1),
            'fill_rate':        f"{r['avg_fulfillment_rate']*100:.1f}%",
            'late_orders_pct':  f"{r['late_delivery_rate']*100:.1f}%",
            'short_delivery':   f"{r['short_delivery_rate']*100:.1f}%",
            'lt_variability':   f"{r['std_lead_actual']:.1f} days",
            'stockouts_caused': int(r['stockout_events_caused']),
            'revenue_at_risk':  float(r['total_revenue_at_risk']),
        }

    if 'LOW' in req:
        suppliers = low
        top_offenders = [make_alert(r) for _,r in suppliers.head(10).iterrows()]
        directive = (f"{len(low)} LOW_RISK suppliers performing well. "
                     f"Best: {low.iloc[0]['supplier_name']} with "
                     f"{low.iloc[0]['avg_fulfillment_rate']*100:.1f}% fill rate.") if not low.empty else 'No LOW_RISK suppliers.'
        return {
            'status': 'ok',
            'requested_tier': req,
            'directive': directive,
            'tier_summary': tier_summary,
            'top_offenders': top_offenders,
            'low_risk_suppliers': top_offenders,
            'risky_supplier_count': len(low),
        }
    elif 'MEDIUM' in req:
        suppliers = medium
        top_offenders = [make_alert(r) for _,r in suppliers.head(10).iterrows()]
        directive = f"{len(medium)} MEDIUM_RISK suppliers on the watch list."
        return {
            'status': 'ok',
            'requested_tier': req,
            'directive': directive,
            'tier_summary': tier_summary,
            'top_offenders': top_offenders,
            'low_risk_suppliers': [],
            'risky_supplier_count': len(medium),
        }
    else:
        suppliers = pd.concat([high, medium]).head(10)
        top_offenders = [make_alert(r) for _,r in suppliers.iterrows()]
        top = high.iloc[0] if not high.empty else medium.iloc[0]
        directive = (f"DUAL-SOURCE: {top['supplier_name']} — "
                     f"fill {top['avg_fulfillment_rate']*100:.1f}% · "
                     f"{top['late_delivery_rate']*100:.1f}% late · "
                     f"{int(top['stockout_events_caused'])} stockouts caused.")
        return {
            'status': 'ok',
            'requested_tier': req,
            'directive': directive,
            'tier_summary': tier_summary,
            'top_offenders': top_offenders,
            'low_risk_suppliers': [],
            'risky_supplier_count': len(high) + len(medium),
            'high_risk_count': len(high),
            'medium_risk_count': len(medium),
        }




TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_forecast",
            "description": "Get 7-day demand forecast for a specific store and SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "sku_id": {"type": "string"},
                    "days": {"type": "integer", "default": 7}
                },
                "required": ["store_id", "sku_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stockout_risk",
            "description": "Get stockout risk score for a store and SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "sku_id": {"type": "string"}
                },
                "required": ["store_id", "sku_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_replenishment_action",
            "description": "Get replenishment order recommendation for a store and SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "sku_id": {"type": "string"}
                },
                "required": ["store_id", "sku_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_phantom_alerts",
            "description": "Detect phantom inventory at a store or network-wide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"},
                    "top_n": {"type": "integer", "default": 10}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_network_status",
            "description": "Get network-wide KPIs including revenue at risk and alert counts.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_store_briefing",
            "description": "Get morning briefing with full order list for a store.",
            "parameters": {
                "type": "object",
                "properties": {
                    "store_id": {"type": "string"}
                },
                "required": ["store_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_supplier_risk",
            "description": "Get supplier performance by risk tier. requested_tier: HIGH_RISK, MEDIUM_RISK, or LOW_RISK.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requested_tier": {"type": "string", "default": "HIGH_RISK"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_csl_what_if",
            "description": "Simulate impact of changing service level target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_csl_pct": {"type": "number"},
                    "foot_traffic_tier": {"type": "string", "default": "All"}
                },
                "required": ["target_csl_pct"]
            }
        }
    }
]
TOOL_FUNCTIONS = {
    'query_forecast':           query_forecast,
    'get_stockout_risk':        get_stockout_risk,
    'get_replenishment_action': get_replenishment_action,
    'get_phantom_alerts':       get_phantom_alerts,
    'get_network_status':       get_network_status,
    'get_store_briefing':       get_store_briefing,
    'get_supplier_risk':        get_supplier_risk,
    'simulate_csl_what_if':     simulate_csl_what_if,
}
def dispatch_tool(name: str, args: dict) -> dict:
    if name not in TOOL_FUNCTIONS:
        return {"status": "error", "message": f"Unknown tool: {name}"}
    try:
        # Type coercion — LLaMA sometimes passes ints as strings
        clean = {}
        for k, v in args.items():
            if k in ("days", "top_n") and isinstance(v, str):
                try: clean[k] = int(v)
                except: clean[k] = v
            elif k in ("days", "top_n") and v is None:
                clean[k] = 7 if k == "days" else 10
            else:
                clean[k] = v
        return TOOL_FUNCTIONS[name](**clean)
    except Exception as e:
        return {"status": "error", "message": str(e), "tool": name}