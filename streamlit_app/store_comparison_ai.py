"""
store_comparison_ai.py — Store vs Store root cause intelligence
Python computes exact comparison, LLaMA explains why
"""
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

C_TEAL="#0D9488";C_RED="#EF4444";C_AMBER="#F59E0B";C_GREEN="#10B981"
C_GRAY="#94A3B8";C_CARD="#1E3352";C_WHITE="#E2E8F0";C_BORDER="#2D4A6A"
C_PURPLE="#8B5CF6"

def DL(title="", height=280):
    return dict(
        title=dict(text=title, font=dict(color=C_WHITE, size=12)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_WHITE, size=10), height=height,
        margin=dict(l=40, r=20, t=36 if title else 16, b=36),
        xaxis=dict(gridcolor=C_BORDER, color=C_GRAY),
        yaxis=dict(gridcolor=C_BORDER, color=C_GRAY),
        hoverlabel=dict(bgcolor=C_CARD, bordercolor=C_BORDER,
                        font=dict(color=C_WHITE, size=10)),
    )

def _llm(prompt):
    try:
        r = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model":"llama3.2:3b","prompt":prompt,"stream":False,
                  "options":{"temperature":0.1,"num_predict":200}},
            timeout=25)
        return r.json().get("response","").strip() if r.status_code==200 else None
    except:
        return None

def _store_summary(na, ns, store_id):
    """Get summary dict for a store."""
    alerts = na[na["store_id"]==store_id].copy() if not na.empty else pd.DataFrame()
    ns_row = ns[ns["store_id"]==store_id].iloc[0] if not ns.empty and store_id in ns["store_id"].values else pd.Series()

    crit  = int((alerts["alert_tier"]=="CRITICAL").sum()) if not alerts.empty and "alert_tier" in alerts.columns else 0
    warn  = int((alerts["alert_tier"]=="WARNING").sum()) if not alerts.empty and "alert_tier" in alerts.columns else 0
    rar   = float(alerts["revenue_at_risk"].sum()) if not alerts.empty and "revenue_at_risk" in alerts.columns else 0
    ph    = int(alerts["is_phantom"].sum()) if not alerts.empty and "is_phantom" in alerts.columns else 0
    uto   = float(alerts["units_to_order"].sum()) if not alerts.empty and "units_to_order" in alerts.columns else 0
    dos   = float(alerts["days_of_supply_current"].mean()) if not alerts.empty and "days_of_supply_current" in alerts.columns else 0

    # Priority factors avg
    f1 = float(alerts["f1_tier"].mean()) if not alerts.empty and "f1_tier" in alerts.columns else 0
    f2 = float(alerts["f2_revenue"].mean()) if not alerts.empty and "f2_revenue" in alerts.columns else 0
    f3 = float(alerts["f3_dos"].mean()) if not alerts.empty and "f3_dos" in alerts.columns else 0
    f4 = float(alerts["f4_supplier"].mean()) if not alerts.empty and "f4_supplier" in alerts.columns else 0
    f5 = float(alerts["f5_seasonal"].mean()) if not alerts.empty and "f5_seasonal" in alerts.columns else 0

    # Worst supplier
    sup_worst = ""
    if not alerts.empty and "supplier_name" in alerts.columns and "revenue_at_risk" in alerts.columns:
        sg = alerts.groupby("supplier_name")["revenue_at_risk"].sum()
        sup_worst = sg.idxmax() if not sg.empty else ""

    urgency = float(ns_row.get("urgency_score",0)) if hasattr(ns_row,"get") else 0

    return {
        "store_id": store_id,
        "critical": crit, "warning": warn,
        "revenue_at_risk": rar, "phantom_skus": ph,
        "units_to_order": uto, "avg_dos": dos,
        "f1_tier": f1, "f2_revenue": f2, "f3_dos": f3,
        "f4_supplier": f4, "f5_seasonal": f5,
        "worst_supplier": sup_worst,
        "urgency_score": urgency,
        "alerts": alerts
    }

def answer(question, na, ns, sc, store_a, store_b, store_c=None):
    if na is None or na.empty:
        return "No alert data available.", None

    q = question.lower()
    NL = "\n"

    sa = _store_summary(na, ns, store_a)
    sb = _store_summary(na, ns, store_b)
    sc_s = _store_summary(na, ns, store_c) if store_c else None

    # ── Why is A worse than B ─────────────────────────────────
    if any(x in q for x in ["why","worse","root cause","reason","different","gap"]):
        worse = sa if sa["revenue_at_risk"] > sb["revenue_at_risk"] else sb
        better = sb if worse["store_id"]==store_a else sa

        # Root cause breakdown using f-scores
        factors = []
        if worse["f1_tier"] > better["f1_tier"]:
            factors.append(f"Alert tier severity (F1): {worse['store_id']} avg {worse['f1_tier']:.0f} vs {better['store_id']} {better['f1_tier']:.0f} — more CRITICAL tier products")
        if worse["f2_revenue"] > better["f2_revenue"]:
            factors.append(f"Revenue at risk per SKU (F2): {worse['store_id']} ${worse['f2_revenue']:.2f} vs {better['store_id']} ${better['f2_revenue']:.2f} — higher-value products at risk")
        if worse["f3_dos"] > better["f3_dos"]:
            factors.append(f"Days of supply urgency (F3): {worse['store_id']} score {worse['f3_dos']:.2f} vs {better['store_id']} {better['f3_dos']:.2f} — running out faster")
        if worse["f4_supplier"] > better["f4_supplier"]:
            factors.append(f"Supplier risk (F4): {worse['store_id']} score {worse['f4_supplier']:.2f} vs {better['store_id']} {better['f4_supplier']:.2f} — supplier reliability gap")
        if worse["phantom_skus"] > better["phantom_skus"]:
            factors.append(f"Phantom inventory: {worse['store_id']} {worse['phantom_skus']} ghost SKUs vs {better['store_id']} {better['phantom_skus']} — inventory accuracy gap")

        # Supplier 4-factor comparison
        sup_lines = []
        if not sc.empty and "supplier_name" in sc.columns and worse["worst_supplier"]:
            ws = sc[sc["supplier_name"]==worse["worst_supplier"]]
            if not ws.empty:
                r = ws.iloc[0]
                sup_lines.append(
                    f"  {r['supplier_name']} (worst supplier at {worse['store_id']}): "
                    f"Fill {float(r.get('avg_fulfillment_rate',0))*100:.1f}% · "
                    f"Late {float(r.get('late_delivery_rate',0))*100:.1f}% · "
                    f"Short {float(r.get('short_delivery_rate',0))*100:.1f}% · "
                    f"Risk: {r.get('risk_tier','?')}")

        # Chart — 5-factor radar comparison
        cats = ["F1 Tier", "F2 Revenue", "F3 DOS", "F4 Supplier", "F5 Seasonal"]
        vals_a = [sa["f1_tier"], sa["f2_revenue"], sa["f3_dos"], sa["f4_supplier"], abs(sa["f5_seasonal"])]
        vals_b = [sb["f1_tier"], sb["f2_revenue"], sb["f3_dos"], sb["f4_supplier"], abs(sb["f5_seasonal"])]
        # Normalize to 0-100
        maxv = [max(a,b,0.01) for a,b in zip(vals_a,vals_b)]
        norm_a = [v/m*100 for v,m in zip(vals_a,maxv)]
        norm_b = [v/m*100 for v,m in zip(vals_b,maxv)]

        fig = go.Figure()
        fig.add_trace(go.Bar(name=store_a, x=cats, y=norm_a,
            marker_color=C_TEAL,
            text=[f"{v:.0f}" for v in norm_a],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.add_trace(go.Bar(name=store_b, x=cats, y=norm_b,
            marker_color=C_AMBER,
            text=[f"{v:.0f}" for v in norm_b],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("5-Factor Priority Score Comparison (higher = more urgent)", height=280),
            barmode="group")

        facts = (
            f"{worse['store_id']} is worse: ${worse['revenue_at_risk']:,.0f} at risk, "
            f"{worse['critical']} CRITICAL, {worse['phantom_skus']} phantoms\n"
            f"{better['store_id']} is better: ${better['revenue_at_risk']:,.0f} at risk, "
            f"{better['critical']} CRITICAL, {better['phantom_skus']} phantoms\n\n"
            "Root causes:\n" + NL.join(factors) +
            ("\n\nWorst supplier detail:\n" + NL.join(sup_lines) if sup_lines else "")
        )
        prompt = (
            "You are a retail store operations expert.\n"
            "EXACT ROOT CAUSE DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: '{worse['store_id']} has ${worse['revenue_at_risk']:,.0f} at risk vs "
            f"{better['store_id']} ${better['revenue_at_risk']:,.0f}.'\n"
            "Explain the specific root causes — which factor is biggest (F1=alert tier, "
            "F2=revenue per SKU, F3=days of supply, F4=supplier risk, F5=seasonal). "
            "Name the worst supplier if present. Be specific and decisive."
        )
        return _llm(prompt) or facts, fig

    # ── Which store to visit first ────────────────────────────
    if any(x in q for x in ["visit","priority","which first","most urgent","go to"]):
        stores = [sa, sb]
        if sc_s:
            stores.append(sc_s)
        stores_sorted = sorted(stores, key=lambda x: x["revenue_at_risk"], reverse=True)

        labels = [s["store_id"] for s in stores_sorted]
        revs   = [s["revenue_at_risk"] for s in stores_sorted]
        crits  = [s["critical"] for s in stores_sorted]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Revenue at Risk", x=labels, y=[r/1e3 for r in revs],
            marker_color=C_RED,
            text=[f"${r/1e3:.0f}K" for r in revs],
            textposition="outside", textfont=dict(color=C_WHITE,size=10)))
        fig.add_trace(go.Bar(name="Critical SKUs", x=labels, y=crits,
            marker_color=C_AMBER,
            text=crits, textposition="outside",
            textfont=dict(color=C_WHITE,size=10)))
        fig.update_layout(**DL("Store priority — which to visit first", height=260),
            barmode="group")

        top = stores_sorted[0]
        facts = (
            "Store priority ranking:\n" +
            NL.join([f"  {i+1}. {s['store_id']}: ${s['revenue_at_risk']:,.0f} at risk · "
                     f"{s['critical']} CRITICAL · {s['phantom_skus']} phantoms · "
                     f"urgency {s['urgency_score']:.0f}"
                     for i,s in enumerate(stores_sorted)])
        )
        prompt = (
            "You are a retail district manager.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: 'Visit {top['store_id']} first — ${top['revenue_at_risk']:,.0f} at risk, "
            f"{top['critical']} CRITICAL SKUs.'\n"
            "Explain what to focus on when you get there. Name the worst supplier to call on-site. "
            "Be action-oriented."
        )
        return _llm(prompt) or facts, fig

    # ── 3-store comparison ────────────────────────────────────
    if any(x in q for x in ["3 store","three store","compare all","side by side","all three"]):
        stores = [sa, sb]
        if sc_s:
            stores.append(sc_s)

        metrics = ["critical","warning","revenue_at_risk","phantom_skus","units_to_order","avg_dos"]
        labels_m = ["Critical","Warning","Rev at Risk","Phantoms","Units to Order","Avg DOS"]

        fig = go.Figure()
        colors = [C_TEAL, C_AMBER, C_GREEN]
        for i,s in enumerate(stores):
            norm_vals = [
                s["critical"], s["warning"],
                s["revenue_at_risk"]/1000,  # in $K
                s["phantom_skus"], s["units_to_order"]/100,
                s["avg_dos"]
            ]
            fig.add_trace(go.Bar(
                name=s["store_id"], x=labels_m, y=norm_vals,
                marker_color=colors[i],
                text=[f"{v:.0f}" for v in norm_vals],
                textposition="outside", textfont=dict(color=C_WHITE,size=8)))
        fig.update_layout(**DL("3-Store comparison — key metrics", height=280),
            barmode="group")

        facts = "3-store comparison:\n" + NL.join([
            f"  {s['store_id']}: ${s['revenue_at_risk']:,.0f} at risk · "
            f"{s['critical']} critical · {s['phantom_skus']} phantoms · "
            f"{s['avg_dos']:.1f} avg days supply"
            for s in stores])
        prompt = (
            "You are a retail district manager.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            "Rank the stores by urgency, name the most critical issue in each. "
            "Give one specific action per store. Use exact numbers."
        )
        return _llm(prompt) or facts, fig

    # ── Supplier comparison ───────────────────────────────────
    if any(x in q for x in ["supplier","vendor","fill rate","reliability"]):
        results = []
        for s_data in [sa, sb]:
            alerts_s = s_data["alerts"]
            if not alerts_s.empty and "supplier_name" in alerts_s.columns:
                sg = alerts_s.groupby("supplier_name").agg(
                    critical=("alert_tier", lambda x: (x=="CRITICAL").sum()),
                    revenue=("revenue_at_risk","sum"),
                    f4=("f4_supplier","mean")
                ).reset_index().nlargest(3,"revenue")
                for _,r in sg.iterrows():
                    sup_row = sc[sc["supplier_name"]==r["supplier_name"]].iloc[0] if not sc.empty and r["supplier_name"] in sc["supplier_name"].values else pd.Series()
                    results.append({
                        "store": s_data["store_id"],
                        "supplier": r["supplier_name"],
                        "critical": int(r["critical"]),
                        "revenue": float(r["revenue"]),
                        "fill_rate": float(sup_row.get("avg_fulfillment_rate",0))*100 if hasattr(sup_row,"get") else 0,
                        "late_rate": float(sup_row.get("late_delivery_rate",0))*100 if hasattr(sup_row,"get") else 0,
                        "short_rate": float(sup_row.get("short_delivery_rate",0))*100 if hasattr(sup_row,"get") else 0,
                        "risk_tier": str(sup_row.get("risk_tier","?")) if hasattr(sup_row,"get") else "?",
                    })

        if not results:
            return "No supplier data available for comparison.", None

        df_r = pd.DataFrame(results)
        fig = px.scatter(df_r, x="fill_rate", y="revenue",
            color="store", size="critical",
            hover_data=["supplier","late_rate","short_rate","risk_tier"],
            color_discrete_map={store_a:C_TEAL, store_b:C_AMBER},
            labels={"fill_rate":"Fill Rate %","revenue":"Revenue at Risk ($)"})
        fig.update_layout(**DL(f"Supplier fill rate vs revenue at risk: {store_a} vs {store_b}", height=280))
        fig.add_vline(x=85, line_dash="dot", line_color=C_AMBER,
            annotation_text="85% minimum", annotation_font=dict(color=C_AMBER,size=9))

        facts_lines = [
            f"  {r['store']} · {r['supplier']}: Fill {r['fill_rate']:.1f}% · "
            f"Late {r['late_rate']:.1f}% · Short {r['short_rate']:.1f}% · "
            f"{r['critical']} critical SKUs · ${r['revenue']:,.0f} at risk · Risk: {r['risk_tier']}"
            for r in results]
        facts = f"Supplier comparison {store_a} vs {store_b}:\n" + NL.join(facts_lines)
        prompt = (
            "You are a retail procurement analyst.\n"
            "EXACT SUPPLIER DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            "Name the worst-performing supplier across both stores, its fill rate, and which store "
            "it impacts most. Recommend whether to dual-source or escalate. Use exact numbers."
        )
        return _llm(prompt) or facts, fig

    # ── Fallback ──────────────────────────────────────────────
    facts = (
        f"{store_a}: ${sa['revenue_at_risk']:,.0f} at risk · {sa['critical']} critical · {sa['phantom_skus']} phantoms\n"
        f"{store_b}: ${sb['revenue_at_risk']:,.0f} at risk · {sb['critical']} critical · {sb['phantom_skus']} phantoms"
    )
    ans = _llm(
        "You are a retail store operations expert.\n"
        "Store comparison data:\n" + facts + "\n"
        "Answer in 3 sentences. Stay in retail store operations scope.\n"
        "Question: " + question
    )
    return ans or "Try: why is A worse · which store to visit first · 3-store comparison · supplier comparison", None