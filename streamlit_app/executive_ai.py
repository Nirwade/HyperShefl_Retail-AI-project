"""
executive_ai.py — True hybrid executive intelligence
Python computes network KPIs, LLaMA explains strategic context
"""
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

C_TEAL="#0D9488";C_TEAL2="#14B8A6";C_RED="#EF4444"
C_AMBER="#F59E0B";C_GREEN="#10B981";C_GRAY="#94A3B8"
C_CARD="#1E3352";C_BG="#0D1B2A";C_WHITE="#E2E8F0"
C_BORDER="#2D4A6A";C_PURPLE="#8B5CF6"

def DL(title="", height=300):
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
                  "options":{"temperature":0.15,"num_predict":200}},
            timeout=25)
        return r.json().get("response","").strip() if r.status_code==200 else None
    except:
        return None

def answer(question, ns, roi_sum):
    if ns is None or ns.empty:
        return "No network data available.", None

    q = question.lower()
    NL = "\n"

    # ── Worst region ──────────────────────────────────────────
    if any(x in q for x in ["worst region","region","southwest","west","northeast","southeast","on fire","burning"]):
        if "region" not in ns.columns:
            return "Region data not available.", None
        rg = ns.groupby("region").agg(
            stores=("store_id","count"),
            critical=("critical_count","sum"),
            warning=("warning_count","sum"),
            revenue=("revenue_at_risk","sum"),
            phantoms=("phantom_skus","sum"),
            units=("units_to_order","sum")
        ).reset_index().sort_values("revenue", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rg["revenue"]/1e6, y=rg["region"], orientation="h",
            name="Revenue at Risk ($M)",
            marker=dict(color=rg["revenue"],
                colorscale=[[0,C_AMBER],[1,C_RED]], showscale=False),
            text=[f"${v:.1f}M" for v in rg["revenue"]/1e6],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Revenue at Risk by Region ($M)", height=260))

        worst = rg.iloc[0]
        best  = rg.iloc[-1]
        facts_str = (
            f"WORST: {worst['region']} — ${worst['revenue']/1e6:.1f}M at risk · "
            f"{int(worst['critical']):,} critical SKUs · {int(worst['stores'])} stores\n"
            f"BEST:  {best['region']} — ${best['revenue']/1e6:.1f}M at risk · "
            f"{int(best['critical']):,} critical SKUs\n\n"
            "All regions:\n" +
            NL.join([f"  {r['region']}: ${r['revenue']/1e6:.1f}M · {int(r['critical']):,} critical · {int(r['stores'])} stores"
                     for _,r in rg.iterrows()])
        )
        prompt = (
            "You are a senior retail network operations director.\n"
            "Regional performance:\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            f"Explain why {worst['region']} is the worst region today, what the numbers mean "
            "for store operations, and what immediate action the regional director should take. "
            "Be specific and decisive."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Worst store format ────────────────────────────────────
    if any(x in q for x in ["format","store format","superstore","convenience","wholesale","hypermarket"]):
        if "store_format" not in ns.columns:
            return "Store format data not available.", None
        fmt = ns.groupby("store_format").agg(
            stores=("store_id","count"),
            critical=("critical_count","sum"),
            revenue=("revenue_at_risk","sum"),
            pct_total=("revenue_at_risk",lambda x: x.sum())
        ).reset_index().sort_values("revenue", ascending=False)
        total_rev = ns["revenue_at_risk"].sum()
        fmt["pct"] = fmt["revenue"]/total_rev*100

        fig = px.pie(fmt, values="revenue", names="store_format",
            color_discrete_sequence=[C_RED,C_AMBER,C_PURPLE,C_TEAL,C_GREEN],
            hole=0.5)
        fig.update_traces(textinfo="label+percent",
            textfont=dict(color=C_WHITE, size=10))
        fig.update_layout(**DL("Revenue at Risk % by Store Format", height=260),
            showlegend=False)

        worst_fmt = fmt.iloc[0]
        facts_str = (
            f"Revenue at risk by store format:\n" +
            NL.join([f"  {r['store_format']}: ${r['revenue']/1e6:.1f}M "
                     f"({r['pct']:.1f}% of total) · {int(r['critical']):,} critical · {int(r['stores'])} stores"
                     for _,r in fmt.iterrows()])
        )
        prompt = (
            "You are a senior retail operations director.\n" + facts_str + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings. "
            f"Explain why {worst_fmt['store_format']} stores carry the most risk, "
            "what operational factors drive this, and what to prioritize. Be specific."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Premium stores at risk ────────────────────────────────
    if any(x in q for x in ["premium","high traffic","foot traffic","vip","flagship"]):
        if "foot_traffic_tier" not in ns.columns:
            return "Traffic tier data not available.", None
        prem = ns[ns["foot_traffic_tier"]=="Premium"].sort_values(
            "revenue_at_risk", ascending=False)
        if prem.empty:
            return "No Premium stores found.", None

        top10 = prem.head(10)
        fig = go.Figure(go.Bar(
            x=top10["revenue_at_risk"]/1e3,
            y=top10["store_id"] + " " + top10.get("store_name","").fillna(""),
            orientation="h",
            marker=dict(color=top10["revenue_at_risk"],
                colorscale=[[0,C_AMBER],[1,C_RED]], showscale=False),
            text=[f"${v/1e3:.0f}K" for v in top10["revenue_at_risk"]],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Top 10 Premium Stores by Revenue at Risk", height=280))
        fig.update_xaxes(title="Revenue at Risk ($K)")

        facts_str = (
            f"Premium store network: {len(prem)} stores, "
            f"${prem['revenue_at_risk'].sum()/1e6:.1f}M total at risk, "
            f"{int(prem['critical_count'].sum()):,} critical SKUs\n\n"
            "Top 5 most urgent Premium stores:\n" +
            NL.join([f"  {r['store_id']} {r.get('store_name','')} ({r.get('city','')}): "
                     f"${r['revenue_at_risk']:,.0f} · {int(r['critical_count'])} critical"
                     for _,r in top10.head(5).iterrows()])
        )
        prompt = (
            "You are a retail VP of Operations.\n" + facts_str + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings. "
            "Premium stores generate highest revenue per sqft — explain why "
            "stockouts here are more damaging than regular stores, "
            "name the most urgent Premium store, and give immediate action. Be urgent."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Recovery if order now ─────────────────────────────────
    if any(x in q for x in ["recover","order now","if i order","action today","revenue recover","roi"]):
        total_rar = float(ns["revenue_at_risk"].sum())
        critical_rar = float(ns[ns["critical_count"]>0]["revenue_at_risk"].sum()) if "critical_count" in ns.columns else total_rar*0.6
        roi_rec = 0
        if roi_sum is not None and not roi_sum.empty:
            roi_rec = float(roi_sum.get("revenue_recovered", [0]).iloc[0] if hasattr(roi_sum.get("revenue_recovered",[0]),"iloc") else 0)
            pct_rec = float(roi_sum.get("pct_recovered",[0]).iloc[0] if hasattr(roi_sum.get("pct_recovered",[0]),"iloc") else 0)
            events_prev = int(roi_sum.get("events_prevented",[0]).iloc[0] if hasattr(roi_sum.get("events_prevented",[0]),"iloc") else 0)
        else:
            pct_rec = 0; events_prev = 0

        # Simulate scenarios
        s80 = total_rar * 0.80
        s60 = total_rar * 0.60
        s40 = total_rar * 0.40

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Act on 80%\nof critical","Act on 60%\nof critical","Act on 40%\nof critical"],
            y=[s80/1e6, s60/1e6, s40/1e6],
            marker_color=[C_GREEN, C_TEAL, C_AMBER],
            text=[f"${v/1e6:.1f}M" for v in [s80,s60,s40]],
            textposition="outside", textfont=dict(color=C_WHITE,size=11)))
        fig.update_layout(**DL("Revenue recoverable if you order today ($M)", height=240))
        fig.update_yaxes(title="$M recoverable")

        facts_str = (
            f"Total network revenue at risk today: ${total_rar/1e6:.1f}M\n"
            f"If 80% of critical orders actioned today: ${s80/1e6:.1f}M recoverable\n"
            f"If 60% actioned: ${s60/1e6:.1f}M recoverable\n"
            f"Historical ROI: ${roi_rec/1e6:.1f}M recovered ({pct_rec:.1f}% of total lost), "
            f"{events_prev:,} stockout events prevented by HyperShelf policy"
        )
        prompt = (
            "You are a retail CFO summarizing inventory ROI.\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Explain what ordering today means in dollar terms, "
            "use the specific recovery amounts, and frame the urgency of acting now vs waiting. "
            "Sound like a CFO presenting to the board."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Store improved most ───────────────────────────────────
    if any(x in q for x in ["improved","best performing","improved most","getting better","low risk"]):
        if "urgency_score" not in ns.columns:
            return "Urgency score data not available.", None
        best = ns.nsmallest(10,"urgency_score")
        fig = go.Figure(go.Bar(
            x=best["urgency_score"], y=best["store_id"],
            orientation="h",
            marker_color=C_GREEN,
            text=[f"{v:,.0f}" for v in best["urgency_score"]],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Lowest urgency score stores — best performing", height=280))

        facts_str = (
            "Best performing stores (lowest urgency score):\n" +
            NL.join([f"  {r['store_id']} {r.get('store_name','')} — "
                     f"urgency {r['urgency_score']:.0f} · {int(r['critical_count'])} critical · "
                     f"${r['revenue_at_risk']:,.0f} at risk"
                     for _,r in best.head(5).iterrows()])
        )
        prompt = (
            "You are a retail network operations expert.\n" + facts_str + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings. "
            "Explain what low urgency score means operationally, "
            "name the best performing store, and suggest what practices they use "
            "that other stores should adopt."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── General network KPIs ──────────────────────────────────
    if any(x in q for x in ["kpi","network","overview","summary","today","dashboard","status"]):
        total_rar = float(ns["revenue_at_risk"].sum())
        total_crit = int(ns["critical_count"].sum())
        total_warn = int(ns["warning_count"].sum())
        total_ph   = int(ns["phantom_skus"].sum())
        total_uto  = int(ns["units_to_order"].sum())
        worst_reg  = ns.groupby("region")["revenue_at_risk"].sum().idxmax() if "region" in ns.columns else "Unknown"
        worst_rev  = ns.groupby("region")["revenue_at_risk"].sum().max() if "region" in ns.columns else 0

        facts_str = (
            f"Network KPIs — {len(ns)} stores:\n"
            f"  Revenue at risk: ${total_rar/1e6:.1f}M\n"
            f"  Critical SKUs: {total_crit:,} (order immediately)\n"
            f"  Warning SKUs: {total_warn:,} (order today)\n"
            f"  Units to order: {total_uto:,}\n"
            f"  Phantom SKUs: {total_ph:,}\n"
            f"  Worst region: {worst_reg} at ${worst_rev/1e6:.1f}M at risk"
        )
        prompt = (
            "You are a retail network operations director giving a morning briefing.\n" +
            facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Summarize the network health, highlight the most urgent issue, "
            "and give the #1 action to take today. Sound like an executive briefing."
        )
        text = _llm(prompt) or facts_str
        return text, None

    # ── Fallback ──────────────────────────────────────────────
    total_rar = float(ns["revenue_at_risk"].sum())
    total_crit = int(ns["critical_count"].sum())
    ctx = (f"Network: {len(ns)} stores, ${total_rar/1e6:.1f}M at risk, "
           f"{total_crit:,} critical SKUs.")
    ans = _llm(
        "You are HyperShelf AI, a retail network operations expert.\n"
        "Context: " + ctx + "\n"
        "Answer in 3 sentences. Stay in retail operations scope.\n"
        "Question: " + question
    )
    return ans or "Try: worst region · store format risk · Premium stores · recover if order now · network KPIs", None