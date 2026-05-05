"""
supplier_ai.py — True hybrid supplier intelligence
Python fetches scorecard data, LLaMA explains strategy
"""
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

C_TEAL="#0D9488";C_TEAL2="#14B8A6";C_RED="#EF4444"
C_AMBER="#F59E0B";C_GREEN="#10B981";C_GRAY="#94A3B8"
C_CARD="#1E3352";C_BG="#0D1B2A";C_WHITE="#E2E8F0";C_BORDER="#2D4A6A"
C_PURPLE="#8B5CF6"

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

def _get_col(sc, *candidates):
    for c in candidates:
        if c in sc.columns:
            return c
    return None

def answer(question, sc, selected_supplier=None):
    """
    sc: supplier scorecard dataframe
    selected_supplier: supplier_name from dropdown (optional)
    Returns (text, fig)
    """
    if sc is None or sc.empty:
        return "No supplier scorecard data available.", None

    q = question.lower()
    NL = "\n"

    fill_col  = _get_col(sc, "avg_fulfillment_rate", "avg_fill_rate")
    late_col  = _get_col(sc, "late_delivery_rate")
    short_col = _get_col(sc, "short_delivery_rate")
    stock_col = _get_col(sc, "stockout_events_caused", "total_stockout_events")
    risk_col  = _get_col(sc, "risk_tier")
    score_col = _get_col(sc, "risk_score", "reliability_score")
    rev_col   = _get_col(sc, "total_revenue_at_risk", "stockout_revenue_lost")
    name_col  = _get_col(sc, "supplier_name")
    lead_col  = _get_col(sc, "avg_lead_actual", "lead_time_days_avg")

    # ── Specific supplier deep dive ───────────────────────────
    if selected_supplier and selected_supplier != "All suppliers":
        sup = sc[sc[name_col] == selected_supplier].iloc[0] if name_col else sc.iloc[0]

        metrics = {}
        if fill_col:  metrics["Fill Rate"] = f"{float(sup[fill_col])*100:.1f}%"
        if late_col:  metrics["Late Delivery"] = f"{float(sup[late_col])*100:.1f}%"
        if short_col: metrics["Short Delivery"] = f"{float(sup[short_col])*100:.1f}%"
        if stock_col: metrics["Stockout Events"] = f"{int(sup[stock_col]):,}"
        if rev_col:   metrics["Revenue at Risk"] = f"${float(sup[rev_col]):,.0f}"
        if score_col: metrics["Risk Score"] = f"{float(sup[score_col]):.1f}"
        if lead_col:  metrics["Avg Lead Time"] = f"{float(sup[lead_col]):.1f} days"
        if risk_col:  metrics["Risk Tier"] = str(sup[risk_col])

        tier = str(sup.get(risk_col,"")) if risk_col else "UNKNOWN"
        tier_color = C_RED if "HIGH" in tier else (C_AMBER if "MEDIUM" in tier else C_GREEN)

        facts = NL.join([f"  {k}: {v}" for k,v in metrics.items()])

        # Radar / bar chart for 4-factor scorecard
        if fill_col and late_col and short_col and stock_col:
            max_stock = sc[stock_col].max() if stock_col else 1
            fill_pct   = float(sup[fill_col]) * 100
            late_pct   = float(sup[late_col]) * 100
            short_pct  = float(sup[short_col]) * 100
            stock_norm = float(sup[stock_col]) / max(max_stock, 1) * 100

            cats = ["Fill Rate", "On-Time %", "Full Delivery %", "Stockout Score (inverted)"]
            # Higher = better: fill rate and on-time and full delivery
            # Lower = better: stockout events (invert it)
            vals = [fill_pct, 100-late_pct, 100-short_pct, 100-stock_norm]
            colors_bar = [C_GREEN if v>=80 else (C_AMBER if v>=60 else C_RED) for v in vals]

            fig = go.Figure(go.Bar(
                x=cats, y=vals,
                marker_color=colors_bar,
                text=[f"{v:.1f}" for v in vals],
                textposition="outside",
                textfont=dict(color=C_WHITE, size=10)))
            fig.add_hline(y=80, line_dash="dot", line_color=C_GREEN,
                annotation_text="Target 80+", annotation_font=dict(color=C_GREEN, size=9))
            fig.update_layout(**DL(f"{selected_supplier} — 4-Factor Scorecard", height=280))
            fig.update_yaxes(range=[0,110])
        else:
            fig = None

        prompt = (
            f"You are a senior retail procurement analyst.\n"
            f"Supplier: {selected_supplier} | Risk Tier: {tier}\n"
            f"4-Factor Scorecard:\n{facts}\n\n"
            f"Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Assess this supplier's performance, identify their biggest weakness, "
            f"and recommend whether to keep, monitor, or replace them. "
            f"If HIGH RISK, recommend dual-sourcing. Be specific with the numbers."
        )
        text = _llm(prompt) or facts
        return text, fig

    # ── Dual source recommendation ────────────────────────────
    if any(x in q for x in ["dual","dual source","backup","alternative","replace","switch"]):
        if risk_col and name_col:
            high = sc[sc[risk_col].str.contains("HIGH", na=False)].copy()
            low  = sc[sc[risk_col].str.contains("LOW",  na=False)].copy()
        else:
            high = sc.nlargest(3, score_col) if score_col else sc.head(3)
            low  = sc.nsmallest(3, score_col) if score_col else sc.tail(3)

        if high.empty:
            return "No HIGH RISK suppliers found.", None

        high_names = high[name_col].tolist() if name_col else []
        low_names  = low[name_col].tolist()[:3] if name_col else []

        high_facts = []
        for _, r in high.iterrows():
            nm = r[name_col] if name_col else "Unknown"
            fr = f"{float(r[fill_col])*100:.1f}%" if fill_col else "?"
            st = f"{int(r[stock_col]):,}" if stock_col else "?"
            rv = f"${float(r[rev_col]):,.0f}" if rev_col else "?"
            high_facts.append(f"  {nm}: Fill Rate {fr} · Stockouts {st} · Revenue at risk {rv}")

        best_alt = low[name_col].iloc[0] if name_col and not low.empty else "a LOW RISK supplier"
        best_fr  = f"{float(low.iloc[0][fill_col])*100:.1f}%" if fill_col and not low.empty else "high"

        # Chart: HIGH RISK vs best LOW RISK side by side
        if fill_col and name_col and not low.empty:
            compare_sups = list(high[name_col]) + [low[name_col].iloc[0]]
            compare_vals = ([float(r[fill_col])*100 for _,r in high.iterrows()] +
                           [float(low.iloc[0][fill_col])*100])
            compare_tiers= (["HIGH RISK"]*len(high) + ["LOW RISK (recommended)"])
            color_map = {"HIGH RISK": C_RED, "LOW RISK (recommended)": C_GREEN}

            fig = px.bar(
                x=compare_sups, y=compare_vals,
                color=compare_tiers,
                color_discrete_map=color_map,
                labels={"x":"Supplier","y":"Fill Rate %","color":"Tier"})
            fig.add_hline(y=85, line_dash="dot", line_color=C_AMBER,
                annotation_text="85% minimum target",
                annotation_font=dict(color=C_AMBER, size=9))
            fig.update_layout(**DL("HIGH RISK vs best alternative fill rate", height=280))
            fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside",
                textfont=dict(color=C_WHITE))
        else:
            fig = None

        facts_str = (
            f"HIGH RISK suppliers needing dual-sourcing:\n" + NL.join(high_facts) + NL +
            f"Best alternative: {best_alt} — Fill Rate {best_fr} (LOW RISK)"
        )
        prompt = (
            "You are a senior retail procurement strategist.\n"
            "Dual-source analysis:\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Name each HIGH RISK supplier, explain why they need a backup, "
            f"and recommend {best_alt} as the alternative with specific reasoning. "
            "Be decisive and quantify the risk."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Most stockouts ────────────────────────────────────────
    if any(x in q for x in ["stockout","causing","most stockout","worst","most damage"]):
        if not stock_col or not name_col:
            return "Stockout data not available.", None

        top = sc.nlargest(10, stock_col)
        facts_lines = []
        for _, r in top.iterrows():
            nm = r[name_col]
            st = int(r[stock_col])
            rv = f"${float(r[rev_col]):,.0f}" if rev_col else ""
            fr = f"{float(r[fill_col])*100:.1f}%" if fill_col else ""
            facts_lines.append(f"  {nm}: {st:,} stockouts · {rv} revenue at risk · Fill Rate {fr}")

        fig = go.Figure(go.Bar(
            x=top[stock_col], y=top[name_col], orientation="h",
            marker=dict(color=top[stock_col],
                colorscale=[[0,C_AMBER],[1,C_RED]], showscale=False),
            text=[f"{int(v):,}" for v in top[stock_col]],
            textposition="outside", textfont=dict(color=C_WHITE, size=9)))
        fig.update_layout(**DL("Suppliers causing most stockout events", height=300))

        facts_str = "Suppliers causing most stockouts:\n" + NL.join(facts_lines)
        prompt = (
            "You are a senior retail procurement analyst.\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Name the worst supplier, quantify the damage in stockouts and revenue, "
            "and recommend immediate action. Sound urgent and specific."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── HIGH RISK overview ────────────────────────────────────
    if any(x in q for x in ["high risk","high_risk","worst supplier","failing","critical supplier"]):
        if risk_col and name_col:
            high = sc[sc[risk_col].str.contains("HIGH", na=False)]
        else:
            high = sc.nlargest(3, score_col) if score_col else sc.head(3)

        if high.empty:
            return "No HIGH RISK suppliers detected.", None

        facts_lines = []
        for _, r in high.iterrows():
            nm = r[name_col] if name_col else "Unknown"
            fr = f"{float(r[fill_col])*100:.1f}%" if fill_col else "?"
            lt = f"{float(r[late_col])*100:.1f}%" if late_col else "?"
            sh = f"{float(r[short_col])*100:.1f}%" if short_col else "?"
            st = f"{int(r[stock_col]):,}" if stock_col else "?"
            rv = f"${float(r[rev_col]):,.0f}" if rev_col else ""
            facts_lines.append(
                f"  {nm}: Fill {fr} · Late {lt} · Short {sh} · Stockouts {st} · {rv}")

        if fill_col and late_col and name_col and not high.empty:
            metrics_df = []
            for _, r in high.iterrows():
                metrics_df.append({
                    "Supplier": r[name_col],
                    "Fill Rate %": float(r[fill_col])*100,
                    "Late %": float(r[late_col])*100,
                    "Short %": float(r[short_col])*100 if short_col else 0,
                })
            import pandas as pd
            mdf = pd.DataFrame(metrics_df).melt(id_vars="Supplier",
                var_name="Metric", value_name="Value")
            fig = px.bar(mdf, x="Supplier", y="Value", color="Metric",
                barmode="group",
                color_discrete_map={"Fill Rate %":C_GREEN,"Late %":C_RED,"Short %":C_AMBER})
            fig.update_layout(**DL("HIGH RISK supplier — 4-factor breakdown", height=280))
        else:
            fig = None

        facts_str = "HIGH RISK suppliers:\n" + NL.join(facts_lines)
        prompt = (
            "You are a senior retail procurement analyst.\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Name each HIGH RISK supplier, quantify their worst metric, "
            "and give a specific action — dual-source, escalate, or replace. Be decisive."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── General / fallback ────────────────────────────────────
    total = len(sc)
    high_n = len(sc[sc[risk_col].str.contains("HIGH",na=False)]) if risk_col else 0
    med_n  = len(sc[sc[risk_col].str.contains("MEDIUM",na=False)]) if risk_col else 0
    low_n  = len(sc[sc[risk_col].str.contains("LOW",na=False)]) if risk_col else 0
    avg_fr = float(sc[fill_col].mean())*100 if fill_col else 0

    ctx = (f"{total} suppliers total. HIGH RISK: {high_n}. MEDIUM: {med_n}. LOW: {low_n}. "
           f"Avg fill rate: {avg_fr:.1f}%.")
    ans = _llm(
        "You are HyperShelf AI, a retail procurement expert.\n"
        "Supplier network context: " + ctx + "\n"
        "Answer in 2-3 sentences. Stay in retail supplier management scope.\n"
        "Question: " + question
    )
    return ans or "Try: dual-source · most stockouts · HIGH RISK suppliers · select a supplier from the dropdown", None