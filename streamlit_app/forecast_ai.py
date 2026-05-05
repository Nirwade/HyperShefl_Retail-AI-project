"""
forecast_ai.py — True hybrid, clean unified output
Python computes exact data, LLaMA writes one clean analyst-style response
Charts rendered inline in chat via st.plotly_chart
"""
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Colors matching app theme ─────────────────────────────────
C_TEAL  = "#0D9488"; C_TEAL2 = "#14B8A6"; C_RED   = "#EF4444"
C_AMBER = "#F59E0B"; C_GREEN = "#10B981"; C_GRAY  = "#94A3B8"
C_CARD  = "#1E3352"; C_BG    = "#0D1B2A"; C_WHITE = "#E2E8F0"
C_BORDER= "#2D4A6A"

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
            json={"model": "llama3.2:3b", "prompt": prompt,
                  "stream": False, "options": {"temperature": 0.15, "num_predict": 180}},
            timeout=25)
        return r.json().get("response", "").strip() if r.status_code == 200 else None
    except:
        return None


def answer(question, wm, acc, pred, na_alerts, v2r, b30, chart_key_prefix="fc"):
    """
    Returns (text_response, fig_or_None).
    text_response: clean unified analyst-style answer, no Python/LLaMA labels.
    fig_or_None: Plotly figure to render inline, or None.
    """
    q = question.lower()
    NL = "\n"

    # ── WAPE spike ────────────────────────────────────────────
    if any(x in q for x in ["spike","black friday","christmas","why wape",
                              "bias","jump","wape increase","wape high"]):
        if wm is None or wm.empty:
            return "No weekly WAPE data available.", None

        avg_w = wm["wape"].mean()
        known = {
            "2025-11-24/2025-11-30": ("Black Friday", -1.719,
                "shelves cleared fast — actual sales hit zero while forecast stayed high. "
                "For every 10 units sold the model predicted 17."),
            "2025-12-22/2025-12-28": ("Christmas", -1.245,
                "pre-holiday buying velocity continued in the forecast after Dec 25 demand collapsed."),
            "2025-12-29/2026-01-04": ("Post-Christmas", -1.823,
                "model was still running on high December training signal after holiday ended. "
                "Worst bias of the year."),
        }
        spikes = wm[wm["wape"] > avg_w * 1.15].sort_values("wape", ascending=False)

        # Build chart
        colors = [C_RED if w > avg_w * 1.15 else C_TEAL for w in wm["wape"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=wm["week"], y=wm["wape"], name="WAPE %",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in wm["wape"]],
            textposition="outside", textfont=dict(color=C_WHITE, size=8)))
        if "bias" in wm.columns:
            fig.add_trace(go.Scatter(
                x=wm["week"], y=wm["bias"], name="Bias",
                line=dict(color=C_AMBER, width=2, dash="dot"),
                yaxis="y2", mode="lines+markers",
                marker=dict(size=5, color=C_AMBER)))
        fig.add_hrect(y0=22, y1=26, fillcolor="rgba(16,185,129,0.08)",
            opacity=1, line_width=0,
            annotation_text="Target 22-26%",
            annotation_font=dict(color=C_GREEN, size=9))
        fig.update_layout(**DL("Weekly WAPE — red bars are spikes", height=260),
            yaxis2=dict(title="Bias", overlaying="y", side="right",
                showgrid=False, tickfont=dict(color=C_AMBER), range=[-3, 2]))
        fig.update_xaxes(tickangle=-40)

        # Build facts for LLM
        spike_facts = []
        for _, r in spikes.iterrows():
            if r["week"] in known:
                name, bias, detail = known[r["week"]]
                spike_facts.append(
                    f"{name}: WAPE {r['wape']:.1f}% (avg {avg_w:.1f}%), "
                    f"Bias {r['bias']:.2f} — {detail}")
        if not spike_facts:
            return f"No significant WAPE spikes. Model stable at {avg_w:.1f}% average across all weeks.", fig

        facts_str = NL.join(spike_facts)
        prompt = (
            "You are a senior retail demand planning analyst writing a briefing note.\n"
            "These WAPE spikes occurred in the forecast model:\n\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one unified paragraph. No bullet points. No headings. "
            "Explain what happened, what caused it, and what it means for inventory decisions. "
            "Use the exact WAPE percentages and bias numbers. Sound like an expert analyst, not a chatbot."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Category accuracy ─────────────────────────────────────
    if any(x in q for x in ["category","worst","best","accurate","accuracy","forecast well"]):
        if acc is None or acc.empty:
            return "No category accuracy data available.", None

        worst = acc.loc[acc["wape"].idxmax()]
        best  = acc.loc[acc["wape"].idxmin()]
        acc_s = acc.sort_values("wape", ascending=True)

        fig = go.Figure(go.Bar(
            x=acc_s["wape"], y=acc_s["category"], orientation="h",
            marker=dict(color=acc_s["wape"],
                colorscale=[[0, C_GREEN], [0.5, C_AMBER], [1, C_RED]],
                showscale=False),
            text=[f"{v:.1f}%" for v in acc_s["wape"]],
            textposition="outside", textfont=dict(color=C_WHITE, size=9)))
        fig.update_layout(**DL("WAPE by Category (lower = better)", height=320))

        facts_str = (
            f"Worst: {worst['category']} at {worst['wape']:.1f}% WAPE "
            f"({int(worst.get('rows',0)):,} training rows, Bias {worst['bias']:.3f}). "
            f"Best: {best['category']} at {best['wape']:.1f}% WAPE "
            f"({int(best.get('rows',0)):,} training rows). "
            f"Overall range: {acc['wape'].min():.1f}% to {acc['wape'].max():.1f}%."
        )
        prompt = (
            "You are a senior retail demand planning analyst.\n"
            "Forecast accuracy by category (lower WAPE = better): " + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            f"Explain why {worst['category']} is the hardest category to forecast in retail — "
            "consider seasonality, promotional volatility, and product lifecycle. "
            "Then briefly note what makes the best category easier. Sound like an expert analyst."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Weekend stockout ──────────────────────────────────────
    if any(x in q for x in ["weekend","stock out","stockout","saturday","sunday",
                              "2 days","this week","running out"]):
        if na_alerts is None or na_alerts.empty or "days_of_supply_current" not in na_alerts.columns:
            return "Alert data not available.", None

        wr = na_alerts[na_alerts["days_of_supply_current"] <= 2].copy()
        if wr.empty:
            return "No products stocking out before this weekend. All stores have more than 2 days of supply.", None

        wr_rev  = float(wr["revenue_at_risk"].sum()) if "revenue_at_risk" in wr.columns else 0
        n_stores = wr["store_id"].nunique()

        # Chart — top stores by revenue at risk
        if "revenue_at_risk" in wr.columns:
            store_risk = (wr.groupby("store_id")["revenue_at_risk"].sum()
                         .nlargest(15).sort_values())
            fig = go.Figure(go.Bar(
                x=store_risk.values, y=store_risk.index, orientation="h",
                marker=dict(color=store_risk.values,
                    colorscale=[[0, C_AMBER], [1, C_RED]], showscale=False),
                text=[f"${v:,.0f}" for v in store_risk.values],
                textposition="outside", textfont=dict(color=C_WHITE, size=8)))
            fig.update_layout(**DL("Top 15 stores by weekend stockout revenue at risk", height=300))
        else:
            fig = None

        # Top urgent items
        top5 = wr.sort_values("days_of_supply_current").head(5)
        top_lines = []
        for _, r in top5.iterrows():
            pname = str(r.get("product_name",""))[:20] if "product_name" in wr.columns else r.get("sku_id","")
            rev = f"${r['revenue_at_risk']:,.0f}" if "revenue_at_risk" in wr.columns else ""
            top_lines.append(f"{r['store_id']} · {pname} · {r['days_of_supply_current']:.0f} day(s) · {rev}")
        top_str = NL.join(top_lines)
        facts_str = (
            f"{len(wr):,} products across {n_stores} stores, ${wr_rev:,.0f} at risk. "
            f"Most urgent: {top_str}"
        )
        prompt = (
            "You are a senior retail operations analyst.\n"
            "Weekend stockout data: " + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Tell the store manager what is at risk, name the most urgent store and product, "
            "and give specific actions to take today. Sound decisive and action-oriented."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Demand uncertainty ────────────────────────────────────
    if any(x in q for x in ["uncertain","band","p10","p90","range","variab","wide","uncertainty"]):
        if pred is None or pred.empty or "lower_bound_90" not in pred.columns:
            return "Demand range data not available.", None

        ut = pred.copy()
        ut["band"] = ut["upper_bound_90"] - ut["lower_bound_90"]
        ut_top = (ut.groupby(["store_id","sku_id","category"])
            .agg(p50=("forecast_units","mean"), p90=("upper_bound_90","mean"),
                 p10=("lower_bound_90","mean"), band=("band","mean"))
            .reset_index().nlargest(12,"band"))

        ut_top["label"] = ut_top["store_id"] + " · " + ut_top["sku_id"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ut_top["band"], y=ut_top["label"], orientation="h",
            name="Uncertainty Band",
            marker=dict(color=ut_top["band"],
                colorscale=[[0, C_AMBER], [1, C_RED]], showscale=False),
            text=[f"{v:.0f}u ({v/max(r['p50'],0.1)*100:.0f}%)"
                  for v, (_, r) in zip(ut_top["band"], ut_top.iterrows())],
            textposition="outside", textfont=dict(color=C_WHITE, size=8)))
        fig.update_layout(**DL("SKUs with widest P10-P90 demand band", height=320))
        fig.update_xaxes(title="Band width (units/day)")

        top1 = ut_top.iloc[0]
        unc1 = top1["band"] / max(top1["p50"], 0.1) * 100
        facts_str = (
            f"Widest band: {top1['store_id']} · {top1['sku_id']} · {top1['category']} "
            f"— P10:{top1['p10']:.0f} P50:{top1['p50']:.0f} P90:{top1['p90']:.0f} "
            f"band {top1['band']:.1f} units ({unc1:.0f}% uncertain). "
            f"Top 12 SKUs shown in chart, all in {ut_top['category'].value_counts().index[0]} "
            f"and {ut_top['category'].value_counts().index[-1] if len(ut_top)>1 else ''} categories."
        )
        prompt = (
            "You are a senior retail inventory planner.\n"
            "Demand uncertainty data: " + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Explain what a wide P10-P90 band means for ordering, why these specific categories "
            "have high uncertainty, and give a clear recommendation on ordering at P90. "
            "Be specific with the numbers provided. Sound like an expert analyst."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Slow movers ───────────────────────────────────────────
    if any(x in q for x in ["slow","capital","tied","tying","overstock","long tail","dead"]):
        if pred is None or pred.empty or "target" not in pred.columns:
            return "Sales data not available.", None

        sm = pred.copy()
        sm_g = (sm.groupby(["store_id","sku_id","category"])
            .agg(avg_s=("target","mean"), avg_f=("forecast_units","mean")).reset_index())
        sm_slow = sm_g[sm_g["avg_s"] < sm_g["avg_f"] * 0.3].nsmallest(12,"avg_s")

        if sm_slow.empty:
            return "No significant slow movers detected — all SKUs selling above 30% of forecast.", None

        sm_slow["label"] = sm_slow["store_id"] + " · " + sm_slow["sku_id"]
        sm_slow["sell_pct"] = sm_slow["avg_s"] / sm_slow["avg_f"].clip(lower=0.1) * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sm_slow["sell_pct"], y=sm_slow["label"], orientation="h",
            marker=dict(color=sm_slow["sell_pct"],
                colorscale=[[0, C_RED], [1, C_AMBER]], showscale=False),
            text=[f"{v:.0f}% of forecast" for v in sm_slow["sell_pct"]],
            textposition="outside", textfont=dict(color=C_WHITE, size=8)))
        fig.update_layout(**DL("Slow movers — actual sales as % of forecast", height=320))
        fig.update_xaxes(title="% of forecast actually sold")

        top1 = sm_slow.iloc[0]
        facts_str = (
            f"{len(sm_slow)} slow-moving SKUs found selling below 30% of forecast. "
            f"Most extreme: {top1['store_id']} · {top1['sku_id']} · {top1['category']} "
            f"selling only {top1['avg_s']:.1f}u/day vs forecast of {top1['avg_f']:.1f}u/day "
            f"({top1['sell_pct']:.0f}% of forecast)."
        )
        prompt = (
            "You are a senior retail buyer.\n"
            "Slow mover data: " + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Explain what slow movers mean for capital efficiency, and recommend specific "
            "actions — reduce reorder quantities, clearance pricing, planogram removal, "
            "or inter-store transfer. Be action-oriented and specific."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Model comparison ──────────────────────────────────────
    if any(x in q for x in ["v2","demandsense","compare","baseline","movingavg","lightgbm","improvement"]):
        if v2r is None:
            return "Model summary not available.", None
        b_wape = float(b30["wape"]) if b30 is not None else 28.89
        diff = b_wape - float(v2r["wape"])
        facts_str = (
            f"DemandSense v2: WAPE {float(v2r['wape']):.2f}%, "
            f"P90 coverage {float(v2r.get('p90_coverage',0)):.2f}%, "
            f"Bias {float(v2r.get('bias',0)):.3f}. "
            f"MovingAvg30 baseline: {b_wape:.2f}% WAPE. "
            f"Improvement: {diff:.1f}% WAPE reduction."
        )
        prompt = (
            "You are a retail AI expert.\n"
            "Model comparison: " + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Explain why LightGBM outperforms a moving average for retail demand — "
            "mention weekday patterns, seasonality, demand variability. Be technical but clear."
        )
        text = _llm(prompt) or facts_str
        return text, None

    # ── General fallback ──────────────────────────────────────
    ctx = ""
    if wm is not None and not wm.empty:
        ctx += f"Model WAPE avg {wm['wape'].mean():.1f}% over {len(wm)} weeks. "
    if acc is not None and not acc.empty:
        w_ = acc.loc[acc["wape"].idxmax()]
        ctx += f"Hardest category: {w_['category']} {w_['wape']:.1f}% WAPE. "
    if v2r is not None:
        b_w = float(b30["wape"]) if b30 is not None else 28.89
        ctx += f"v2 WAPE {float(v2r['wape']):.2f}% vs {b_w:.2f}% baseline."

    ans = _llm(
        "You are HyperShelf AI, a retail demand planning expert.\n"
        "Context: " + ctx + "\n"
        "Write 2-3 sentences as a direct answer. Stay in retail forecasting scope.\n"
        "Question: " + question
    )
    return (ans or
            "Try asking: WAPE spikes · category accuracy · weekend stockout · "
            "demand uncertainty · slow movers · model comparison"), None