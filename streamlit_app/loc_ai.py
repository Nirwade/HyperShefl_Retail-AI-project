"""
loc_ai.py — Localization intelligence
Python finds exact mismatch data, LLaMA explains retail strategy
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

def answer(question, ls, lc, ln, pred, na, store_a=None, store_b=None, category=None):
    if lc is None or lc.empty:
        return "No localization data available.", None

    q = question.lower()
    NL = "\n"

    # ── Biggest revenue opportunity store ─────────────────────
    if any(x in q for x in ["biggest opportunity","most opportunity","revenue opportunity",
                              "which store","best opportunity"]):
        if ls is None or ls.empty:
            return "Localization scores not available.", None
        worst = ls.sort_values("localization_score").head(10)
        fig = go.Figure(go.Bar(
            x=worst["localization_score"],
            y=worst["store_id"] + " " + worst.get("store_name","").fillna(""),
            orientation="h",
            marker=dict(color=worst["localization_score"],
                colorscale=[[0,C_RED],[1,C_AMBER]], showscale=False),
            text=[f"{v:.1f}" for v in worst["localization_score"]],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Biggest localization opportunity (lowest score)", height=280))

        top = worst.iloc[0]
        mm_cats = lc[lc["store_id"]==top["store_id"]] if not lc.empty else pd.DataFrame()
        mm_count = int((mm_cats["mismatch_flag"].isin(["MISMATCH","EXTREME_MISMATCH"])).sum()) if not mm_cats.empty else 0
        worst_cat = mm_cats.loc[mm_cats["rev_gap_pct"].idxmin(),"category"] if not mm_cats.empty and "rev_gap_pct" in mm_cats.columns else "Unknown"

        facts = (
            f"Biggest revenue opportunity: {top['store_id']} {top.get('store_name','')} "
            f"({top.get('city','')}, {top.get('store_format','')})\n"
            f"Localization score: {top['localization_score']:.1f}/100\n"
            f"Mismatched categories: {mm_count}\n"
            f"Worst category: {worst_cat}\n\n"
            "Top 5 stores by opportunity:\n" +
            NL.join([f"  {r['store_id']} {r.get('store_name','')} — score {r['localization_score']:.1f}, "
                     f"{r.get('mismatch_cats',0)} mismatched cats"
                     for _,r in worst.head(5).iterrows()])
        )
        prompt = (
            "You are a retail localization strategist.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: '{top['store_id']} has the biggest localization opportunity with a score of {top['localization_score']:.1f}.'\n"
            "Explain what a low localization score means — the store's product mix doesn't match local demand. "
            "Name the worst category and estimate the revenue uplift potential. Be specific."
        )
        return _llm(prompt) or facts, fig

    # ── Which category to localize first ─────────────────────
    if any(x in q for x in ["category","localize first","which category","worst category","fix first"]):
        if ln is None or ln.empty:
            return "Network category data not available.", None
        up_col = "total_rev_gap" if "total_rev_gap" in ln.columns else None
        if not up_col:
            return "Revenue gap data not available.", None

        ln_s = ln.sort_values(up_col)
        fig = go.Figure(go.Bar(
            x=abs(ln_s[up_col]), y=ln_s["category"], orientation="h",
            marker=dict(color=abs(ln_s[up_col]),
                colorscale=[[0,C_AMBER],[1,C_RED]], showscale=False),
            text=[f"${abs(v):,.0f}" for v in ln_s[up_col]],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Category revenue gap — fix these first", height=280))

        worst_cat = ln_s.iloc[0]
        facts = (
            f"Category localization priority (by total revenue gap):\n" +
            NL.join([f"  {r['category']}: ${abs(r[up_col]):,.0f} gap · "
                     f"{int(r.get('stores_with_mismatch',0))} stores mismatched · "
                     f"{int(r.get('extreme_count',0))} extreme"
                     for _,r in ln_s.iterrows()])
        )
        prompt = (
            "You are a retail category manager.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: '{worst_cat['category']} has the largest revenue gap at ${abs(worst_cat[up_col]):,.0f} across the network.'\n"
            "Explain why fixing category localization improves revenue — right products for local demand. "
            "Give specific actions: add SKUs, remove slow movers, adjust planogram. Be actionable."
        )
        return _llm(prompt) or facts, fig

    # ── Store vs store comparison ─────────────────────────────
    if any(x in q for x in ["compare","vs","versus","s0064","s0001","s0","store a","store b"]):
        if store_a and store_b and not lc.empty:
            da = lc[lc["store_id"]==store_a].copy()
            db = lc[lc["store_id"]==store_b].copy()
            if category and category != "All categories":
                da = da[da["category"]==category]
                db = db[db["category"]==category]

            if da.empty or db.empty:
                return f"No comparison data for {store_a} vs {store_b}.", None

            merged = da[["category","store_avg_daily_rev","rev_gap_pct","mismatch_flag"]].merge(
                db[["category","store_avg_daily_rev","rev_gap_pct","mismatch_flag"]],
                on="category", suffixes=(f"_{store_a}",f"_{store_b}"))

            fig = go.Figure()
            fig.add_trace(go.Bar(name=store_a, x=merged["category"],
                y=merged[f"store_avg_daily_rev_{store_a}"],
                marker_color=C_TEAL,
                text=[f"${v:.0f}" for v in merged[f"store_avg_daily_rev_{store_a}"]],
                textposition="outside", textfont=dict(color=C_WHITE,size=8)))
            fig.add_trace(go.Bar(name=store_b, x=merged["category"],
                y=merged[f"store_avg_daily_rev_{store_b}"],
                marker_color=C_AMBER,
                text=[f"${v:.0f}" for v in merged[f"store_avg_daily_rev_{store_b}"]],
                textposition="outside", textfont=dict(color=C_WHITE,size=8)))
            fig.update_layout(**DL(f"{store_a} vs {store_b} — daily revenue by category", height=300))
            fig.update_layout(barmode="group")
            fig.update_xaxes(tickangle=-30)

            a_better = merged[merged[f"store_avg_daily_rev_{store_a}"] > merged[f"store_avg_daily_rev_{store_b}"]]["category"].tolist()
            b_better = merged[merged[f"store_avg_daily_rev_{store_b}"] > merged[f"store_avg_daily_rev_{store_a}"]]["category"].tolist()

            facts = (
                f"{store_a} vs {store_b} comparison:\n"
                f"{store_a} outperforms in: {', '.join(a_better) or 'none'}\n"
                f"{store_b} outperforms in: {', '.join(b_better) or 'none'}\n\n"
                "Category detail:\n" +
                NL.join([f"  {r['category']}: {store_a} ${r[f'store_avg_daily_rev_{store_a}']:.0f}/day "
                         f"({r[f'mismatch_flag_{store_a}']}) vs "
                         f"{store_b} ${r[f'store_avg_daily_rev_{store_b}']:.0f}/day "
                         f"({r[f'mismatch_flag_{store_b}']})"
                         for _,r in merged.iterrows()])
            )
            prompt = (
                "You are a retail localization analyst.\n"
                "EXACT COMPARISON DATA:\n" + facts + "\n\n"
                "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
                f"Name which categories {store_a} outperforms {store_b} in and vice versa. "
                "Explain what this means for inventory allocation — the underperforming store "
                "may need more SKUs or better planogram placement. Be specific with the numbers."
            )
            return _llm(prompt) or facts, fig
        return "Select Store A and Store B above to compare.", None

    # ── Slow movers / capital efficiency ──────────────────────
    if any(x in q for x in ["slow mover","capital","tied","tying","overstock","long tail",
                              "stop reorder","opportunity cost","holding"]):
        if pred is None or pred.empty:
            return "Prediction data not available for slow mover analysis.", None

        sm = pred.copy()
        if "target" not in sm.columns or "forecast_units" not in sm.columns:
            return "Target and forecast columns needed for slow mover analysis.", None

        sm_g = (sm.groupby(["store_id","sku_id","category"])
            .agg(avg_sales=("target","mean"), avg_fc=("forecast_units","mean"))
            .reset_index())
        sm_g = sm_g[sm_g["avg_fc"] > 0]
        sm_g["sell_pct"] = sm_g["avg_sales"]/sm_g["avg_fc"]*100

        # Filter to selected store if applicable
        if store_a and store_a != "All stores":
            sm_g = sm_g[sm_g["store_id"]==store_a]

        slow = sm_g[sm_g["sell_pct"] < 70].nsmallest(12,"sell_pct")
        if slow.empty:
            return "No significant slow movers — all SKUs selling above 30% of forecast.", None

        # Estimate opportunity cost
        if na is not None and not na.empty and "unit_price" in na.columns:
            price_map = na.groupby("sku_id")["unit_price"].mean().to_dict()
            slow["unit_price"] = slow["sku_id"].map(price_map).fillna(30)
        else:
            slow["unit_price"] = 30
        slow["opportunity_cost"] = slow["avg_fc"] * slow["unit_price"] * 14  # 14-day holding
        slow["should_stop"] = slow["sell_pct"] < 40

        fig = go.Figure(go.Bar(
            x=slow["sell_pct"],
            y=slow["sku_id"] + " " + slow["category"],
            orientation="h",
            marker=dict(color=slow["sell_pct"],
                colorscale=[[0,C_RED],[1,C_AMBER]], showscale=False),
            text=[f"{v:.0f}% · Stop:{r['should_stop']}" for v,(i,r) in zip(slow["sell_pct"],slow.iterrows())],
            textposition="outside", textfont=dict(color=C_WHITE,size=8)))
        fig.update_layout(**DL("Slow movers — actual sales as % of forecast", height=300))
        fig.update_xaxes(title="% of forecast actually sold")

        total_opp = slow["opportunity_cost"].sum()
        facts = (
            f"{len(slow)} slow-moving SKUs, ${total_opp:,.0f} opportunity cost (14-day holding)\n\n"
            "Top slow movers (should stop reordering if <15%):\n" +
            NL.join([f"  {r['store_id']} · {r['sku_id']} · {r['category']} — "
                     f"{r['sell_pct']:.0f}% of forecast · ${r['opportunity_cost']:,.0f} opp cost · "
                     f"{'STOP REORDER' if r['should_stop'] else 'REDUCE QTY'}"
                     for _,r in slow.iterrows()])
        )
        prompt = (
            "You are a retail buyer.\n"
            "EXACT SLOW MOVER DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with the total opportunity cost (${total_opp:,.0f}) tied up in slow movers.\n"
            "Explain what opportunity cost means — capital that could be deployed on faster-moving SKUs. "
            "Recommend: stop reordering SKUs below 15% sell-through, reduce quantities for 15-30%. "
            "Be specific with names."
        )
        return _llm(prompt) or facts, fig

    # ── Fallback ──────────────────────────────────────────────
    total_mm = int((lc["mismatch_flag"].isin(["MISMATCH","EXTREME_MISMATCH"])).sum()) if "mismatch_flag" in lc.columns else 0
    ctx = f"Network: {total_mm} mismatched store-category combinations."
    ans = _llm(
        "You are HyperShelf AI, a retail localization expert.\n"
        "Context: " + ctx + "\n"
        "Answer in 3 sentences. Stay in retail localization scope.\n"
        "Question: " + question
    )
    return ans or "Try: biggest opportunity store · category to fix first · compare two stores · slow movers", None