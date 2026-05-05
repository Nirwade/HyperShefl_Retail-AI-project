"""
phantom_ai.py — True hybrid phantom inventory intelligence
Python finds exact phantom data, LLaMA explains action
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

def _daily_rev(row):
    """Estimate daily revenue at risk from rolling avg and price."""
    rev = row.get("revenue_at_risk_daily", 0)
    if rev and float(rev) > 0:
        return float(rev)
    avg = float(row.get("rolling_14d_avg", 0) or 0)
    price = float(row.get("unit_price", 0) or 0)
    return avg * price

def answer(question, ph, store_id=None, min_days=3, confidence_filter="All"):
    """
    ph: phantom_confidence dataframe
    store_id: selected store (optional)
    min_days: minimum consec_zero_days filter
    confidence_filter: All / High / Medium / Low
    Returns (text, fig)
    """
    if ph is None or ph.empty:
        return "No phantom inventory data available.", None

    q = question.lower()
    NL = "\n"

    # Apply filters
    ph_f = ph[ph["is_phantom_candidate"] == True].copy()
    ph_f = ph_f[ph_f["consec_zero_days"] >= min_days]
    if confidence_filter != "All":
        ph_f = ph_f[ph_f["phantom_confidence"] == confidence_filter]
    if store_id and store_id != "All stores":
        ph_f = ph_f[ph_f["store_id"] == store_id]

    if ph_f.empty:
        return f"No phantom candidates found with current filters (min {min_days} days, {confidence_filter} confidence).", None

    # Pre-compute daily revenue
    ph_f["daily_rev"] = ph_f.apply(_daily_rev, axis=1)

    # ── Aisle check / which to check first ───────────────────
    if any(x in q for x in ["aisle","check first","check","inspect","physical","walk","audit"]):
        # Sort by High confidence first, then by days descending
        conf_order = {"High": 0, "Medium": 1, "Low": 2}
        ph_f["conf_rank"] = ph_f["phantom_confidence"].map(conf_order).fillna(3)
        top = ph_f.sort_values(["conf_rank","consec_zero_days"],
                               ascending=[True,False]).head(15)

        # Chart — bar by confidence
        conf_counts = ph_f["phantom_confidence"].value_counts()
        fig = go.Figure(go.Bar(
            x=conf_counts.index,
            y=conf_counts.values,
            marker_color=[C_RED if c=="High" else (C_AMBER if c=="Medium" else C_PURPLE)
                         for c in conf_counts.index],
            text=conf_counts.values,
            textposition="outside",
            textfont=dict(color=C_WHITE, size=12)))
        fig.update_layout(**DL(
            f"Phantom candidates by confidence{' — '+store_id if store_id else ''}",
            height=240))

        facts_lines = []
        for _, r in top.iterrows():
            facts_lines.append(
                f"  {r['store_id']} · {r['sku_id']} · {str(r.get('product_name',''))[:25]} · "
                f"{r['phantom_confidence']} confidence · {int(r['consec_zero_days'])} zero-sales days · "
                f"${r['daily_rev']:,.0f}/day at risk")
        facts_str = (
            f"{len(ph_f):,} phantom candidates{' at '+store_id if store_id else ''} "
            f"(min {min_days} days). Top priority for physical check:\n" +
            NL.join(facts_lines)
        )
        prompt = (
            "You are a retail store operations expert.\n"
            "Phantom inventory check list:\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Tell the store manager which products to check first, why High confidence "
            "phantoms are most urgent, and what physical action to take in the store. "
            "Be specific with product names and days. Sound decisive."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Capital locked ────────────────────────────────────────
    if any(x in q for x in ["capital","locked","cost","revenue","money","value","how much"]):
        total_daily = ph_f["daily_rev"].sum()
        high_daily  = ph_f[ph_f["phantom_confidence"]=="High"]["daily_rev"].sum()
        by_cat = ph_f.groupby("category")["daily_rev"].sum().sort_values(ascending=False)

        fig = go.Figure(go.Bar(
            x=by_cat.values, y=by_cat.index, orientation="h",
            marker=dict(color=by_cat.values,
                colorscale=[[0,C_PURPLE],[1,C_RED]], showscale=False),
            text=[f"${v:,.0f}/day" for v in by_cat.values],
            textposition="outside", textfont=dict(color=C_WHITE, size=9)))
        fig.update_layout(**DL("Daily revenue at risk by category (phantom inventory)", height=300))

        facts_str = (
            f"Total phantom daily revenue at risk: ${total_daily:,.0f}/day\n"
            f"High confidence only: ${high_daily:,.0f}/day\n"
            f"Total candidates: {len(ph_f):,} phantom SKUs\n"
            f"Top categories: " +
            ", ".join([f"{cat}: ${val:,.0f}/day" for cat,val in by_cat.head(5).items()])
        )
        prompt = (
            "You are a retail inventory expert.\n"
            "Phantom inventory capital at risk:\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            "Explain what phantom inventory means for capital efficiency, "
            "quantify the daily revenue loss, and recommend what action resolves it fastest. "
            "Be specific with the numbers."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Worst category ────────────────────────────────────────
    if any(x in q for x in ["category","worst","most","which category","beverages","produce"]):
        by_cat = ph_f.groupby("category").agg(
            count=("sku_id","count"),
            high=("phantom_confidence", lambda x: (x=="High").sum()),
            daily_rev=("daily_rev","sum")
        ).sort_values("count", ascending=False)

        fig = px.bar(by_cat.reset_index(),
            x="count", y="category", orientation="h",
            color="high",
            color_continuous_scale=[[0,C_PURPLE],[1,C_RED]],
            labels={"count":"Phantom SKUs","high":"High Confidence","category":"Category"})
        fig.update_layout(**DL("Phantom SKUs by category (color = high confidence count)", height=300),
            coloraxis_showscale=False)
        fig.update_traces(texttemplate="%{x}", textposition="outside",
            textfont=dict(color=C_WHITE))

        top_cat = by_cat.index[0]
        facts_str = (
            f"Phantom inventory by category:\n" +
            NL.join([f"  {cat}: {int(r['count'])} SKUs · {int(r['high'])} high confidence · "
                     f"${r['daily_rev']:,.0f}/day at risk"
                     for cat, r in by_cat.iterrows()])
        )
        prompt = (
            "You are a retail inventory expert.\n" + facts_str + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings. "
            f"Explain why {top_cat} has the most phantom inventory in retail, "
            "what causes phantom inventory in that category specifically, "
            "and what immediate action to take. Be specific."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Age distribution ──────────────────────────────────────
    if any(x in q for x in ["age","how long","days","duration","old","3 days","7 days","14 days"]):
        bins = [3,7,14,22]
        labels = ["3-6 days","7-13 days","14-21 days","22+ days"]
        ph_f2 = ph_f.copy()
        ph_f2["age_band"] = pd.cut(ph_f2["consec_zero_days"],
            bins=[3,7,14,22,999], labels=labels, right=False)
        age_g = ph_f2.groupby("age_band").agg(
            count=("sku_id","count"),
            high=("phantom_confidence", lambda x: (x=="High").sum()),
            daily_rev=("daily_rev","sum")
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=age_g.index.tolist(), y=age_g["count"],
            name="All phantoms",
            marker_color=[C_PURPLE,C_AMBER,C_RED,C_RED],
            text=age_g["count"], textposition="outside",
            textfont=dict(color=C_WHITE,size=10)))
        fig.add_trace(go.Bar(
            x=age_g.index.tolist(), y=age_g["high"],
            name="High confidence",
            marker_color="rgba(239,68,68,0.5)",
            text=age_g["high"], textposition="outside",
            textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Phantom age distribution — how long have they been sitting?",
            height=280), barmode="group")

        facts_str = (
            "Phantom inventory age distribution:\n" +
            NL.join([f"  {band}: {int(row['count'])} SKUs · {int(row['high'])} high confidence · "
                     f"${row['daily_rev']:,.0f}/day at risk"
                     for band, row in age_g.iterrows()])
        )
        prompt = (
            "You are a retail store operations expert.\n" + facts_str + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings. "
            "Explain what the age distribution means — older phantoms are harder to recover. "
            "Recommend prioritizing the longest-sitting phantoms for physical audit. Be specific."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Store ranking ─────────────────────────────────────────
    if any(x in q for x in ["store","which store","worst store","most phantom","store ranking"]):
        by_store = ph_f.groupby("store_id").agg(
            count=("sku_id","count"),
            high=("phantom_confidence", lambda x: (x=="High").sum()),
            daily_rev=("daily_rev","sum")
        ).sort_values("count", ascending=False)

        top15 = by_store.head(15)
        fig = go.Figure(go.Bar(
            x=top15["count"], y=top15.index, orientation="h",
            marker=dict(color=top15["count"],
                colorscale=[[0,C_PURPLE],[1,C_RED]], showscale=False),
            text=[f"{int(v)}" for v in top15["count"]],
            textposition="outside", textfont=dict(color=C_WHITE,size=9)))
        fig.update_layout(**DL("Top 15 stores by phantom SKU count", height=300))

        top_store = by_store.index[0]
        facts_str = (
            f"Top 5 stores by phantom count:\n" +
            NL.join([f"  {store}: {int(r['count'])} phantoms · {int(r['high'])} high conf · "
                     f"${r['daily_rev']:,.0f}/day"
                     for store, r in by_store.head(5).iterrows()])
        )
        prompt = (
            "You are a retail store operations expert.\n" + facts_str + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings. "
            f"Name the worst store ({top_store}), explain the operational impact, "
            "and give a specific action plan. Be decisive and urgent."
        )
        text = _llm(prompt) or facts_str
        return text, fig

    # ── Fallback ──────────────────────────────────────────────
    total = len(ph_f)
    high  = int((ph_f["phantom_confidence"]=="High").sum())
    total_rev = ph_f["daily_rev"].sum()
    ctx = (f"{total:,} phantom candidates, {high} high confidence, "
           f"${total_rev:,.0f}/day revenue at risk, "
           f"min {min_days} zero-sales days filter applied.")
    ans = _llm(
        "You are HyperShelf AI, a retail inventory expert.\n"
        "Phantom inventory context: " + ctx + "\n"
        "Answer in 3 sentences. Stay in retail phantom inventory scope.\n"
        "Question: " + question
    )
    return ans or "Try: aisle check · capital locked · worst category · age distribution · store ranking", None