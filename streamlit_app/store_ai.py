"""
store_ai.py — True hybrid store deep dive intelligence
Python fetches exact alert data, LLaMA explains action
"""
import requests
import pandas as pd
import plotly.graph_objects as go

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

def answer(question, alerts, store_id):
    if alerts is None or alerts.empty:
        return "No alert data available for this store.", None

    q = question.lower()
    NL = "\n"

    # Filter to store
    s = alerts[alerts["store_id"]==store_id].copy() if store_id else alerts.copy()
    if s.empty:
        return f"No alerts found for store {store_id}.", None

    # ── What to order right now ───────────────────────────────
    if any(x in q for x in ["order now","what to order","order right now","order today",
                              "what should i order","replenish","restock"]):
        crit = s[s["alert_tier"]=="CRITICAL"].sort_values("priority_score", ascending=False)
        if crit.empty:
            return f"No CRITICAL alerts at store {store_id}. All products have adequate stock.", None

        total_units = int(crit["units_to_order"].sum())
        total_rev   = float(crit["revenue_at_risk"].sum())

        top = crit.head(10)
        order_lines = []
        for _, r in top.iterrows():
            pname = str(r.get("product_name",""))[:28]
            dos   = float(r.get("days_of_supply_current", 0))
            units = int(r.get("units_to_order", 0))
            sup   = str(r.get("supplier_name","Unknown"))
            lt    = float(r.get("lead_time_final", 7))
            rev   = float(r.get("revenue_at_risk", 0))
            pack  = int(r.get("pack_size", 1))
            order_lines.append(
                f"  {r['sku_id']} · {pname} · {dos:.0f} day(s) left · "
                f"Order {units}u ({units//max(pack,1)} packs) · "
                f"Supplier: {sup} · LT: {lt:.1f} days · ${rev:,.0f} at risk")

        # Chart — revenue at risk by SKU
        top_chart = crit.head(12)
        fig = go.Figure(go.Bar(
            x=top_chart["revenue_at_risk"],
            y=top_chart["sku_id"] + " " + top_chart.get("product_name","").str[:15].fillna(""),
            orientation="h",
            marker=dict(color=top_chart["revenue_at_risk"],
                colorscale=[[0,C_AMBER],[1,C_RED]], showscale=False),
            text=[f"${v:,.0f}" for v in top_chart["revenue_at_risk"]],
            textposition="outside", textfont=dict(color=C_WHITE, size=8)))
        fig.update_layout(**DL(f"CRITICAL SKUs revenue at risk — {store_id}", height=300))

        facts = (
            f"Store {store_id} — {len(crit)} CRITICAL SKUs, "
            f"{total_units:,} units to order, ${total_rev:,.0f} total at risk\n\n"
            "Top 10 order priority (by priority score):\n" + NL.join(order_lines)
        )
        top1 = top.iloc[0]
        prompt = (
            "You are a retail store manager's AI assistant.\n"
            "EXACT ORDER LIST:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: 'You have {len(crit)} CRITICAL SKUs totalling ${total_rev:,.0f} at risk — order these now.'\n"
            f"Name the most urgent product ({top1['sku_id']} · {top1.get('product_name','')}), "
            f"its days left ({top1['days_of_supply_current']:.0f}), and which supplier to call. "
            "Be specific and action-oriented."
        )
        text = _llm(prompt) or facts
        return text, fig

    # ── Daily revenue loss ────────────────────────────────────
    if any(x in q for x in ["daily loss","revenue loss","cost if i do nothing","daily revenue",
                              "how much am i losing","loss per day","cost of inaction"]):
        crit = s[s["alert_tier"]=="CRITICAL"].copy()
        if "demand_for_calc" in crit.columns and "unit_price" in crit.columns:
            crit["daily_cost"] = crit["demand_for_calc"] * crit["unit_price"].fillna(0)
        elif "revenue_at_risk" in crit.columns:
            crit["daily_cost"] = crit["revenue_at_risk"] / 7
        else:
            crit["daily_cost"] = 0

        total_daily = crit["daily_cost"].sum()
        weekly_loss = total_daily * 7

        top5 = crit.nlargest(5, "daily_cost")
        loss_lines = []
        for _, r in top5.iterrows():
            pname = str(r.get("product_name",""))[:28]
            loss_lines.append(
                f"  {r['sku_id']} · {pname}: ${r['daily_cost']:,.0f}/day")

        fig = go.Figure(go.Bar(
            x=top5["daily_cost"],
            y=top5["sku_id"] + " " + top5.get("product_name","").str[:15].fillna(""),
            orientation="h",
            marker_color=C_RED,
            text=[f"${v:,.0f}/day" for v in top5["daily_cost"]],
            textposition="outside", textfont=dict(color=C_WHITE, size=9)))
        fig.update_layout(**DL(f"Daily revenue loss by SKU — {store_id}", height=240))

        facts = (
            f"Store {store_id} daily revenue loss if no action taken:\n"
            f"  Total: ${total_daily:,.0f}/day = ${weekly_loss:,.0f}/week\n\n"
            "Top 5 products causing the most daily loss:\n" + NL.join(loss_lines)
        )
        prompt = (
            "You are a retail store operations analyst.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: 'Every day without action costs ${total_daily:,.0f} in lost revenue at store {store_id}.'\n"
            "Name the top product causing the most daily loss with exact amount. "
            "Emphasize the cumulative weekly loss and urgency of ordering today."
        )
        text = _llm(prompt) or facts
        return text, fig

    # ── Supplier to call first ────────────────────────────────
    if any(x in q for x in ["supplier","call first","which supplier","supplier call",
                              "supplier contact","vendor"]):
        crit = s[s["alert_tier"]=="CRITICAL"].copy()
        if crit.empty or "supplier_name" not in crit.columns:
            return "No critical alerts or supplier data available.", None

        sup_g = crit.groupby("supplier_name").agg(
            sku_count=("sku_id","count"),
            revenue=("revenue_at_risk","sum"),
            avg_lt=("lead_time_final","mean"),
            avg_fill=("avg_fill_rate","mean"),
            units=("units_to_order","sum")
        ).reset_index().sort_values("revenue", ascending=False)

        fig = go.Figure(go.Bar(
            x=sup_g["revenue"],
            y=sup_g["supplier_name"],
            orientation="h",
            marker=dict(color=sup_g["revenue"],
                colorscale=[[0,C_AMBER],[1,C_RED]], showscale=False),
            text=[f"${v:,.0f} · {int(c)} SKUs" for v,c in zip(sup_g["revenue"],sup_g["sku_count"])],
            textposition="outside", textfont=dict(color=C_WHITE, size=8)))
        fig.update_layout(**DL(f"Supplier revenue at risk — {store_id}", height=max(200, len(sup_g)*28)))

        sup_lines = []
        for _, r in sup_g.iterrows():
            sup_lines.append(
                f"  {r['supplier_name']}: {int(r['sku_count'])} critical SKUs · "
                f"${r['revenue']:,.0f} at risk · LT {r['avg_lt']:.1f} days · "
                f"Fill rate {r['avg_fill']*100:.1f}% · {int(r['units'])} units to order")

        top_sup = sup_g.iloc[0]
        facts = (
            f"Suppliers with CRITICAL SKUs at store {store_id}:\n" + NL.join(sup_lines)
        )
        prompt = (
            "You are a retail store manager.\n"
            "EXACT SUPPLIER DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: 'Call {top_sup['supplier_name']} first — they have {int(top_sup['sku_count'])} critical SKUs "
            f"worth ${top_sup['revenue']:,.0f} at risk with a {top_sup['avg_lt']:.1f}-day lead time.'\n"
            "Explain what fill rate means for ordering confidence. "
            "Name the second most urgent supplier to contact after."
        )
        text = _llm(prompt) or facts
        return text, fig

    # ── Phantom SKUs for aisle check ─────────────────────────
    if any(x in q for x in ["phantom","aisle","ghost","check physically","zero sales"]):
        phantoms = s[s["is_phantom"]==True].copy() if "is_phantom" in s.columns else pd.DataFrame()
        if phantoms.empty:
            return f"No phantom inventory detected at store {store_id}.", None

        phantoms["daily_loss"] = phantoms["demand_for_calc"]*phantoms["unit_price"].fillna(0) if "demand_for_calc" in phantoms.columns else phantoms["revenue_at_risk"]/7
        total_phantom_loss = phantoms["daily_loss"].sum()

        top = phantoms.nlargest(10, "daily_loss")
        ph_lines = []
        for _, r in top.iterrows():
            pname = str(r.get("product_name",""))[:28]
            ph_lines.append(
                f"  {r['sku_id']} · {pname} · {r['category']} · "
                f"${r['daily_loss']:,.0f}/day · check physically")

        fig = go.Figure(go.Bar(
            x=top["daily_loss"],
            y=top["sku_id"] + " " + top.get("product_name","").str[:15].fillna(""),
            orientation="h",
            marker_color=C_PURPLE,
            text=[f"${v:,.0f}/day" for v in top["daily_loss"]],
            textposition="outside", textfont=dict(color=C_WHITE, size=8)))
        fig.update_layout(**DL(f"Phantom SKUs daily revenue at risk — {store_id}", height=240))

        facts = (
            f"Store {store_id} phantom inventory: {len(phantoms)} ghost SKUs, "
            f"${total_phantom_loss:,.0f}/day at risk\n\n"
            "Send associate to check these physically:\n" + NL.join(ph_lines)
        )
        prompt = (
            "You are a retail store operations expert.\n"
            "EXACT PHANTOM DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: 'Store {store_id} has {len(phantoms)} phantom SKUs costing ${total_phantom_loss:,.0f}/day in lost sales.'\n"
            "Explain what phantom inventory means (stock shows in system but not on shelf). "
            "Give specific instruction to send an associate to check these aisles today."
        )
        text = _llm(prompt) or facts
        return text, fig

    # ── Weekend stockout ─────────────────────────────────────
    if any(x in q for x in ["weekend","stock out","stockout","before weekend","saturday","2 days"]):
        wr = s[s["days_of_supply_current"]<=2].copy() if "days_of_supply_current" in s.columns else pd.DataFrame()
        if wr.empty:
            return f"No products stocking out before the weekend at store {store_id}.", None

        wr_rev = float(wr["revenue_at_risk"].sum())
        top = wr.sort_values("days_of_supply_current").head(10)
        wr_lines = []
        for _, r in top.iterrows():
            pname = str(r.get("product_name",""))[:25]
            wr_lines.append(
                f"  {r['sku_id']} · {pname} · {r['days_of_supply_current']:.0f} day(s) · "
                f"${r['revenue_at_risk']:,.0f}")

        facts = (
            f"Store {store_id}: {len(wr)} products stocking out before weekend — ${wr_rev:,.0f} at risk\n\n" +
            NL.join(wr_lines)
        )
        prompt = (
            "You are a retail store manager.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with: '{len(wr)} products at store {store_id} will stock out before the weekend, "
            f"putting ${wr_rev:,.0f} at risk.'\n"
            "Name the most urgent product. Give a specific emergency action to take today. "
            "Be urgent and decisive."
        )
        text = _llm(prompt) or facts
        return text, None

    # ── Fallback ──────────────────────────────────────────────
    total_crit = int((s["alert_tier"]=="CRITICAL").sum()) if "alert_tier" in s.columns else 0
    total_rar  = float(s["revenue_at_risk"].sum()) if "revenue_at_risk" in s.columns else 0
    ctx = f"Store {store_id}: {total_crit} CRITICAL SKUs, ${total_rar:,.0f} revenue at risk."
    ans = _llm(
        "You are HyperShelf AI, a retail store operations expert.\n"
        "Context: " + ctx + "\n"
        "Answer in 3 sentences. Stay in retail store operations scope.\n"
        "Question: " + question
    )
    return ans or "Try: order now · daily loss · supplier to call · phantom check · weekend stockout", None