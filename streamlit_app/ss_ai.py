"""
ss_ai.py — Safety Stock AI module
Python computes exact data from the filtered repl DataFrame.
LLaMA 3.2 explains the result in plain English.
Returns (text: str, fig: go.Figure | None)
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

C_BG="#0A1628";C_CARD="#0F2040";C_TEAL="#0D9488";C_TEAL2="#14B8A6"
C_RED="#EF4444";C_AMBER="#F59E0B";C_GREEN="#10B981";C_WHITE="#F0F9FF";C_GRAY="#64748B"

def _dl(title="",height=280):
    return dict(
        title=dict(text=title,font=dict(color=C_WHITE,size=13)),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_WHITE,family="IBM Plex Sans",size=11),height=height,
        margin=dict(l=48,r=20,t=40,b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)",zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)",zerolinecolor="rgba(255,255,255,0.1)"),
    )

def _llm(system,user,temperature=0.1):
    try:
        resp=requests.post("http://localhost:11434/api/generate",json={
            "model":"llama3.2:3b",
            "prompt":f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>",
            "stream":False,"options":{"temperature":temperature,"num_predict":280}},timeout=25)
        return resp.json().get("response","").strip()
    except Exception as e:
        return f"[LLaMA unavailable: {e}]"

def _has(q,*words):
    q=q.lower();return any(w in q for w in words)

def _handle_understocked(df,store_id):
    if "reorder_point" not in df.columns or "target" not in df.columns:
        return "Cannot compute understocked SKUs — reorder_point or target column missing.",None
    df=df.copy();df["units_short"]=(df["reorder_point"]-df["target"]).clip(lower=0)
    below=df[df["units_short"]>0].copy()
    if below.empty:
        return f"All SKUs at {store_id} are currently above their reorder point. Store is well stocked.",None
    below=below.sort_values("units_short",ascending=False);n=len(below)
    if "mu_daily" in df.columns:
        below["days_oos"]=(below["target"]/below["mu_daily"].clip(lower=0.1)).round(1)
        urgent=int((below["days_oos"]<3).sum())
    else:
        below["days_oos"]=None;urgent=None
    top=below.head(10)
    sku_col="sku_id" if "sku_id" in top.columns else None
    labels=top[sku_col].astype(str).tolist() if sku_col else [str(i) for i in top.index]
    vals=top["units_short"].tolist()
    cats=top["category"].tolist() if "category" in top.columns else [""]*len(labels)
    fig=go.Figure(go.Bar(x=vals,y=labels,orientation="h",
        marker_color=[C_RED if v>20 else C_AMBER for v in vals],
        text=[f"{v:.0f}u short" for v in vals],textposition="outside",
        textfont=dict(color=C_WHITE,size=10),customdata=cats,
        hovertemplate="%{y}<br>%{customdata}<br>Units short: %{x:.0f}<extra></extra>"))
    layout=_dl(f"Top 10 understocked SKUs — {store_id}",height=300)
    layout["yaxis"]["autorange"]="reversed";fig.update_layout(**layout)
    top5=below.head(5);lines=[]
    for _,row in top5.iterrows():
        lines.append(f"  {row.get('sku_id','')} ({row.get('category','')}): {row['units_short']:.0f} units short, ~{row.get('days_oos','?'):.1f} days until OOS" if row.get("days_oos") else f"  {row.get('sku_id','')} ({row.get('category','')}): {row['units_short']:.0f} units short")
    summary=(f"Store: {store_id}\nTotal SKUs below ROP: {n}\n"
        +(f"SKUs with less than 3 days until OOS: {urgent}\n" if urgent else "")
        +"Top 5:\n"+("\n".join(lines)))
    system="You are a retail supply chain analyst. Answer ONLY using the data provided. Do not calculate or estimate. Be direct. Max 4 sentences."
    return _llm(system,f"Data:\n{summary}\n\nQuestion: Which SKUs are most understocked right now?"),fig

def _handle_overstock_capital(df,store_id):
    if not all(c in df.columns for c in ["target","reorder_point","safety_stock_units"]):
        return "Cannot compute overstock — target, reorder_point, or safety_stock_units missing.",None
    df=df.copy();threshold=df["reorder_point"]+df["safety_stock_units"]*2
    over=df[df["target"]>threshold].copy();over["excess_units"]=(over["target"]-threshold).round(0)
    if over.empty:
        return f"No overstocked SKUs at {store_id}. Inventory within safety stock bounds.",None
    if "unit_price_actual" in over.columns:
        over["capital_locked"]=over["excess_units"]*over["unit_price_actual"]
        total_capital=over["capital_locked"].sum();capital_str=f"${total_capital:,.0f}"
    else:
        total_capital=None;capital_str="price data not available"
    n=len(over);fig=None
    if "category" in over.columns and total_capital:
        cat_g=over.groupby("category")["capital_locked"].sum().reset_index().sort_values("capital_locked",ascending=False)
        fig=go.Figure(go.Bar(x=cat_g["capital_locked"],y=cat_g["category"],orientation="h",
            marker_color=C_AMBER,text=[f"${v:,.0f}" for v in cat_g["capital_locked"]],
            textposition="outside",textfont=dict(color=C_WHITE,size=10)))
        layout=_dl(f"Capital locked by category — {store_id}",height=280)
        layout["yaxis"]["autorange"]="reversed";fig.update_layout(**layout)
    top=over.sort_values("excess_units",ascending=False).head(5);lines=[]
    for _,row in top.iterrows():
        cap=f"${row['capital_locked']:,.0f}" if "capital_locked" in row else f"{row['excess_units']:.0f} excess units"
        lines.append(f"  {row.get('sku_id','')} ({row.get('category','')}): {row['excess_units']:.0f} excess units — {cap}")
    summary=f"Store: {store_id}\nOverstocked SKUs: {n}\nCapital locked: {capital_str}\nTop 5:\n"+("\n".join(lines))
    system="You are a retail inventory analyst. Answer ONLY using the data provided. Focus on actionable insight. Max 4 sentences."
    return _llm(system,f"Data:\n{summary}\n\nQuestion: How much capital is locked in overstock?"),fig

def _handle_below_rop(df,store_id):
    if not all(c in df.columns for c in ["reorder_point","target","mu_daily"]):
        return "Cannot compute — reorder_point, target, or mu_daily missing.",None
    df=df.copy();df["units_short"]=(df["reorder_point"]-df["target"]).clip(lower=0)
    df["days_until_oos"]=(df["target"]/df["mu_daily"].clip(lower=0.1)).round(1)
    below=df[df["units_short"]>0].sort_values("days_until_oos")
    if below.empty:
        return f"All SKUs at {store_id} are above ROP. No immediate reorders needed.",None
    critical=below[below["days_until_oos"]<3];warning=below[(below["days_until_oos"]>=3)&(below["days_until_oos"]<7)]
    colors=below["days_until_oos"].apply(lambda d:C_RED if d<3 else(C_AMBER if d<7 else C_TEAL2))
    sku_labels=below["sku_id"].astype(str) if "sku_id" in below.columns else below.index.astype(str)
    cat_labels=below["category"].astype(str) if "category" in below.columns else pd.Series([""]*len(below))
    fig=go.Figure();fig.add_trace(go.Scatter(x=below["days_until_oos"],y=below["units_short"],
        mode="markers",marker=dict(color=colors,size=7,opacity=0.85),
        text=sku_labels,customdata=cat_labels,
        hovertemplate="%{text}<br>%{customdata}<br>Days until OOS: %{x:.1f}<br>Units short: %{y:.0f}<extra></extra>"))
    layout=_dl(f"SKUs below ROP — {store_id} (red=<3 days)",height=280)
    fig.update_layout(**layout);fig.add_vline(x=3,line_dash="dash",line_color=C_RED,opacity=0.5)
    fig.add_vline(x=7,line_dash="dash",line_color=C_AMBER,opacity=0.5)
    fig.update_xaxes(title_text="Days until stockout");fig.update_yaxes(title_text="Units short of ROP")
    top5=below.head(5);lines=[]
    for _,row in top5.iterrows():
        lines.append(f"  {row.get('sku_id','')} ({row.get('category','')}): {row['days_until_oos']:.1f} days, {row['units_short']:.0f} units short")
    summary=(f"Store: {store_id}\nBelow ROP: {len(below)}\nCritical (<3 days): {len(critical)}\nWarning (3-7 days): {len(warning)}\nTop 5:\n"+("\n".join(lines)))
    system="You are a retail operations analyst. Answer ONLY using the data. Prioritise urgency. Max 4 sentences."
    return _llm(system,f"Data:\n{summary}\n\nQuestion: Show SKUs below reorder point with days until stockout."),fig

def _handle_simulate(df,store_id,pct=-10):
    from scipy import stats as sc
    if not all(c in df.columns for c in ["safety_stock_units","sigma_daily","lead_time_days_avg","mu_daily"]):
        return "Cannot simulate — safety_stock_units, sigma_daily, lead_time_days_avg, or mu_daily missing.",None
    df=df.copy();df["ss_new"]=(df["safety_stock_units"]*(1+pct/100)).round(0)
    df["rop_new"]=(df["mu_daily"]*df["lead_time_days_avg"]+df["ss_new"]).round(0)
    denom=(df["sigma_daily"]*np.sqrt(df["lead_time_days_avg"])).clip(lower=0.01)
    df["z_new"]=(df["ss_new"]/denom).clip(upper=3.5)
    df["sl_new"]=df["z_new"].apply(lambda z:sc.norm.cdf(z)*100).round(1)
    avg_ss_old=df["safety_stock_units"].mean();avg_ss_new=df["ss_new"].mean()
    avg_rop_old=(df["mu_daily"]*df["lead_time_days_avg"]+df["safety_stock_units"]).mean()
    avg_rop_new=df["rop_new"].mean();avg_sl_new=df["sl_new"].mean();carry_delta=(avg_ss_new-avg_ss_old)*30
    fig=go.Figure()
    fig.add_trace(go.Bar(name="Current",x=["Safety Stock","Avg ROP"],y=[avg_ss_old,avg_rop_old],
        marker_color=C_GRAY,text=[f"{avg_ss_old:.0f}u",f"{avg_rop_old:.0f}u"],
        textposition="outside",textfont=dict(color=C_WHITE,size=10)))
    fig.add_trace(go.Bar(name=f"Simulated ({pct:+d}%)",x=["Safety Stock","Avg ROP"],y=[avg_ss_new,avg_rop_new],
        marker_color=C_RED if pct<0 else C_GREEN,text=[f"{avg_ss_new:.0f}u",f"{avg_rop_new:.0f}u"],
        textposition="outside",textfont=dict(color=C_WHITE,size=10)))
    layout=_dl(f"Safety stock simulation ({pct:+d}%) — {store_id}",height=260);layout["barmode"]="group"
    fig.update_layout(**layout)
    summary=(f"Store: {store_id}\nSimulation: {pct:+d}% SS change\n"
        f"Current avg SS: {avg_ss_old:.1f}u\nNew avg SS: {avg_ss_new:.1f}u (delta: {avg_ss_new-avg_ss_old:+.1f}u)\n"
        f"New service level: {avg_sl_new:.1f}%\nNew avg ROP: {avg_rop_new:.1f}u\n"
        f"Carrying cost change/SKU/month: ${carry_delta:+,.0f}")
    direction="reduced" if pct<0 else "increased"
    system="You are a retail supply chain strategist. Answer ONLY using the simulation data. Give a clear recommendation. Max 4 sentences."
    return _llm(system,f"Data:\n{summary}\n\nQuestion: What happens if we {direction} SS by {abs(pct)}%?",0.15),fig

def answer(question,df,store_id):
    """Main router. Returns (text, fig)."""
    q=question.strip().lower()
    if df is None or df.empty:
        return f"No data loaded for {store_id}. Select a store with replenishment data.",None
    if _has(q,"simulat","what if","what would","reduce","increase","decrease","lower","raise","change"):
        pct=-10;m=re.search(r'([+-]?\d+)\s*%',question)
        if m: pct=int(m.group(1))
        elif _has(q,"increase","raise"): pct=10
        return _handle_simulate(df,store_id,pct)
    if _has(q,"capital","locked","overstock","overstocked","excess","too much"):
        return _handle_overstock_capital(df,store_id)
    if _has(q,"below rop","below reorder","days until","reorder point"):
        return _handle_below_rop(df,store_id)
    if _has(q,"understock","understocked","which sku","sku","short","shortage"):
        return _handle_understocked(df,store_id)
    n_total=len(df)
    n_below=int((df["target"]<df["reorder_point"]).sum()) if all(c in df.columns for c in ["target","reorder_point"]) else "unknown"
    avg_ss=f"{df['safety_stock_units'].mean():.1f}u" if "safety_stock_units" in df.columns else "unknown"
    summary=f"Store: {store_id}\nTotal SKUs: {n_total}\nSKUs below ROP: {n_below}\nAvg safety stock: {avg_ss}"
    system="You are a retail supply chain analyst. Answer using only the store data provided. Max 4 sentences."
    return _llm(system,f"Data:\n{summary}\n\nQuestion: {question}",0.15),None
