"""
HyperShelf AI Chat — llm_chat.py
Clean rewrite with all features built in properly:
- Casual vs data intent detection
- Python computes all numbers from real CSVs
- Hybrid LLM for flexible natural language responses
- Conversation memory with follow-up detection
- Supplier tier routing (HIGH/MEDIUM/LOW)
- Store name lookup by partial match
"""
import sys, os, re, warnings, random
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

PROJECT  = Path(__file__).resolve().parent.parent
PROCESSED= PROJECT / "data/processed/training"
NEXUS    = PROJECT / "data/processed/nexus"
CSV      = PROJECT / "data/raw/output/csv"

st.set_page_config(page_title="HyperShelf AI", layout="wide",
                   initial_sidebar_state="expanded")

BG="#0D1B2A";NAVY="#0F1E35";CARD="#1E3352";BORDER="#2D4A6A"
TEAL="#0D9488";TEAL2="#14B8A6";RED="#EF4444";AMBER="#F59E0B"
GREEN="#10B981";PURPLE="#8B5CF6";GRAY="#94A3B8";WHITE="#E2E8F0"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"],*{{font-family:'IBM Plex Sans',sans-serif!important;}}
.stApp{{background:{BG}!important;}}
.stAppHeader{{display:none!important;}}
section[data-testid="stSidebar"]{{background:{NAVY}!important;border-right:1px solid {BORDER};}}
section[data-testid="stSidebar"] *{{color:{WHITE}!important;}}
section[data-testid="stSidebar"] .stRadio>div>label{{padding:8px 12px!important;border-radius:6px!important;font-size:12px!important;cursor:pointer!important;display:block!important;}}
section[data-testid="stSidebar"] .stRadio>div>label:hover{{background:{CARD}!important;}}
section[data-testid="stSidebar"] .stRadio>div>label>div:first-child{{display:none!important;}}
section[data-testid="stSidebar"] .stRadio>label{{display:none!important;}}
[data-testid="stSidebarCollapseButton"]{{display:none!important;}}
.block-container{{padding:0 2rem 2rem!important;max-width:100%!important;background:{BG};}}
div[data-baseweb="select"]>div{{background:{NAVY}!important;border-color:{BORDER}!important;color:{WHITE}!important;}}
div[data-baseweb="menu"]{{background:{NAVY}!important;border-color:{BORDER}!important;}}
div[role="option"]{{color:{WHITE}!important;background:{NAVY}!important;}}
div[role="option"]:hover{{background:{CARD}!important;}}
h1,h2,h3,h4,h5,h6{{color:{WHITE}!important;}} p,span,div,label{{color:{WHITE};}}
hr{{border-color:{BORDER}!important;margin:12px 0!important;}}
.stButton>button{{background:{CARD};color:{TEAL2};border:1px solid {BORDER};border-radius:6px;font-size:11px;font-weight:700;padding:6px 10px;width:100%;text-align:left;}}
.stButton>button:hover{{background:{NAVY};border-color:{TEAL};color:{WHITE};}}
::-webkit-scrollbar{{width:4px;}}
.chat-area{{max-height:60vh;overflow-y:auto;padding:4px 0 16px;display:flex;flex-direction:column;gap:8px;}}
.bubble-user-wrap{{display:flex;justify-content:flex-end;}}
.bubble-ai-wrap{{display:flex;justify-content:flex-start;}}
.bubble-user{{background:{TEAL};color:#fff;border-radius:16px 16px 4px 16px;padding:10px 15px;max-width:70%;font-size:13px;line-height:1.6;}}
.bubble-ai{{background:{CARD};border:1px solid {BORDER};border-radius:4px 16px 16px 16px;padding:13px 17px;max-width:92%;width:auto;font-size:13px;line-height:1.8;color:{WHITE};}}
.bubble-label{{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:{GRAY};margin-bottom:3px;}}
.mode-badge{{display:inline-block;font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:1px;padding:2px 8px;border-radius:10px;margin-bottom:6px;}}
.kpi-chip{{background:{CARD};border:1px solid {BORDER};border-radius:6px;padding:8px 12px;text-align:center;}}
.kpi-chip-val{{font-size:18px;font-weight:800;line-height:1.1;}}
.kpi-chip-lbl{{font-size:9px;color:{GRAY};text-transform:uppercase;letter-spacing:1px;margin-top:2px;}}
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def load():
    def rd(p):
        try: return pd.read_csv(p)
        except: return pd.DataFrame()
    stores = rd(CSV/"stores.csv")
    ns = rd(NEXUS/"allstore"/"network_store_summary.csv")
    ns = ns.rename(columns={"critical":"critical_count","warning":"warning_count",
                             "phantom_count":"phantom_skus","monitor":"monitor_count"})
    if not ns.empty and not stores.empty:
        for col in ["store_name","city","state","region","store_format","foot_traffic_tier"]:
            if col not in ns.columns and col in stores.columns:
                ns = ns.merge(stores[["store_id",col]], on="store_id", how="left")
    return dict(
        ns=ns, na=rd(NEXUS/"allstore"/"network_master_alerts.csv"),
        stores=stores, products=rd(CSV/"products.csv"),
        ms=rd(PROCESSED/"demandSense_model_summary.csv"),
        wm=rd(PROCESSED/"weekly_monitor_demandsense.csv"),
        bt=rd(PROCESSED/"backtest_summary_demandsense.csv"),
        acc=rd(NEXUS/"forecast"/"accuracy_by_category.csv"),
    )


# ── Tool dispatch ─────────────────────────────────────────────
def _dispatch(name, args):
    try:
        from src.tools import dispatch_tool
        return dispatch_tool(name, args)
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Conversation memory ───────────────────────────────────────
# Stores (role, content, store_ids_mentioned) for follow-up detection
_memory = []

def mem_add(role, content):
    global _memory
    store_ids = re.findall(r'S\d{4}', content)
    _memory.append({"role": role, "content": content, "store_ids": store_ids})
    if len(_memory) > 8:
        _memory = _memory[-8:]

def mem_get_store_ids():
    """Return store IDs from recent AI responses for follow-up questions."""
    for m in reversed(_memory):
        if m["role"] == "assistant" and m["store_ids"]:
            return m["store_ids"]
    return []

def mem_get_context(max_chars=600):
    if not _memory:
        return ""
    lines = ["Recent conversation:"]
    for m in _memory[-4:]:
        prefix = "User" if m["role"] == "user" else "AI"
        lines.append(f"  {prefix}: {m['content'][:200]}")
    return "\n".join(lines)


# ── Supplier tier detector ────────────────────────────────────
def detect_supplier_tier(q):
    q = q.lower()
    if any(x in q for x in ["low risk","safe supplier","reliable","best supplier",
                              "good supplier","performing well","low",
                              "best performing","best","top supplier",
                              "who are my best","most reliable"]):
        return "LOW_RISK"
    if any(x in q for x in ["medium risk","watch","borderline","medium"]):
        return "MEDIUM_RISK"
    return "HIGH_RISK"


# ── Store name resolver ───────────────────────────────────────
def resolve_store_name(question, stores_df):
    """Find store ID from partial name in question."""
    if stores_df.empty: return None
    q = question.lower()
    for _, row in stores_df.iterrows():
        name = str(row.get("store_name","")).lower()
        parts = [p for p in name.split() if len(p) > 3]
        if any(p in q for p in parts):
            return row["store_id"]
    return None


# ── Follow-up ordinal detector ────────────────────────────────
def detect_followup_ordinal(q):
    """Return index (0-based) if question references an ordinal like 'first one'."""
    ordinals = {
        "first":0,"second":1,"third":2,"fourth":3,"fifth":4,
        "sixth":5,"seventh":6,"eighth":7,"ninth":8,"tenth":9,"last":-1,
        "1st":0,"2nd":1,"3rd":2,"4th":3,"5th":4,
    }
    for word, idx in ordinals.items():
        if word in q:
            return idx
    return None


# ── LLM helper ───────────────────────────────────────────────
@st.cache_resource
def get_agent(persona):
    try:
        from src.agent import DemandSenseAgent
        return DemandSenseAgent(persona=persona)
    except: return None

def check_ollama():
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except: return False

def llm_explain(data_summary, question, persona, ollama_ok, intent_hint=""):
    """
    Core hybrid engine: Python fetches data, LLM explains it.
    LLM receives only the pre-computed data — cannot hallucinate numbers.
    """
    if not ollama_ok:
        return data_summary

    ctx = mem_get_context()
    compressed = data_summary[:2800]  # ~700 tokens

    system = (
        "You are HyperShelf AI, a retail supply chain assistant. "
        "You receive real data from the system. Your job: explain it clearly. "
        "Rules: "
        "1. Use ONLY the numbers in the data — never calculate or estimate. "
        "2. If asked about LOW RISK suppliers, discuss low risk ones only. "
        "3. If asked about HIGH RISK, discuss high risk only. "
        "4. CRITICAL = 3 days or less of stock. WARNING = 7 days or less. Do not mix. "
        "5. If the data does not have what the user asked, say: "
        "'I don't have that specific data. Try: morning briefing for [store ID].' "
        "6. Keep response under 8 lines. Be direct and conversational."
        + (f" 7. The user is asking about {intent_hint}." if intent_hint else "")
    )

    prompt = "\n\n".join(filter(None, [
        ctx,
        f"REAL DATA:\n{compressed}",
        f"QUESTION: {question}\n\nAnswer using only the data above."
    ]))

    try:
        agent = get_agent(persona)
        if agent:
            old_hist = agent.history[:]
            agent.history = [{"role":"system","content":system}]
            r = agent.chat(prompt)
            agent.history = old_hist
            resp = r.get("response","")
            if resp and not resp.strip().startswith("{") and len(resp) > 10:
                return resp
    except: pass
    return data_summary


# ── Main data engine ──────────────────────────────────────────
def answer(question, D, persona, ollama_ok):
    """
    All answers come from here.
    Python reads CSVs → formats data → LLM explains (for flexible queries)
    Returns (response_text, mode_label)
    """
    ns = D["ns"]; na = D["na"]; ms = D["ms"]
    q  = question.lower().strip()
    wm = D.get("wm", __import__("pandas").DataFrame())
    acc = D.get("acc", __import__("pandas").DataFrame())

    # ── WAPE / forecast questions — check FIRST before any other routing
    if any(x in q for x in ["wape","spike","black friday","christmas","post-christmas",
                              "why did wape","why wape","forecast accuracy",
                              "worst category","best category","hardest to forecast"]):
        _NL = chr(10)
        if any(x in q for x in ["spike","black friday","christmas","why wape","why did"]):
            if not wm.empty:
                _avg = wm["wape"].mean()
                _sp = wm[wm["wape"]>_avg*1.15].sort_values("wape",ascending=False)
                _known = {
                    "2025-11-24/2025-11-30":("Black Friday",-1.719,"Model over-predicted demand. Stores cleared stock fast — actual sales hit zero while forecast stayed high. For every 10 units sold model predicted 17."),
                    "2025-12-22/2025-12-28":("Christmas",-1.245,"Holiday surge then sharp drop after Dec 25. Model kept forecasting high after demand collapsed."),
                    "2025-12-29/2026-01-04":("Post-Christmas",-1.823,"Worst week. Demand collapsed post-holiday. Model trained on high December volumes kept predicting high."),
                }
                _facts=[]
                for _,_r in _sp.iterrows():
                    if _r["week"] in _known:
                        _n,_b,_d=_known[_r["week"]]
                        _facts.append(f"**{_n}** ({_r['week']}): WAPE {_r['wape']:.1f}% vs avg {_avg:.1f}%, Bias {_r['bias']:.2f} — {_d}")
                    else:
                        _facts.append(f"{_r['week']}: WAPE {_r['wape']:.1f}%, Bias {_r['bias']:.2f}")
                if _facts:
                    return "**WAPE Spike Analysis**" + _NL + _NL + _NL.join(_facts) + _NL + _NL + f"Avg WAPE: {_avg:.1f}% across {len(wm)} weeks.", "Forecast"
        if any(x in q for x in ["category","worst","best","accurate","accuracy","hardest"]):
            if not acc.empty:
                _w=acc.loc[acc["wape"].idxmax()]; _b=acc.loc[acc["wape"].idxmin()]
                _resp="**Forecast Accuracy by Category**"+_NL+_NL
                _resp+=f"**Worst:** {_w['category']} — WAPE {_w['wape']:.1f}%, Bias {_w['bias']:.3f}, {int(_w.get('rows',0)):,} rows"+_NL
                _resp+=f"**Best:** {_b['category']} — WAPE {_b['wape']:.1f}%, Bias {_b['bias']:.3f}"+_NL+_NL
                for _,_r in acc.sort_values("wape",ascending=False).iterrows():
                    _resp+=f"  {_r['category']}: {_r['wape']:.1f}% WAPE · {int(_r.get('rows',0)):,} rows"+_NL
                return _resp, "Forecast"


    # ── Extract store/SKU IDs ─────────────────────────────────
    s_match = re.search(r'\bS(\d{4})\b', question.upper())
    k_match = re.search(r'\bP(\d{5})\b', question.upper())
    sid = f"S{s_match.group(1)}" if s_match else None
    sku = f"P{k_match.group(1)}" if k_match else None

    # ── Bare store ID — 'S0021' or 'tell me about S0021' ────────
    if sid and not sku and len(q.split()) <= 6 and not any(x in q for x in [
        'briefing','brief','morning','replenishment','forecast','risk',
        'stockout','phantom','order','how many','at risk','supplier']):
        # User typed just a store ID or 'tell me about S0021'
        # Route to store risk overview
        if not ns.empty and sid in ns['store_id'].values:
            row  = ns[ns['store_id']==sid].iloc[0]
            crit = int(row.get('critical_count',0))
            warn = int(row.get('warning_count',0))
            rar  = float(row.get('revenue_at_risk',0))
            name = f"{row.get('store_name',sid)} ({row.get('city','')}, {row.get('state','')})"
            lines = [
                f'**{name} [{sid}]**',
                '',
                f'🔴 {crit} products need orders right now (CRITICAL)',
                f'🟠 {warn} products need orders today (WARNING)',
                f'Revenue at risk: ${rar:,.0f}',
                '',
                f'Try:',
                f'  • *morning briefing for {sid}* — full order list',
                f'  • *forecast for {sid} P00055* — demand forecast',
                f'  • *replenishment for {sid} P00055* — order details',
            ]
            return chr(10).join(lines), 'Store Risk'

    # ── Bare store ID — 'S0021' or 'tell me about S0021' ────────
    if sid and not sku and len(q.split()) <= 6 and not any(x in q for x in [
        'briefing','brief','morning','replenishment','forecast','risk',
        'stockout','phantom','order','how many','at risk','supplier']):
        # User typed just a store ID or 'tell me about S0021'
        # Route to store risk overview
        if not ns.empty and sid in ns['store_id'].values:
            row  = ns[ns['store_id']==sid].iloc[0]
            crit = int(row.get('critical_count',0))
            warn = int(row.get('warning_count',0))
            rar  = float(row.get('revenue_at_risk',0))
            name = f"{row.get('store_name',sid)} ({row.get('city','')}, {row.get('state','')})"
            lines = [
                f'**{name} [{sid}]**',
                '',
                f'🔴 {crit} products need orders right now (CRITICAL)',
                f'🟠 {warn} products need orders today (WARNING)',
                f'Revenue at risk: ${rar:,.0f}',
                '',
                f'Try:',
                f'  • *morning briefing for {sid}* — full order list',
                f'  • *forecast for {sid} P00055* — demand forecast',
                f'  • *replenishment for {sid} P00055* — order details',
            ]
            return chr(10).join(lines), 'Store Risk'

    # ── Follow-up: "tell me about the third one" ──────────────
    ordinal = detect_followup_ordinal(q)
    is_followup = any(x in q for x in [
        "first one","second one","third one","fourth one","fifth one",
        "sixth one","seventh one","eighth one","ninth one","tenth one",
        "last one","that store","tell me more","more about","the one",
        "1st one","2nd one","3rd one","4th one","5th one"
    ])

    if is_followup and ordinal is not None and not sid:
        store_ids = mem_get_store_ids()
        if store_ids:
            try:
                sid = store_ids[ordinal]
            except IndexError:
                sid = store_ids[-1]

    # ── Store name lookup if no ID found ─────────────────────
    if not sid:
        sid = resolve_store_name(question, D["stores"])

    # ══════════════════════════════════════════════════════════
    # MORNING BRIEFING
    # ══════════════════════════════════════════════════════════
    if sid and any(x in q for x in ["briefing","brief","morning","what to order",
                                     "what should i order","all alerts","full list",
                                     "tell me more","more about","more detail"]) and not sku:
        r = _dispatch("get_store_briefing", {"store_id": sid})
        if not na.empty and "alert_tier" in na.columns:
            _ss = na[na["store_id"]==sid]
            r["critical_count"] = int((_ss["alert_tier"]=="CRITICAL").sum())
            r["warning_count"] = int((_ss["alert_tier"]=="WARNING").sum())
        if r.get("status") == "ok":
            crit = r.get("critical_orders",[])
            warn = r.get("warning_orders",[])
            by_cat = defaultdict(list)
            for o in crit:
                by_cat[o.get("category","Other")].append(o)
            sup_count = defaultdict(int)
            for o in crit[:10]:
                sup_count[o.get("supplier","?")] += 1
            top_sup     = max(sup_count, key=sup_count.get) if sup_count else "—"
            top_sup_arr = crit[0].get("arrival_date","—") if crit else "—"
            lines = [
                f"**{r.get('store_name',sid)} [{sid}]** — {r.get('city','')} {r.get('state','')}",
                f"🔴 {r.get('critical_count',0)} CRITICAL · 🟠 {r.get('warning_count',0)} WARNING orders",
                f"Total order value: ${r.get('total_order_value',0):,.0f}",
                "",
                "**Top 5 urgent:**",
            ]
            # Check if user asked for more
            show_n = 20 if any(x in q for x in ['complete','full list','all','show more','15 more','more products']) else 5
            for o in crit[:show_n]:
                dos = o.get("days_of_supply",0)
                dos_txt = f"{dos:.0f} day{'s' if dos!=1 else ''} left"
                lines.append(f"  🔴 **{o['product_name'][:32]}** — {o['order_qty']} units · {dos_txt} · {o['aisle']}")
            if len(crit) > 5:
                lines.append(f"  ... and {len(crit)-5} more")
            lines.append("\n**By category:**")
            for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1]))[:5]:
                avg_dos = sum(o.get("days_of_supply",0) for o in items)/len(items)
                lines.append(f"  • {cat}: {len(items)} SKUs · avg {avg_dos:.0f} days left")
            lines.append(f"\n**Call first:** {top_sup} · stock arrives {top_sup_arr}")
            return "\n".join(lines), "Store Briefing"
        return f"No briefing data for {sid}.", "Store Briefing"

    # ══════════════════════════════════════════════════════════
    # REPLENISHMENT (store + SKU)
    # ══════════════════════════════════════════════════════════
    if sid and sku and any(x in q for x in ["replenishment","reorder","order",
                                              "how many","quantity","stock up","repl"]):
        r = _dispatch("get_replenishment_action", {"store_id":sid,"sku_id":sku})
        if r.get("status") == "ok":
            lines = [
                f"**{r.get('product_name',sku)}**",
                f"{r.get('store_name',sid)} ({r.get('city','')} {r.get('state','')})",
                "",
                f"Order **{r.get('recommended_order_qty',0)} units** from {r.get('supplier_name','')}",
                f"  • Order today before 3 PM",
                f"  • Arrives: {r.get('expected_arrival','')} ({r.get('lead_time_days',0)} days)",
                f"  • Put on: {r.get('aisle_location','check planogram')}",
                f"  • Runs out: {r.get('projected_stockout_date','')} ({r.get('days_of_supply_current',0):.0f} days left)",
                f"  • Demand: {r.get('mu_daily',0):.0f} units/day",
            ]
            if r.get("requires_approval"):
                lines.append(f"  ⚠ Needs manager sign-off — ${r.get('order_value_usd',0):,.0f}")
            if r.get("weekend_risk"):
                lines.append(f"  ⚠ Weekend spike — {r.get('weekend_demand_est',0):.0f} units/day Sat/Sun")
            return "\n".join(lines), "Replenishment"
        # Not found — show available SKUs
        try:
            from src.model_registry import registry
            skus = registry.repl_inputs[registry.repl_inputs["store_id"]==sid]["sku_id"].unique()[:8].tolist()
            return f"{sku} not in replenishment data for {sid}.\nAvailable SKUs: {', '.join(skus)}", "Replenishment"
        except:
            return f"{sku} not found at {sid}.", "Replenishment"

    # ══════════════════════════════════════════════════════════
    # FORECAST (store + SKU)
    # ══════════════════════════════════════════════════════════
    if sid and sku and any(x in q for x in ["forecast","demand","predict",
                                              "how much","selling","trend"]):
        r = _dispatch("query_forecast", {"store_id":sid,"sku_id":sku,"days":7})
        if r.get("status") == "ok":
            lines = [
                f"**{r.get('product_name',sku)}** at {r.get('store_name',sid)}",
                "",
                f"Expected: **{r.get('avg_forecast',0)} units/day** · trend {r.get('trend','')}",
                f"Peak day: **{r.get('peak_day','')}** at {r.get('peak_forecast',0)} units",
            ]
            if r.get("weekend_spike"):
                lines.append(f"⚠ Weekend spike — {r.get('weekend_avg_demand',0)} units Sat/Sun "
                             f"vs {r.get('weekday_avg_demand',0)} weekdays · stock up by Friday")
            daily = r.get("daily_forecast",[])
            if daily:
                lines.append("\nThis week:")
                for d in daily:
                    bar = "▓" * min(int(d['forecast_p50']/3), 18)
                    flag = " 🔴" if d.get("is_weekend") else ""
                    lines.append(f"  {d['day_name'][:3]}  {bar} {d['forecast_p50']:.0f}u{flag}")
            return "\n".join(lines), "Forecast"
        try:
            from src.model_registry import registry
            skus = registry.predictions[registry.predictions["store_id"]==sid]["sku_id"].unique()[:8].tolist()
            return f"{sku} not found at {sid}.\nSKUs with forecast data: {', '.join(skus)}", "Forecast"
        except:
            return f"{sku} not found at {sid}.", "Forecast"

    # ══════════════════════════════════════════════════════════
    # STOCKOUT RISK (store + SKU)
    # ══════════════════════════════════════════════════════════
    if sid and sku and any(x in q for x in ["risk","stockout","stock out","run out",
                                              "shortage","when will","daily revenue",
                                              "revenue will we lose","exact stockout"]):
        r = _dispatch("get_stockout_risk", {"store_id":sid,"sku_id":sku})
        if r.get("status") == "ok":
            tier_emoji = {"HIGH":"🔴","MEDIUM":"🟠","LOW":"🟡"}.get(r.get("risk_tier",""),"⚪")
            lines = [
                f"**{r.get('product_name',sku)}** at {r.get('store_name',sid)}",
                "",
                f"{tier_emoji} Risk: **{r.get('risk_tier','')}** (score {r.get('risk_score',0):.2f})",
                f"Runs out: **{r.get('projected_stockout_date','')}** ({r.get('days_of_supply',0):.0f} days)",
                f"Daily revenue at risk: **${r.get('daily_revenue_at_risk',0):,.0f}**",
                f"Demand: {r.get('mu_daily',0):.0f} units/day · trend {r.get('demand_trend','')}",
                f"Supplier: {r.get('supplier_name','')} · {r.get('lead_time_days',0):.0f}-day lead · "
                f"{r.get('supplier_reliability_pct',0):.0f}% reliable",
            ]
            return "\n".join(lines), "Stockout Risk"
        else:
            try:
                from src.model_registry import registry
                ri = registry.repl_inputs
                store_skus = ri[ri["store_id"]==sid]["sku_id"].unique()[:8].tolist()
                preds = registry.predictions
                pred_skus = preds[preds["store_id"]==sid]["sku_id"].unique()[:8].tolist()
                available = list(set(store_skus + pred_skus))[:8]
                if available:
                    return (f"{sku} is not in the data for {sid}.\n"
                            f"SKUs available at {sid}: {', '.join(available)}"), "Stockout Risk"
                else:
                    return f"No data found for {sid}. Check the store ID.", "Stockout Risk"
            except Exception as e:
                return f"{sku} not found at {sid}. Try: S0001 P00055", "Stockout Risk"

    # ══════════════════════════════════════════════════════════
    # STORE-LEVEL RISK (store, no SKU)
    # ══════════════════════════════════════════════════════════
    if sid and not sku and any(x in q for x in ["how many","at risk","products","critical",
                                                  "warning","need orders","alerts","issues",
                                                  "tell me more","more about","more detail",
                                                  "what about","first one","second one",
                                                  "third one","fourth one","fifth one"]):
        if not ns.empty and sid in ns["store_id"].values:
            row  = ns[ns["store_id"]==sid].iloc[0]
            crit = int(row.get("critical_count",0))
            warn = int(row.get("warning_count",0))
            mon  = int(row.get("monitor_count",0))
            rar  = float(row.get("revenue_at_risk",0))
            ph   = int(row.get("phantom_skus",0))
            uto  = int(row.get("units_to_order",0))
            name = f"{row.get('store_name',sid)} ({row.get('city','')}, {row.get('state','')})"
            lines = [
                f"**{name} [{sid}]**",
                "",
                f"🔴 **{crit} products** need orders right now (3 days or less of stock)",
                f"🟠 **{warn} products** need orders today (7 days or less)",
                f"🟡 {mon} products to monitor",
            ]
            if ph > 0:
                lines.append(f"👻 {ph} phantom SKUs detected")
            lines.append(f"\nTotal to order: {uto:,} units · Revenue at risk: ${rar:,.0f}")
            lines.append(f"\nFor the full order list: *morning briefing for {sid}*")
            data_summary = "\n".join(lines)
            resp = llm_explain(data_summary, question, persona, ollama_ok,
                               f"store {sid} ({row.get('store_name','')})")
            return resp, "Store Risk"

    # ══════════════════════════════════════════════════════════
    # NETWORK STATUS
    # ══════════════════════════════════════════════════════════
    if any(x in q for x in ["what is going on","what's going on","going on","status",
                              "overview","network","summary","kpi","whats up",
                              "what's up","today","how many stores"]):
        if not ns.empty:
            def g(c): return float(ns[c].sum()) if c in ns.columns else 0
            rar=g("revenue_at_risk"); crit=int(g("critical_count"))
            warn=int(g("warning_count")); ph=int(g("phantom_skus")); uto=int(g("units_to_order"))
            sort_c = "urgency_score" if "urgency_score" in ns.columns else "revenue_at_risk"
            top3 = ns.nlargest(3, sort_c)
            data_lines = [
                f"Network status — {datetime.now().strftime('%A %B %d')}",
                f"Revenue at risk: ${rar/1e6:.1f}M",
                f"CRITICAL SKUs (order immediately): {crit:,}",
                f"WARNING SKUs (order today): {warn:,}",
                f"Phantom SKUs: {ph}",
                f"Total units to order: {uto:,}",
                f"Stores monitored: {len(ns)}",
                "",
                "3 most urgent stores:",
            ]
            for _,r in top3.iterrows():
                data_lines.append(
                    f"  {r.get('store_name',r['store_id'])} [{r['store_id']}] "
                    f"({r.get('city','')}, {r.get('state','')}) — "
                    f"{int(r.get('critical_count',0))} critical · ${float(r.get('revenue_at_risk',0)):,.0f} at risk"
                )
            if "region" in ns.columns:
                top_reg = ns.groupby("region")["revenue_at_risk"].sum().nlargest(3)
                data_lines.append("\nRevenue at risk by region:")
                for reg, val in top_reg.items():
                    data_lines.append(f"  {reg}: ${val/1e6:.2f}M")
            data_summary = "\n".join(data_lines)
            resp = llm_explain(data_summary, question, persona, ollama_ok, "network-wide status")
            return resp, "Network Status"

    # ══════════════════════════════════════════════════════════
    # TOP STORES RANKING
    # ══════════════════════════════════════════════════════════
    if any(x in q for x in ["which store","top store","worst store","most urgent",
                              "most critical","store need","store attention",
                              "need attention","urgent store"]):
        if not ns.empty:
            sort_c = "urgency_score" if "urgency_score" in ns.columns else "revenue_at_risk"
            top10  = ns.nlargest(10, sort_c)
            lines  = [f"**Top 10 stores needing attention — {datetime.now().strftime('%b %d')}**", ""]
            for i,(_, r) in enumerate(top10.iterrows(), 1):
                crit_ = int(r.get("critical_count",0))
                rar_  = float(r.get("revenue_at_risk",0))
                ph_   = int(r.get("phantom_skus",0))
                ph_txt = f" · {ph_} phantoms" if ph_ > 0 else ""
                lines.append(
                    f"  {i}. **{r.get('store_name',r['store_id'])}** [{r['store_id']}] "
                    f"({r.get('city','')}, {r.get('state','')}) "
                    f"— {crit_} critical · ${rar_:,.0f} at risk{ph_txt}"
                )
            lines.append("")
            lines.append("Ask *morning briefing for [store ID]* for full order list.")
            return "\n".join(lines), "Store Ranking"

    # ══════════════════════════════════════════════════════════
    # SUPPLIER RISK (all tiers)
    # ══════════════════════════════════════════════════════════
    if any(x in q for x in ["supplier","vendor","fill rate","late delivery",
                              "supplier risk","risky","supplier issue","supply chain"]):
        req_tier = detect_supplier_tier(q)
        r = _dispatch("get_supplier_risk", {"requested_tier": req_tier})
        if r.get("status") == "ok":
            alerts      = r.get("top_offenders",[])
            tier_summary= r.get("tier_summary",{})
            low_risk    = r.get("low_risk_suppliers",[])
            total       = tier_summary.get("total",85)

            if req_tier == "LOW_RISK":
                data_lines = [
                    f"Supplier network: {total} total",
                    f"LOW RISK (performing well): {tier_summary.get('LOW_RISK',0)}",
                    f"MEDIUM RISK (watch): {tier_summary.get('MEDIUM_RISK',0)}",
                    f"HIGH RISK (action needed): {tier_summary.get('HIGH_RISK',0)}",
                    "",
                    "LOW RISK suppliers — best performers:",
                ]
                for s in (low_risk or alerts)[:8]:
                    data_lines.append(
                        f"  {s.get('supplier_name','?')} — "
                        f"fill rate {s.get('fill_rate','?')} · "
                        f"{s.get('late_orders_pct','?')} late · "
                        f"{s.get('stockouts_caused',0):,} stockouts"
                    )
                resp = llm_explain("\n".join(data_lines), question, persona, ollama_ok,
                                   "LOW RISK suppliers that are performing well")
                return resp, "Supplier Risk"

            elif req_tier == "MEDIUM_RISK":
                med_alerts = [a for a in alerts if a.get("risk_tier")=="MEDIUM_RISK"]
                data_lines = [
                    f"MEDIUM RISK suppliers ({tier_summary.get('MEDIUM_RISK',0)} total):",
                    f"These are underperforming but not yet critical.",
                    "",
                ]
                for a in med_alerts[:8]:
                    data_lines.append(
                        f"  {a['supplier_name']} — "
                        f"fill {a.get('fill_rate','?')} · "
                        f"late {a.get('late_orders_pct','?')} · "
                        f"short {a.get('short_delivery','?')} · "
                        f"{a['stockouts_caused']:,} stockouts"
                    )
                resp = llm_explain("\n".join(data_lines), question, persona, ollama_ok,
                                   "MEDIUM RISK suppliers on the watch list")
                return resp, "Supplier Risk"

            else:  # HIGH RISK
                high_alerts = [a for a in alerts if a.get("risk_tier")=="HIGH_RISK"]
                med_alerts  = [a for a in alerts if a.get("risk_tier")=="MEDIUM_RISK"]
                high_names  = " and ".join(a["supplier_name"] for a in high_alerts[:2])
                total_so    = sum(a["stockouts_caused"] for a in high_alerts)
                total_rev   = sum(a.get("revenue_at_risk",0) for a in high_alerts)
                data_lines  = [
                    f"HIGH RISK suppliers: {len(high_alerts)} need immediate action",
                    f"Names: {high_names}",
                    f"Combined: {total_so:,} stockout events, ${total_rev/1e6:.1f}M revenue at risk",
                    f"MEDIUM RISK watch list: {len(med_alerts)} suppliers",
                    "",
                    "HIGH RISK details:",
                ]
                for a in high_alerts:
                    data_lines.append(
                        f"  {a['supplier_name']}: fill={a.get('fill_rate','?')} "
                        f"late={a.get('late_orders_pct','?')} "
                        f"short={a.get('short_delivery','?')} "
                        f"stockouts={a['stockouts_caused']:,} "
                        f"revenue_at_risk=${a.get('revenue_at_risk',0):,.0f}"
                    )
                for a in med_alerts[:3]:
                    data_lines.append(
                        f"  {a['supplier_name']}: fill={a.get('fill_rate','?')} "
                        f"late={a.get('late_orders_pct','?')} "
                        f"stockouts={a['stockouts_caused']:,}"
                    )
                resp = llm_explain("\n".join(data_lines), question, persona, ollama_ok,
                                   "HIGH RISK suppliers causing stockouts")
                return resp, "Supplier Risk"

    # ══════════════════════════════════════════════════════════
    # PHANTOM INVENTORY
    # ══════════════════════════════════════════════════════════
    if any(x in q for x in ["phantom","ghost","ghost stock","fake stock"]):
        args = {"store_id":sid,"top_n":8} if sid else {"top_n":8}
        r = _dispatch("get_phantom_alerts", args)
        if r.get("status") == "ok":
            total = r.get("total_suspects",0)
            scope = sid if sid else "all stores"
            if total == 0:
                return f"No phantom inventory detected at {scope}.", "Phantom"
            lines = [f"**Phantom inventory at {scope}**",
                     f"{total} products show as in-stock but have zero recent sales",""]
            for a in r.get("alerts",[])[:5]:
                lines.append(f"  👻 {a['product_name'][:35]} — {a['aisle_location']} "
                             f"· confidence {a['confidence']} · ${a.get('capital_locked',0):,.0f} locked")
            return "\n".join(lines), "Phantom"

    # ══════════════════════════════════════════════════════════
    # CSL WHAT-IF
    # ══════════════════════════════════════════════════════════
    if any(x in q for x in ["what if","simulate","csl","service level",
                              "change service","increase service"]) and \
       any(c.isdigit() for c in question):
        pct_nums = re.findall(r"([0-9]+(?:[.][0-9]+)?)\s*%", question)
        target   = float(pct_nums[0]) if pct_nums else 97.5
        tier     = "All"
        for t in ["premium","high","low","medium"]:
            if t in q: tier = t.title(); break
        r = _dispatch("simulate_csl_what_if", {"target_csl_pct":target,"foot_traffic_tier":tier})
        if r.get("status") == "ok":
            curr = float(r.get("current_safety_stock_units",0))
            sim  = float(r.get("simulated_safety_stock_units",0))
            diff = float(r.get("net_unit_change",0))
            pct  = ((sim-curr)/curr*100) if curr > 0 else 0
            scope = f"{tier} stores" if tier != "All" else "all stores"
            lines = [
                f"**What-If: {target}% Service Level for {scope}**","",
                f"Current safety stock:   **{curr:,.0f} units**",
                f"Simulated safety stock: **{sim:,.0f} units**",
                f"Change: **{'+' if diff>=0 else ''}{diff:,.0f} units ({pct:+.1f}%)**","",
                (f"⚠ {target}% SL means holding {diff:,.0f} more units — fewer stockouts but higher cost."
                 if diff > 0 else
                 f" {target}% SL frees {abs(diff):,.0f} units — lower cost, slightly more stockout risk.")
            ]
            return "\n".join(lines), "What-If"

    # ══════════════════════════════════════════════════════════
    # MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════
    if any(x in q for x in ["wape","model","accuracy","p90","bias","lightgbm",
                              "perform","backtest","how accurate","v2"]) \
            and not any(x in q for x in ["spike","black friday","christmas","category","worst","best","why wape","why did"]):
        lines = []
        if not ms.empty:
            v2 = ms[ms["forecast_method"].str.contains("DemandSense_v2",na=False)]
            b  = ms[ms["forecast_method"].str.contains("MovingAvg30",na=False)]
            if not v2.empty:
                w=float(v2.iloc[0]["wape"]); p=float(v2.iloc[0].get("p90_coverage",0))
                bi=float(v2.iloc[0].get("bias",0)); bw=float(b.iloc[0]["wape"]) if not b.empty else 28.89
                lines = [
                    "**DemandSense v2 — Model Performance**","",
                    f"• WAPE: **{w:.2f}%** (was {bw:.2f}% — {((bw-w)/bw*100):.1f}% better)",
                    f"• P90 coverage: **{p:.2f}%** (target 88–92%)",
                    f"• Bias: {bi:.4f} (near zero = balanced)",
                    f"• Status: **CHAMPION — in production**",
                ]
        bt=D["bt"]
        if not bt.empty:
            old=bt[bt["policy"]=="OLD"]; new=bt[bt["policy"]=="NEW"]
            if not old.empty and not new.empty:
                so=(float(old.iloc[0]["stockout_events"])-float(new.iloc[0]["stockout_events"]))/max(float(old.iloc[0]["stockout_events"]),1)*100
                lu=(float(old.iloc[0]["lost_units"])-float(new.iloc[0]["lost_units"]))/max(float(old.iloc[0]["lost_units"]),1)*100
                lines += ["","**Black Friday + Christmas backtest:**",
                          f"• {so:.1f}% fewer stockout events",f"• {lu:.1f}% fewer lost units"]
        if lines: return "\n".join(lines), "Model Performance"

    # ══════════════════════════════════════════════════════════
    # SAFETY STOCK FORMULA
    # ══════════════════════════════════════════════════════════
    # ── FORMULA / EDUCATION ──────────────────────────────────────
    formula_triggers = [
        "safety stock", "formula", "how does", "z score", "explain",
        "reorder point", "lead time", "replenishment formula",
        "critical mean", "warning mean", "critical vs", "vs warning",
        "what does critical", "what does warning", "difference between",
        "phantom formula", "phantom detection", "alert tier",
        "what is critical", "what is warning", "threshold"]
    if any(x in q for x in formula_triggers):
        if any(x in q for x in ["replenishment", "reorder", "order quantity"]):
            msg = (
                "Replenishment formula:\n\n"
                "Target = mu + Z x sigma\n\n"
                "  mu = average daily demand (LightGBM P50 forecast)\n"
                "  Z  = service level factor (1.96 = 97.5% CSL)\n"
                "  sigma = demand variability (standard deviation)\n\n"
                "Example: mu=34, sigma=6, Z=1.96\n"
                "  Target = 34 + (1.96 x 6) = 45.8 rounded to 46 units\n\n"
                "Old policy used moving average only.\n"
                "New AI policy adds Z x sigma safety buffer on top."
            )
            return msg, "Formula"
        elif any(x in q for x in ["phantom", "ghost", "detection"]):
            msg = (
                "How phantom inventory is detected:\n\n"
                "A product is flagged as phantom when:\n"
                "  1. System shows positive stock (units > 0)\n"
                "  2. Actual sales near zero for 7+ consecutive days\n"
                "  3. LightGBM forecast predicts demand above 5 units/day\n\n"
                "Phantom Score = (1 - actual/forecast) x (mu / max_mu)\n"
                "  Score > 0.65 = High confidence phantom\n"
                "  Score 0.50-0.65 = Medium confidence\n\n"
                "Action: send an associate to physically check the aisle."
            )
            return msg, "Formula"
        elif any(x in q for x in [
                "critical", "warning", "monitor", "threshold",
                "what does", "difference", "tier", "alert"]):
            msg = (
                "Alert tiers explained:\n\n"
                "CRITICAL - 3 days or less of stock remaining\n"
                "  You are inside the supplier lead time window.\n"
                "  Action: place order immediately today.\n\n"
                "WARNING - 4 to 7 days of stock remaining\n"
                "  Small buffer but lead time may eat into it.\n"
                "  Action: place order today.\n\n"
                "MONITOR - 8 to 14 days of stock remaining\n"
                "  Healthy but trending down.\n"
                "  Action: schedule reorder before it hits WARNING.\n\n"
                "OK - More than 14 days. No action needed."
            )
            return msg, "Formula"
        else:
            msg = (
                "Safety stock formula:\n\n"
                "SS = Z x sqrt(LT x sigmaD^2 + D^2 x sigmaLT^2) x SF\n\n"
                "  Z = service level factor (1.96 = 97.5% no stockout)\n"
                "  LT = supplier lead time in days\n"
                "  sigmaD = daily sales variability\n"
                "  D = average daily sales from LightGBM\n"
                "  sigmaLT = delivery time variability\n"
                "  SF = seasonal factor\n\n"
                "CRITICAL = 3 days or less. WARNING = 7 days or less.\n\n"
                "Ask: explain the replenishment formula or explain the phantom formula"
            )
            return msg, "Formula"

    # FALLBACK — LLM with network context
    # ══════════════════════════════════════════════════════════
    if not ns.empty:
        def g(c): return float(ns[c].sum()) if c in ns.columns else 0
        facts = (
            f"Network: ${g('revenue_at_risk')/1e6:.1f}M at risk, "
            f"{int(g('critical_count')):,} CRITICAL, "
            f"{int(g('warning_count')):,} WARNING, "
            f"{len(ns)} stores."
        )
        if not ms.empty:
            v2 = ms[ms["forecast_method"].str.contains("DemandSense_v2",na=False)]
            if not v2.empty:
                facts += f" Model WAPE={float(v2.iloc[0]['wape']):.2f}%."
        ctx = mem_get_context()
        prompt = "\n\n".join(filter(None,[
            ctx,
            f"REAL DATA: {facts}",
            f"QUESTION: {question}",
            ("If this is a follow-up question, use the conversation context above. "
             "If you cannot answer from the data, say: 'I don't have that specific data. "
             "Try: morning briefing for S0001, which stores need attention, or supplier risk.'")
        ]))
        try:
            agent = get_agent(persona)
            if agent and ollama_ok:
                old = agent.history[:]
                agent.history = []
                r   = agent.chat(prompt)
                agent.history = old
                resp = r.get("response","")
                if resp and not resp.strip().startswith("{") and len(resp) > 5:
                    return resp, ""
        except: pass

    # ── WAPE spike / forecast accuracy ─────────────────────────
    if any(x in q for x in ["wape","spike","black friday","christmas","post-christmas",
                              "forecast accuracy","bias","worst category","best category",
                              "why wape","why did wape","model accurate"]):
        wm = D.get("wm", pd.DataFrame())
        acc = D.get("acc", pd.DataFrame())
        NL = chr(10)
        if any(x in q for x in ["spike","black friday","christmas","bias","why wape","why did"]):
            if not wm.empty:
                avg_w = wm["wape"].mean()
                spikes = wm[wm["wape"] > avg_w*1.15].sort_values("wape",ascending=False)
                known = {
                    "2025-11-24/2025-11-30": ("Black Friday",-1.719,"Model over-predicted demand. Stores cleared stock fast — actual sales hit zero while forecast stayed high."),
                    "2025-12-22/2025-12-28": ("Christmas",-1.245,"Holiday surge then sharp drop after Dec 25. Model kept forecasting high after demand collapsed."),
                    "2025-12-29/2026-01-04": ("Post-Christmas",-1.823,"Worst week. Demand collapsed post-holiday. Model trained on high December volumes kept predicting high."),
                }
                facts = []
                for _, r in spikes.iterrows():
                    if r["week"] in known:
                        name,bias,detail = known[r["week"]]
                        facts.append(f"**{name}** ({r['week']}): WAPE {r['wape']:.1f}% vs avg {avg_w:.1f}%, Bias {r['bias']:.2f} — {detail}")
                    else:
                        facts.append(f"{r['week']}: WAPE {r['wape']:.1f}%, Bias {r['bias']:.2f}")
                if facts:
                    resp = "**WAPE Spike Analysis**" + NL + NL + NL.join(facts)
                    resp += NL + NL + f"Network avg WAPE: {avg_w:.1f}% across {len(wm)} weeks."
                    return resp, "Forecast"
        if any(x in q for x in ["category","worst","best","accurate","accuracy"]):
            if not acc.empty:
                worst = acc.loc[acc["wape"].idxmax()]
                best  = acc.loc[acc["wape"].idxmin()]
                resp = "**Forecast Accuracy by Category**" + NL + NL
                resp += f"**Worst:** {worst['category']} — WAPE {worst['wape']:.1f}%, Bias {worst['bias']:.3f}, {int(worst.get('rows',0)):,} rows" + NL
                resp += f"**Best:** {best['category']} — WAPE {best['wape']:.1f}%, Bias {best['bias']:.3f}" + NL + NL
                for _,r in acc.sort_values("wape",ascending=False).iterrows():
                    resp += f"  {r['category']}: {r['wape']:.1f}% WAPE · {int(r.get('rows',0)):,} rows" + NL
                return resp, "Forecast"
        if not wm.empty:
            avg_w = wm["wape"].mean()
            return f"Model WAPE: {avg_w:.1f}% avg. Highest: {wm['wape'].max():.1f}% · Lowest: {wm['wape'].min():.1f}%", "Forecast"

    return "", ""


# ── Casual replies ────────────────────────────────────────────
CASUAL_RE = re.compile(
    r"^(hi+|hey+|hello+|howdy|yo+|sup|what'?s? ?up|how ?are ?you|"
    r"good (morning|afternoon|evening)|thank(s| you)|ok(ay)?|cool|"
    r"great|got it|nice|sure|lol|haha|wow|dude|bro|mate|bye|cya)[\s!?.]*$",
    re.IGNORECASE)

# end of answer()

# end of answer()

def is_casual(t): return bool(CASUAL_RE.match(t.strip()))

def casual_reply(t):
    t = t.lower()
    if any(x in t for x in ["thank","thanks"]): return random.choice(["You're welcome!","Happy to help!"])
    if any(x in t for x in ["bye","cya","later"]): return "See you later!"
    if "good morning" in t: return "Good morning! Your network has 6,544 critical SKUs right now. Want a briefing?"
    return random.choice(["Hey! What would you like to check today?",
                          "Hi there! Ask me about a store, product, or the network.",
                          "Hello! Ready when you are."])


# ── Ollama check ──────────────────────────────────────────────
D = load()
ollama_ok = check_ollama()

for k, v in [("messages",[]),("persona","supply_chain_planner")]:
    if k not in st.session_state: st.session_state[k] = v

# SIDEBAR
with st.sidebar:
    st.markdown(f"""<div style="padding:16px 8px 14px">
      <div style="font-size:16px;font-weight:800;color:{WHITE}">HyperShelf AI</div>
      <div style="font-size:9px;color:{TEAL2};text-transform:uppercase;letter-spacing:2px;margin-top:2px">DemandSense v2 · LLaMA 3.2</div>
    </div>""", unsafe_allow_html=True)
    dot = GREEN if ollama_ok else RED
    lbl = "Ollama connected" if ollama_ok else "Ollama offline — run: ollama serve"
    st.markdown(f'<div style="background:{CARD};border:1px solid {dot}33;border-radius:6px;padding:8px 12px;margin-bottom:12px;font-size:10px;color:{dot}">{"●" if ollama_ok else "○"} {lbl}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;color:{GRAY};text-transform:uppercase;letter-spacing:1px;font-weight:700;margin:12px 0 6px">Persona</div>', unsafe_allow_html=True)
    persona = st.radio("p",["supply_chain_planner","store_manager"],
        format_func=lambda x:"Supply Chain Planner" if x=="supply_chain_planner" else "Store Manager",
        label_visibility="collapsed", key="persona_sel")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;color:{GRAY};text-transform:uppercase;letter-spacing:1px;font-weight:700;margin:12px 0 6px">Quick Questions</div>', unsafe_allow_html=True)
    quick = [
        ("Network status",           "What is going on across the network today?"),
        ("Top urgent stores",        "Which stores need immediate attention?"),
        ("Morning briefing S0001",   "Morning briefing for S0001"),
        ("Morning briefing S0064",   "Morning briefing for S0064"),
        ("Replenishment S0001 P00055","Replenishment for S0001 P00055"),
        ("Forecast S0001 P00055",    "Forecast for S0001 P00055"),
        ("Stockout risk S0001 P00055","Stockout risk for S0001 P00055"),
        ("Phantom inventory",        "Any phantom inventory issues?"),
        ("Suppliers HIGH RISK",      "Which suppliers are HIGH RISK right now?"),
        ("Suppliers LOW RISK",       "Which suppliers are LOW RISK right now?"),
        ("Suppliers MEDIUM RISK",    "Which suppliers are MEDIUM RISK?"),
        ("What-If 99% CSL",          "What if we increase service level to 99%?"),
        ("What-If Premium 98%",      "What if we change CSL to 98% for Premium stores only?"),
        ("Model performance",        "How is the forecast model performing?"),
        ("Safety stock formula",     "Explain the safety stock formula"),
    ]
    for lbl_, q_ in quick:
        if st.button(lbl_, key=f"qq_{lbl_}"):
            st.session_state["_pending"] = q_
            st.rerun()
    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Clear conversation", key="clear"):
        st.session_state.messages = []
        _memory.clear()
        st.rerun()

# HEADER
st.markdown(f"""<div style="background:{NAVY};border-bottom:1px solid {BORDER};padding:14px 24px;margin:-1rem -2rem 1.2rem -2rem;display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-size:16px;font-weight:800;color:{WHITE}">HyperShelf AI Assistant</div>
    <div style="font-size:11px;color:{GRAY};margin-top:2px">DemandSense v2 · Powered by LLaMA 3.2 3B · 478 stores · 38,240 alerts · Real-time Retail Intelligence</div>
  </div>
  <div style="font-size:11px;color:{GRAY}">{datetime.now().strftime("%A, %B %d · %I:%M %p")}</div>
</div>""", unsafe_allow_html=True)

# KPI STRIP


# CHAT
msgs_html = '<div class="chat-area">'
if not st.session_state.messages:
    _w = (
        '<div style="margin-bottom:16px;">'
        f'<div style="background:linear-gradient(135deg,#0F2A3F,#0D3D4A);border:1px solid {TEAL};border-radius:12px;box-shadow:0 0 20px {TEAL}22;padding:24px 28px;">'
        f'<div style="font-size:20px;font-weight:800;color:#C0C8D8;margin-bottom:2px;letter-spacing:0.3px;">HyperShelf <span style="color:#14B8A6;">AI</span></div>' +
        f'<div style="font-size:11px;color:#5E7A8A;text-transform:uppercase;letter-spacing:2px;margin-bottom:14px;">Real-time retail intelligence</div>'
        f'<div style="border-top:1px solid #2D4A6A;margin-bottom:14px;"></div>'
        f'<div style="font-size:13px;color:#94A3B8;line-height:1.9;margin-bottom:4px;"><span style="color:#14B8A6;font-weight:600;">&#8594;</span>  Decide <strong style="color:#E2E8F0;">what to reorder</strong> before shelves go empty</div>'
        f'<div style="font-size:13px;color:#94A3B8;line-height:1.9;margin-bottom:4px;"><span style="color:#14B8A6;font-weight:600;">&#8594;</span>  Reduce <strong style="color:#E2E8F0;">stockouts</strong> across 478 stores</div>'
        f'<div style="font-size:13px;color:#94A3B8;line-height:1.9;margin-bottom:14px;"><span style="color:#14B8A6;font-weight:600;">&#8594;</span>  Manage <strong style="color:#E2E8F0;">store risk</strong> with real-time alerts</div>'
        f'<div style="border-top:1px solid #2D4A6A;margin-bottom:12px;"></div>'
        f'<div style="font-size:13px;color:#94A3B8;">What can I help you with today?  <em style="color:#14B8A6;">what should I reorder today</em></div>'
        f'</div></div>'
    )
    msgs_html += _w

for m in st.session_state.messages:
    ts = m.get("ts",""); mode = m.get("mode","")
    if m["role"] == "user":
        txt = m["content"].replace("<","&lt;").replace(">","&gt;")
        msgs_html += f"""<div class="bubble-user-wrap"><div>
          <div class="bubble-label" style="text-align:right">You · {ts}</div>
          <div class="bubble-user">{txt}</div></div></div>"""
    else:
        mc = {"Store Briefing":TEAL,"Replenishment":AMBER,"Forecast":TEAL2,
              "Stockout Risk":RED,"Network Status":TEAL,"Store Ranking":AMBER,
              "Store Risk":AMBER,"Model Performance":GREEN,"Phantom":PURPLE,
              "Formula":TEAL,"Supplier Risk":RED,"What-If":TEAL2}.get(mode,GRAY)
        badge = (f'<span class="mode-badge" style="background:{mc}22;color:{mc};border:1px solid {mc}44">{mode}</span><br>'
                 if mode else "")
        raw = m["content"]
        raw = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', raw)
        raw = re.sub(r'```(.*?)```', r'<code style="background:#0a1628;padding:6px 10px;border-radius:4px;display:block;margin:5px 0;font-family:monospace;font-size:11px">\1</code>', raw, flags=re.DOTALL)
        raw = raw.replace("\n","<br>")
        msgs_html += f"""<div class="bubble-ai-wrap"><div>
          <div class="bubble-label">HyperShelf AI · {ts}</div>
          {badge}<div class="bubble-ai">{raw}</div></div></div>"""

msgs_html += "</div>"
st.markdown(msgs_html, unsafe_allow_html=True)

# INPUT
pending = st.session_state.pop("_pending", None)
# Suggested prompts when chat is empty
if not st.session_state.get('messages'):
    _pcols = st.columns(3)
    _prompts = [
        'Morning briefing for S0064',
        'Which stores need attention?',
        'Replenishment for S0001 P00113',
        'Which suppliers are HIGH RISK?',
        'Any phantom inventory?',
        'What if service level is 99%?',
    ]
    for _pi, _pr in enumerate(_prompts):
        with _pcols[_pi % 3]:
            if st.button(_pr, key=f'sugg_{_pi}', use_container_width=True):
                st.session_state['_pend_prompt'] = _pr
                st.rerun()
_pend = st.session_state.pop('_pend_prompt', None)
user_in = st.chat_input("Ask about a store, product, or the network...", key="cin")
user_in = user_in or _pend
user_in = user_in or pending

if user_in:
    ts_now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role":"user","content":user_in,"ts":ts_now})

    if is_casual(user_in):
        resp = casual_reply(user_in)
        mode = ""
    else:
        mem_add("user", user_in)
        resp, mode = answer(user_in, D, persona, ollama_ok)
        if not resp:
            resp = ("I only have data for your 478 stores and products. "
                    "Try: *morning briefing for S0001*, *which stores need attention*, "
                    "*supplier risk*, or *forecast for S0001 P00055*.")
            mode = ""
        mem_add("assistant", resp)

    st.session_state.messages.append({"role":"assistant","content":resp,"mode":mode,"ts":ts_now})
    st.rerun()