"""
whatif_ai.py — What-If Simulator AI intelligence
Python computes exact scenario data, LLaMA explains strategy
"""
import requests

C_TEAL="#0D9488";C_GREEN="#10B981";C_RED="#EF4444";C_AMBER="#F59E0B"
C_GRAY="#94A3B8";C_WHITE="#E2E8F0"

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

def answer(question, mu, sig, lt, lt_extra, z_val, unit_cost, holding_days, rel, sku_id, store_id):
    import numpy as np
    from scipy import stats as sc

    q = question.lower()
    NL = "\n"

    eff_mu   = mu
    eff_lt   = lt + lt_extra
    eff_sig  = sig
    ss_curr  = z_val * eff_sig * np.sqrt(eff_lt)
    rop_curr = eff_mu * eff_lt + ss_curr
    sl_curr  = float(sc.norm.cdf(z_val)) * 100
    carry    = ss_curr * unit_cost * holding_days / 30

    scenarios = [
        ("95% SL",  1.645),
        ("97.5% SL", 1.960),
        ("99% SL",  2.326),
    ]
    sc_data = []
    for label, z in scenarios:
        ss_ = z * eff_sig * np.sqrt(eff_lt)
        rop_ = eff_mu * eff_lt + ss_
        carry_ = ss_ * unit_cost * holding_days / 30
        sl_ = float(sc.norm.cdf(z)) * 100
        sc_data.append({"label":label,"z":z,"ss":ss_,"rop":rop_,"carry":carry_,"sl":sl_})

    # ── Explain Z score ───────────────────────────────────────
    if any(x in q for x in ["z score","what is z","explain z","z mean","z value"]):
        facts = (
            f"Current Z score: {z_val:.2f} → service level {sl_curr:.1f}%\n"
            f"  Z=1.645 → 95% service level (stockout 1 in 20 cycles)\n"
            f"  Z=1.960 → 97.5% service level (stockout 1 in 40 cycles)\n"
            f"  Z=2.326 → 99% service level (stockout 1 in 100 cycles)\n"
            f"Current SS: {ss_curr:.1f}u | ROP: {rop_curr:.1f}u"
        )
        prompt = (
            "You are a retail supply chain expert explaining safety stock to a store manager.\n"
            "EXACT DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Explain what Z={z_val:.2f} means in plain English — what happens 1 in N cycles. "
            "Explain why higher Z means more safety stock but also higher holding cost. "
            "Recommend whether the current setting is right for a retail environment. Use the exact numbers."
        )
        return _llm(prompt) or facts

    # ── Supplier delay impact ─────────────────────────────────
    if any(x in q for x in ["supplier","delay","slow","lead time","late","slower"]):
        days_list = [0, 3, 7, 14]
        delay_lines = []
        for d in days_list:
            ss_d = z_val * eff_sig * np.sqrt(lt + d)
            rop_d = eff_mu * (lt + d) + ss_d
            carry_d = ss_d * unit_cost * holding_days / 30
            delta_ss = ss_d - ss_curr
            delay_lines.append(
                f"  +{d} days delay: SS {ss_d:.1f}u (+{delta_ss:.1f}u), "
                f"ROP {rop_d:.1f}u, Carrying cost ${carry_d:,.0f}")

        facts = (
            f"Supplier delay impact for {sku_id} at {store_id} "
            f"(base LT {lt:.0f}d, current delay {lt_extra}d):\n" +
            NL.join(delay_lines)
        )
        prompt = (
            "You are a retail procurement expert.\n"
            "EXACT DELAY IMPACT DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            f"Start with the current situation: {lt_extra} days extra delay, SS {ss_curr:.1f}u.\n"
            "Explain how each additional week of delay compounds safety stock requirements. "
            "Recommend the maximum acceptable supplier delay before dual-sourcing is needed. "
            "Use the exact numbers."
        )
        return _llm(prompt) or facts

    # ── 95 vs 97.5 vs 99 comparison ──────────────────────────
    if any(x in q for x in ["95","97","99","compare","scenario","service level","three"]):
        sc_lines = [
            f"  {r['label']}: Z={r['z']:.3f}, SS {r['ss']:.1f}u, "
            f"ROP {r['rop']:.1f}u, SL {r['sl']:.1f}%, Carrying ${r['carry']:,.0f}"
            for r in sc_data]
        diff_ss = sc_data[2]["ss"] - sc_data[0]["ss"]
        diff_carry = sc_data[2]["carry"] - sc_data[0]["carry"]
        facts = (
            f"3-scenario comparison for {sku_id} at {store_id}:\n" +
            NL.join(sc_lines) +
            f"\nGoing from 95% to 99% SL requires {diff_ss:.1f}u more safety stock "
            f"and ${diff_carry:,.0f} more carrying cost."
        )
        prompt = (
            "You are a retail inventory strategist.\n"
            "EXACT SCENARIO DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            "Explain the trade-off between service level and carrying cost. "
            "For a retail store, recommend which service level makes most sense and why. "
            "Use the exact SS, ROP, and cost numbers. Be specific."
        )
        return _llm(prompt) or facts

    # ── Apply to Premium stores ───────────────────────────────
    if any(x in q for x in ["premium","apply","all stores","tier","rollout"]):
        facts = (
            f"Current settings: Z={z_val:.2f} ({sl_curr:.1f}% SL), "
            f"SS {ss_curr:.1f}u, ROP {rop_curr:.1f}u, Carrying ${carry:,.0f}/SKU/month\n"
            f"If applied to all Premium stores (est. 120 stores × 80 SKUs = 9,600 SKUs):\n"
            f"  Total additional safety stock: {ss_curr*9600:,.0f}u\n"
            f"  Total carrying cost: ${carry*9600:,.0f}/month"
        )
        prompt = (
            "You are a retail VP of Supply Chain.\n"
            "EXACT ROLLOUT DATA:\n" + facts + "\n\n"
            "Write 3-4 sentences as one paragraph. No bullet points. No headings.\n"
            "Explain what applying these settings to all Premium stores means in total cost and inventory. "
            "Recommend whether to roll out to all Premium stores or test on a subset first. "
            "Use the exact numbers."
        )
        return _llm(prompt) or facts

    # ── Fallback ──────────────────────────────────────────────
    facts = (
        f"Current simulation: {sku_id} at {store_id}\n"
        f"Z={z_val:.2f} → {sl_curr:.1f}% SL | SS {ss_curr:.1f}u | "
        f"ROP {rop_curr:.1f}u | LT {eff_lt:.0f}d | Carrying ${carry:,.0f}"
    )
    ans = _llm(
        "You are a retail inventory expert.\n"
        "Current simulation: " + facts + "\n"
        "Answer in 3 sentences using retail inventory expertise.\n"
        "Question: " + question
    )
    return ans or "Try: explain Z score · supplier delay impact · compare 95 vs 97.5 vs 99% · apply to Premium stores"