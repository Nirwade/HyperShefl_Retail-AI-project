# ─────────────────────────────────────────────────────────────────
# src/agent.py
# DemandSense LLM agent — wraps Ollama with tool calling.
# Uses native ollama Python client (no LangChain).
# ─────────────────────────────────────────────────────────────────
import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

import json
import ollama
from src.tools import TOOL_SCHEMAS, dispatch_tool
from src.prompts import STORE_MANAGER_PROMPT, SUPPLY_CHAIN_PROMPT, ESCALATION_RULES

MODEL_NAME    = "llama3.2:3b"
MAX_HISTORY   = 20   # keep last N messages to avoid context overflow


class DemandSenseAgent:
    """
    Conversational agent for retail shelf optimization.
    Supports two personas: store_manager and supply_chain_planner.
    """

    def __init__(self, persona: str = "store_manager"):
        assert persona in ("store_manager", "supply_chain_planner"), \
            "persona must be 'store_manager' or 'supply_chain_planner'"

        self.persona = persona
        self.history = []

        # Select system prompt
        base_prompt = STORE_MANAGER_PROMPT if persona == "store_manager" \
                      else SUPPLY_CHAIN_PROMPT
        self.system_prompt = base_prompt + "\n\n" + ESCALATION_RULES

        print(f"[Agent] DemandSenseAgent ready — persona: {persona} · model: {MODEL_NAME}")

    def chat(self, user_message: str) -> dict:
        """
        Send a message and get a response.
        Returns: {response_text, tool_calls_made, requires_approval, escalation_reason}
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Trim history to avoid context overflow
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        tool_calls_made = []
        requires_approval = False
        escalation_reason = None

        # ── First call — let model decide which tool to call ──────
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": self.system_prompt}] + self.history,
            tools=TOOL_SCHEMAS,
        )

        msg = response["message"]

        # ── Tool calling loop ─────────────────────────────────────
        while msg.get("tool_calls"):
            tool_results = []

            for tc in msg["tool_calls"]:
                tool_name = tc["function"]["name"]
                tool_args = tc["function"]["arguments"]

                # Parse args if string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Dispatch to ML function
                result = dispatch_tool(tool_name, tool_args)

                # Check escalation
                if result.get("requires_approval"):
                    requires_approval = True
                    escalation_reason = (
                        f"Order value ${result.get('order_value_usd', 0):,.2f} "
                        f"for {result.get('product_name', tool_args.get('sku_id', ''))} "
                        f"at {result.get('store_name', tool_args.get('store_id', ''))} "
                        f"requires approval."
                    )

                tool_calls_made.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result_summary": _summarise_result(result),
                })

                tool_results.append({
                    "role":    "tool",
                    "content": json.dumps(result, default=str),
                })

            # Add assistant message with tool calls to history
            self.history.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": msg["tool_calls"]})
            # Add tool results
            self.history.extend(tool_results)

            # Second call — let model generate natural language response
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": self.system_prompt}] + self.history,
                tools=TOOL_SCHEMAS,
            )
            msg = response["message"]

        # Final text response — strip raw JSON tool calls that leaked
        response_text = msg.get("content", "")
        if not response_text or response_text.strip().startswith('{"name":'):
            # LLM printed tool call as text instead of executing — recover gracefully
            response_text = ("I processed your request using live data. "
                             "Could you rephrase or try a more specific question? "
                             "Example: 'morning briefing for S0001' or "
                             "'replenishment for S0001 P00055'")
        self.history.append({"role": "assistant", "content": response_text})

        return {
            "response":         response_text,
            "tool_calls_made":  tool_calls_made,
            "requires_approval": requires_approval,
            "escalation_reason": escalation_reason,
            "persona":          self.persona,
        }

    def reset(self):
        """Clear conversation history."""
        self.history = []
        print("[Agent] Conversation history cleared.")


def _summarise_result(result: dict) -> str:
    """One-line summary of a tool result for logging."""
    if result.get("status") == "error":
        return f"ERROR: {result.get('message')}"
    if "risk_tier" in result:
        return f"Risk: {result['risk_tier']} ({result['risk_score']}) for {result.get('product_name','')}"
    if "recommended_order_qty" in result:
        return f"Order {result['recommended_order_qty']} units of {result.get('product_name','')} — approval={'YES' if result.get('requires_approval') else 'no'}"
    if "alerts" in result:
        return f"{len(result['alerts'])} phantom alerts found"
    if "daily_forecast" in result:
        return f"Forecast avg {result.get('avg_forecast',0)} units/day for {result.get('product_name','')}"
    return "OK"