"""
Conversational agent. Gemini interprets the question, chooses tools and writes the explanation;
every number comes from the deterministic tools. A post-check flags numbers in the answer that
no tool returned, so an invented or miscalculated figure is visible to the analyst.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from .tools import AirportTools

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s*(%|million|billion|[kKmMbB])(?![a-zA-Z]))?")
SCALE = {"k": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9}

SYSTEM_PROMPT = """You are an airport investment analyst assistant for a firm that invests in US airport \
modernization projects. You help analysts find airports where terminal renovation or expansion is most likely to \
pay off because flight and passenger demand is growing or capacity is under pressure.

DATA AVAILABLE (through tools only)
{data_context}

HOW TO WORK
- Use tools for every fact. Resolve place names with resolve_airports unless the user gave an IATA code.
- Terminal expansion candidates: rank_airports(region=..., score_type="expansion").
- Congestion comparisons: compare_airports (raw delay, cancellation and taxi-out rates) and the congestion_index.
- Long-haul share: get_long_haul_share with its default thresholds unless the user asks for others (there is no
  single industry definition, so name the thresholds). Shares are of departures; give passenger-service and
  cargo-inclusive figures separately when they differ.
- Unmet demand: get_demand_pressure. It is a proxy (index + seat shortfall vs a target load factor), not an
  observed quantity. Explain "why" using the component drivers.
- Use explain_score when the user asks why a score is high or low; get_data_provenance for data questions;
  get_live_status only when the user asks about current conditions.
- Follow-up questions refer to airports and results from earlier in the conversation.

NUMBERS
- Only state numbers that appear in tool results (rounding is fine). Do not compute new figures yourself;
  differences are provided by compare_airports. If a value is null, say it is not available.

ANSWER FORMAT (markdown, concise)
1. **Answer** - one or two sentences that directly answer the question.
2. **Key figures** - a table for rankings and comparisons, otherwise bullets.
3. **Why / how it was calculated** - which components drove the result (weights and percentiles).
4. **Assumptions & limitations** - the data period, data_status notes (e.g. partial On-Time coverage) and the
   relevant caveat from the tool output. Never present a screening score as proof of profitability.
Do not add decorative emoji or unsolicited follow-up suggestions. For simple follow-ups, answer directly
without repeating the full structure; retain the relevant period and limitations.
"""


@dataclass
class AgentReply:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)


def tool_numbers(tool_calls: list[dict]) -> list[float]:
    """Every number appearing in tool results (including inside strings); fractions also as percentages."""
    numbers = []
    for match in NUMBER_PATTERN.finditer(json.dumps([c["result"] for c in tool_calls])):
        value = float(match.group(1).replace(",", ""))
        numbers.append(value)
        if value <= 1:
            numbers.append(value * 100)
    return numbers


def find_unverified_numbers(answer: str, known: list[float]) -> list[str]:
    """Numbers in the answer that do not match any tool value within their displayed rounding."""
    unverified = []
    for match in NUMBER_PATTERN.finditer(answer):
        digits, unit = match.group(1), (match.group(2) or "").lower()
        value = float(digits.replace(",", ""))
        is_integer = "." not in digits
        if is_integer and not unit and (value <= 12 or 1990 <= value <= 2100):
            continue  # ranks, counts, months and years
        scale = SCALE.get(unit, 1.0)
        decimals = len(digits.split(".")[1]) if not is_integer else 0
        tolerance = 0.5 * 10 ** -decimals * scale + 1e-9
        if not any(abs(value * scale - k) <= tolerance for k in known):
            unverified.append(match.group(0).strip())
    return list(dict.fromkeys(unverified))


class AirportAgent:
    def __init__(self, api_key: str, tools: AirportTools, model: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")
        self.tools = tools
        self.tool_log: list[dict] = []
        periods = tools.metrics.periods()
        data_context = (
            f"- Analysis window (12 months): {periods['current_window']}; YoY vs {periods['prior_year_window']}; "
            f"3-year CAGR vs {periods['three_years_back_window']}.\n"
            f"- On-Time Performance months inside the window: {', '.join(periods['otp_months_in_current_window']) or 'none'}.\n"
            f"- Rankable airports: {len(tools.scored.table)} US airports with at least "
            f"{tools.config['analysis']['min_passengers_for_ranking']:,} passengers in the window."
        )
        self._client = genai.Client(api_key=api_key)  # must outlive __init__, or its HTTP session is closed
        declarations = [types.FunctionDeclaration(name=s["name"], description=s["description"],
                                                  parameters_json_schema=s["parameters"]) for s in tools.specs]
        self._chat = self._client.chats.create(model=model, config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(data_context=data_context),
            tools=[types.Tool(function_declarations=declarations)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        ))

    def ask(self, question: str) -> AgentReply:
        calls: list[dict] = []
        response = self._send(question)
        for _ in range(MAX_TOOL_ROUNDS):
            if not response.function_calls:
                break
            parts = []
            for call in response.function_calls:
                args = dict(call.args or {})
                result = self.tools.call(call.name, args)
                calls.append({"tool": call.name, "args": args, "result": result})
                parts.append(types.Part.from_function_response(name=call.name, response={"result": result}))
            response = self._send(parts)
        self.tool_log.extend(calls)

        if response.function_calls:
            return AgentReply("I could not finish within the tool-call limit. Please ask a narrower question.", calls)
        text = response.text or "The model returned an empty response."
        unverified = find_unverified_numbers(text, tool_numbers(self.tool_log))
        if unverified:
            text += ("\n\n> Verification warning: These numbers were not found in any tool output and may be model-generated: "
                     + ", ".join(unverified))
        return AgentReply(text, calls, unverified)

    def _send(self, message):
        """Send to Gemini, retrying transient rate-limit / overload errors."""
        for attempt in range(4):
            try:
                return self._chat.send_message(message)
            except Exception as e:  # the SDK raises several error types; only transient ones are retried
                text = str(e)
                transient = any(s in text for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
                if not transient or attempt == 3:
                    raise
                hint = re.search(r"retry in ([\d.]+)s", text, re.I)
                delay = min(float(hint.group(1)) + 1 if hint else 5 * (attempt + 1), 65)
                logger.warning("Gemini transient error, retrying in %.0fs", delay)
                time.sleep(delay)
