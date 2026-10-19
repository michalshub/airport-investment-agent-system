"""Streamlit chat interface for the Airport Investment Intelligence Agent."""

import logging

import pandas as pd
import streamlit as st

from src.agent import AirportAgent
from src.metrics import AirportMetrics, Warehouse
from src.scoring import ScoringEngine
from src.tools import AirportTools
from src.utils import get_config, setup_environment

logging.basicConfig(level=logging.INFO)

EXAMPLE_QUESTIONS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

COMPONENT_LABELS = {
    "delay_rate_15_pct": "Departures delayed 15+ minutes",
    "avg_dep_delay_min": "Average departure delay",
    "cancellation_rate_pct": "Cancellation rate",
    "avg_taxi_out_min": "Average taxi-out time",
    "load_factor_pct": "Load factor",
    "pax_minus_seat_growth_pp": "Passenger vs. seat growth",
    "congestion_index": "Congestion index",
    "pax_cagr_3y_pct": "Passenger growth (3-year CAGR)",
    "departure_growth_yoy_pct": "Departure growth (YoY)",
    "passengers_ttm": "Departing passengers (12 months)",
}

st.set_page_config(page_title="Airport Investment Agent", layout="wide")


@st.cache_resource(show_spinner="Loading aviation data...")
def load_tools() -> AirportTools:
    config = get_config()
    return AirportTools(AirportMetrics(Warehouse.load(), config), ScoringEngine(config), config)


def new_agent(tools: AirportTools) -> AirportAgent:
    env = setup_environment()
    return AirportAgent(env["api_key"], tools, env["model"])


try:
    tools = load_tools()
except FileNotFoundError as e:
    st.error(f"{e}\n\nBuild the data first: `python -m src.etl`")
    st.stop()

if "agent" not in st.session_state:
    try:
        st.session_state.agent = new_agent(tools)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.session_state.messages = []

periods = tools.metrics.periods()
with st.sidebar:
    st.header("Data in use")
    st.markdown(
        f"**Analysis window:** {periods['current_window']}  \n"
        f"**YoY comparison:** {periods['prior_year_window'] or 'not available'}  \n"
        f"**3-year CAGR base:** {periods['three_years_back_window'] or 'not available'}  \n"
        f"**Rankable airports:** {len(tools.scored.table)} "
        f"(≥ {tools.config['analysis']['min_passengers_for_ranking']:,} passengers)"
    )
    otp_months = periods["otp_months_in_current_window"]
    full_months = tools.config["analysis"]["full_coverage_otp_months"]
    if len(otp_months) < full_months:
        st.warning(f"On-Time data covers {len(otp_months)} of {full_months} months in the window. "
                   "Congestion results are partial and may be seasonal.")

    st.header("Scoring weights")
    for score_type, weights in tools.scoring.weights.items():
        with st.expander(score_type.replace("_", " ").title()):
            frame = pd.DataFrame({"Component": [COMPONENT_LABELS[key] for key in weights],
                                  "Weight": [f"{value:.0%}" for value in weights.values()]})
            st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("Components use national percentiles among rankable airports.")

    if st.button("New conversation", width="stretch"):
        st.session_state.agent = new_agent(tools)
        st.session_state.messages = []
        st.rerun()

st.title("Airport Investment Intelligence Agent")
st.caption("US airport expansion screening, grounded in BTS traffic and on-time data.")

if not st.session_state.messages:
    st.markdown("**Example questions**")
    columns = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if columns[i % 2].button(question, width="stretch"):
            st.session_state.pending = question
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tool_calls"):
            with st.expander("Sources and calculations"):
                for call in message["tool_calls"]:
                    st.markdown(f"`{call['tool']}` {call['args']}")
                    st.json(call["result"], expanded=False)

prompt = st.chat_input("Ask about airport investment opportunities...") or st.session_state.pop("pending", None)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Analyzing..."):
        try:
            reply = st.session_state.agent.ask(prompt)
            st.session_state.messages.append(
                {"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
        except Exception:  # keep technical details in logs, not in analyst-facing responses
            logging.exception("Agent error")
            st.session_state.messages.append(
                {"role": "assistant", "content": "The request could not be completed. Please try again."})
    st.rerun()

st.divider()
st.caption("Screening tool only: not a profitability forecast.")
