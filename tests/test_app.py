"""Offline UI checks with a synthetic warehouse and a stubbed conversational agent."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.agent import AgentReply

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def app(monkeypatch, warehouse, test_config):
    class StubAgent:
        def __init__(self, api_key, tools, model):
            self.tools = tools

        def ask(self, question):
            if question == "Trigger failure":
                raise RuntimeError("Internal diagnostic details")
            result = self.tools.call("rank_airports", {"region": "New England"})
            return AgentReply("Synthetic screening result.", [
                {"tool": "rank_airports", "args": {"region": "New England"}, "result": result},
            ])

    monkeypatch.setattr("src.agent.AirportAgent", StubAgent)
    monkeypatch.setattr("src.metrics.Warehouse.load", lambda: warehouse)
    monkeypatch.setattr("src.utils.get_config", lambda: test_config)
    monkeypatch.setattr("src.utils.setup_environment", lambda: {"api_key": "", "model": "stub"})
    st.cache_resource.clear()
    yield AppTest.from_file(str(APP_PATH)).run(timeout=20)
    st.cache_resource.clear()


def test_initial_ui_shows_examples_coverage_and_readable_weights(app):
    assert not app.exception
    assert app.title[0].value == "Airport Investment Intelligence Agent"
    assert any(item.value == "**Example questions**" for item in app.markdown)
    assert any("On-Time data covers 2 of 12 months" in item.value for item in app.warning)
    assert "Component" in app.dataframe[0].value.columns
    assert "Departures delayed 15+ minutes" in app.dataframe[0].value["Component"].tolist()
    assert any(item.value == "Screening tool only: not a profitability forecast." for item in app.caption)


def test_chat_keeps_evidence_and_new_conversation_clears_messages(app):
    next(button for button in app.button if button.label.startswith("Which airports")).click().run()
    assert not app.exception
    assert app.session_state.messages[0]["role"] == "user"
    assert app.session_state.messages[1]["content"] == "Synthetic screening result."
    assert any(item.label == "Sources and calculations" for item in app.expander)
    next(button for button in app.button if button.label == "New conversation").click().run()
    assert not app.exception
    assert app.session_state.messages == []


def test_request_failure_is_visible_without_internal_details(app, caplog):
    app.chat_input[0].set_value("Trigger failure").run()
    assert not app.exception
    assert app.session_state.messages[-1]["content"] == (
        "The request could not be completed. Please try again.")
    assert "Internal diagnostic details" in caplog.text
    assert not any("Internal diagnostic details" in item.value for item in app.markdown)
