"""Scoring: percentiles, missing-data reweighting, order-independent ranking and real stability testing."""

import pandas as pd
import pytest

from src.scoring import ScoredAirports, ScoringEngine, percentile_rank, weighted_score


def test_percentile_rank_keeps_missing_values_missing():
    result = percentile_rank(pd.Series([10, 20, None, 40]))
    assert result.tolist()[:2] == pytest.approx([100 / 3, 200 / 3])
    assert pd.isna(result.iloc[2])
    assert result.iloc[3] == pytest.approx(100)


def test_weighted_score_rescales_weights_when_a_component_is_missing():
    percentiles = pd.DataFrame({"a": [80.0, 80.0], "b": [40.0, None]}, index=["X", "Y"])
    score, coverage = weighted_score(percentiles, {"a": 0.75, "b": 0.25})
    assert score["X"] == pytest.approx(70)
    assert score["Y"] == pytest.approx(80)      # not imputed: only 'a' is used
    assert coverage["Y"] == pytest.approx(0.75)


def test_contributions_sum_to_score(tools):
    explanation = tools.scoring.explain(tools.scored, "AAA", "expansion")
    total = sum(c["contribution_points"] or 0 for c in explanation["components"])
    assert total == pytest.approx(explanation["score"])


def test_ranking_does_not_depend_on_input_order(tools):
    forward = tools.scoring.rank(tools.scored, ["AAA", "BBB", "CCC", "DDD"], "expansion")
    backward = tools.scoring.rank(tools.scored, ["DDD", "CCC", "BBB", "AAA"], "expansion")
    assert forward.index.tolist() == backward.index.tolist()


def _engine_with(percentiles: dict) -> tuple[ScoringEngine, ScoredAirports]:
    config = {"analysis": {"full_coverage_otp_months": 12},
              "congestion_weights": {"a": 0.5, "b": 0.5}, "demand_pressure_weights": {"a": 1.0},
              "expansion_weights": {"a": 0.5, "b": 0.5},
              "stability": {"weight_perturbations_pct": [-20, 20], "top_n": 1}}
    frame = pd.DataFrame(percentiles)
    return ScoringEngine(config), ScoredAirports(table=frame, percentiles=frame)


def test_stability_detects_a_fragile_top_position():
    engine, scored = _engine_with({"a": {"X": 90.0, "Y": 50.0}, "b": {"X": 10.0, "Y": 48.0}})
    stability = engine.stability(scored, ["X", "Y"], "expansion")  # base: X 50 vs Y 49
    assert stability["X"]["rank_range"] == [1, 2]
    assert stability["X"]["in_top_n_pct"] < 100


def test_stability_confirms_a_dominant_airport():
    engine, scored = _engine_with({"a": {"X": 90.0, "Y": 20.0}, "b": {"X": 90.0, "Y": 20.0}})
    assert engine.stability(scored, ["X", "Y"], "expansion")["X"] == {
        "rank_range": [1, 1], "top_n": 1, "in_top_n_pct": 100.0, "scenarios": 4}


def test_status_is_partial_when_on_time_coverage_is_short(tools):
    status = tools.scoring.status(tools.scored, "AAA", "congestion")
    assert status["level"] == "partial"
    assert "2 of 12 months" in status["notes"][0]
