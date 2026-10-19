"""Tools: name resolution, ranking, comparison, long-haul, demand pressure and error handling."""

import math

from src.tools import clean


def test_resolve_alias_city_state_and_region(tools):
    assert tools.call("resolve_airports", {"query": "Windsor Locks airport"})["data"]["airports"][0]["code"] == "CCC"
    assert tools.call("resolve_airports", {"query": "anchorage"})["data"]["matched_by"].startswith("city")
    assert [a["code"] for a in tools.call("resolve_airports", {"query": "MA"})["data"]["airports"]] == ["BBB"]
    region = tools.call("resolve_airports", {"query": "New England"})["data"]
    assert {a["code"] for a in region["airports"]} == {"BBB", "CCC"}


def test_la_alias_is_flagged_as_not_louisiana(tools, monkeypatch):
    monkeypatch.setitem(tools.config["airport_aliases"], "LA", ["AAA"])
    result = tools.call("resolve_airports", {"query": "LA"})["data"]
    assert result["airports"][0]["code"] == "AAA"
    assert "not the state" in result["matched_by"]


def test_rank_region_returns_scores_drivers_and_stability(tools):
    result = tools.call("rank_airports", {"region": "New England", "score_type": "expansion"})
    assert result["ok"]
    ranked = result["data"]["ranked"]
    assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    assert ranked[0]["score"] >= ranked[-1]["score"]
    assert {"rank_range", "in_top_n_pct"} <= set(ranked[0]["stability"])
    assert result["meta"]["caveats"]


def test_rank_reports_unknown_airports_instead_of_inventing_scores(tools):
    result = tools.call("rank_airports", {"airport_codes": ["AAA", "ZZZ"], "score_type": "congestion"})
    assert [r["code"] for r in result["data"]["ranked"]] == ["AAA"]
    assert result["data"]["not_scored"][0]["code"] == "ZZZ"


def test_compare_provides_differences_and_monthly_consistency(tools):
    data = tools.call("compare_airports", {"airport_codes": ["AAA", "BBB"]})["data"]
    passengers = data["metrics"]["passengers_ttm"]
    assert passengers["difference_vs_AAA"]["BBB"] == passengers["values"]["BBB"] - passengers["values"]["AAA"]
    assert data["monthly_consistency"]["AAA_vs_BBB"]["passengers"]["months_compared"] == 12


def test_long_haul_flags_cargo_heavy_airport(tools):
    data = tools.call("get_long_haul_share", {"airport_code": "DDD", "thresholds_miles": [2500]})["data"]
    assert data["notes"] and "Cargo" in data["notes"][0]


def test_demand_pressure_has_drivers_and_seat_shortfall(tools):
    data = tools.call("get_demand_pressure", {"airport_code": "DDD"})["data"]  # seat load factor 83.3% > 80%
    assert data["demand_pressure_index"] is not None
    assert all("signal" in c for c in data["components"])
    assert data["seat_shortfall_proxy"]["additional_seats_per_year"] == round(24000 / 0.8 - 28800, 2)


def test_unknown_airport_and_bad_arguments_return_errors(tools):
    assert not tools.call("get_airport_profile", {"airport_code": "ZZZ"})["ok"]
    assert "Invalid arguments" in tools.call("get_airport_profile", {"wrong": 1})["error"]
    assert not tools.call("no_such_tool", {})["ok"]


def test_live_status_is_marked_as_context_only(tools):
    result = tools.call("get_live_status", {"airport_code": "AAA"})
    assert result["ok"] and "not used" in result["meta"]["note"]


def test_clean_makes_values_json_safe():
    assert clean({"a": float("nan"), "b": [math.inf, 1.23456]}) == {"a": None, "b": [None, 1.23]}
