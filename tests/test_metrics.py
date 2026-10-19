"""Metrics: common 12-month windows, growth, load factor, On-Time rates and long-haul shares."""

import pytest


def test_window_is_twelve_months_ending_at_latest_complete_month(metrics):
    # 2026-02 is only partially reported, so the window ends at 2026-01.
    assert metrics.periods()["current_window"] == "2025-02 to 2026-01"
    assert metrics.periods()["prior_year_window"] == "2024-02 to 2025-01"
    assert metrics.periods()["three_years_back_window"] == "2022-02 to 2023-01"
    assert metrics.table.at["AAA", "passengers_ttm"] == 12 * 1000


def test_constant_traffic_has_zero_growth(metrics):
    row = metrics.table.loc["AAA"]
    assert row["pax_growth_yoy_pct"] == pytest.approx(0)
    assert row["departure_growth_yoy_pct"] == pytest.approx(0)
    assert row["pax_cagr_3y_pct"] == pytest.approx(0)


def test_growth_and_passenger_minus_seat_gap(metrics):
    row = metrics.table.loc["BBB"]
    assert row["pax_growth_yoy_pct"] == pytest.approx(20)
    assert row["seat_growth_yoy_pct"] == pytest.approx(0)
    assert row["pax_minus_seat_growth_pp"] == pytest.approx(20)
    assert row["pax_cagr_3y_pct"] == pytest.approx(100 * (1.2 ** (1 / 3) - 1))


def test_load_factor_uses_reported_seats(metrics):
    assert metrics.table.at["AAA", "load_factor_pct"] == pytest.approx(80)
    assert metrics.table.at["AAA", "seat_load_factor_pct"] == pytest.approx(80)


def test_on_time_rates_use_sums_and_correct_denominators(metrics):
    row = metrics.table.loc["AAA"]  # 2025-12: 2 on-time flights; 2026-01: 10 flights, 2 cancelled, 2 delayed
    assert row["otp_months"] == 2
    assert row["delay_rate_15_pct"] == pytest.approx(100 * 2 / 10)
    assert row["cancellation_rate_pct"] == pytest.approx(100 * 2 / 12)
    assert row["avg_dep_delay_min"] == pytest.approx(90 / 10)
    assert row["avg_taxi_out_min"] == pytest.approx(10)


def test_airport_without_on_time_data_has_missing_not_zero_rates(metrics):
    assert metrics.table["delay_rate_15_pct"].isna()["CCC"]


def test_foreign_origin_is_excluded_from_us_table(metrics):
    assert "FFF" not in metrics.table.index


def test_hub_classes_from_enplanement_share(metrics):
    assert set(metrics.table["hub_class"]) <= {"large_hub", "medium_hub", "small_hub", "nonhub"}
    assert metrics.table["enplanement_share_pct"].sum() == pytest.approx(100)


def test_long_haul_passenger_and_cargo_scopes_differ(metrics):
    result = metrics.long_haul("DDD", [2500])["thresholds"]["2500_miles"]
    assert result["passenger_service"]["share_of_departures_pct"] == pytest.approx(0)
    assert result["all_services_incl_cargo"]["share_of_departures_pct"] == pytest.approx(60)
    assert result["all_services_incl_cargo"]["top_destinations_by_departures"] == {"BBB": 360}
    assert metrics.long_haul("DDD", [2500])["total_departures"]["cargo_only"] == 360
