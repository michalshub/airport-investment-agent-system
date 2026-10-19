"""ETL: combining yearly/monthly files, real seats, service classes and correct On-Time denominators."""

import pandas as pd
import pytest

from src.etl import build_otp_airport_months, build_t100_routes, validate_airport_coordinates


def test_t100_files_are_combined_and_newest_file_wins_overlaps(built, warehouse):
    routes = warehouse.routes
    june_2023 = routes[(routes["year_month"] == "2023-06-01") & (routes["origin"] == "AAA")]
    assert june_2023["passengers"].sum() == 1000  # the stale 999,999 re-download was skipped
    assert any("t100_2023_06_old.csv" in w for w in built["provenance"]["t100"]["warnings"])
    assert built["provenance"]["t100"]["missing_months"] == []


def test_t100_keeps_reported_seats_and_sums_carriers(warehouse):
    row = warehouse.routes[(warehouse.routes["year_month"] == "2025-03-01") & (warehouse.routes["origin"] == "AAA")]
    assert row["passengers"].item() == 1000
    assert row["seats"].item() == 1250
    assert row["departures"].item() == 10


def test_t100_separates_passenger_and_cargo_service(warehouse):
    ddd = warehouse.routes[(warehouse.routes["origin"] == "DDD") & (warehouse.routes["year_month"] == "2025-03-01")]
    assert dict(zip(ddd["service"], ddd["departures"])) == {"passenger": 20, "cargo": 30}


def test_t100_missing_required_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"YEAR": [2025], "MONTH": [1], "ORIGIN": ["AAA"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required T-100 columns"):
        build_t100_routes([path], {"passenger": ["F"]})


def test_otp_cancelled_flights_are_not_counted_as_on_time(warehouse):
    jan = warehouse.otp[(warehouse.otp["origin"] == "AAA") & (warehouse.otp["year_month"] == "2026-01-01")].iloc[0]
    assert jan["flights"] == 10
    assert jan["cancelled"] == 2
    assert jan["dep_observed"] == 8          # cancelled flights excluded from the delay denominator
    assert jan["dep_delayed_15"] == 2
    assert jan["dep_delay_min_sum"] == 90
    assert jan["taxi_out_observed"] == 8
    assert jan["nas_delay_min"] == 45


def test_otp_prezip_zip_with_camelcase_headers_is_supported(warehouse):
    dec = warehouse.otp[(warehouse.otp["origin"] == "AAA") & (warehouse.otp["year_month"] == "2025-12-01")]
    assert dec["flights"].item() == 2


def test_otp_signed_dep_delay_is_clipped_when_dep_delay_new_is_absent(tmp_path):
    path = tmp_path / "otp_dep_delay.csv"
    pd.DataFrame({"YEAR": [2026] * 3, "MONTH": [2] * 3, "ORIGIN": ["AAA"] * 3, "CANCELLED": [0, 0, 1],
                  "DEP_DEL15": [0, 1, None], "DEP_DELAY": [-8, 20, None], "TAXI_OUT": [10, 12, None]}
                 ).to_csv(path, index=False)
    otp, _ = build_otp_airport_months([path])
    row = otp.iloc[0]
    assert row["dep_observed"] == 2
    assert row["dep_delay_min_sum"] == 20  # early departure counts as 0, not -8


def test_otp_missing_required_column_is_reported(tmp_path):
    path = tmp_path / "bad_otp.csv"
    pd.DataFrame({"YEAR": [2026], "MONTH": [1], "ORIGIN": ["AAA"], "CANCELLED": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required On-Time columns"):
        build_otp_airport_months([path])


def test_airport_with_contradicting_coordinates_is_rejected():
    airports = pd.DataFrame({"iata": ["AAA", "BBB", "XXX"], "latitude": [33.94, 42.36, 10.0],
                             "longitude": [-118.41, -71.01, 10.0]})
    routes = pd.DataFrame({"origin": ["AAA", "XXX"], "dest": ["BBB", "AAA"], "distance_miles": [2600, 300]})
    kept, warnings = validate_airport_coordinates(airports, routes)
    assert kept["iata"].tolist() == ["AAA", "BBB"]
    assert "XXX" in warnings[0]


def test_unresolved_foreign_origin_is_listed_in_provenance(built):
    assert "FFF" in built["provenance"]["airports"]["unresolved_origins_by_passengers"]
