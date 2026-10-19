"""
Shared fixtures: a small synthetic BTS data set with hand-checkable answers, built through the real ETL.

Airports (coordinates roughly LAX, BOS, BDL, ANC) plus one foreign origin that must be excluded:
  AAA -> BBB  passenger, constant 1,000 pax / 1,250 seats / 10 departures per month (zero growth)
  BBB -> CCC  passenger, 1,000 pax per month, 1,200 from 2025-02 onward; seats constant 1,500
  CCC -> BBB  passenger, 500 pax per month
  DDD -> AAA  passenger, 20 departures per month (~2,340 miles)
  DDD -> BBB  all-cargo (class G), 30 departures per month (~3,250 miles)
  FFF -> AAA  foreign origin, not returned by the airport API
Months 2022-01 .. 2026-01 are complete; 2026-02 is a partially reported month that must be ignored.
"""

import copy
import io
import os
import zipfile

import pandas as pd
import pytest

from src.etl import _great_circle_miles, build
from src.metrics import AirportMetrics, Warehouse
from src.scoring import ScoringEngine
from src.tools import AirportTools
from src.utils import get_config

AIRPORTS = {
    "AAA": ("KAAA", "Los Angeles", "CA", 33.94, -118.41),
    "BBB": ("KBBB", "Boston", "MA", 42.36, -71.01),
    "CCC": ("KCCC", "Windsor Locks", "CT", 41.94, -72.68),
    "DDD": ("PDDD", "Anchorage", "AK", 61.17, -150.00),
}
FFF_COORDS = (51.47, -0.45)


def distance(a: str, b: str) -> float:
    lat1, lon1 = AIRPORTS[a][3:] if a in AIRPORTS else FFF_COORDS
    lat2, lon2 = AIRPORTS[b][3:]
    return round(float(_great_circle_miles([lat1], [lon1], [lat2], [lon2]).iloc[0]))


def t100_row(month: pd.Timestamp, origin, dest, cls, pax, seats, deps):
    return {"YEAR": month.year, "MONTH": month.month, "ORIGIN": origin, "DEST": dest, "CLASS": cls,
            "PASSENGERS": pax, "SEATS": seats, "DEPARTURES_PERFORMED": deps, "DEPARTURES_SCHEDULED": deps,
            "DISTANCE": distance(origin, dest), "FREIGHT": 1000 if cls == "G" else 0}


def t100_month(month: pd.Timestamp) -> list[dict]:
    bbb_pax = 1200 if month >= pd.Timestamp("2025-02-01") else 1000
    return [
        t100_row(month, "AAA", "BBB", "F", 600, 750, 6),  # two carriers on the same route
        t100_row(month, "AAA", "BBB", "F", 400, 500, 4),
        t100_row(month, "BBB", "CCC", "F", bbb_pax, 1500, 10),
        t100_row(month, "CCC", "BBB", "F", 500, 700, 5),
        t100_row(month, "DDD", "AAA", "F", 2000, 2400, 20),
        t100_row(month, "DDD", "BBB", "G", 0, 0, 30),
        t100_row(month, "FFF", "AAA", "F", 5000, 6000, 20),
    ]


def otp_rows(month: pd.Timestamp, origin: str, n: int, cancelled=0, delays=()) -> list[dict]:
    """n flights; the first `cancelled` are cancelled, then one delayed flight per entry in `delays` (minutes)."""
    rows = []
    for i in range(n):
        is_cancelled = i < cancelled
        delay = delays[i - cancelled] if cancelled <= i < cancelled + len(delays) else 0
        rows.append({
            "YEAR": month.year, "MONTH": month.month, "ORIGIN": origin, "DEST": "BBB",
            "CANCELLED": 1 if is_cancelled else 0,
            "DEP_DEL15": None if is_cancelled else int(delay >= 15),
            "DEP_DELAY_NEW": None if is_cancelled else delay,
            "TAXI_OUT": None if is_cancelled else 10,
            "CARRIER_DELAY": None if delay < 15 else delay / 2,
            "WEATHER_DELAY": None if delay < 15 else 0,
            "NAS_DELAY": None if delay < 15 else delay / 2,
            "LATE_AIRCRAFT_DELAY": None if delay < 15 else 0,
        })
    return rows


def fake_airport_fetcher(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"iata": code, "icao": icao, "airport_name": f"{city} Intl", "city": city, "state": state,
         "latitude": lat, "longitude": lon, "runway_count": 2, "longest_runway_ft": 10000}
        for code, (icao, city, state, lat, lon) in AIRPORTS.items() if code in codes
    ])


def write_raw_data(raw_dir) -> None:
    t100_dir, otp_dir = raw_dir / "t100", raw_dir / "otp"
    t100_dir.mkdir(parents=True)
    otp_dir.mkdir(parents=True)

    months = pd.date_range("2022-01-01", "2026-01-01", freq="MS")
    early = [r for m in months if m.year <= 2023 for r in t100_month(m)]
    late = [r for m in months if m.year >= 2024 for r in t100_month(m)]
    partial = [t100_row(pd.Timestamp("2026-02-01"), "AAA", "BBB", "F", 1000, 1250, 10)]
    pd.DataFrame(early).to_csv(t100_dir / "t100_2022_2023.csv", index=False)
    pd.DataFrame(late + partial).to_csv(t100_dir / "t100_2024_2026.csv", index=False)

    # An older, overlapping re-download of 2023-06 with wrong values: the newer file must win.
    stale = pd.DataFrame(t100_month(pd.Timestamp("2023-06-01"))).assign(PASSENGERS=999_999)
    stale_path = t100_dir / "t100_2023_06_old.csv"
    stale.to_csv(stale_path, index=False)
    os.utime(stale_path, (1_000_000_000, 1_000_000_000))

    jan = pd.Timestamp("2026-01-01")
    otp = otp_rows(jan, "AAA", 10, cancelled=2, delays=(30, 60)) + otp_rows(jan, "BBB", 4)
    pd.DataFrame(otp).to_csv(otp_dir / "otp_2026_01.csv", index=False)

    # PREZIP-style monthly zip with CamelCase headers (2025-12): AAA, two on-time flights.
    prezip = pd.DataFrame(otp_rows(pd.Timestamp("2025-12-01"), "AAA", 2)).rename(columns={
        "YEAR": "Year", "MONTH": "Month", "ORIGIN": "Origin", "DEST": "Dest", "CANCELLED": "Cancelled",
        "DEP_DEL15": "DepDel15", "DEP_DELAY_NEW": "DepDelayMinutes", "TAXI_OUT": "TaxiOut",
        "CARRIER_DELAY": "CarrierDelay", "WEATHER_DELAY": "WeatherDelay", "NAS_DELAY": "NASDelay",
        "LATE_AIRCRAFT_DELAY": "LateAircraftDelay"})
    buffer = io.StringIO()
    prezip.to_csv(buffer, index=False)
    with zipfile.ZipFile(otp_dir / "On_Time_2025_12.zip", "w") as archive:
        archive.writestr("readme.html", "<html></html>")
        archive.writestr("On_Time_2025_12.csv", buffer.getvalue())


@pytest.fixture(scope="session")
def test_config():
    config = copy.deepcopy(get_config())
    config["analysis"]["min_passengers_for_ranking"] = 1000
    return config


@pytest.fixture(scope="session")
def built(tmp_path_factory, test_config):
    root = tmp_path_factory.mktemp("data")
    write_raw_data(root / "raw")
    provenance = build(raw_dir=root / "raw", processed_dir=root / "processed", reference_dir=root / "reference",
                       airport_fetcher=fake_airport_fetcher, config=test_config)
    return {"root": root, "provenance": provenance}


@pytest.fixture(scope="session")
def warehouse(built):
    return Warehouse.load(built["root"] / "processed")


@pytest.fixture(scope="session")
def metrics(warehouse, test_config):
    return AirportMetrics(warehouse, test_config)


class FakeLive:
    def status(self, iata, icao):
        return {"airport": iata, "weather": {"flight_category": "VFR"}, "faa_status": None, "errors": []}


@pytest.fixture(scope="session")
def tools(metrics, test_config):
    return AirportTools(metrics, ScoringEngine(test_config), test_config, live=FakeLive())
