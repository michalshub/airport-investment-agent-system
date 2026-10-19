"""
Build the processed data set from raw BTS downloads and the aviationweather.gov API.

Raw inputs (any number of files; how they are split by year or month does not matter):
  data/raw/t100/*.csv         BTS T-100 Segment (All Carriers)
  data/raw/otp/*.csv | *.zip  BTS Reporting Carrier On-Time Performance (TranStats CSV or PREZIP zip)

Outputs in data/processed/:
  t100_route_month.parquet    origin x destination x service x month traffic
  otp_airport_month.parquet   origin x month sums and counts for delay metrics
  airports.parquet            US airport reference (name, city, state, runways)
  provenance.json             files, months present, row counts, warnings

Usage:
  python -m src.etl                     build everything (airport metadata from cache if present)
  python -m src.etl --refresh-airports  re-query airport metadata from the API
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd
import requests

from .utils import PROJECT_ROOT, get_config

logger = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

T100_URL = "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FMG"
OTP_URL = "https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ"
AIRPORT_API_URL = "https://aviationweather.gov/api/data/airport"

T100_REQUIRED = ["YEAR", "MONTH", "ORIGIN", "DEST", "CLASS", "PASSENGERS", "SEATS", "DEPARTURES_PERFORMED", "DISTANCE"]
T100_OPTIONAL = ["DEPARTURES_SCHEDULED", "FREIGHT"]

OTP_REQUIRED = ["YEAR", "MONTH", "ORIGIN", "CANCELLED", "DEP_DEL15", "DEP_DELAY_NEW", "TAXI_OUT"]
OTP_CAUSES = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
OTP_OPTIONAL = ["DIVERTED", *OTP_CAUSES]
# Monthly PREZIP files use CamelCase headers instead of the TranStats download names.
OTP_PREZIP_COLUMNS = {
    "Year": "YEAR", "Month": "MONTH", "Origin": "ORIGIN", "Cancelled": "CANCELLED", "Diverted": "DIVERTED",
    "DepDel15": "DEP_DEL15", "DepDelayMinutes": "DEP_DELAY_NEW", "TaxiOut": "TAXI_OUT",
    "CarrierDelay": "CARRIER_DELAY", "WeatherDelay": "WEATHER_DELAY", "NASDelay": "NAS_DELAY",
    "SecurityDelay": "SECURITY_DELAY", "LateAircraftDelay": "LATE_AIRCRAFT_DELAY",
}
OTP_CHUNK_ROWS = 500_000
OTP_SUM_COLUMNS = [
    "flights", "cancelled", "dep_observed", "dep_delayed_15", "dep_delay_min_sum",
    "taxi_out_observed", "taxi_out_min_sum",
]

# US airports outside the contiguous states whose ICAO code is not simply "K"/"P" + IATA code.
NON_CONTIGUOUS_ICAO = {
    "OGG": "PHOG", "KOA": "PHKO", "LIH": "PHLI", "ITO": "PHTO", "MKK": "PHMK", "LNY": "PHNY", "JHM": "PHJH",
    "SJU": "TJSJ", "BQN": "TJBQ", "PSE": "TJPS", "VQS": "TJVQ", "CPX": "TJCP",
    "STT": "TIST", "STX": "TISX", "SPN": "PGSN",
    "FAI": "PAFA", "JNU": "PAJN", "KTN": "PAKT", "SIT": "PASI", "BET": "PABE", "ADQ": "PADQ", "OME": "PAOM",
    "OTZ": "PAOT", "BRW": "PABR", "CDV": "PACV", "ENA": "PAEN", "HOM": "PAHO", "DLG": "PADL", "AKN": "PAKN",
    "DUT": "PADU", "SCC": "PASC", "WRG": "PAWG", "PSG": "PAPG", "YAK": "PAYA", "VDZ": "PAVD", "GST": "PAGS",
    "UNK": "PAUN", "GAL": "PAGA", "MCG": "PAMC", "ADK": "PADK", "CDB": "PACD", "SDP": "PASD", "KSM": "PASM",
}
EARTH_RADIUS_MILES = 3958.8
MIN_PASSENGERS_FOR_METADATA = 1_000


@contextmanager
def _open_csv(path: Path) -> Iterator:
    """Yield a binary handle for a plain CSV, or for the first CSV inside a zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            member = next((n for n in archive.namelist() if n.lower().endswith(".csv")), None)
            if member is None:
                raise ValueError(f"{path.name}: zip contains no CSV file")
            with archive.open(member) as handle:
                yield handle
    else:
        with open(path, "rb") as handle:
            yield handle


def _read_header(path: Path) -> list[str]:
    with _open_csv(path) as handle:
        return list(pd.read_csv(handle, nrows=0).columns)


def _year_month(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(pd.DataFrame({"year": df["YEAR"], "month": df["MONTH"], "day": 1}))


def _month_labels(months: Iterable) -> list[str]:
    return sorted(pd.Timestamp(m).strftime("%Y-%m") for m in set(months))


def _newest_first(files: Iterable[Path]) -> list[Path]:
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def _drop_months_already_loaded(df: pd.DataFrame, path: Path, seen: set, warnings: list[str]) -> pd.DataFrame:
    """Keep each month from one file only (the newest), so overlapping downloads are not double counted."""
    months = set(df["year_month"].unique())
    overlap = months & seen
    if overlap:
        warnings.append(f"{path.name}: skipped {len(overlap)} month(s) already loaded from a newer file: "
                        f"{', '.join(_month_labels(overlap))}")
        df = df[~df["year_month"].isin(overlap)]
    seen.update(months)
    return df


def _missing_months(months: Iterable) -> list[str]:
    months = sorted(set(pd.Timestamp(m) for m in months))
    if not months:
        return []
    expected = pd.date_range(months[0], months[-1], freq="MS")
    return _month_labels(set(expected) - set(months))


def build_t100_routes(files: list[Path], service_classes: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    """Load T-100 segment files and aggregate to origin x destination x service x month."""
    class_to_service = {cls: service for service, classes in service_classes.items() for cls in classes}
    frames, file_info, warnings, seen = [], [], [], set()

    for path in _newest_first(files):
        header = _read_header(path)
        missing = [c for c in T100_REQUIRED if c not in header]
        if missing:
            raise ValueError(f"{path.name}: missing required T-100 columns {missing}")
        columns = T100_REQUIRED + [c for c in T100_OPTIONAL if c in header]
        with _open_csv(path) as handle:
            df = pd.read_csv(handle, usecols=columns, dtype={"ORIGIN": str, "DEST": str, "CLASS": str})
        for column in T100_OPTIONAL:
            if column not in df:
                df[column] = float("nan")
        df["year_month"] = _year_month(df)
        df = _drop_months_already_loaded(df, path, seen, warnings)
        file_info.append({"file": path.name, "rows": len(df), "months": _month_labels(df["year_month"].unique())})
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    raw["service"] = raw["CLASS"].map(class_to_service)
    unknown = raw["service"].isna()
    if unknown.any():
        classes = sorted(raw.loc[unknown, "CLASS"].dropna().unique())
        warnings.append(f"T-100: dropped {int(unknown.sum())} rows with unmapped service class {classes}")
        raw = raw[~unknown]

    raw["rpm"] = raw["PASSENGERS"] * raw["DISTANCE"]
    raw["asm"] = raw["SEATS"] * raw["DISTANCE"]
    routes = (
        raw.groupby(["year_month", "ORIGIN", "DEST", "service"], as_index=False)
        .agg(
            departures=("DEPARTURES_PERFORMED", "sum"),
            departures_scheduled=("DEPARTURES_SCHEDULED", "sum"),
            passengers=("PASSENGERS", "sum"),
            seats=("SEATS", "sum"),
            freight_lbs=("FREIGHT", "sum"),
            distance_miles=("DISTANCE", "max"),
            rpm=("rpm", "sum"),
            asm=("asm", "sum"),
        )
        .rename(columns={"ORIGIN": "origin", "DEST": "dest"})
    )
    months = routes["year_month"].unique()
    info = {
        "source": "BTS T-100 Segment (All Carriers)",
        "url": T100_URL,
        "files": file_info,
        "months": _month_labels(months),
        "missing_months": _missing_months(months),
        "rows_by_month": {k.strftime("%Y-%m"): int(v) for k, v in raw.groupby("year_month").size().items()},
        "warnings": warnings,
    }
    return routes, info


def _aggregate_otp_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Turn flight rows into additive per origin-month sums and counts (never averages)."""
    cancelled = chunk["CANCELLED"].fillna(0) > 0
    departed = ~cancelled
    dep_observed = departed & chunk["DEP_DEL15"].notna()
    taxi_observed = departed & chunk["TAXI_OUT"].notna()
    sums = pd.DataFrame({
        "origin": chunk["ORIGIN"],
        "year_month": _year_month(chunk),
        "flights": 1,
        "cancelled": cancelled.astype(int),
        "dep_observed": dep_observed.astype(int),
        "dep_delayed_15": (dep_observed & (chunk["DEP_DEL15"] == 1)).astype(int),
        "dep_delay_min_sum": chunk["DEP_DELAY_NEW"].where(dep_observed, 0).fillna(0),
        "taxi_out_observed": taxi_observed.astype(int),
        "taxi_out_min_sum": chunk["TAXI_OUT"].where(taxi_observed, 0),
    })
    if "DIVERTED" in chunk:
        sums["diverted"] = chunk["DIVERTED"].fillna(0)
    for cause in OTP_CAUSES:
        if cause in chunk:
            sums[f"{cause.lower()}_min"] = chunk[cause].fillna(0)
    return sums.groupby(["origin", "year_month"], as_index=False).sum()


def _load_otp_file(path: Path) -> pd.DataFrame:
    header = _read_header(path)
    rename = {c: OTP_PREZIP_COLUMNS[c] for c in header if c in OTP_PREZIP_COLUMNS}
    available = {rename.get(c, c) for c in header}
    # Signed DEP_DELAY (early = negative) is accepted when DEP_DELAY_NEW was not downloaded; it is clipped at 0.
    derive_delay_new = "DEP_DELAY_NEW" not in available and "DEP_DELAY" in available
    if derive_delay_new:
        available.add("DEP_DELAY_NEW")
    missing = [c for c in OTP_REQUIRED if c not in available]
    if missing:
        raise ValueError(f"{path.name}: missing required On-Time columns {missing}")
    wanted = OTP_REQUIRED + OTP_OPTIONAL + (["DEP_DELAY"] if derive_delay_new else [])
    source_columns = [c for c in header if rename.get(c, c) in wanted]
    origin_column = next(c for c in header if rename.get(c, c) == "ORIGIN")

    parts = []
    with _open_csv(path) as handle:
        reader = pd.read_csv(handle, usecols=source_columns, chunksize=OTP_CHUNK_ROWS, dtype={origin_column: str})
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            if derive_delay_new:
                chunk["DEP_DELAY_NEW"] = chunk.pop("DEP_DELAY").clip(lower=0)
            parts.append(_aggregate_otp_chunk(chunk))
    return pd.concat(parts).groupby(["origin", "year_month"], as_index=False).sum()


def build_otp_airport_months(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Load On-Time files (any month split) into per origin-month sums and counts."""
    frames, file_info, warnings, seen = [], [], [], set()
    for path in _newest_first(files):
        df = _drop_months_already_loaded(_load_otp_file(path), path, seen, warnings)
        file_info.append({"file": path.name, "flights": int(df["flights"].sum()),
                          "months": _month_labels(df["year_month"].unique())})
        frames.append(df)

    otp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["origin", "year_month", *OTP_SUM_COLUMNS])
    causes = [c for c in otp.columns if c.endswith("_delay_min")]
    info = {
        "source": "BTS Reporting Carrier On-Time Performance",
        "url": OTP_URL,
        "files": file_info,
        "months": _month_labels(otp["year_month"].unique()),
        "delay_causes_available": causes,
        "warnings": warnings if files else ["No On-Time Performance files found; congestion metrics unavailable."],
    }
    return otp, info


def _parse_airport(code: str, record: dict) -> dict:
    city, _, name = (record.get("name") or "").partition("/")
    runway_lengths = []
    for runway in record.get("runways") or []:
        length = str(runway.get("dimension", "")).split("x")[0]
        if not str(runway.get("id", "")).startswith("H") and length.isdigit():
            runway_lengths.append(int(length))
    return {
        "iata": code,
        "icao": record.get("icaoId"),
        "airport_name": (name or city).strip().title(),
        "city": city.strip().title(),
        "state": record.get("state"),
        "latitude": record.get("lat"),
        "longitude": record.get("lon"),
        "runway_count": len(runway_lengths),
        "longest_runway_ft": max(runway_lengths) if runway_lengths else None,
    }


def _query_airport_api(session: requests.Session, icao_ids: list[str], timeout: int, skipped: list[str]) -> list[dict]:
    """Query a batch; if a filter or server rejects it, bisect so only the offending ids are skipped."""
    try:
        response = session.get(AIRPORT_API_URL, timeout=timeout, params={"ids": ",".join(icao_ids), "format": "json"})
        response.raise_for_status()
        records = response.json() if response.content else []
        return records if isinstance(records, list) else []
    except requests.RequestException:
        if len(icao_ids) == 1:
            skipped.extend(icao_ids)
            return []
        middle = len(icao_ids) // 2
        return (_query_airport_api(session, icao_ids[:middle], timeout, skipped)
                + _query_airport_api(session, icao_ids[middle:], timeout, skipped))


def fetch_airport_metadata(codes: list[str], timeout: int = 60) -> pd.DataFrame:
    """Look up US airports on the aviationweather.gov airport API. Non-US codes simply do not resolve."""
    session = requests.Session()
    found: dict[str, dict] = {}
    skipped: list[str] = []
    icao_rules: list[Callable[[str], str | None]] = [
        lambda c: "K" + c, lambda c: "P" + c, NON_CONTIGUOUS_ICAO.get,
    ]
    for to_icao in icao_rules:
        pending = {to_icao(c): c for c in codes if c not in found and to_icao(c)}
        icao_ids = list(pending)
        for start in range(0, len(icao_ids), 100):
            for record in _query_airport_api(session, icao_ids[start:start + 100], timeout, skipped):
                code = pending.get(record.get("icaoId"))
                if code and record.get("country") == "US" and code in (record.get("iataId"), record.get("faaId")):
                    found[code] = record
    if skipped:
        logger.warning("Airport API rejected %d id(s): %s", len(skipped), ", ".join(skipped[:20]))
    return pd.DataFrame([_parse_airport(code, record) for code, record in found.items()])


def _great_circle_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (pd.Series(v, dtype=float).map(math.radians) for v in (lat1, lon1, lat2, lon2))
    a = ((lat2 - lat1) / 2).map(math.sin) ** 2 + lat1.map(math.cos) * lat2.map(math.cos) * ((lon2 - lon1) / 2).map(math.sin) ** 2
    return 2 * EARTH_RADIUS_MILES * a.map(lambda x: math.asin(math.sqrt(min(1.0, x))))


def validate_airport_coordinates(airports: pd.DataFrame, routes: pd.DataFrame, max_error: float = 0.15) -> tuple[pd.DataFrame, list[str]]:
    """Reject API matches whose coordinates contradict BTS route distances (i.e. the code matched the wrong airport)."""
    coords = airports.set_index("iata")[["latitude", "longitude"]]
    pairs = routes.groupby(["origin", "dest"], as_index=False)["distance_miles"].max()
    pairs = pairs[pairs["origin"].isin(coords.index) & pairs["dest"].isin(coords.index) & (pairs["distance_miles"] >= 100)]
    if pairs.empty:
        return airports, []
    o, d = coords.loc[pairs["origin"]].reset_index(drop=True), coords.loc[pairs["dest"]].reset_index(drop=True)
    computed = _great_circle_miles(o["latitude"], o["longitude"], d["latitude"], d["longitude"])
    pairs = pairs.reset_index(drop=True).assign(error=(computed - pairs["distance_miles"].values).abs() / pairs["distance_miles"].values)
    median_error = pairs.groupby("origin")["error"].median()
    rejected = sorted(median_error[median_error > max_error].index)
    warnings = [f"Airport metadata rejected (coordinates contradict BTS distances): {rejected}"] if rejected else []
    return airports[~airports["iata"].isin(rejected)], warnings


def load_airports(routes: pd.DataFrame, reference_dir: Path, refresh: bool,
                  fetcher: Callable[[list[str]], pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    cache = reference_dir / "airports.csv"
    warnings: list[str] = []
    if cache.exists() and not refresh:
        airports = pd.read_csv(cache, dtype={"iata": str, "icao": str, "state": str})
        origin = f"cached copy ({cache.relative_to(PROJECT_ROOT) if cache.is_relative_to(PROJECT_ROOT) else cache})"
    else:
        passenger_origins = (routes[routes["service"] == "passenger"].groupby("origin")["passengers"].sum()
                             .sort_values(ascending=False))
        codes = passenger_origins[passenger_origins >= MIN_PASSENGERS_FOR_METADATA].index.tolist()
        logger.info("Querying airport metadata for %d origin codes", len(codes))
        airports, warnings = validate_airport_coordinates(fetcher(codes), routes)
        reference_dir.mkdir(parents=True, exist_ok=True)
        airports.to_csv(cache, index=False)
        origin = "live API query"

    pax = routes[routes["service"] == "passenger"].groupby("origin")["passengers"].sum()
    unresolved = pax[~pax.index.isin(airports["iata"])].sort_values(ascending=False)
    info = {
        "source": "aviationweather.gov airport API (FAA airport data)",
        "url": AIRPORT_API_URL,
        "loaded_from": origin,
        "airports": int(len(airports)),
        "note": "Origins not resolved to a US airport (mostly foreign airports) are excluded from US rankings.",
        "unresolved_origins_by_passengers": unresolved.head(15).astype(int).to_dict(),
        "warnings": warnings,
    }
    return airports, info


def build(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR, reference_dir: Path = REFERENCE_DIR,
          refresh_airports: bool = False, airport_fetcher: Callable[[list[str]], pd.DataFrame] = fetch_airport_metadata,
          config: dict | None = None) -> dict:
    config = config or get_config()
    t100_files = sorted((raw_dir / "t100").glob("*.csv"))
    if not t100_files:
        raise FileNotFoundError(f"No T-100 CSV files found in {raw_dir / 't100'}")
    otp_files = sorted([*(raw_dir / "otp").glob("*.csv"), *(raw_dir / "otp").glob("*.zip")])

    logger.info("Loading %d T-100 file(s)", len(t100_files))
    routes, t100_info = build_t100_routes(t100_files, config["service_classes"])
    logger.info("Loading %d On-Time file(s)", len(otp_files))
    otp, otp_info = build_otp_airport_months(otp_files)
    airports, airports_info = load_airports(routes, reference_dir, refresh_airports, airport_fetcher)

    processed_dir.mkdir(parents=True, exist_ok=True)
    routes.to_parquet(processed_dir / "t100_route_month.parquet", index=False)
    otp.to_parquet(processed_dir / "otp_airport_month.parquet", index=False)
    airports.to_parquet(processed_dir / "airports.parquet", index=False)

    provenance = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "t100": t100_info,
        "otp": otp_info,
        "airports": airports_info,
    }
    (processed_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed aviation data from raw BTS files.")
    parser.add_argument("--refresh-airports", action="store_true", help="re-query airport metadata from the API")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    provenance = build(refresh_airports=args.refresh_airports)
    t100, otp = provenance["t100"], provenance["otp"]
    print(f"T-100 months: {t100['months'][0]} to {t100['months'][-1]} ({len(t100['months'])} months)")
    print(f"On-Time months: {', '.join(otp['months']) or 'none'}")
    print(f"US airports: {provenance['airports']['airports']} ({provenance['airports']['loaded_from']})")
    for section in ("t100", "otp", "airports"):
        for warning in provenance[section]["warnings"]:
            print(f"WARNING [{section}]: {warning}")


if __name__ == "__main__":
    main()
