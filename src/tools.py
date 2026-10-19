"""
Agent tools: thin, deterministic wrappers around metrics and scoring.

Every tool returns {"ok": bool, "data": ..., "meta": {...}} with floats rounded, missing values as null,
and the analysis period, sources and caveats attached so the LLM can cite them.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .live import LiveStatusClient
from .metrics import AirportMetrics
from .scoring import SCORE_COLUMNS, ScoringEngine

SCORE_TYPES = list(SCORE_COLUMNS)

STATE_CODES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA", "COLORADO": "CO",
    "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA",
    "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT",
    "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK STATE": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY", "PUERTO RICO": "PR", "GUAM": "GU",
    "US VIRGIN ISLANDS": "VI",
}

COMPARE_METRICS = [
    "passengers_ttm", "departures_ttm", "pax_growth_yoy_pct", "seat_growth_yoy_pct", "departure_growth_yoy_pct",
    "pax_cagr_3y_pct", "load_factor_pct",
    "delay_rate_15_pct", "avg_dep_delay_min", "cancellation_rate_pct", "avg_taxi_out_min",
    "nas_share_of_delay_min_pct", "otp_months", "congestion_index", "demand_pressure_index", "expansion_score",
    "runway_count",
]

TOOL_SPECS: list[dict] = [
    {
        "name": "resolve_airports",
        "description": "Turn a place name into US airport codes: an IATA code, city ('Santa Ana'), alias ('LA'), "
                       "state ('Texas', 'TX') or region ('New England'). Regions and states return airports with "
                       "at least the ranking passenger threshold, largest first.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
    {
        "name": "get_airport_profile",
        "description": "All metrics for one airport over the common 12-month window: traffic, growth, load factor, "
                       "delays, cancellations, taxi-out, all three scores with data status, and runway context.",
        "parameters": {"type": "object", "properties": {"airport_code": {"type": "string"}}, "required": ["airport_code"]},
    },
    {
        "name": "rank_airports",
        "description": "Rank airports by a deterministic score. Give either airport_codes or a region/state/city name. "
                       "score_type: 'expansion' (terminal expansion opportunity), 'congestion' or 'demand_pressure'. "
                       "Returns scores, top drivers, data status and a weight-sensitivity stability check.",
        "parameters": {
            "type": "object",
            "properties": {
                "airport_codes": {"type": "array", "items": {"type": "string"}},
                "region": {"type": "string"},
                "score_type": {"type": "string", "enum": SCORE_TYPES},
                "top_n": {"type": "integer"},
            },
        },
    },
    {
        "name": "compare_airports",
        "description": "Side-by-side comparison of 2-5 airports on traffic, growth, load factor, delay, cancellation, "
                       "taxi-out and scores, including month-by-month consistency of the difference.",
        "parameters": {
            "type": "object",
            "properties": {"airport_codes": {"type": "array", "items": {"type": "string"}}},
            "required": ["airport_codes"],
        },
    },
    {
        "name": "get_long_haul_share",
        "description": "Share of departures and passengers on segments at or above distance thresholds (miles), "
                       "reported separately for passenger service and for all services including cargo.",
        "parameters": {
            "type": "object",
            "properties": {
                "airport_code": {"type": "string"},
                "thresholds_miles": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["airport_code"],
        },
    },
    {
        "name": "get_demand_pressure",
        "description": "Unmet-demand proxy for one airport: Demand Pressure Index with component drivers, the seat "
                       "shortfall versus a target load factor, passenger-vs-seat growth and runway context.",
        "parameters": {"type": "object", "properties": {"airport_code": {"type": "string"}}, "required": ["airport_code"]},
    },
    {
        "name": "explain_score",
        "description": "Break a score into components: raw value, national percentile, weight and contribution points.",
        "parameters": {
            "type": "object",
            "properties": {"airport_code": {"type": "string"}, "score_type": {"type": "string", "enum": SCORE_TYPES}},
            "required": ["airport_code", "score_type"],
        },
    },
    {
        "name": "get_data_provenance",
        "description": "Data sources, loaded months, analysis windows, coverage gaps, warnings and scoring weights.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_live_status",
        "description": "Current weather (METAR) and active FAA delays/ground stops/closures for an airport. "
                       "Context only; never part of the historical scores.",
        "parameters": {"type": "object", "properties": {"airport_code": {"type": "string"}}, "required": ["airport_code"]},
    },
]


def clean(value: Any) -> Any:
    """Make tool output JSON-safe: round floats, turn NaN/inf into None, unwrap numpy/pandas scalars."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4 if abs(value) < 1 else 2)  # keep precision for weights and fractions
    if value is pd.NA:
        return None
    return value


def _ok(data: dict, meta: dict | None = None) -> dict:
    return {"ok": True, "data": data, "meta": meta or {}}


def _error(message: str, data: dict | None = None) -> dict:
    return {"ok": False, "error": message, "data": data or {}}


def _normalize_code(code: str) -> str:
    return str(code).strip().upper()


class AirportTools:
    def __init__(self, metrics: AirportMetrics, scoring: ScoringEngine, config: dict,
                 live: LiveStatusClient | None = None):
        self.metrics = metrics
        self.scoring = scoring
        self.config = config
        self.scored = scoring.score(metrics.table)
        self.live = live or LiveStatusClient()
        self.specs = TOOL_SPECS
        self._handlers = {spec["name"]: getattr(self, spec["name"]) for spec in TOOL_SPECS}

    def call(self, name: str, args: dict | None = None) -> dict:
        handler = self._handlers.get(name)
        if handler is None:
            return _error(f"Unknown tool '{name}'")
        try:
            return clean(handler(**(args or {})))
        except (TypeError, ValueError) as e:
            return _error(f"Invalid arguments for {name}: {e}")

    @property
    def table(self) -> pd.DataFrame:
        return self.metrics.table

    def _meta(self, caveat_keys: list[str], **extra) -> dict:
        caveats = self.config["caveats"]
        return {
            "period": self.metrics.periods(),
            "sources": ["BTS T-100 Segment (All Carriers)", "BTS Reporting Carrier On-Time Performance",
                        "FAA airport data via aviationweather.gov API"],
            "caveats": [caveats[k] for k in caveat_keys],
            **extra,
        }

    def _summary(self, code: str) -> dict:
        row = self.table.loc[code]
        return {"code": code, "name": row["airport_name"], "city": row["city"], "state": row["state"],
                "hub_class": row["hub_class"], "passengers_ttm": row["passengers_ttm"]}

    def _not_found(self, code: str) -> dict:
        return _error(f"'{code}' is not a US airport with passenger service in {self.metrics.window.label()}. "
                      "Use resolve_airports to look up names.")

    def _not_scored_reason(self, code: str, score_type: str) -> str:
        if code not in self.table.index:
            return "not found as a US airport with passenger service in the current window"
        if code not in self.scored.table.index:
            return f"below the ranking threshold of {self.config['analysis']['min_passengers_for_ranking']:,} passengers"
        return "; ".join(self.scoring.status(self.scored, code, score_type)["notes"]) or "insufficient data"

    def _rankable_in_states(self, states: list[str]) -> list[str]:
        rows = self.table[self.table["state"].isin(states) & self.table["rankable"]]
        return rows.index.tolist()

    def _resolve(self, query: str) -> tuple[list[str], str]:
        words = [w for w in str(query).upper().replace(".", " ").split() if w not in {"AIRPORT", "AIRPORTS", "INTL"}]
        q = " ".join(words)
        regions = {name.upper(): states for name, states in self.config["regions"].items()}
        aliases = self.config["airport_aliases"]
        if q in regions:
            return self._rankable_in_states(regions[q]), f"region '{q.title()}' = states {', '.join(regions[q])}"
        if q in aliases and any(c in self.table.index for c in aliases[q]):
            note = f"alias '{q}'"
            if q in STATE_CODES.values():
                note += f" (interpreted as an airport, not the state {q})"
            return aliases[q], note
        if len(q) == 3 and q in self.table.index:
            return [q], "IATA code"
        state = STATE_CODES.get(q) or (q if q in STATE_CODES.values() else None)
        if state:
            return self._rankable_in_states([state]), f"state {state}"
        cities = self.table["city"].fillna("").str.upper()
        names = self.table["airport_name"].fillna("").str.upper()
        for mask, how in ((cities == q, "city"), (cities.str.contains(q, regex=False), "city (partial)"),
                          (names.str.contains(q, regex=False), "airport name")):
            if mask.any():
                return self.table.index[mask].tolist()[:10], how
        return [], "no match"

    def resolve_airports(self, query: str) -> dict:
        codes, how = self._resolve(query)
        found = [c for c in codes if c in self.table.index]
        if not found:
            return _error(f"No US airport matched '{query}'.")
        return _ok({"query": query, "matched_by": how, "airports": [self._summary(c) for c in found]})

    def get_airport_profile(self, airport_code: str) -> dict:
        code = _normalize_code(airport_code)
        row = self.metrics.airport(code)
        if row is None:
            return self._not_found(code)
        if code in self.scored.table.index:
            scores = {SCORE_COLUMNS[t]: {"value": self.scored.table.at[code, SCORE_COLUMNS[t]],
                                         "status": self.scoring.status(self.scored, code, t)} for t in SCORE_TYPES}
        else:
            scores = {"note": self._not_scored_reason(code, "expansion")}
        def pick(keys: list[str]) -> dict[str, Any]:
            return {key: row.get(key) for key in keys}

        return _ok({
            "airport": self._summary(code),
            "traffic_passenger_service": pick(["passengers_ttm", "seats_ttm", "departures_ttm", "load_factor_pct",
                                               "seat_load_factor_pct", "enplanement_share_pct"]),
            "growth": pick(["pax_growth_yoy_pct", "seat_growth_yoy_pct", "departure_growth_yoy_pct",
                            "pax_minus_seat_growth_pp", "pax_cagr_3y_pct"]),
            "cargo": pick(["cargo_departures_ttm", "cargo_share_of_departures_pct"]),
            "operations_on_time": pick(["otp_months", "otp_flights", "delay_rate_15_pct", "avg_dep_delay_min",
                                        "cancellation_rate_pct", "avg_taxi_out_min", "carrier_share_of_delay_min_pct",
                                        "weather_share_of_delay_min_pct", "nas_share_of_delay_min_pct",
                                        "late_aircraft_share_of_delay_min_pct"]),
            "scores": scores,
            "infrastructure_context": {**pick(["runway_count", "longest_runway_ft"]),
                                       "note": "FAA airport data; context only, not scored"},
        }, self._meta(["expansion", "otp_coverage"]))

    def rank_airports(self, airport_codes: list[str] | None = None, region: str | None = None,
                      score_type: str = "expansion", top_n: int = 10) -> dict:
        if score_type not in SCORE_COLUMNS:
            return _error(f"score_type must be one of {SCORE_TYPES}")
        resolution = None
        if airport_codes:
            codes = [_normalize_code(c) for c in airport_codes]
        elif region:
            codes, resolution = self._resolve(region)
        else:
            return _error("Provide airport_codes or region.")

        ranked = self.scoring.rank(self.scored, codes, score_type)
        not_scored = [{"code": c, "reason": self._not_scored_reason(c, score_type)}
                      for c in dict.fromkeys(codes) if c not in ranked.index]
        if ranked.empty:
            return _error("None of the airports could be scored.", {"not_scored": not_scored})

        weights = self.scoring.weights[score_type]
        stability = self.scoring.stability(self.scored, ranked.index.tolist(), score_type)
        rows = []
        for position, (code, score) in enumerate(ranked.head(max(1, int(top_n))).items(), start=1):
            explanation = self.scoring.explain(self.scored, code, score_type)
            rows.append({
                "rank": position,
                **self._summary(code),
                "score": score,
                "data_status": explanation["status"]["level"],
                "top_drivers": [{k: c[k] for k in ("component", "raw_value", "national_percentile", "contribution_points")}
                                for c in explanation["components"][:2]],
                "key_metrics": {c: self.scored.table.at[code, c] for c in weights},
                "stability": stability[code],
            })
        return _ok({
            "score_type": score_type,
            "resolution": resolution,
            "airports_considered": len(dict.fromkeys(codes)),
            "ranked": rows,
            "not_scored": not_scored,
            "method": {
                "weights": weights,
                "normalization": f"national percentile among {len(self.scored.table)} rankable US airports",
                "stability_test": "each weight changed by "
                                  f"{self.config['stability']['weight_perturbations_pct']}% one at a time; "
                                  "rank_range and in_top_n_pct show how robust each position is",
            },
        }, self._meta([score_type, "otp_coverage"]))

    def compare_airports(self, airport_codes: list[str]) -> dict:
        codes = list(dict.fromkeys(_normalize_code(c) for c in airport_codes))
        if not 2 <= len(codes) <= 5:
            return _error("Provide 2 to 5 airport codes.")
        unknown = [c for c in codes if c not in self.table.index]
        if unknown:
            return self._not_found(", ".join(unknown))

        combined = self.table.loc[codes].join(self.scored.table[list(SCORE_COLUMNS.values())], how="left")
        base = codes[0]
        comparison = {}
        for metric in COMPARE_METRICS:
            values = combined[metric]
            comparison[metric] = {
                "values": values.to_dict(),
                "highest": values.idxmax() if values.notna().any() else None,
                f"difference_vs_{base}": {c: values[c] - values[base] for c in codes[1:]},
            }

        base_monthly = self.metrics.monthly(base)
        consistency = {}
        for other in codes[1:]:
            other_monthly = self.metrics.monthly(other)
            entry = {}
            for metric in ("passengers", "delay_rate_15_pct"):
                if metric in base_monthly and metric in other_monthly:
                    both = base_monthly[[metric]].join(other_monthly[[metric]], rsuffix="_other", how="inner").dropna()
                    entry[metric] = {"months_compared": len(both),
                                     f"months_{base}_higher": int((both[metric] > both[f"{metric}_other"]).sum())}
            consistency[f"{base}_vs_{other}"] = entry

        hub_classes = combined["hub_class"].to_dict()
        notes = []
        if len(set(hub_classes.values())) > 1:
            notes.append(f"Airports have different estimated FAA-style hub categories {hub_classes}; scores use national percentiles, "
                         "so compare raw rates for like-for-like operational differences.")
        return _ok({
            "airports": [self._summary(c) for c in codes],
            "metrics": comparison,
            "monthly_consistency": consistency,
            "notes": notes,
        }, self._meta(["congestion", "otp_coverage"]))

    def get_long_haul_share(self, airport_code: str, thresholds_miles: list[int] | None = None) -> dict:
        code = _normalize_code(airport_code)
        if code not in self.table.index:
            return self._not_found(code)
        thresholds = [int(t) for t in (thresholds_miles or self.config["long_haul_thresholds_miles"]) if int(t) > 0]
        result = self.metrics.long_haul(code, thresholds)
        notes = []
        cargo_share = self.table.at[code, "cargo_share_of_departures_pct"]
        if cargo_share >= 25:
            notes.append(f"Cargo is {cargo_share:.1f}% of all departures here, so the passenger-only and "
                         "cargo-inclusive answers differ materially.")
        return _ok({
            "airport": self._summary(code),
            **result,
            "definition": "Long-haul = segment distance (BTS great-circle miles) >= threshold; departures from this airport.",
            "notes": notes,
        }, self._meta(["long_haul"]))

    def get_demand_pressure(self, airport_code: str) -> dict:
        code = _normalize_code(airport_code)
        if code not in self.scored.table.index:
            return _error(f"{code}: {self._not_scored_reason(code, 'demand_pressure')}")
        row = self.scored.table.loc[code]
        explanation = self.scoring.explain(self.scored, code, "demand_pressure")
        for component in explanation["components"]:
            pct = component["national_percentile"]
            component["signal"] = None if pd.isna(pct) else "strong" if pct >= 70 else "moderate" if pct >= 40 else "weak"

        target = self.config["analysis"]["target_seat_load_factor_pct"]
        if row["seat_load_factor_pct"] > target:
            extra_seats = row["passengers_ttm"] / (target / 100) - row["seats_ttm"]
            seats_per_departure = row["seats_ttm"] / row["departures_ttm"]
            shortfall = {
                "target_seat_load_factor_pct": target,
                "current_seat_load_factor_pct": row["seat_load_factor_pct"],
                "additional_seats_per_year": extra_seats,
                "approx_additional_departures_per_day": extra_seats / seats_per_departure / 365,
                "method": "Seats needed to bring seat load factor down to the target at current passenger volume. "
                          "Illustrative: demand that never booked (spill) is not observable in public data.",
            }
        else:
            shortfall = {"target_seat_load_factor_pct": target,
                         "current_seat_load_factor_pct": row["seat_load_factor_pct"],
                         "note": "Seat load factor is at or below the target; no seat shortfall implied."}
        return _ok({
            "airport": self._summary(code),
            "demand_pressure_index": explanation["score"],
            "components": explanation["components"],
            "data_status": explanation["status"],
            "seat_shortfall_proxy": shortfall,
            "growth_context": {k: row[k] for k in ("pax_growth_yoy_pct", "seat_growth_yoy_pct",
                                                   "departure_growth_yoy_pct", "pax_cagr_3y_pct")},
            "infrastructure_context": {"runway_count": row["runway_count"], "longest_runway_ft": row["longest_runway_ft"],
                                       "note": "Context only; runway count alone does not measure capacity."},
        }, self._meta(["demand_pressure", "otp_coverage"]))

    def explain_score(self, airport_code: str, score_type: str = "expansion") -> dict:
        code = _normalize_code(airport_code)
        if score_type not in SCORE_COLUMNS:
            return _error(f"score_type must be one of {SCORE_TYPES}")
        if code not in self.scored.table.index:
            return _error(f"{code}: {self._not_scored_reason(code, score_type)}")
        explanation = self.scoring.explain(self.scored, code, score_type)
        return _ok({"airport": self._summary(code), **explanation,
                    "how_to_read": "contribution_points = national_percentile x effective_weight; they sum to the score."},
                   self._meta([score_type, "otp_coverage"]))

    def get_data_provenance(self) -> dict:
        provenance = self.metrics.wh.provenance
        t100, otp, airports = provenance["t100"], provenance["otp"], provenance["airports"]
        return _ok({
            "built_at_utc": provenance["built_at"],
            "analysis_periods": self.metrics.periods(),
            "t100": {"source": t100["source"], "url": t100["url"],
                     "months_loaded": f"{t100['months'][0]} to {t100['months'][-1]} ({len(t100['months'])} months)",
                     "missing_months": t100["missing_months"], "files": len(t100["files"]), "warnings": t100["warnings"]},
            "on_time": {"source": otp["source"], "url": otp["url"], "months_loaded": otp["months"],
                        "delay_causes_available": otp["delay_causes_available"], "warnings": otp["warnings"]},
            "airports": {k: airports[k] for k in ("source", "url", "loaded_from", "airports", "note", "warnings")},
            "universe": {"us_airports_with_passenger_service": len(self.table),
                         "rankable_airports": len(self.scored.table),
                         "min_passengers_for_ranking": self.config["analysis"]["min_passengers_for_ranking"]},
            "scoring_weights": self.scoring.weights,
        }, self._meta(["otp_coverage"]))

    def get_live_status(self, airport_code: str) -> dict:
        code = _normalize_code(airport_code)
        airports = self.metrics.wh.airports.set_index("iata")
        if code not in airports.index:
            return self._not_found(code)
        status = self.live.status(code, airports.at[code, "icao"])
        if status["weather"] is None and status["faa_status"] is None:
            return _error("Live data unavailable: " + "; ".join(status["errors"] or ["no observation returned"]), status)
        return _ok(status, {"sources": ["aviationweather.gov METAR API", "FAA NAS Status API"],
                            "note": "Current conditions only; not used in any historical score or ranking."})
