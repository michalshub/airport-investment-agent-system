"""
Deterministic airport metrics computed from the processed BTS data.

All windows share one end month (the latest fully reported T-100 month), so every airport
is measured over exactly the same 12 calendar months:
  current window     end-11 .. end
  prior-year window  the same 12 months one year earlier (for YoY growth)
  3-years-back window the same 12 months three years earlier (for 3-year CAGR)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .etl import PROCESSED_DIR

TRAFFIC_COLUMNS = ["passengers", "seats", "departures", "departures_scheduled", "rpm", "asm"]
OTP_SUM_COLUMNS = ["flights", "cancelled", "dep_observed", "dep_delayed_15", "dep_delay_min_sum",
                   "taxi_out_observed", "taxi_out_min_sum"]


@dataclass
class Warehouse:
    """The processed data set written by `python -m src.etl`."""

    routes: pd.DataFrame
    otp: pd.DataFrame
    airports: pd.DataFrame
    provenance: dict

    @classmethod
    def load(cls, processed_dir: Path = PROCESSED_DIR) -> "Warehouse":
        files = {name: processed_dir / name for name in
                 ("t100_route_month.parquet", "otp_airport_month.parquet", "airports.parquet", "provenance.json")}
        missing = [name for name, path in files.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Processed data missing ({', '.join(missing)}). Run: python -m src.etl")
        return cls(
            routes=pd.read_parquet(files["t100_route_month.parquet"]),
            otp=pd.read_parquet(files["otp_airport_month.parquet"]),
            airports=pd.read_parquet(files["airports.parquet"]),
            provenance=json.loads(files["provenance.json"].read_text(encoding="utf-8")),
        )


@dataclass(frozen=True)
class Window:
    end: pd.Timestamp
    months: int = 12

    def bounds(self, years_back: int = 0) -> tuple[pd.Timestamp, pd.Timestamp]:
        end = self.end - pd.DateOffset(years=years_back)
        return end - pd.DateOffset(months=self.months - 1), end

    def month_starts(self, years_back: int = 0) -> set[pd.Timestamp]:
        return set(pd.date_range(*self.bounds(years_back), freq="MS"))

    def label(self, years_back: int = 0) -> str:
        start, end = self.bounds(years_back)
        return f"{start:%Y-%m} to {end:%Y-%m}"


def _in_window(frame: pd.DataFrame, window: Window, years_back: int = 0) -> pd.DataFrame:
    start, end = window.bounds(years_back)
    return frame[(frame["year_month"] >= start) & (frame["year_month"] <= end)]


def _growth_pct(current: pd.Series, previous: pd.Series) -> pd.Series:
    return 100 * (current / previous.where(previous > 0) - 1)


def _to_python(value: Any) -> Any:
    if isinstance(value, float) and pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


class AirportMetrics:
    """Builds one row of metrics per US airport for the common analysis window."""

    def __init__(self, warehouse: Warehouse, config: dict):
        self.wh = warehouse
        self.config = config
        self.analysis = config["analysis"]
        self.window = Window(self._latest_complete_month(), self.analysis["window_months"])
        self.t100_months = set(warehouse.routes["year_month"].unique())
        self.otp_months_in_window = sorted(set(_in_window(warehouse.otp, self.window)["year_month"].unique()))
        self.table = self._build_table()

    def _latest_complete_month(self) -> pd.Timestamp:
        """Newest T-100 month whose row count is not far below the prior 12 months (i.e. not partially reported)."""
        rows = self.wh.routes.groupby("year_month").size().sort_index()
        min_ratio = self.analysis["latest_month_min_completeness"]
        for month in reversed(rows.index):
            prior = rows[rows.index < month].tail(12)
            if prior.empty or rows[month] >= min_ratio * prior.median():
                return month
        return rows.index[-1]

    def _window_loaded(self, years_back: int) -> bool:
        return self.window.month_starts(years_back) <= self.t100_months

    def _traffic(self, service: str, years_back: int) -> pd.DataFrame:
        frame = self.wh.routes[self.wh.routes["service"] == service]
        return _in_window(frame, self.window, years_back).groupby("origin")[TRAFFIC_COLUMNS].sum()

    def _build_table(self) -> pd.DataFrame:
        current = self._traffic("passenger", 0)
        t = current.add_suffix("_ttm")
        t["load_factor_pct"] = 100 * t["rpm_ttm"] / t["asm_ttm"].where(t["asm_ttm"] > 0)
        t["seat_load_factor_pct"] = 100 * t["passengers_ttm"] / t["seats_ttm"].where(t["seats_ttm"] > 0)

        growth_columns = ["pax_growth_yoy_pct", "seat_growth_yoy_pct", "departure_growth_yoy_pct", "pax_cagr_3y_pct"]
        t[growth_columns] = float("nan")
        if self._window_loaded(1):
            prior = self._traffic("passenger", 1).reindex(t.index)
            t["pax_growth_yoy_pct"] = _growth_pct(t["passengers_ttm"], prior["passengers"])
            t["seat_growth_yoy_pct"] = _growth_pct(t["seats_ttm"], prior["seats"])
            t["departure_growth_yoy_pct"] = _growth_pct(t["departures_ttm"], prior["departures"])
        t["pax_minus_seat_growth_pp"] = t["pax_growth_yoy_pct"] - t["seat_growth_yoy_pct"]
        if self._window_loaded(3):
            base = self._traffic("passenger", 3)["passengers"].reindex(t.index)
            t["pax_cagr_3y_pct"] = 100 * ((t["passengers_ttm"] / base.where(base > 0)) ** (1 / 3) - 1)

        cargo = self._traffic("cargo", 0)["departures"].reindex(t.index).fillna(0)
        t["cargo_departures_ttm"] = cargo
        t["cargo_share_of_departures_pct"] = 100 * cargo / (cargo + t["departures_ttm"])

        t = t.join(self._otp_metrics())
        t = t.join(self.wh.airports.set_index("iata"), how="inner")  # keeps US airports only

        t["enplanement_share_pct"] = 100 * t["passengers_ttm"] / t["passengers_ttm"].sum()
        t["hub_class"] = t["enplanement_share_pct"].map(self._hub_class)
        t["rankable"] = t["passengers_ttm"] >= self.analysis["min_passengers_for_ranking"]
        return t.sort_values("passengers_ttm", ascending=False)

    def _hub_class(self, share_pct: float) -> str:
        thresholds = self.config["hub_classes_min_enplanement_share_pct"]
        for name, minimum in sorted(thresholds.items(), key=lambda kv: -kv[1]):
            if share_pct >= minimum:
                return name
        return "nonhub"

    def _otp_metrics(self) -> pd.DataFrame:
        otp = _in_window(self.wh.otp, self.window)
        grouped = otp.groupby("origin")
        s = grouped[OTP_SUM_COLUMNS].sum()
        observed = s["dep_observed"].where(s["dep_observed"] > 0)
        m = pd.DataFrame(index=s.index)
        m["otp_months"] = grouped["year_month"].nunique()
        m["otp_flights"] = s["flights"]
        m["delay_rate_15_pct"] = 100 * s["dep_delayed_15"] / observed
        m["avg_dep_delay_min"] = s["dep_delay_min_sum"] / observed
        m["cancellation_rate_pct"] = 100 * s["cancelled"] / s["flights"].where(s["flights"] > 0)
        m["avg_taxi_out_min"] = s["taxi_out_min_sum"] / s["taxi_out_observed"].where(s["taxi_out_observed"] > 0)

        causes = [c for c in otp.columns if c.endswith("_delay_min")]
        cause_minutes = grouped[causes].sum(min_count=1)
        total = cause_minutes.sum(axis=1)
        for cause in causes:
            m[f"{cause.removesuffix('_delay_min')}_share_of_delay_min_pct"] = 100 * cause_minutes[cause] / total.where(total > 0)

        too_few = m["otp_months"] < self.analysis["min_otp_months_for_congestion"]
        m.loc[too_few, m.columns.difference(["otp_months", "otp_flights"])] = float("nan")
        return m

    def periods(self) -> dict:
        return {
            "current_window": self.window.label(0),
            "prior_year_window": self.window.label(1) if self._window_loaded(1) else None,
            "three_years_back_window": self.window.label(3) if self._window_loaded(3) else None,
            "otp_months_in_current_window": [m.strftime("%Y-%m") for m in self.otp_months_in_window],
        }

    def airport(self, code: str) -> dict | None:
        if code not in self.table.index:
            return None
        return {"iata": code, **{k: _to_python(v) for k, v in self.table.loc[code].items()}}

    def monthly(self, code: str) -> pd.DataFrame:
        """Passenger-service traffic and On-Time metrics by month within the current window."""
        routes = _in_window(self.wh.routes, self.window)
        traffic = (routes[(routes["origin"] == code) & (routes["service"] == "passenger")]
                   .groupby("year_month")[["passengers", "seats", "departures"]].sum())
        otp = _in_window(self.wh.otp, self.window)
        otp = otp[otp["origin"] == code].set_index("year_month")
        if not otp.empty:
            traffic["delay_rate_15_pct"] = 100 * otp["dep_delayed_15"] / otp["dep_observed"].where(otp["dep_observed"] > 0)
            traffic["cancellation_rate_pct"] = 100 * otp["cancelled"] / otp["flights"].where(otp["flights"] > 0)
        return traffic

    def long_haul(self, code: str, thresholds: list[int]) -> dict | None:
        routes = _in_window(self.wh.routes, self.window)
        routes = routes[routes["origin"] == code]
        if routes.empty:
            return None
        scopes = {"passenger_service": routes[routes["service"] == "passenger"], "all_services_incl_cargo": routes}
        departures = {name: int(f["departures"].sum()) for name, f in scopes.items()}
        result: dict[str, Any] = {
            "total_departures": {**departures,
                                 "cargo_only": departures["all_services_incl_cargo"] - departures["passenger_service"]},
            "total_passengers": int(scopes["passenger_service"]["passengers"].sum()),
            "thresholds": {},
        }
        for threshold in sorted(thresholds):
            entry = {}
            for name, frame in scopes.items():
                long = frame[frame["distance_miles"] >= threshold]
                departures, total = long["departures"].sum(), frame["departures"].sum()
                top = long.groupby("dest")["departures"].sum().nlargest(5)
                entry[name] = {
                    "departures": int(departures),
                    "share_of_departures_pct": 100 * departures / total if total else None,
                    "passengers": int(long["passengers"].sum()),
                    "share_of_passengers_pct": (100 * long["passengers"].sum() / frame["passengers"].sum()
                                                if frame["passengers"].sum() else None),
                    "top_destinations_by_departures": {d: int(v) for d, v in top.items()},
                }
            result["thresholds"][f"{threshold}_miles"] = entry
        return result
