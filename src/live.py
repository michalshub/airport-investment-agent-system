"""
Live airport conditions from public APIs. Context only: never used in historical scores.

  aviationweather.gov METAR API   current weather observation
  FAA NAS Status API              active ground stops, ground delay programs, delays, closures
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

METAR_URL = "https://aviationweather.gov/api/data/metar"
NAS_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"


class LiveStatusClient:
    def __init__(self, timeout: int = 10, session: requests.Session | None = None):
        self.timeout = timeout
        self.session = session or requests.Session()

    def status(self, iata: str, icao: str | None) -> dict:
        result = {
            "airport": iata,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "weather": None,
            "faa_status": None,
            "errors": [],
        }
        try:
            result["weather"] = self.weather(icao or f"K{iata}")
        except (requests.RequestException, ValueError) as e:
            result["errors"].append(f"weather API: {e}")
        try:
            result["faa_status"] = self.faa_events(iata)
        except (requests.RequestException, ET.ParseError) as e:
            result["errors"].append(f"FAA NAS status API: {e}")
        return result

    def weather(self, icao: str) -> dict | None:
        response = self.session.get(METAR_URL, params={"ids": icao, "format": "json"}, timeout=self.timeout)
        response.raise_for_status()
        observations = response.json() if response.content else []
        if not observations:
            return None
        obs = observations[0]
        return {
            "observed_at_utc": obs.get("reportTime"),
            "flight_category": obs.get("fltCat"),
            "wind_direction_deg": obs.get("wdir"),
            "wind_speed_kt": obs.get("wspd"),
            "wind_gust_kt": obs.get("wgst"),
            "visibility_sm": obs.get("visib"),
            "clouds": obs.get("clouds"),
            "temperature_c": obs.get("temp"),
            "raw_metar": obs.get("rawOb"),
        }

    def faa_events(self, iata: str) -> dict:
        response = self.session.get(NAS_STATUS_URL, timeout=self.timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        events = []
        for delay_type in root.iter("Delay_type"):
            category = delay_type.findtext("Name")
            for element in delay_type.iter():
                if element.findtext("ARPT") == iata:
                    details = {child.tag: (child.text or "").strip() for child in element if child.tag != "ARPT"}
                    events.append({"category": category, "details": details})
        return {
            "feed_updated": root.findtext("Update_Time"),
            "active_events": events,
            "summary": f"{len(events)} active FAA event(s)" if events else "No FAA-reported delays, ground stops or closures",
        }
