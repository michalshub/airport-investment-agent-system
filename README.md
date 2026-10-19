# Airport Investment Intelligence Agent

A conversational screening tool for US airport modernization opportunities. Gemini selects analyses and
explains results; Python computes the metrics, rankings and score contributions.

## Run

Requires Python 3.11+ and a [Gemini API key](https://aistudio.google.com/apikey).
Processed data and cached airport metadata are included; downloading raw BTS files is not needed to run the app.

```powershell
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Set GOOGLE_API_KEY in .env, then:
python -m streamlit run app.py
```

Open http://localhost:8501. Chat requires internet access to Gemini; historical calculations run locally.


## What it supports

| Requirement | Implementation |
|---|---|
| Airport ranking and comparison | Deterministic expansion, congestion and demand-pressure scores; component explanations and weight sensitivity |
| Public aviation APIs | FAA airport metadata, aviationweather.gov METAR and FAA NAS status; live context is separate from historical scores |
| Conversational follow-ups | Gemini tool calling with conversation history |
| Chat interface | Streamlit with example questions and expandable sources/calculations; voice is not implemented |
| Assumptions and uncertainty | Period, coverage status, missing components and caveats in tool results and answers |

Try New England expansion candidates, LAX vs. Santa Ana congestion, Anchorage long-haul share, or SFO unmet
demand. See [ARCHITECTURE.md](ARCHITECTURE.md) for methodology, tradeoffs and AI responsibilities.

## Data and refresh

The bundled snapshot uses T-100 from January 2022 through July 2026. Its common analysis window is
August 2025-July 2026.

| Source | Local location | Purpose |
|---|---|---|
| BTS T-100 Segment (All Carriers) | `data/raw/t100/*.csv` | Passengers, reported seats, departures, distance and service class |
| BTS Reporting Carrier On-Time Performance | `data/raw/otp/*.csv` or `*.zip` | Delays, cancellations, taxi-out and delay causes |
| FAA airport API | `data/reference/airports.csv` | Cached airport names, location and runway context |
| Processed snapshot | `data/processed/` | Parquet tables and source provenance used by the app |

Raw files are excluded from version control. To refresh, place downloads in the corresponding raw folder;
no renaming or manual union is needed. The newest file wins overlapping months, with a warning.

- **T-100 required:** `YEAR, MONTH, ORIGIN, DEST, CLASS, PASSENGERS, SEATS, DEPARTURES_PERFORMED, DISTANCE`.
- **On-Time required:** `YEAR, MONTH, ORIGIN, CANCELLED, DEP_DEL15, TAXI_OUT` and either `DEP_DELAY_NEW`
  or `DEP_DELAY`. Negative signed delays become zero; missing observations remain missing.
- Optional fields include scheduled departures/freight (T-100), and diversion/delay-cause fields (On-Time).
  Monthly BTS PREZIP archives are also supported.

```powershell
python -m src.etl
# Optional airport metadata refresh: python -m src.etl --refresh-airports
```

Restart Streamlit after a rebuild to reload cached data. Adding July 2026 On-Time data requires no code change.

## Validation and scope

```powershell
python -m pytest tests -q
```

Tests use synthetic data with hand-checked expectations and require neither network access nor an API key.
Source modules in `src/` separate ETL, metrics, scoring, tools, live APIs and the agent; `app.py` is the chat UI.
Weights, thresholds and aliases are in `config.json`.

Scores indicate traffic and capacity pressure, **not profitability or a proven terminal bottleneck**.
No capex, lease, concession, gate or terminal-area data is used. On-Time coverage is limited to reporting
carriers; missing components are excluded and remaining weights rescaled. Unmet demand is a proxy, not an
observed count of passengers unable to fly.
