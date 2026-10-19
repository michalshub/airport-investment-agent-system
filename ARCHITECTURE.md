# Architecture and methodology

## Data flow

```text
BTS CSV/ZIP + FAA airport API
    -> ETL -> Parquet tables + provenance
    -> metrics -> deterministic scores -> agent tools
    -> Gemini tool calling -> Streamlit chat

METAR + FAA NAS APIs -> live context tool (never changes historical scores)
```

`src/etl.py` builds the snapshot; `src/metrics.py` calculates airport metrics; `src/scoring.py` normalizes,
ranks and explains them. `src/tools.py` exposes nine tools with periods, sources, coverage and caveats.
`src/agent.py` maintains conversation history and coordinates Gemini; `app.py` displays answers and tool evidence.

## Data and metrics

- **T-100:** monthly origin/destination/service totals, using reported seats rather than estimates.
  Classes F/L are passenger service; G/P are cargo. Long-haul results separate passenger and cargo-inclusive scopes.
- **On-Time:** flight records become monthly sums and observed counts. Cancelled flights are excluded from
  departure-delay and taxi-out denominators; rates use combined counts, not averages of monthly averages.
  Signed departure delays are clipped at zero, preserving missing values.
- **Airport metadata:** FAA airport API records are cached. Country/code checks and route-distance consistency
  help reject foreign code collisions. Runways are context, not scoring inputs.
- **Overlap and provenance:** the newest file owns an overlapping month. Source files, loaded months,
  warnings and unresolved airports are recorded in `data/processed/provenance.json`.

All airports share a 12-month window ending at the latest accepted T-100 month. A month is excluded if its
aggregated route-row count is below 85% of the prior-12-month median. This is a completeness heuristic,
not official confirmation of reporting completeness. YoY compares the same months one year earlier;
three-year CAGR compares the same window three years earlier.

| Metric | Definition |
|---|---|
| Traffic | Passenger-service departing passengers, seats and performed departures |
| Passenger CAGR | `(current passengers / passengers three years earlier)^(1/3) - 1` |
| Load factor | `sum(passengers * distance) / sum(seats * distance)` |
| Passenger-minus-seat growth | Passenger YoY growth minus seat YoY growth, in percentage points |
| Delay rate | Departures at least 15 minutes late / departures with an observed delay |
| Average delay / taxi-out | Total observed minutes / corresponding observed flights |
| Cancellation rate | Cancelled flights / scheduled flights |
| Long-haul share | Departures on segments at or above the distance threshold / scoped departures |

Long-haul thresholds default to 2,500 and 3,000 miles; neither is a universal definition.
Hub categories use FAA-style enplanement-share thresholds applied to this dataset, not official FAA designations.

## Deterministic scoring

Airports with at least 100,000 departing passengers in the window form the national comparison set.
Raw components become percentiles from 0-100 using average ranks for ties. The resulting congestion index is
reused directly in the other two scores, without re-ranking it. Higher means more pressure or opportunity.
Weights and thresholds are configured in `config.json`.

| Score | Components and weights |
|---|---|
| Congestion | Delay rate 35%; average departure delay 25%; cancellations 20%; taxi-out 20% |
| Demand pressure | Load factor 45%; passenger-minus-seat growth 35%; congestion 20% |
| Expansion opportunity | Three-year passenger CAGR 30%; congestion 25%; departure YoY growth 20%; load factor 15%; passenger scale 10% |

Expansion rewards sustained growth and operating strain, with a smaller scale component. These are transparent
screening choices, not empirically calibrated return forecasts. Percentiles are relative to the candidate
population, not engineering capacity thresholds.

Missing components are dropped and remaining weights rescaled, never filled with zero. Component explanations
show raw values, percentiles, weights and contribution points that sum to the score. Scores report `full`,
`partial` or `not_scored`, together with the share of available weight. At least one On-Time month is required
for congestion; fewer than 12 months remains partial, so seasonal airports need particular caution.

The stability tool perturbs each weight by -20%, -10%, +10% and +20%, re-ranks the candidates, and reports rank
ranges and top-N retention. This measures sensitivity to chosen weights, not statistical confidence.

**Unmet demand:** demand pressure is a proxy. An additional seat-shortfall calculation estimates seats needed
to reach an 80% seat load factor at current passenger volume, plus approximate extra departures per day.
It uses passengers/seats, whereas the scoring load factor is distance-weighted. Neither estimates spilled demand.

## Where AI is used

Gemini interprets the question and follow-ups, chooses tools/arguments, and explains results. Deterministic
tools resolve airport aliases/regions and calculate metrics, comparisons, scores and sensitivity.
The prompt requires tool-backed figures and relevant scope/coverage caveats; simple follow-ups stay concise.

A post-check flags answer numbers that do not match accumulated tool outputs within displayed rounding.
It is a heuristic warning, not factual verification: it cannot establish that a matching number has the right
meaning, and some ranks, small counts and dates are exempt. Expandable tool results remain the audit trail.
Gemini receives the conversation and tool results, not raw flight records. Network/API failures are surfaced;
there is no simulated successful response. Voice is outside this implementation's scope.

## Tradeoffs and limitations

| Choice | Reason and cost |
|---|---|
| Local BTS snapshots and Parquet | Reproducible, fast analysis without per-question bulk downloads; freshness requires an ETL refresh |
| Pandas, no database | Few moving parts for a one-day build and a roughly 21 MB snapshot; not intended for concurrent warehouse-scale queries |
| National percentiles | Comparable scores across regions; hub-peer differences can still influence rankings |
| Departure-based operations | Relevant to the origin airport; weather, ATC and airline behavior are confounders |
| Live APIs as context only | Current conditions are useful but not evidence for long-term investment returns |
| LLM explanation, deterministic calculations | Auditable arithmetic with conversational access; model explanations still require analyst review |

The current window is August 2025-July 2026; On-Time has 11 months, with July missing.
Reporting-carrier coverage varies by airport, and partial-year observations can be seasonal. Airports with
no On-Time data have their remaining components reweighted. Such scores are not equally supported by evidence.

High load factor is not terminal occupancy; delays do not prove an infrastructure bottleneck. No capex, lease,
concession, gate or terminal-area data is included, so the agent cannot establish project profitability.
Next priorities are the missing On-Time month, airport capacity/financial data, and historical validation of
whether the screening signals predict useful expansion outcomes.
