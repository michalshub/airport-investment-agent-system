"""
Deterministic scoring.

Every component is first converted to a national percentile (0-100) among rankable US airports,
so components with different units can be combined. Higher always means "stronger signal".

  Congestion Index        = weighted percentiles of delay rate, average departure delay,
                            cancellation rate and taxi-out time
  Demand Pressure Index   = weighted percentiles of load factor and passenger-minus-seat growth,
                            plus the Congestion Index  (proxy for unmet demand)
  Expansion Opportunity   = weighted percentiles of 3-year passenger CAGR, YoY departure growth,
                            load factor and traffic scale, plus the Congestion Index

Missing components are excluded and the remaining weights are rescaled (never imputed);
the share of weight actually used is reported as `coverage`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SCORE_WEIGHT_KEYS = {
    "congestion": "congestion_weights",
    "demand_pressure": "demand_pressure_weights",
    "expansion": "expansion_weights",
}
SCORE_COLUMNS = {
    "congestion": "congestion_index",
    "demand_pressure": "demand_pressure_index",
    "expansion": "expansion_score",
}


def percentile_rank(values: pd.Series) -> pd.Series:
    """Percentile (0-100] among non-missing values, ties averaged. Missing stays missing."""
    return values.rank(pct=True, method="average") * 100


def weighted_score(percentiles: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """Weighted mean of available percentiles; returns (score, share of total weight that was available)."""
    w = pd.Series(weights, dtype=float)
    values = percentiles[w.index]
    available_weight = values.notna().mul(w).sum(axis=1)
    score = values.fillna(0).mul(w).sum(axis=1) / available_weight.where(available_weight > 0)
    return score, available_weight / w.sum()


@dataclass
class ScoredAirports:
    table: pd.DataFrame        # metrics plus scores and coverage, one row per rankable airport
    percentiles: pd.DataFrame  # component percentiles used by the scores


class ScoringEngine:
    def __init__(self, config: dict):
        self.config = config
        self.weights = {score_type: config[key] for score_type, key in SCORE_WEIGHT_KEYS.items()}
        self.full_otp_months = config["analysis"]["full_coverage_otp_months"]

    def score(self, metrics_table: pd.DataFrame) -> ScoredAirports:
        table = metrics_table[metrics_table["rankable"]].copy()
        raw_components = {c for w in self.weights.values() for c in w if c != "congestion_index"}
        percentiles = pd.DataFrame({c: percentile_rank(table[c]) for c in sorted(raw_components)}, index=table.index)

        for score_type in ("congestion", "demand_pressure", "expansion"):
            score, coverage = weighted_score(percentiles, self.weights[score_type])
            column = SCORE_COLUMNS[score_type]
            table[column], table[f"{score_type}_coverage"] = score, coverage
            if score_type == "congestion":
                percentiles["congestion_index"] = score
        return ScoredAirports(table=table, percentiles=percentiles)

    def status(self, scored: ScoredAirports, code: str, score_type: str) -> dict:
        """Data completeness behind one airport's score."""
        row = scored.table.loc[code]
        missing = [c for c in self.weights[score_type] if pd.isna(scored.percentiles.at[code, c])]
        uses_otp = score_type == "congestion" or "congestion_index" in self.weights[score_type]
        otp_months = 0 if pd.isna(row["otp_months"]) else int(row["otp_months"])
        notes = []
        if missing:
            notes.append(f"missing components (weights rescaled): {', '.join(missing)}")
        if uses_otp and otp_months < self.full_otp_months:
            notes.append(f"On-Time data covers {otp_months} of {self.full_otp_months} months (seasonal bias possible)")
        if pd.isna(row[SCORE_COLUMNS[score_type]]):
            level = "not_scored"
        else:
            level = "partial" if notes else "full"
        return {"level": level, "weight_coverage_pct": 100 * row[f"{score_type}_coverage"], "notes": notes}

    def explain(self, scored: ScoredAirports, code: str, score_type: str) -> dict:
        """Component-by-component breakdown: raw value, percentile, weight, contribution."""
        weights = self.weights[score_type]
        percentiles = scored.percentiles.loc[code]
        available = sum(w for c, w in weights.items() if pd.notna(percentiles[c]))
        components = []
        for component, weight in weights.items():
            pct = percentiles[component]
            effective = weight / available if pd.notna(pct) and available else 0.0
            components.append({
                "component": component,
                "raw_value": scored.table.at[code, component],
                "national_percentile": pct,
                "configured_weight": weight,
                "effective_weight": effective,
                "contribution_points": pct * effective if pd.notna(pct) else None,
            })
        components.sort(key=lambda c: -(c["contribution_points"] or 0))
        return {
            "score_type": score_type,
            "score": scored.table.at[code, SCORE_COLUMNS[score_type]],
            "components": components,
            "status": self.status(scored, code, score_type),
            "peer_group": f"all rankable US airports (n={len(scored.table)})",
        }

    def rank(self, scored: ScoredAirports, codes: list[str], score_type: str) -> pd.Series:
        column = SCORE_COLUMNS[score_type]
        present = [c for c in dict.fromkeys(codes) if c in scored.table.index]
        return scored.table.loc[present, column].dropna().sort_values(ascending=False, kind="stable")

    def stability(self, scored: ScoredAirports, ranked_codes: list[str], score_type: str) -> dict[str, dict]:
        """Re-rank under one-at-a-time weight perturbations; report each airport's rank range and top-N retention."""
        stability_config = self.config["stability"]
        top_n = min(stability_config["top_n"], len(ranked_codes))
        weights = self.weights[score_type]
        percentiles = scored.percentiles.loc[ranked_codes]
        ranks: dict[str, list[int]] = {c: [] for c in ranked_codes}

        for component in weights:
            for change in stability_config["weight_perturbations_pct"]:
                perturbed = {**weights, component: weights[component] * (1 + change / 100)}
                score, _ = weighted_score(percentiles, perturbed)
                for position, code in enumerate(score.sort_values(ascending=False, kind="stable").index, start=1):
                    ranks[code].append(position)

        scenarios = len(next(iter(ranks.values()), []))
        return {
            code: {
                "rank_range": [min(r), max(r)],
                "top_n": top_n,
                "in_top_n_pct": 100 * sum(p <= top_n for p in r) / scenarios,
                "scenarios": scenarios,
            }
            for code, r in ranks.items()
        }
