"""Temporal evidence analysis for TrustSOC.

Implements evidence freshness scoring and temporal trust adjustment.
Core insight: stale threat intelligence should not drive high-confidence conclusions.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..evidence_extractor import safe_text


# ---------------------------------------------------------------------------
# CTI Freshness Scoring
# ---------------------------------------------------------------------------

# Default half-life in days for different CTI source types
CTI_HALF_LIFE = {
    "1_otx_threat_intel.csv": 90,  # OTX pulses age faster
    "2_cve_vulnerabilities.csv": 365,  # CVEs stay relevant longer
    "3_malicious_domains.csv": 60,  # Domains rotate quickly
    "4_malicious_ips.csv": 30,  # IPs change most frequently
    "default": 180,
}


def temporal_decay(age_days: float, half_life_days: float = 180.0) -> float:
    """Exponential decay function for evidence freshness.
    
    Returns a value in [0, 1] where:
    - 1.0 = brand new evidence
    - 0.5 = evidence is half_life_days old
    - approaches 0 as evidence gets very old
    
    Formula: freshness = 2^(-age / half_life)
    """
    if age_days <= 0:
        return 1.0
    return float(2.0 ** (-age_days / max(half_life_days, 1.0)))


def compute_cti_freshness(
    cti_matches: list[dict[str, Any]],
    reference_date: datetime | None = None,
    half_lives: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute freshness scores for CTI matches.
    
    Parameters
    ----------
    cti_matches : list of CTI match dictionaries
    reference_date : the date to measure freshness against
    half_lives : custom half-life values per source type
    
    Returns
    -------
    dict with freshness metrics
    """
    if half_lives is None:
        half_lives = CTI_HALF_LIFE
    if reference_date is None:
        reference_date = datetime.now()

    if not cti_matches:
        return {
            "mean_freshness": 0.0,
            "min_freshness": 0.0,
            "max_freshness": 0.0,
            "stale_ratio": 1.0,
            "n_stale": 0,
            "n_total": 0,
        }

    freshness_scores = []
    n_stale = 0

    for match in cti_matches:
        source = safe_text(match.get("source", "default"))
        half_life = half_lives.get(source, half_lives.get("default", 180))

        # Try to extract date from match
        match_date = match.get("date") or match.get("created") or match.get("timestamp")
        if match_date:
            try:
                if isinstance(match_date, str):
                    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
                        try:
                            match_dt = datetime.strptime(match_date, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        match_dt = reference_date
                elif isinstance(match_date, datetime):
                    match_dt = match_date
                else:
                    match_dt = reference_date
                age_days = (reference_date - match_dt).days
            except (ValueError, TypeError):
                age_days = half_life  # Assume average age if parsing fails
        else:
            # No date available — assume moderate staleness
            age_days = half_life * 0.5

        freshness = temporal_decay(age_days, half_life)
        freshness_scores.append(freshness)
        if freshness < 0.3:
            n_stale += 1

    arr = np.array(freshness_scores)
    return {
        "mean_freshness": float(arr.mean()),
        "min_freshness": float(arr.min()),
        "max_freshness": float(arr.max()),
        "stale_ratio": float(n_stale / max(len(freshness_scores), 1)),
        "n_stale": n_stale,
        "n_total": len(freshness_scores),
    }


# ---------------------------------------------------------------------------
# Evidence Reliability Decay
# ---------------------------------------------------------------------------

def decay_adjusted_reliability(
    reliability_score: float,
    freshness: float,
    decay_weight: float = 0.3,
) -> float:
    """Adjust evidence reliability based on temporal freshness.
    
    Formula: adjusted = reliability * (1 - decay_weight * (1 - freshness))
    
    When evidence is fresh (freshness=1.0), reliability is unchanged.
    When evidence is stale (freshness→0), reliability degrades.
    """
    adjustment = 1.0 - decay_weight * (1.0 - freshness)
    return float(np.clip(reliability_score * adjustment, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Temporal Trust Adjustment
# ---------------------------------------------------------------------------

def temporal_trust_adjustment(
    trust_score: float,
    cti_freshness: dict[str, float],
    mitre_count: int,
    evidence_consistency: float,
) -> dict[str, float]:
    """Adjust trust score based on temporal evidence analysis.
    
    Key insight: if all CTI evidence is stale, the model's trust
    should be penalized because it may be reasoning on outdated intelligence.
    
    Returns
    -------
    dict with adjusted_trust, temporal_penalty, and reasoning
    """
    mean_freshness = cti_freshness.get("mean_freshness", 0.5)
    stale_ratio = cti_freshness.get("stale_ratio", 0.0)
    n_total = cti_freshness.get("n_total", 0)

    # Temporal penalty factors
    freshness_penalty = 0.0
    if n_total > 0:
        # Penalize if evidence is stale
        freshness_penalty = max(0.0, 0.15 * (1.0 - mean_freshness))
        # Extra penalty if most evidence is stale
        if stale_ratio > 0.5:
            freshness_penalty += 0.10

    # Mitre coverage bonus (MITRE techniques are relatively stable)
    mitre_stability = min(0.05, 0.01 * mitre_count)

    temporal_penalty = max(0.0, freshness_penalty - mitre_stability)
    adjusted_trust = float(np.clip(trust_score - temporal_penalty, 0.0, 1.0))

    return {
        "original_trust": trust_score,
        "adjusted_trust": adjusted_trust,
        "temporal_penalty": temporal_penalty,
        "freshness_penalty": freshness_penalty,
        "mitre_stability_bonus": mitre_stability,
        "mean_evidence_freshness": mean_freshness,
        "stale_evidence_ratio": stale_ratio,
    }


# ---------------------------------------------------------------------------
# Batch Temporal Analysis
# ---------------------------------------------------------------------------

def batch_temporal_analysis(
    df: pd.DataFrame,
    trust_scores: np.ndarray,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Apply temporal analysis to a full dataframe.
    
    Adds temporal columns for analysis and paper tables.
    """
    import json as json_module

    if reference_date is None:
        reference_date = datetime.now()

    records = []
    for idx, (_, row) in enumerate(df.iterrows()):
        # Parse CTI matches
        cti_str = safe_text(row.get("cti_matches_json", "[]"))
        try:
            cti_matches = json_module.loads(cti_str) if cti_str else []
        except (json_module.JSONDecodeError, TypeError):
            cti_matches = []

        freshness = compute_cti_freshness(cti_matches, reference_date)
        trust = float(trust_scores[idx]) if idx < len(trust_scores) else 0.5

        adjustment = temporal_trust_adjustment(
            trust,
            freshness,
            int(row.get("mitre_count", 0)),
            float(row.get("evidence_consistency", 1.0)),
        )

        records.append({
            "case_id": safe_text(row.get("case_id", f"idx_{idx}")),
            "original_trust": trust,
            "adjusted_trust": adjustment["adjusted_trust"],
            "temporal_penalty": adjustment["temporal_penalty"],
            "mean_freshness": freshness["mean_freshness"],
            "stale_ratio": freshness["stale_ratio"],
            "n_cti": freshness["n_total"],
        })

    return pd.DataFrame(records)
