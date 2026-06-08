"""Explainability module for TrustSOC trust decisions.

Provides SHAP-based feature importance, evidence attribution,
counterfactual analysis, and per-case explanation generation
for the trust calibration layer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..evidence_extractor import safe_text


# ---------------------------------------------------------------------------
# Feature Importance via Logistic Regression Coefficients
# ---------------------------------------------------------------------------

BASE_META_FEATURE_NAMES = [
    "confidence",
    "uncertainty",
    "reliability",
    "contradiction",
    "adversarial_noise",
    "risk_score",
    "cti_match_score",
    "evidence_consistency",
]

TRANSFORMER_META_FEATURE_NAMES = BASE_META_FEATURE_NAMES + [
    "threat_margin",
    "severity_margin",
    "label_margin",
    "min_margin",
    "max_margin",
]

META_FEATURE_NAMES = BASE_META_FEATURE_NAMES


def resolve_feature_names(meta_features: np.ndarray | list[float], feature_names: list[str] | None = None) -> list[str]:
    if feature_names is not None:
        return feature_names
    n_features = int(np.asarray(meta_features).shape[-1])
    if n_features == len(TRANSFORMER_META_FEATURE_NAMES):
        return TRANSFORMER_META_FEATURE_NAMES
    return BASE_META_FEATURE_NAMES[:n_features]


def trust_feature_importance(calibrator, feature_names: list[str] | None = None) -> dict[str, float]:
    """Extract feature importance from the trust calibrator (LogisticRegression).
    
    Uses absolute coefficient values as importance proxy.
    """
    if not hasattr(calibrator, "coef_"):
        return {}

    coefs = calibrator.coef_.flatten()
    feature_names = resolve_feature_names(coefs, feature_names)
    n = min(len(coefs), len(feature_names))
    importance = {}
    for i in range(n):
        importance[feature_names[i]] = float(coefs[i])

    # Rank by absolute importance
    ranked = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    return {k: v for k, v in ranked}


# ---------------------------------------------------------------------------
# SHAP-based Explanation (optional dependency)
# ---------------------------------------------------------------------------

def shap_trust_explanations(
    calibrator,
    meta_features: np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 100,
) -> dict[str, Any] | None:
    """Compute SHAP values for trust decisions.
    
    Returns None if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        return None

    n = min(max_samples, len(meta_features))
    sample = meta_features[:n]
    feature_names = resolve_feature_names(sample, feature_names)

    explainer = shap.LinearExplainer(calibrator, sample)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    n_features = min(len(mean_abs_shap), len(feature_names))

    importance = {feature_names[i]: float(mean_abs_shap[i]) for i in range(n_features)}
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "shap_values": shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
        "feature_importance": {k: v for k, v in ranked},
        "top_3_features": [k for k, _ in ranked[:3]],
        "n_samples": n,
    }


# ---------------------------------------------------------------------------
# Evidence Attribution
# ---------------------------------------------------------------------------

def evidence_importance_for_case(
    case_row: pd.Series,
    calibrator,
    meta_features_single: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Attribute trust decision to specific evidence for a single case.
    
    Performs leave-one-out analysis: for each meta-feature, set it to
    its neutral value and measure the change in trust score.
    """
    feature_names = resolve_feature_names(meta_features_single, feature_names)

    baseline_score = float(calibrator.predict_proba(meta_features_single.reshape(1, -1))[:, 1][0])

    # Neutral values for each feature
    neutral_values = {
        "confidence": 0.5,
        "uncertainty": 0.5,
        "reliability": 0.7,
        "contradiction": 0.0,
        "adversarial_noise": 0.0,
        "risk_score": 0.5,
        "cti_match_score": 0.0,
        "evidence_consistency": 1.0,
        "threat_margin": 0.0,
        "severity_margin": 0.0,
        "label_margin": 0.0,
        "min_margin": 0.0,
        "max_margin": 0.0,
    }

    attributions = {}
    n = min(len(meta_features_single), len(feature_names))
    for i in range(n):
        perturbed = meta_features_single.copy()
        perturbed[i] = neutral_values.get(feature_names[i], 0.5)
        perturbed_score = float(calibrator.predict_proba(perturbed.reshape(1, -1))[:, 1][0])
        delta = baseline_score - perturbed_score
        attributions[feature_names[i]] = {
            "original_value": float(meta_features_single[i]),
            "neutral_value": neutral_values.get(feature_names[i], 0.5),
            "trust_delta": float(delta),
            "direction": "increases_trust" if delta > 0 else "decreases_trust",
        }

    # Sort by absolute impact
    sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]["trust_delta"]), reverse=True)

    return {
        "case_id": safe_text(case_row.get("case_id", "unknown")),
        "baseline_trust_score": baseline_score,
        "attributions": {k: v for k, v in sorted_attrs},
        "top_positive_factor": next((k for k, v in sorted_attrs if v["trust_delta"] > 0), "none"),
        "top_negative_factor": next((k for k, v in sorted_attrs if v["trust_delta"] < 0), "none"),
    }


# ---------------------------------------------------------------------------
# Counterfactual Analysis
# ---------------------------------------------------------------------------

def counterfactual_analysis(
    calibrator,
    meta_features_single: np.ndarray,
    current_action: str,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """What-if analysis: what minimal changes would flip the trust decision?
    
    Tests: what if contradiction were 0? What if reliability were higher?
    """
    feature_names = resolve_feature_names(meta_features_single, feature_names)

    baseline_score = float(calibrator.predict_proba(meta_features_single.reshape(1, -1))[:, 1][0])

    scenarios = {
        "perfect_reliability": {"reliability": 1.0},
        "no_contradiction": {"contradiction": 0.0, "adversarial_noise": 0.0},
        "high_confidence": {"confidence": 0.95, "uncertainty": 0.05},
        "with_cti_evidence": {"cti_match_score": 0.8},
        "perfect_consistency": {"evidence_consistency": 1.0},
        "worst_case": {"contradiction": 0.8, "adversarial_noise": 0.9, "confidence": 0.3},
    }

    results = {}
    n = min(len(meta_features_single), len(feature_names))
    for scenario_name, changes in scenarios.items():
        perturbed = meta_features_single.copy()
        for feat_name, new_val in changes.items():
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if idx < n:
                    perturbed[idx] = new_val
        new_score = float(calibrator.predict_proba(perturbed.reshape(1, -1))[:, 1][0])
        results[scenario_name] = {
            "trust_score": new_score,
            "delta": new_score - baseline_score,
            "changes_applied": changes,
        }

    return {
        "baseline_trust_score": baseline_score,
        "current_action": current_action,
        "scenarios": results,
    }


# ---------------------------------------------------------------------------
# Natural Language Explanation Generation
# ---------------------------------------------------------------------------

def generate_explanation(
    case_row: pd.Series,
    trust_score: float,
    expected_action: str,
    attributions: dict[str, dict[str, float]] | None = None,
) -> str:
    """Generate a human-readable explanation for a trust decision.
    
    Suitable for case study sections in academic papers.
    """
    case_id = safe_text(case_row.get("case_id", "unknown"))
    contradiction = float(case_row.get("contradiction_score", 0.0))
    reliability = float(case_row.get("reliability_score", 0.7))
    cti_count = int(case_row.get("cti_match_count", 0))
    mitre_count = int(case_row.get("mitre_count", 0))
    noise = float(case_row.get("adversarial_noise_score", 0.0))

    lines = [f"Case {case_id}: Trust Score = {trust_score:.4f}, Action = {expected_action}"]

    if expected_action == "refuse":
        lines.append(f"  → REFUSED because: contradiction={contradiction:.2f}, noise={noise:.2f}")
        if contradiction >= 0.60:
            lines.append("  → High contradiction in evidence suggests conflicting signals.")
        if noise >= 0.70:
            lines.append("  → Adversarial noise detected — evidence may be manipulated.")

    elif expected_action == "escalate":
        risk = float(case_row.get("risk_score", 0.0))
        lines.append(f"  → ESCALATED: risk={risk:.1f}, trust too low for autonomous conclusion.")

    elif expected_action == "investigate":
        lines.append(f"  → INVESTIGATE: evidence incomplete (CTI={cti_count}, MITRE={mitre_count})")
        if cti_count == 0:
            lines.append("  → No CTI matches found — external intelligence needed.")

    else:  # conclude
        lines.append(f"  → CONCLUDED: high trust, low contradiction={contradiction:.2f}, reliable={reliability:.2f}")

    if attributions:
        top_factors = list(attributions.items())[:3]
        if top_factors:
            lines.append("  Key factors:")
            for name, attrs in top_factors:
                direction = attrs.get("direction", "unknown")
                delta = attrs.get("trust_delta", 0.0)
                lines.append(f"    - {name}: {direction} (Δ={delta:+.4f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch Explanation for Paper Case Studies
# ---------------------------------------------------------------------------

def generate_case_study_batch(
    test_df: pd.DataFrame,
    trust_scores: np.ndarray,
    expected_actions: np.ndarray,
    calibrator,
    meta_features: np.ndarray,
    n_cases: int = 5,
    feature_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Select and explain representative cases for paper.
    
    Selects cases that cover all 4 action types, with preference
    for interesting (adversarial, borderline) cases.
    """
    if feature_names is None:
        feature_names = META_FEATURE_NAMES

    cases = []
    used_actions = set()

    # Priority: one case per action type
    for action in ["refuse", "escalate", "investigate", "conclude"]:
        mask = expected_actions == action
        if not mask.any():
            continue
        indices = np.where(mask)[0]
        # Pick the most "interesting" case (closest to threshold)
        scores = trust_scores[indices]
        mid_idx = np.argmin(np.abs(scores - 0.5))  # most borderline
        idx = indices[mid_idx]

        attribution = evidence_importance_for_case(
            test_df.iloc[idx],
            calibrator,
            meta_features[idx],
            feature_names,
        )
        explanation = generate_explanation(
            test_df.iloc[idx],
            float(trust_scores[idx]),
            str(expected_actions[idx]),
            attribution.get("attributions"),
        )
        counterfactual = counterfactual_analysis(
            calibrator,
            meta_features[idx],
            str(expected_actions[idx]),
            feature_names,
        )

        cases.append({
            "index": int(idx),
            "case_id": safe_text(test_df.iloc[idx].get("case_id", "unknown")),
            "action": action,
            "trust_score": float(trust_scores[idx]),
            "attribution": attribution,
            "explanation": explanation,
            "counterfactual": counterfactual,
        })
        used_actions.add(action)

        if len(cases) >= n_cases:
            break

    # Fill remaining slots with adversarial cases if available
    if len(cases) < n_cases:
        adv_mask = test_df["adversarial_type"].ne("normal_case")
        adv_indices = np.where(adv_mask.to_numpy())[0]
        for idx in adv_indices[:n_cases - len(cases)]:
            attribution = evidence_importance_for_case(
                test_df.iloc[idx],
                calibrator,
                meta_features[idx],
                feature_names,
            )
            explanation = generate_explanation(
                test_df.iloc[idx],
                float(trust_scores[idx]),
                str(expected_actions[idx]),
                attribution.get("attributions"),
            )
            cases.append({
                "index": int(idx),
                "case_id": safe_text(test_df.iloc[idx].get("case_id", "unknown")),
                "action": str(expected_actions[idx]),
                "trust_score": float(trust_scores[idx]),
                "attribution": attribution,
                "explanation": explanation,
            })

    return cases
