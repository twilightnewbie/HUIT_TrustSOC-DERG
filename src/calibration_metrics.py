"""Calibration metrics for TrustSOC trust layer evaluation.

Includes ECE, ACE, classwise ECE, Brier score, selective prediction metrics,
AURC (Area Under Risk-Coverage curve), and trust alignment scoring.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Standard ECE: weighted average of |accuracy - confidence| per bin."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            weight = mask.sum() / len(confidence)
            ece += abs(confidence[mask].mean() - correct[mask].mean()) * weight
    return float(ece)


# ---------------------------------------------------------------------------
# Adaptive Calibration Error (ACE) — less biased than ECE
# ---------------------------------------------------------------------------

def adaptive_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ACE uses adaptive bin boundaries (equal-mass bins) instead of fixed edges.
    
    This produces more stable estimates especially when confidence distributions
    are skewed (common in neural networks).
    """
    n = len(confidence)
    if n == 0:
        return 0.0
    sorted_indices = np.argsort(confidence)
    bin_size = max(1, n // n_bins)
    ace = 0.0
    count = 0
    for start in range(0, n, bin_size):
        end = min(start + bin_size, n)
        idx = sorted_indices[start:end]
        bin_conf = confidence[idx].mean()
        bin_acc = correct[idx].mean()
        ace += abs(bin_acc - bin_conf) * len(idx)
        count += len(idx)
    return float(ace / max(count, 1))


# ---------------------------------------------------------------------------
# Classwise ECE
# ---------------------------------------------------------------------------

def classwise_ece(
    confidence: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 15,
) -> dict[str, float]:
    """Per-class ECE — measures miscalibration for each class separately.
    
    Useful for detecting if the model is well-calibrated for common classes
    but poorly calibrated for rare classes.
    """
    classes = sorted(set(y_true) | set(y_pred))
    correct = (y_true == y_pred).astype(float)
    results = {}
    for cls in classes:
        mask = y_pred == cls
        if mask.sum() < 2:
            results[str(cls)] = 0.0
            continue
        results[str(cls)] = expected_calibration_error(confidence[mask], correct[mask], bins)
    results["mean_classwise_ece"] = float(np.mean(list(results.values())))
    return results


# ---------------------------------------------------------------------------
# Reliability Diagram Data
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int = 15,
) -> dict[str, list[float]]:
    """Export data for plotting reliability diagrams (calibration plots).
    
    Returns bin midpoints, bin accuracy, bin confidence, and bin counts.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    midpoints = []
    accuracies = []
    confidences = []
    counts = []

    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count > 0:
            midpoints.append(float((lo + hi) / 2))
            accuracies.append(float(correct[mask].mean()))
            confidences.append(float(confidence[mask].mean()))
            counts.append(count)

    return {
        "midpoints": midpoints,
        "accuracies": accuracies,
        "confidences": confidences,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Selective Prediction Metrics
# ---------------------------------------------------------------------------

def selective_prediction_metrics(
    trust_score: np.ndarray,
    correctness: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """Coverage-accuracy tradeoff analysis for selective prediction.
    
    At each trust threshold, compute:
    - coverage: fraction of samples the model chooses to predict on
    - selective_accuracy: accuracy on the accepted (non-refused) samples
    - risk: 1 - selective_accuracy
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)

    coverages = []
    selective_accuracies = []
    risks = []

    for t in thresholds:
        accept = trust_score >= t
        coverage = float(accept.mean())
        if accept.any():
            sel_acc = float(correctness[accept].mean())
        else:
            sel_acc = 0.0
        coverages.append(coverage)
        selective_accuracies.append(sel_acc)
        risks.append(1.0 - sel_acc)

    # AURC — Area Under Risk-Coverage curve (lower is better)
    if hasattr(np, "trapezoid"):
        aurc = float(np.trapezoid(risks, coverages)) if len(coverages) > 1 else 0.0
    elif hasattr(np, "trapz"):
        aurc = float(np.trapz(risks, coverages)) if len(coverages) > 1 else 0.0
    else:
        # custom simple trapezoidal rule integration if neither exists
        y = np.asarray(risks)
        x = np.asarray(coverages)
        # Note: np.trapz integrates along the axis. np.diff(x) computes difference.
        # Since coverages is typically descending (from 1.0 to 0.0) or ascending,
        # we compute the trapezoidal sum.
        aurc = float(0.5 * np.sum((y[:-1] + y[1:]) * np.diff(x))) if len(coverages) > 1 else 0.0

    # Find optimal threshold: best coverage at >= 95% selective accuracy
    best_coverage_at_95 = 0.0
    best_threshold_at_95 = 1.0
    for t, cov, acc in zip(thresholds, coverages, selective_accuracies):
        if acc >= 0.95 and cov > best_coverage_at_95:
            best_coverage_at_95 = cov
            best_threshold_at_95 = float(t)

    return {
        "thresholds": [float(t) for t in thresholds],
        "coverages": coverages,
        "selective_accuracies": selective_accuracies,
        "risks": risks,
        "aurc": aurc,
        "best_coverage_at_95_accuracy": best_coverage_at_95,
        "best_threshold_at_95_accuracy": best_threshold_at_95,
    }


# ---------------------------------------------------------------------------
# Trust Alignment Score (enhanced)
# ---------------------------------------------------------------------------

def trust_alignment_score(
    trust_score: np.ndarray,
    correctness: np.ndarray,
    threshold: float,
    adversarial_mask: np.ndarray | None = None,
    consistency_score: np.ndarray | None = None,
) -> dict[str, float]:
    accept = trust_score >= threshold
    coverage = float(accept.mean())

    if accept.any():
        accept_precision = float(correctness[accept].mean())
    else:
        accept_precision = 0.0

    incorrect = ~correctness.astype(bool)
    if incorrect.any():
        refusal_correctness = float((~accept[incorrect]).mean())
    else:
        refusal_correctness = 1.0

    correlation = float(np.corrcoef(trust_score, correctness.astype(float))[0, 1]) if len(trust_score) > 1 else 0.0
    if math.isnan(correlation):
        correlation = 0.0

    overconfidence_mask = (trust_score >= 0.8) & (~correctness.astype(bool))
    overconfidence_rate = float(overconfidence_mask.mean())
    underconfidence_mask = (trust_score <= 0.4) & correctness.astype(bool)
    underconfidence_rate = float(underconfidence_mask.mean())

    adversarial_penalty = 0.0
    if adversarial_mask is not None and adversarial_mask.any():
        adversarial_penalty = float(
            ((trust_score[adversarial_mask] >= 0.8) & (~correctness[adversarial_mask].astype(bool))).mean()
        )

    consistency_bonus = 0.0
    if consistency_score is not None and len(consistency_score) == len(trust_score):
        consistency_bonus = float(np.clip(np.corrcoef(trust_score, consistency_score)[0, 1], -1.0, 1.0))
        if math.isnan(consistency_bonus):
            consistency_bonus = 0.0

    score = (
        0.30 * accept_precision
        + 0.20 * refusal_correctness
        + 0.20 * max(correlation, 0.0)
        + 0.15 * coverage
        + 0.15 * max(consistency_bonus, 0.0)
        - 0.25 * adversarial_penalty
        - 0.15 * overconfidence_rate
        - 0.05 * underconfidence_rate
    )

    return {
        "coverage": coverage,
        "accept_precision": accept_precision,
        "refusal_correctness": refusal_correctness,
        "trust_correctness_correlation": correlation,
        "overconfidence_rate": overconfidence_rate,
        "underconfidence_rate": underconfidence_rate,
        "adversarial_overconfidence_penalty": adversarial_penalty,
        "consistency_alignment": consistency_bonus,
        "trust_alignment_score": float(score),
    }


# ---------------------------------------------------------------------------
# Trust Score Decomposition
# ---------------------------------------------------------------------------

def trust_score_decomposition(
    trust_score: np.ndarray,
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    reliability: np.ndarray,
    contradiction: np.ndarray,
    adversarial_noise: np.ndarray,
    evidence_consistency: np.ndarray,
) -> dict[str, float]:
    """Analyze which factors most influence trust scores.
    
    Computes correlation between trust score and each input factor,
    useful for explaining WHY the trust layer behaves as it does.
    """
    factors = {
        "confidence": confidence,
        "uncertainty": uncertainty,
        "reliability": reliability,
        "contradiction": contradiction,
        "adversarial_noise": adversarial_noise,
        "evidence_consistency": evidence_consistency,
    }
    decomposition = {}
    for name, values in factors.items():
        corr = np.corrcoef(trust_score, values)[0, 1] if len(trust_score) > 1 else 0.0
        decomposition[f"trust_vs_{name}_correlation"] = float(0.0 if math.isnan(corr) else corr)

    # Rank factors by absolute correlation
    ranked = sorted(
        [(k, abs(v)) for k, v in decomposition.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    decomposition["top_factor"] = ranked[0][0].replace("trust_vs_", "").replace("_correlation", "") if ranked else "unknown"
    return decomposition


# ---------------------------------------------------------------------------
# Combined Calibration Summary (enhanced)
# ---------------------------------------------------------------------------

def calibration_summary(
    trust_score: np.ndarray,
    correctness: np.ndarray,
    threshold: float,
    adversarial_mask: np.ndarray | None = None,
    consistency_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full calibration summary with ECE, ACE, Brier, and trust alignment."""
    summary = trust_alignment_score(trust_score, correctness, threshold, adversarial_mask, consistency_score)
    summary["ece"] = expected_calibration_error(trust_score, correctness.astype(int))
    summary["ace"] = adaptive_calibration_error(trust_score, correctness.astype(int))
    summary["brier"] = float(brier_score_loss(correctness.astype(int), np.clip(trust_score, 0.0, 1.0)))

    # Selective prediction
    selective = selective_prediction_metrics(trust_score, correctness.astype(bool))
    summary["aurc"] = selective["aurc"]
    summary["best_coverage_at_95_accuracy"] = selective["best_coverage_at_95_accuracy"]

    # Reliability diagram data
    summary["reliability_diagram"] = reliability_diagram_data(trust_score, correctness.astype(int))

    return summary
