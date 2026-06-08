"""Trust calibration layer for TrustSOC.

Provides trust scoring with adaptive threshold learning,
temperature scaling for post-hoc calibration, and
trust decision decomposition for explainability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .calibration_metrics import calibration_summary


@dataclass
class TrustOutputs:
    trust_score: np.ndarray
    uncertainty_score: np.ndarray
    reliability_score: np.ndarray
    expected_action: np.ndarray
    threshold: float
    calibration_metrics: dict[str, Any]
    trust_decomposition: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Meta-Feature Construction
# ---------------------------------------------------------------------------

def build_meta_features(
    confidence: np.ndarray,
    uncertainty: np.ndarray,
    reliability: np.ndarray,
    contradiction: np.ndarray,
    adversarial_noise: np.ndarray,
    risk_score: np.ndarray,
    cti_match_score: np.ndarray,
    evidence_consistency: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            confidence,
            uncertainty,
            reliability,
            contradiction,
            adversarial_noise,
            risk_score,
            cti_match_score,
            evidence_consistency,
        ]
    )


# ---------------------------------------------------------------------------
# Temperature Scaling (Post-hoc Calibration)
# ---------------------------------------------------------------------------

def temperature_scale(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to logits for post-hoc calibration.
    
    Lower temperature -> sharper (more confident) predictions.
    Higher temperature -> softer (less confident) predictions.
    """
    scaled = logits / max(temperature, 1e-6)
    # Softmax
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exps = np.exp(np.clip(scaled, -40.0, 40.0))
    return exps / exps.sum(axis=1, keepdims=True)


def learn_temperature(
    logits: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 50,
) -> float:
    """Learn optimal temperature on validation set by minimizing NLL.
    
    Simple grid search over temperature values.
    """
    from sklearn.metrics import log_loss

    best_temp = 1.0
    best_loss = float("inf")

    for temp in np.linspace(0.1, 5.0, n_steps):
        probs = temperature_scale(logits, float(temp))
        try:
            loss = log_loss(y_true, probs, labels=sorted(set(y_true)))
            if loss < best_loss:
                best_loss = loss
                best_temp = float(temp)
        except ValueError:
            continue

    return best_temp


# ---------------------------------------------------------------------------
# Adaptive Trust Threshold Learning
# ---------------------------------------------------------------------------

def learn_adaptive_threshold(
    trust_scores: np.ndarray,
    correctness: np.ndarray,
    adversarial_mask: np.ndarray | None = None,
    consistency: np.ndarray | None = None,
    n_candidates: int = 37,
    objective: str = "trust_alignment_score",
) -> tuple[float, dict[str, Any]]:
    """Learn optimal trust threshold from validation data.
    
    Instead of using fixed threshold=0.50, search for the threshold
    that maximizes the trust alignment score (or another objective).
    
    Parameters
    ----------
    trust_scores : trust scores from the calibrator
    correctness : boolean array of correct predictions
    adversarial_mask : boolean mask for adversarial samples
    consistency : evidence consistency scores
    n_candidates : number of threshold candidates to try
    objective : which metric to optimize
    
    Returns
    -------
    (best_threshold, best_metrics)
    """
    best_threshold = 0.50
    best_metrics: dict[str, Any] | None = None
    best_score = -float("inf")

    for threshold in np.linspace(0.05, 0.95, n_candidates):
        current = calibration_summary(
            trust_scores,
            correctness.astype(bool),
            float(threshold),
            adversarial_mask,
            consistency,
        )
        score = current.get(objective, current.get("trust_alignment_score", 0.0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = current

    assert best_metrics is not None
    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# Action Decision Logic (Enhanced with context-aware rules)
# ---------------------------------------------------------------------------

def decide_actions(
    trust_score: np.ndarray,
    uncertainty: np.ndarray,
    reliability: np.ndarray,
    contradiction: np.ndarray,
    adversarial_noise: np.ndarray,
    risk_score: np.ndarray,
) -> np.ndarray:
    """Decide expected action based on multi-factor analysis.
    
    Actions:
    - conclude: high trust, low contradiction, low uncertainty
    - investigate: moderate uncertainty, needs more evidence
    - escalate: high risk with insufficient trust
    - refuse: strong contradiction or adversarial noise detected
    """
    actions = []
    for t_score, unc, rel, contra, noise, risk in zip(
        trust_score, uncertainty, reliability, contradiction, adversarial_noise, risk_score, strict=False
    ):
        if contra >= 0.60 or noise >= 0.70:
            actions.append("refuse" if t_score < 0.65 else "escalate")
        elif risk >= 0.75 and (unc >= 0.25 or t_score < 0.80):
            actions.append("escalate")
        elif t_score >= 0.75 and contra < 0.35 and unc < 0.25 and rel >= 0.55:
            actions.append("conclude")
        else:
            actions.append("investigate")
    return np.asarray(actions)


# ---------------------------------------------------------------------------
# Trust Calibrator Training (Enhanced)
# ---------------------------------------------------------------------------

def fit_trust_calibrator(
    meta_train: np.ndarray,
    correctness_train: np.ndarray,
    meta_eval: np.ndarray,
    correctness_eval: np.ndarray,
    adversarial_mask_eval: np.ndarray,
    consistency_eval: np.ndarray,
) -> tuple[LogisticRegression, float, dict[str, Any]]:
    """Fit trust calibrator and learn adaptive threshold.
    
    Uses LogisticRegression on meta-features to predict whether
    the primary model's prediction is correct. Then uses adaptive
    threshold learning to find the optimal trust threshold.
    """
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    calibrator.fit(meta_train, correctness_train.astype(int))

    eval_score = calibrator.predict_proba(meta_eval)[:, 1]

    best_threshold, best_metrics = learn_adaptive_threshold(
        eval_score,
        correctness_eval,
        adversarial_mask_eval,
        consistency_eval,
    )

    return calibrator, best_threshold, best_metrics
