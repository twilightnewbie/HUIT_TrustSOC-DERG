"""Statistical testing utilities for TrustSOC research.

Provides bootstrap confidence intervals, paired comparison tests,
and multi-model ranking tests for rigorous academic evaluation.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


# ---------------------------------------------------------------------------
# Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a given metric function.

    Parameters
    ----------
    y_true : ground truth labels or values
    y_pred : predicted labels or values
    metric_fn : callable(y_true, y_pred, **kwargs) -> float
    n_bootstrap : number of bootstrap samples
    confidence_level : CI level (0.95 = 95% CI)
    seed : random seed for reproducibility

    Returns
    -------
    dict with keys: mean, std, ci_lower, ci_upper, ci_level
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        try:
            scores[i] = metric_fn(y_true[indices], y_pred[indices], **metric_kwargs)
        except (ValueError, ZeroDivisionError):
            scores[i] = np.nan

    scores = scores[~np.isnan(scores)]
    if len(scores) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_level": confidence_level}

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": confidence_level,
    }


def bootstrap_all_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
    y_true_reg: np.ndarray,
    y_pred_reg: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap CIs for standard TrustSOC evaluation metrics."""
    results = {}

    results["accuracy"] = bootstrap_metric(
        y_true_cls, y_pred_cls, accuracy_score, n_bootstrap=n_bootstrap, seed=seed
    )
    results["f1_macro"] = bootstrap_metric(
        y_true_cls, y_pred_cls, f1_score, n_bootstrap=n_bootstrap, seed=seed, average="macro", zero_division=0
    )
    results["f1_weighted"] = bootstrap_metric(
        y_true_cls, y_pred_cls, f1_score, n_bootstrap=n_bootstrap, seed=seed, average="weighted", zero_division=0
    )
    results["mae"] = bootstrap_metric(
        y_true_reg, y_pred_reg, mean_absolute_error, n_bootstrap=n_bootstrap, seed=seed
    )
    return results


# ---------------------------------------------------------------------------
# Paired Model Comparison Tests
# ---------------------------------------------------------------------------

def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict[str, float]:
    """McNemar's test for paired model comparison on classification.

    Tests whether two models have significantly different error rates.

    Returns
    -------
    dict with keys: statistic, p_value, significant_005, significant_001
    """
    correct_a = (y_true == y_pred_a)
    correct_b = (y_true == y_pred_b)

    # b = A correct, B wrong
    b = int(np.sum(correct_a & ~correct_b))
    # c = A wrong, B correct
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_005": False, "significant_001": False}

    # McNemar's test with continuity correction
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - stats.chi2.cdf(statistic, df=1))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    seed: int = 42,
    **metric_kwargs,
) -> dict[str, float]:
    """Paired bootstrap test (two-sided) for comparing two models.

    Tests H0: metric(A) == metric(B).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstrap, dtype=float)

    observed_diff = metric_fn(y_true, y_pred_a, **metric_kwargs) - metric_fn(y_true, y_pred_b, **metric_kwargs)

    for i in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        try:
            score_a = metric_fn(y_true[indices], y_pred_a[indices], **metric_kwargs)
            score_b = metric_fn(y_true[indices], y_pred_b[indices], **metric_kwargs)
            diffs[i] = score_a - score_b
        except (ValueError, ZeroDivisionError):
            diffs[i] = np.nan

    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"observed_diff": float(observed_diff), "p_value": 1.0, "significant_005": False, "significant_001": False}

    # Two-sided p-value
    p_value = float(np.mean(np.abs(diffs - np.mean(diffs)) >= abs(observed_diff - np.mean(diffs))))

    return {
        "observed_diff": float(observed_diff),
        "mean_diff": float(np.mean(diffs)),
        "std_diff": float(np.std(diffs)),
        "p_value": p_value,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
    }


# ---------------------------------------------------------------------------
# Multi-Model Ranking
# ---------------------------------------------------------------------------

def friedman_test(score_matrix: np.ndarray) -> dict[str, float]:
    """Friedman test for comparing multiple models across multiple datasets/folds.

    Parameters
    ----------
    score_matrix : shape (n_datasets, n_models)
        Each row is a dataset/fold, each column is a model.

    Returns
    -------
    dict with statistic, p_value, and significance flags
    """
    if score_matrix.shape[0] < 3 or score_matrix.shape[1] < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant_005": False}

    stat, p_value = stats.friedmanchisquare(*[score_matrix[:, j] for j in range(score_matrix.shape[1])])
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": float(p_value) < 0.05,
    }


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two groups."""
    n_a, n_b = len(group_a), len(group_b)
    mean_diff = np.mean(group_a) - np.mean(group_b)
    pooled_std = math.sqrt(((n_a - 1) * np.var(group_a, ddof=1) + (n_b - 1) * np.var(group_b, ddof=1)) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float(mean_diff / pooled_std)


# ---------------------------------------------------------------------------
# Convenience: Run All Pairwise Comparisons
# ---------------------------------------------------------------------------

def pairwise_model_comparisons(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Run McNemar's test for all pairs of models.

    Parameters
    ----------
    y_true : ground truth labels
    predictions : dict mapping model_name -> predicted labels

    Returns
    -------
    list of comparison result dicts
    """
    model_names = sorted(predictions.keys())
    results = []
    for i, name_a in enumerate(model_names):
        for name_b in model_names[i + 1:]:
            test = mcnemar_test(y_true, predictions[name_a], predictions[name_b])
            test["model_a"] = name_a
            test["model_b"] = name_b
            results.append(test)
    return results


def format_ci(result: dict[str, float], precision: int = 4) -> str:
    """Format a bootstrap result as 'mean ± std (CI: [lower, upper])'."""
    return (
        f"{result['mean']:.{precision}f} ± {result['std']:.{precision}f} "
        f"({int(result['ci_level'] * 100)}% CI: [{result['ci_lower']:.{precision}f}, {result['ci_upper']:.{precision}f}])"
    )
