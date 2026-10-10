"""
evaluation/metrics.py — Research-grade streaming evaluation metrics.

All metrics are designed to be statistically meaningful and reproducible.
The key design principle: post-drift metrics NEVER look before drift_point
(fixes the recovery_speed=0 bug present in rolling-window implementations
that let pre-drift samples inflate the post-drift window).

Public API
----------
rolling_accuracy(errors, window)            → np.ndarray
recovery_speed(errors, dp, pre_acc, ...)    → int | None
adaptation_lag(errors, dp, pre_err, ...)    → int | None
auc_accuracy(errors, normalise)             → float
stability_variance(errors, window)          → float
post_drift_drop(errors, dp, ...)            → float | None
recovery_curve(errors, dp, ...)             → np.ndarray
bootstrap_ci(errors, ...)                   → (mean, lower, upper)
paired_test(homeo_vals, base_vals, test)    → dict
full_evaluation(errors, drift_points, ...) → dict
"""

import numpy as np
from typing import List, Optional, Tuple, Dict


# ── Rolling accuracy ─────────────────────────────────────────────────────────

def rolling_accuracy(errors: np.ndarray, window: int = 50) -> np.ndarray:
    """Per-step rolling accuracy with a backward-looking sliding window.

    Args:
        errors: 1-D array of {0, 1} per-step error indicators.
        window: Number of recent steps to average.

    Returns:
        np.ndarray of shape (n,) with smoothed accuracy values.
    """
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    acc = np.empty(n)
    for i in range(n):
        start = max(0, i - window + 1)
        acc[i] = 1.0 - float(np.mean(errors[start:i + 1]))
    return acc


# ── Recovery speed ───────────────────────────────────────────────────────────

def recovery_speed(
    errors: np.ndarray,
    drift_point: int,
    pre_acc: Optional[float],
    threshold: float = 0.80,
    window: int = 20,
    max_steps: int = 400,
) -> Optional[int]:
    """Steps from drift_point until post-drift rolling accuracy ≥ threshold × pre_acc.

    CRITICAL: the rolling window is clamped to [drift_point, step] so that
    pre-drift samples never inflate the post-drift mean.  Without this clamp
    the window at step=drift_point still contains (window-1) high-accuracy
    pre-drift samples, making recovery_speed=0 on every run.

    Args:
        errors:      1-D {0,1} error array.
        drift_point: Step index of drift.
        pre_acc:     Accuracy before drift (recovery target denominator).
        threshold:   Fraction of pre_acc to recover to (default 0.80).
        window:      Rolling window for smoothing.
        max_steps:   Search horizon past drift_point.

    Returns:
        Steps to recovery (int), or None if not recovered within max_steps.
    """
    if pre_acc is None or pre_acc < 0.01:
        return None
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    if drift_point >= n:
        return None

    target = pre_acc * threshold
    for step in range(drift_point, min(drift_point + max_steps, n)):
        # Clamp to drift_point — never include pre-drift samples
        start   = max(drift_point, step - window + 1)
        rolling = 1.0 - float(np.mean(errors[start:step + 1]))
        if rolling >= target:
            return step - drift_point
    return None


# ── Adaptation lag ───────────────────────────────────────────────────────────

def adaptation_lag(
    errors: np.ndarray,
    drift_point: int,
    pre_error_rate: Optional[float],
    multiplier: float = 2.0,
    window: int = 10,
    max_steps: int = 300,
) -> Optional[int]:
    """Steps from drift_point until error rate drops to ≤ multiplier × pre-drift rate.

    Measures how quickly the model stops making "drift mistakes" — a tighter
    indicator than recovery_speed because it only requires the error rate to
    fall, not the full accuracy to recover.

    Args:
        pre_error_rate: Mean error rate in the pre-drift window (= 1 − pre_acc).
        multiplier:     Acceptable error multiplier (default 2×).
    """
    if pre_error_rate is None or pre_error_rate >= 1.0:
        return None
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    if drift_point >= n:
        return None

    target = min(pre_error_rate * multiplier, 0.50)
    for step in range(drift_point, min(drift_point + max_steps, n)):
        start = max(drift_point, step - window + 1)
        err   = float(np.mean(errors[start:step + 1]))
        if err <= target:
            return step - drift_point
    return None


# ── AUC ──────────────────────────────────────────────────────────────────────

def auc_accuracy(errors: np.ndarray, normalise: bool = True) -> float:
    """Area under the per-step accuracy curve (trapezoidal rule).

    Args:
        normalise: Divide by stream length so result ∈ [0, 1].
    """
    errors = np.asarray(errors, dtype=float)
    acc    = 1.0 - errors
    auc    = float(np.trapezoid(acc))
    if normalise:
        auc /= max(1, len(errors))
    return auc


# ── Stability ────────────────────────────────────────────────────────────────

def stability_variance(errors: np.ndarray, window: int = 50) -> float:
    """Variance of rolling accuracy.  Lower = more stable trajectory."""
    roll = rolling_accuracy(errors, window=window)
    return float(np.var(roll))


# ── Post-drift drop ──────────────────────────────────────────────────────────

def post_drift_drop(
    errors: np.ndarray,
    drift_point: int,
    window_pre: int = 50,
    window_imm: int = 10,
) -> Optional[float]:
    """Accuracy drop from pre-drift window to the immediate post-drift window.

    Returns:
        Positive value = degradation; negative = unexpected improvement.
        None if either window is empty.
    """
    errors = np.asarray(errors, dtype=float)
    n = len(errors)
    pre  = errors[max(0, drift_point - window_pre):drift_point]
    post = errors[drift_point:min(n, drift_point + window_imm)]
    if len(pre) == 0 or len(post) == 0:
        return None
    # positive = more errors = worse
    return float(np.mean(post)) - float(np.mean(pre))


# ── Recovery curve ───────────────────────────────────────────────────────────

def recovery_curve(
    errors: np.ndarray,
    drift_point: int,
    post_window: int = 200,
    smooth_window: int = 20,
) -> np.ndarray:
    """Accuracy array zoomed to the post-drift region, with rolling smoothing.

    Useful for side-by-side recovery plots.  Time axis starts at drift_point.
    """
    errors      = np.asarray(errors, dtype=float)
    n           = len(errors)
    post_errors = errors[drift_point:min(n, drift_point + post_window)]
    return rolling_accuracy(post_errors, window=smooth_window)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    errors: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for overall accuracy.

    Returns:
        (mean, lower_bound, upper_bound)
    """
    rng    = np.random.RandomState(seed)
    errors = np.asarray(errors, dtype=float)
    n      = len(errors)
    means  = np.array([
        1.0 - float(np.mean(rng.choice(errors, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])
    alpha  = 1.0 - confidence
    return (
        float(1.0 - np.mean(errors)),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


# ── Statistical tests ────────────────────────────────────────────────────────

def paired_test(
    homeo_values: List[float],
    base_values: List[float],
    test: str = "wilcoxon",
) -> Dict:
    """Paired statistical test: homeostatic vs baseline across multiple seeds.

    Args:
        homeo_values: Per-seed metric values for the homeostatic model.
        base_values:  Per-seed metric values for the baseline model.
        test:         'wilcoxon' (non-parametric, preferred for n < 30)
                      or 'ttest' (parametric paired t-test).

    Returns:
        dict with statistic, p_value, significant, effect_size (Cohen's d),
        mean_delta, std_delta.
    """
    from scipy import stats as scipy_stats

    homeo = np.array(homeo_values, dtype=float)
    base  = np.array(base_values,  dtype=float)
    diff  = homeo - base

    try:
        if test == "wilcoxon":
            stat, pval = scipy_stats.wilcoxon(homeo, base, alternative="greater")
        else:
            stat, pval = scipy_stats.ttest_rel(homeo, base)
    except ValueError:
        stat, pval = 0.0, 1.0

    effect_size = float(np.mean(diff) / (np.std(diff) + 1e-8))

    return {
        "test":        test,
        "statistic":   float(stat),
        "p_value":     float(pval),
        "significant": bool(pval < 0.05),
        "effect_size": effect_size,
        "mean_delta":  float(np.mean(diff)),
        "std_delta":   float(np.std(diff)),
        "n":           len(homeo),
    }


# ── Full evaluation ──────────────────────────────────────────────────────────

def full_evaluation(
    errors: np.ndarray,
    drift_points: List[int],
    label: str = "model",
    window: int = 50,
) -> Dict:
    """Complete evaluation of a single error sequence.

    Args:
        errors:       1-D {0,1} error array.
        drift_points: List of step indices where concept drift occurs.
        label:        Model name for reporting.
        window:       Rolling window size (used for pre/post analysis and smoothing).

    Returns:
        Dict with all metrics including per-drift breakdown.
    """
    errors = np.asarray(errors, dtype=float)
    n      = len(errors)
    roll   = rolling_accuracy(errors, window=window)

    result: Dict = {
        "label":              label,
        "n_steps":            n,
        "overall_accuracy":   float(1.0 - np.mean(errors)),
        "final_accuracy":     float(1.0 - np.mean(errors[-window:])),
        "auc":                auc_accuracy(errors),
        "stability_variance": stability_variance(errors, window=window),
        "stability":          float(1.0 - np.std(roll)),
        "rolling_accuracy":   roll,
        "cumulative_regret":  int(np.sum(errors)),
        "mean_acc_ci":        bootstrap_ci(errors, seed=0),
    }

    per_drift = []
    for dp in drift_points:
        if dp >= n:
            continue
        pre_errors  = errors[max(0, dp - window):dp]
        post_errors = errors[dp:min(n, dp + window)]

        pre_acc  = float(1.0 - np.mean(pre_errors))  if len(pre_errors)  > 0 else None
        post_acc = float(1.0 - np.mean(post_errors)) if len(post_errors) > 0 else None
        pre_err  = float(np.mean(pre_errors))         if len(pre_errors)  > 0 else None

        per_drift.append({
            "drift_point":    dp,
            "pre_acc":        pre_acc,
            "post_acc":       post_acc,
            "drop":           post_drift_drop(errors, dp, window_pre=window),
            "recovery_speed": recovery_speed(errors, dp, pre_acc, window=window // 2),
            "adaptation_lag": adaptation_lag(errors, dp, pre_err),
            "recovery_curve": recovery_curve(errors, dp, post_window=min(window * 2, n - dp)),
        })

    result["per_drift"] = per_drift

    if per_drift:
        def _agg(key):
            vals = [d[key] for d in per_drift if d.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        result["mean_pre_drift_acc"]   = _agg("pre_acc")
        result["mean_post_drift_acc"]  = _agg("post_acc")
        result["mean_post_drift_drop"] = _agg("drop")
        result["mean_recovery_speed"]  = _agg("recovery_speed")
        result["mean_adaptation_lag"]  = _agg("adaptation_lag")

    return result
