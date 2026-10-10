"""
experiments/run_experiments.py — Centralized multi-seed experiment runner.

Single source of truth for all Digital Homeostasis experiments.
Produces mean ± std tables, statistical significance tests, and returns
structured results for downstream visualization.

Usage
-----
    python experiments/run_experiments.py                   # full suite
    python experiments/run_experiments.py --seeds 5         # quick (5 seeds)
    python experiments/run_experiments.py --drift sudden    # single drift type
    python experiments/run_experiments.py --steps 2000      # longer run

Protocol
--------
  • n_warmup  = 200  samples (initial SGD training)
  • n_steps   = 1000 streaming steps per run
  • drift_point = 500 (midpoint of stream)
  • seeds      = 10 per drift type (42–51)
  • drift types = regular, sudden, gradual, recurring
  • tests      = Wilcoxon signed-rank (non-parametric, paired, one-sided)
  • recurring  uses drift_point=200 (5 full cycles in 1000 steps)
"""

import sys
import os
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_model.ml_base_model import BaselineModel
from base_model.base_runner import BaselineSystem
from system_architecture.input_data import StreamingDataGenerator
from system_architecture.prediction_model import PredictionModel
from system_architecture.performance_monitor import PerformanceMonitor
from system_architecture.homeostasis_controller import HomeostasisController
from system_architecture.adaptive_mechanisms import AdaptiveMechanisms
from system_architecture.model_self_regulation import ModelSelfRegulation
from evaluation.metrics import (
    full_evaluation, paired_test, rolling_accuracy,
    auc_accuracy, recovery_speed,
)


# ── Experiment parameters ─────────────────────────────────────────────────────

DEFAULT_SEEDS       = list(range(42, 52))          # 10 seeds: 42–51
DEFAULT_N_WARMUP    = 200
DEFAULT_N_STEPS     = 1000
DEFAULT_DRIFT_POINT = 500
DEFAULT_DRIFT_TYPES = ["regular", "sudden", "gradual", "recurring"]

DRIFT_POINT_OVERRIDE = {
    "recurring": 200,   # 5 cycles × 200-step period in 1000 steps
}


# ── System builders ───────────────────────────────────────────────────────────

def _build_homeo():
    model      = PredictionModel()
    monitor    = PerformanceMonitor()
    controller = HomeostasisController()
    adaptive   = AdaptiveMechanisms(model, controller)
    system     = ModelSelfRegulation(model, monitor, controller, adaptive)
    return model, system


def _build_baseline():
    model  = BaselineModel()
    system = BaselineSystem(model)
    return model, system


# ── Single-seed run ───────────────────────────────────────────────────────────

def run_single(
    drift_type: str,
    n_steps: int = DEFAULT_N_STEPS,
    drift_point: int = DEFAULT_DRIFT_POINT,
    n_warmup: int = DEFAULT_N_WARMUP,
    seed: int = 42,
) -> dict:
    """One full experiment: warm-up → streaming → evaluation.

    Returns a dict with raw error arrays, rolling accuracy, action counts,
    and summary metrics split at drift_point.

    Pipeline:
        1. Generate warm-up data (deterministic with seed).
        2. train_initial() on both models.
        3. calibrate_on_warmup() → set controller targets to actual warm-up
           performance so the controller's setpoint matches reality.
        4. Streaming loop: predict → monitor → controller → adapt.
        5. Full evaluation via evaluation.metrics.full_evaluation().
    """
    np.random.seed(seed)

    dp = DRIFT_POINT_OVERRIDE.get(drift_type, drift_point)

    gen = StreamingDataGenerator(
        n_features=10,
        drift_point=dp,
        drift_type=drift_type,
    )

    base_model,  base_system  = _build_baseline()
    homeo_model, homeo_system = _build_homeo()

    # ── Warm-up ───────────────────────────────────────────────────────────
    X_init, y_init = [], []
    for _ in range(n_warmup):
        X, y = gen.generate_sample()
        X_init.append(X)
        y_init.append(y)
    X_init = np.array(X_init)
    y_init = np.array(y_init)

    base_model.train_initial(X_init, y_init)
    homeo_model.train_initial(X_init, y_init)

    # Calibrate controller targets to match warm-up performance
    homeo_system.calibrate_on_warmup(X_init, y_init)

    # ── Streaming loop ────────────────────────────────────────────────────
    base_errors   = np.empty(n_steps, dtype=float)
    homeo_errors  = np.empty(n_steps, dtype=float)
    action_counts = defaultdict(int)
    homeo_scores  = np.empty(n_steps, dtype=float)
    drift_flags   = np.zeros(n_steps, dtype=int)
    decision_logs = []

    for i in range(n_steps):
        X, y = gen.generate_sample()

        base_system.step(X, y)
        base_errors[i] = float(base_system.errors[-1])

        state = homeo_system.step(X, y)
        homeo_errors[i] = float(int(state["prediction"] != y))
        ctrl            = state["controller"]
        action_counts[ctrl["action"]] += 1
        homeo_scores[i] = ctrl["homeostasis_score"]
        drift_flags[i]  = int(state.get("drift_detected", False))
        decision_logs.append(ctrl.get("decision_log", {}))

    # ── Evaluate ──────────────────────────────────────────────────────────
    drift_points = [dp]
    base_eval  = full_evaluation(base_errors,  drift_points, label="baseline",     window=50)
    homeo_eval = full_evaluation(homeo_errors, drift_points, label="homeostatic",  window=50)

    return {
        # Identifiers
        "drift_type":   drift_type,
        "seed":         seed,
        "n_steps":      n_steps,
        "drift_point":  dp,
        # Raw arrays (for plotting)
        "base_errors":  base_errors,
        "homeo_errors": homeo_errors,
        "homeo_scores": homeo_scores,
        "drift_flags":  drift_flags,
        "decision_logs": decision_logs,
        # Evaluations
        "base_eval":    base_eval,
        "homeo_eval":   homeo_eval,
        # Action distribution
        "action_counts":     dict(action_counts),
        "action_counts_pct": {k: v / n_steps * 100 for k, v in action_counts.items()},
        "n_drift_flags":     int(np.sum(drift_flags)),
        # Quick-access accuracy values (for multi-seed aggregation)
        "base_overall":  base_eval["overall_accuracy"],
        "homeo_overall": homeo_eval["overall_accuracy"],
        "base_post":     base_eval.get("mean_post_drift_acc"),
        "homeo_post":    homeo_eval.get("mean_post_drift_acc"),
        "base_final":    base_eval["final_accuracy"],
        "homeo_final":   homeo_eval["final_accuracy"],
        "base_auc":      base_eval["auc"],
        "homeo_auc":     homeo_eval["auc"],
        "base_stability":  base_eval["stability_variance"],
        "homeo_stability": homeo_eval["stability_variance"],
        "base_recovery":   base_eval.get("mean_recovery_speed"),
        "homeo_recovery":  homeo_eval.get("mean_recovery_speed"),
        "base_lag":        base_eval.get("mean_adaptation_lag"),
        "homeo_lag":       homeo_eval.get("mean_adaptation_lag"),
    }


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_multi_seed(
    drift_types=None,
    seeds=None,
    n_steps=DEFAULT_N_STEPS,
    drift_point=DEFAULT_DRIFT_POINT,
    n_warmup=DEFAULT_N_WARMUP,
    verbose=True,
) -> dict:
    """Run experiments across all (seed × drift_type) combinations.

    Returns:
        {drift_type: {"runs": [...], "stats": {...}}}
    """
    if drift_types is None: drift_types = DEFAULT_DRIFT_TYPES
    if seeds       is None: seeds       = DEFAULT_SEEDS

    results = {}

    for dtype in drift_types:
        runs = []
        if verbose:
            print(f"\n  [{dtype.upper()}]  {len(seeds)} seeds × {n_steps} steps", flush=True)

        for seed in seeds:
            r = run_single(dtype, n_steps=n_steps, drift_point=drift_point,
                           n_warmup=n_warmup, seed=seed)
            runs.append(r)
            if verbose:
                delta = (r["homeo_post"] or 0) - (r["base_post"] or 0)
                rec   = r["homeo_recovery"]
                print(f"    seed={seed}  post_acc: base={r['base_post'] or 0:.3f} "
                      f"homeo={r['homeo_post'] or 0:.3f}  Δ={delta:+.3f}  "
                      f"recovery={rec}", flush=True)

        stats = _aggregate_stats(runs)
        results[dtype] = {"runs": runs, "stats": stats}

    return results


def _aggregate_stats(runs: list) -> dict:
    """Compute mean ± std and statistical tests across seeds."""
    keys = ["base_overall", "homeo_overall", "base_post", "homeo_post",
            "base_final", "homeo_final", "base_auc", "homeo_auc",
            "base_stability", "homeo_stability",
            "base_recovery", "homeo_recovery",
            "base_lag", "homeo_lag"]

    stats = {}
    for k in keys:
        vals = [r[k] for r in runs if r.get(k) is not None]
        if vals:
            stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    # Statistical tests: homeostatic vs baseline on post-drift accuracy
    hp = [r["homeo_post"] for r in runs if r.get("homeo_post") is not None]
    bp = [r["base_post"]  for r in runs if r.get("base_post")  is not None]
    if len(hp) >= 3 and len(bp) >= 3:
        stats["significance_post"] = paired_test(hp, bp, test="wilcoxon")

    ha = [r["homeo_auc"] for r in runs if r.get("homeo_auc") is not None]
    ba = [r["base_auc"]  for r in runs if r.get("base_auc")  is not None]
    if len(ha) >= 3 and len(ba) >= 3:
        stats["significance_auc"] = paired_test(ha, ba, test="wilcoxon")

    return stats


# ── Formatted report ─────────────────────────────────────────────────────────

def print_summary(results: dict):
    """Print a publication-quality summary table."""
    DRIFT_TYPES = ["regular", "sudden", "gradual", "recurring"]

    print(f"\n{'='*90}")
    print("  DIGITAL HOMEOSTASIS — MULTI-SEED SUMMARY")
    print(f"{'='*90}")

    # ── Post-drift accuracy table ─────────────────────────────────────────
    print(f"\n  POST-DRIFT ACCURACY  (mean ± std over seeds)\n")
    print(f"  {'Drift':<12} {'Base post':>14} {'Homeo post':>14}  {'Δpost':>8}  "
          f"{'p-val':>8}  {'sig':>4}")
    print(f"  {'-'*66}")

    for dtype in DRIFT_TYPES:
        if dtype not in results:
            continue
        s = results[dtype]["stats"]
        bp  = s.get("base_post",  {})
        hp  = s.get("homeo_post", {})
        bm, bs = bp.get("mean"), bp.get("std")
        hm, hs = hp.get("mean"), hp.get("std")
        delta  = (hm - bm) if (hm is not None and bm is not None) else None

        sig_d  = s.get("significance_post", {})
        pval   = sig_d.get("p_value")
        sig    = "★" if sig_d.get("significant") else ""

        def fs(m, sd):
            return f"{m:.3f}±{sd:.3f}" if m is not None else "   N/A   "

        delta_s = f"{delta:+.3f}" if delta is not None else "   N/A"
        pval_s  = f"{pval:.3f}"   if pval  is not None else "  N/A "

        print(f"  {dtype:<12} {fs(bm,bs):>14} {fs(hm,hs):>14}  {delta_s:>8}  "
              f"{pval_s:>8}  {sig:>4}")

    # ── AUC table ─────────────────────────────────────────────────────────
    print(f"\n  AUC ACCURACY  (mean ± std)\n")
    print(f"  {'Drift':<12} {'Base AUC':>14} {'Homeo AUC':>14}  {'ΔAUC':>8}  {'p-val':>8}")
    print(f"  {'-'*66}")

    for dtype in DRIFT_TYPES:
        if dtype not in results:
            continue
        s  = results[dtype]["stats"]
        ba = s.get("base_auc",  {})
        ha = s.get("homeo_auc", {})
        bm, bs = ba.get("mean"), ba.get("std")
        hm, hs = ha.get("mean"), ha.get("std")
        delta  = (hm - bm) if (hm is not None and bm is not None) else None
        sig_d  = s.get("significance_auc", {})
        pval   = sig_d.get("p_value")

        def fs(m, sd): return f"{m:.3f}±{sd:.3f}" if m is not None else "   N/A   "
        delta_s = f"{delta:+.3f}" if delta is not None else "   N/A"
        pval_s  = f"{pval:.3f}"   if pval  is not None else "  N/A "
        print(f"  {dtype:<12} {fs(bm,bs):>14} {fs(hm,hs):>14}  {delta_s:>8}  {pval_s:>8}")

    # ── Recovery speed table ──────────────────────────────────────────────
    print(f"\n  RECOVERY SPEED  (mean steps ± std; None = not recovered)\n")
    print(f"  {'Drift':<12} {'Base rec':>12} {'Homeo rec':>12}  {'Δrec':>8}")
    print(f"  {'-'*52}")

    for dtype in DRIFT_TYPES:
        if dtype not in results:
            continue
        s  = results[dtype]["stats"]
        br = s.get("base_recovery",  {})
        hr = s.get("homeo_recovery", {})
        bm, bs = br.get("mean"), br.get("std")
        hm, hs = hr.get("mean"), hr.get("std")
        delta  = (hm - bm) if (hm is not None and bm is not None) else None

        def fs(m, sd): return f"{m:.1f}±{sd:.1f}" if m is not None else "     N/A  "
        delta_s = f"{delta:+.1f}" if delta is not None else "   N/A"
        print(f"  {dtype:<12} {fs(bm,bs):>12} {fs(hm,hs):>12}  {delta_s:>8}")

    # ── Stability (variance) table ────────────────────────────────────────
    print(f"\n  STABILITY VARIANCE  (lower = more stable)\n")
    print(f"  {'Drift':<12} {'Base stab':>12} {'Homeo stab':>12}  {'Δstab':>8}")
    print(f"  {'-'*52}")

    for dtype in DRIFT_TYPES:
        if dtype not in results:
            continue
        s  = results[dtype]["stats"]
        bs = s.get("base_stability",  {})
        hs = s.get("homeo_stability", {})
        bm, bsd = bs.get("mean"), bs.get("std")
        hm, hsd = hs.get("mean"), hs.get("std")
        delta  = (hm - bm) if (hm is not None and bm is not None) else None

        def fs(m, sd): return f"{m:.4f}±{sd:.4f}" if m is not None else "    N/A   "
        delta_s = f"{delta:+.4f}" if delta is not None else "   N/A"
        print(f"  {dtype:<12} {fs(bm,bsd):>12} {fs(hm,hsd):>12}  {delta_s:>8}")

    print(f"\n  ★ = p < 0.05 (Wilcoxon signed-rank, one-sided, paired)\n")
    print(f"{'='*90}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Digital Homeostasis experiment runner")
    p.add_argument("--seeds",  type=int, default=10,
                   help="Number of seeds (starting at 42)")
    p.add_argument("--steps",  type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--warmup", type=int, default=DEFAULT_N_WARMUP)
    p.add_argument("--drift",  type=str, default=None,
                   help="Single drift type (default: all four)")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip visualization")
    return p.parse_args()


if __name__ == "__main__":
    args   = _parse_args()
    seeds  = list(range(42, 42 + args.seeds))
    dtypes = [args.drift] if args.drift else DEFAULT_DRIFT_TYPES

    print(f"\n  Seeds: {seeds}")
    print(f"  Drift types: {dtypes}")
    print(f"  Steps per run: {args.steps}  |  Warm-up: {args.warmup}")

    results = run_multi_seed(
        drift_types=dtypes,
        seeds=seeds,
        n_steps=args.steps,
        n_warmup=args.warmup,
        verbose=True,
    )
    print_summary(results)

    if not args.no_plot:
        try:
            from visualization.plots import plot_all
            plot_all(results, save_dir="figures")
            print("  Figures saved to figures/")
        except ImportError as e:
            print(f"  [visualization skipped: {e}]")
