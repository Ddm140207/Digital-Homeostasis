"""
evaluation/drift_evaluator.py — Drift-aware evaluation metrics.

Metrics computed:
    overall_accuracy        — mean accuracy over the full run
    pre_drift_accuracy      — mean accuracy in the window before each drift point
    post_drift_accuracy     — mean accuracy in the window after each drift point
    immediate_drop          — accuracy in the first 10 steps after drift
    recovery_speed          — steps to recover to 80% of pre-drift accuracy
    final_accuracy          — mean accuracy over the last `window` steps
    stability               — 1 - std(rolling accuracy)  [higher = smoother]
    cumulative_regret       — total number of errors (lower = better)
    rolling_accuracy        — per-step rolling accuracy array (window-smoothed)

Compare two runs with DriftEvaluator.compare(baseline_errors, homeo_errors).
"""

import numpy as np
from collections import defaultdict


class DriftEvaluator:

    def __init__(self, drift_points, window=50):
        """
        Args:
            drift_points: List of step indices where concept drift occurs.
            window:       Pre/post analysis window (steps on each side of drift).
        """
        self.drift_points = list(drift_points)
        self.window       = window

    # ── Public API ───────────────────────────────────────────────────────

    def evaluate(self, errors, label="model"):
        """Compute all drift metrics for one error sequence.

        Args:
            errors: 1D array-like of {0, 1} per-step error indicators.
            label:  String label for printing.

        Returns:
            dict with all metrics.  Arrays (rolling_accuracy, per_drift) are
            included for downstream plotting.
        """
        errors = np.asarray(errors, dtype=float)
        n      = len(errors)
        results = {"label": label, "n_steps": n}

        results["overall_accuracy"]  = float(1.0 - np.mean(errors))
        results["cumulative_regret"] = int(np.sum(errors))
        results["final_accuracy"]    = float(1.0 - np.mean(errors[-self.window:]))

        rolling = self._rolling_accuracy(errors, self.window)
        results["rolling_accuracy"]  = rolling
        results["stability"]         = float(1.0 - np.std(rolling))

        # Per-drift-point analysis
        per_drift = []
        for dp in self.drift_points:
            if dp >= n:
                continue

            pre_errors  = errors[max(0, dp - self.window):dp]
            post_errors = errors[dp:min(n, dp + self.window)]
            imm_errors  = errors[dp:min(n, dp + 10)]

            pre_acc  = float(1.0 - np.mean(pre_errors))  if len(pre_errors)  > 0 else None
            post_acc = float(1.0 - np.mean(post_errors)) if len(post_errors) > 0 else None
            imm_acc  = float(1.0 - np.mean(imm_errors))  if len(imm_errors)  > 0 else None
            delta    = (post_acc - pre_acc) if (pre_acc is not None and post_acc is not None) else None

            recovery = self._recovery_speed(errors, dp, pre_acc)
            per_drift.append({
                "drift_point":    dp,
                "pre_acc":        pre_acc,
                "post_acc":       post_acc,
                "immediate_acc":  imm_acc,
                "delta":          delta,
                "recovery_speed": recovery,
            })

        results["per_drift"] = per_drift

        if per_drift:
            valid_delta   = [d["delta"]          for d in per_drift if d["delta"]          is not None]
            valid_recov   = [d["recovery_speed"]  for d in per_drift if d["recovery_speed"] is not None]
            results["mean_pre_drift_acc"]    = float(np.mean([d["pre_acc"]  for d in per_drift if d["pre_acc"]  is not None]))
            results["mean_post_drift_acc"]   = float(np.mean([d["post_acc"] for d in per_drift if d["post_acc"] is not None]))
            results["mean_post_drift_delta"] = float(np.mean(valid_delta)) if valid_delta else None
            results["mean_recovery_speed"]   = float(np.mean(valid_recov)) if valid_recov else None

        return results

    def compare(self, baseline_errors, homeo_errors, label_base="baseline",
                label_homeo="homeostatic"):
        """Side-by-side evaluation table.

        Returns:
            (base_results, homeo_results) dicts.
        """
        base_r  = self.evaluate(baseline_errors, label=label_base)
        homeo_r = self.evaluate(homeo_errors,    label=label_homeo)

        def fv(v):
            return f"{v:>9.4f}" if v is not None else "      N/A"

        def fd(b, h):
            return f"{h - b:>+10.4f}" if (b is not None and h is not None) else "       N/A"

        rows = [
            ("Overall accuracy",         "overall_accuracy"),
            ("Final accuracy",            "final_accuracy"),
            ("Stability (1-std rolling)", "stability"),
            ("Cumulative regret",         "cumulative_regret"),
            ("Mean pre-drift acc",        "mean_pre_drift_acc"),
            ("Mean post-drift acc",       "mean_post_drift_acc"),
            ("Mean post-drift delta",     "mean_post_drift_delta"),
            ("Mean recovery speed (steps)","mean_recovery_speed"),
        ]

        print(f"\n{'='*72}")
        print(f"  DRIFT EVALUATION  |  drift points: {self.drift_points}  |  window: {self.window}")
        print(f"{'='*72}")
        print(f"  {'Metric':<34} {label_base:>10} {label_homeo:>12}  Delta")
        print(f"  {'-'*68}")
        for name, key in rows:
            bv = base_r.get(key)
            hv = homeo_r.get(key)
            print(f"  {name:<34} {fv(bv)} {fv(hv)} {fd(bv, hv)}")

        print(f"\n  Per-drift breakdown:")
        for pd_b, pd_h in zip(base_r.get("per_drift", []), homeo_r.get("per_drift", [])):
            dp = pd_b["drift_point"]
            print(f"    drift @ step {dp:>4}:")
            print(f"      pre-drift  acc: base={fv(pd_b['pre_acc'])}  homeo={fv(pd_h['pre_acc'])}")
            print(f"      post-drift acc: base={fv(pd_b['post_acc'])} homeo={fv(pd_h['post_acc'])}  Δ={fd(pd_b['post_acc'], pd_h['post_acc'])}")
            print(f"      recovery (steps): base={pd_b['recovery_speed']}  homeo={pd_h['recovery_speed']}")

        return base_r, homeo_r

    # ── Multi-seed experiment runner ─────────────────────────────────────

    @staticmethod
    def multi_seed_summary(run_fn, seeds, drift_types, n_steps, drift_point):
        """Run experiments across multiple seeds and summarise.

        Args:
            run_fn:      Callable(drift_type, n_steps, drift_point, seed) → result dict
                         (same signature as test_all.run_experiment).
            seeds:       List of integer seeds.
            drift_types: List of drift type strings.
            n_steps:     Steps per run.
            drift_point: Drift point index.

        Returns:
            summary dict: {drift_type: {metric: (mean, std)}}
        """
        from scipy import stats as scipy_stats   # optional — guarded import

        all_results = defaultdict(lambda: defaultdict(list))

        for seed in seeds:
            for dtype in drift_types:
                r = run_fn(dtype, n_steps=n_steps, drift_point=drift_point, seed=seed)
                all_results[dtype]["homeo_post_acc"].append(r.get("homeo_post_acc") or 0)
                all_results[dtype]["base_post_acc"].append(r.get("base_post_acc")   or 0)
                all_results[dtype]["homeo_final_acc"].append(r.get("homeo_final_acc") or 0)
                all_results[dtype]["base_final_acc"].append(r.get("base_final_acc")   or 0)
                all_results[dtype]["mean_homeo_score"].append(r.get("mean_homeo_score") or 0)

        summary = {}
        print(f"\n{'='*80}")
        print(f"  MULTI-SEED SUMMARY  ({len(seeds)} seeds × {len(drift_types)} drift types)")
        print(f"{'='*80}")
        header = f"  {'Drift':<12} {'HomPost(μ±σ)':>18} {'BasePost(μ±σ)':>18} {'Δpost':>8} {'p-val':>8}"
        print(header)
        print(f"  {'-'*76}")

        for dtype in drift_types:
            hp = np.array(all_results[dtype]["homeo_post_acc"])
            bp = np.array(all_results[dtype]["base_post_acc"])
            delta = np.mean(hp) - np.mean(bp)

            try:
                _, pval = scipy_stats.ttest_ind(hp, bp)
                pval_str = f"{pval:.3f}"
            except Exception:
                pval_str = "  N/A"

            print(
                f"  {dtype:<12} "
                f"{np.mean(hp):>7.4f}±{np.std(hp):.4f}  "
                f"{np.mean(bp):>7.4f}±{np.std(bp):.4f}  "
                f"{delta:>+8.4f}  "
                f"{pval_str:>8}"
            )
            summary[dtype] = {
                "homeo_post": (float(np.mean(hp)), float(np.std(hp))),
                "base_post":  (float(np.mean(bp)), float(np.std(bp))),
                "delta_post": float(delta),
            }

        return summary

    # ── Private helpers ──────────────────────────────────────────────────

    def _rolling_accuracy(self, errors, window):
        n   = len(errors)
        acc = np.zeros(n)
        for i in range(n):
            start  = max(0, i - window + 1)
            acc[i] = 1.0 - float(np.mean(errors[start:i + 1]))
        return acc

    def _recovery_speed(self, errors, drift_point, pre_acc, threshold=0.80,
                        max_steps=500):
        """Steps until post-drift rolling accuracy (window=20) reaches threshold × pre_acc.

        BUG FIX: window is clamped to [drift_point, step] so pre-drift samples
        never inflate the post-drift rolling mean, which previously caused
        recovery_speed = 0 on every run (the window still contained 19 high-
        accuracy pre-drift samples at the very first post-drift step).
        """
        if pre_acc is None or pre_acc < 0.01:
            return None
        target = pre_acc * threshold
        n      = len(errors)
        win    = 20
        for step in range(drift_point, min(drift_point + max_steps, n)):
            # FIXED: clamp start to drift_point — never look before the drift
            start   = max(drift_point, step - win + 1)
            rolling = 1.0 - float(np.mean(errors[start:step + 1]))
            if rolling >= target:
                return step - drift_point
        return None   # did not recover within max_steps
