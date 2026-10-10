"""
visualization/plots.py — Paper-quality figures for Digital Homeostasis.

Produces four figures suitable for a research paper:

  Figure 1 — accuracy_over_time.png
      Per-step accuracy for baseline vs homeostatic model across all drift
      types.  Drift point marked with a vertical dashed line.

  Figure 2 — rolling_accuracy.png
      Smoothed (rolling window) accuracy.  Highlights the stable pre-drift
      plateau, the post-drift collapse, and the recovery trajectory.

  Figure 3 — recovery_zoom.png
      Post-drift zoom (0–200 steps after drift).  Shows how quickly each
      model recovers relative to the pre-drift baseline.

  Figure 4 — benchmark_comparison.png
      Multi-seed bar chart comparing post-drift accuracy (mean ± std) across
      all drift types.  Statistical significance annotations included.

Usage
-----
    from visualization.plots import plot_all, plot_single_run
    plot_all(multi_seed_results, save_dir="figures")
    plot_single_run(single_run_result, save_dir="figures")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional, Dict, List

# ── Style ─────────────────────────────────────────────────────────────────────

STYLE = {
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.grid":           True,
    "grid.alpha":          0.35,
    "grid.linestyle":      "--",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.family":         "sans-serif",
    "font.size":           10,
    "axes.titlesize":      11,
    "axes.labelsize":      10,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "lines.linewidth":     1.5,
    "lines.antialiased":   True,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
}

COLORS = {
    "baseline":    "#E05A45",   # warm red
    "homeostatic": "#2E7FBF",   # steel blue
    "drift":       "#555555",   # neutral grey for drift marker
    "pre_target":  "#888888",   # dashed pre-drift target line
}

DRIFT_LABELS = {
    "regular":   "Regular (control)",
    "sudden":    "Sudden drift",
    "gradual":   "Gradual drift",
    "recurring": "Recurring drift",
}

DRIFT_ORDER = ["regular", "sudden", "gradual", "recurring"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _mark_drift(ax, drift_point: int, n_steps: int, label: bool = True):
    ax.axvline(x=drift_point, color=COLORS["drift"], linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=5)
    if label:
        ax.text(drift_point + n_steps * 0.01, ax.get_ylim()[1] * 0.97,
                "drift", fontsize=8, color=COLORS["drift"], va="top")


def _rolling(errors: np.ndarray, window: int = 50) -> np.ndarray:
    n = len(errors)
    acc = np.empty(n)
    for i in range(n):
        s = max(0, i - window + 1)
        acc[i] = 1.0 - float(np.mean(errors[s:i + 1]))
    return acc


def _legend_handles():
    return [
        Line2D([0], [0], color=COLORS["baseline"],    linewidth=1.8, label="Baseline"),
        Line2D([0], [0], color=COLORS["homeostatic"], linewidth=1.8, label="Homeostatic"),
        Line2D([0], [0], color=COLORS["drift"],       linewidth=1.2,
               linestyle="--", label="Drift point"),
    ]


# ── Figure 1: Accuracy over time ──────────────────────────────────────────────

def plot_accuracy_over_time(
    runs_by_dtype: dict,
    save_dir: str = "figures",
    smooth_window: int = 30,
):
    """One sub-plot per drift type; per-step accuracy (lightly smoothed)."""
    _ensure_dir(save_dir)
    dtypes   = [d for d in DRIFT_ORDER if d in runs_by_dtype]
    n_plots  = len(dtypes)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.0 * n_plots, 3.4),
                             sharey=True)
    if n_plots == 1:
        axes = [axes]

    with plt.rc_context(STYLE):
        for ax, dtype in zip(axes, dtypes):
            data    = runs_by_dtype[dtype]
            runs    = data["runs"]
            dp      = runs[0]["drift_point"]
            n       = runs[0]["n_steps"]

            # Average over seeds
            base_roll  = np.mean([_rolling(r["base_errors"],  smooth_window) for r in runs], axis=0)
            homeo_roll = np.mean([_rolling(r["homeo_errors"], smooth_window) for r in runs], axis=0)
            base_std   = np.std( [_rolling(r["base_errors"],  smooth_window) for r in runs], axis=0)
            homeo_std  = np.std( [_rolling(r["homeo_errors"], smooth_window) for r in runs], axis=0)

            x = np.arange(n)
            ax.plot(x, base_roll,  color=COLORS["baseline"],    alpha=0.85, zorder=3)
            ax.plot(x, homeo_roll, color=COLORS["homeostatic"], alpha=0.85, zorder=4)
            ax.fill_between(x, base_roll  - base_std,  base_roll  + base_std,
                            color=COLORS["baseline"],    alpha=0.12, zorder=2)
            ax.fill_between(x, homeo_roll - homeo_std, homeo_roll + homeo_std,
                            color=COLORS["homeostatic"], alpha=0.12, zorder=2)

            ax.axvline(x=dp, color=COLORS["drift"], linewidth=1.0, linestyle="--", alpha=0.8)
            ax.set_title(DRIFT_LABELS.get(dtype, dtype), fontweight="bold")
            ax.set_xlabel("Step")
            ax.set_ylim(0.3, 1.05)
            ax.set_xlim(0, n)

        axes[0].set_ylabel("Accuracy (rolling)")
        axes[0].legend(handles=_legend_handles(), loc="lower left", framealpha=0.85)

        fig.suptitle("Accuracy over Time — Baseline vs Homeostatic",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(save_dir, "accuracy_over_time.png")
        fig.savefig(path, format="png")
        plt.close(fig)
    print(f"    saved: {path}")


# ── Figure 2: Rolling accuracy (wide window) ──────────────────────────────────

def plot_rolling_accuracy(
    runs_by_dtype: dict,
    save_dir: str = "figures",
    window: int = 50,
):
    """Same layout as Fig 1 but with a wider rolling window for stability."""
    _ensure_dir(save_dir)
    dtypes  = [d for d in DRIFT_ORDER if d in runs_by_dtype]
    n_plots = len(dtypes)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.0 * n_plots, 3.4),
                             sharey=True)
    if n_plots == 1:
        axes = [axes]

    with plt.rc_context(STYLE):
        for ax, dtype in zip(axes, dtypes):
            data  = runs_by_dtype[dtype]
            runs  = data["runs"]
            dp    = runs[0]["drift_point"]
            n     = runs[0]["n_steps"]

            base_roll  = np.mean([_rolling(r["base_errors"],  window) for r in runs], axis=0)
            homeo_roll = np.mean([_rolling(r["homeo_errors"], window) for r in runs], axis=0)
            base_std   = np.std( [_rolling(r["base_errors"],  window) for r in runs], axis=0)
            homeo_std  = np.std( [_rolling(r["homeo_errors"], window) for r in runs], axis=0)

            x = np.arange(n)
            ax.plot(x, base_roll,  color=COLORS["baseline"],    alpha=0.85, zorder=3)
            ax.plot(x, homeo_roll, color=COLORS["homeostatic"], alpha=0.85, zorder=4)
            ax.fill_between(x, base_roll  - base_std,  base_roll  + base_std,
                            color=COLORS["baseline"],    alpha=0.12, zorder=2)
            ax.fill_between(x, homeo_roll - homeo_std, homeo_roll + homeo_std,
                            color=COLORS["homeostatic"], alpha=0.12, zorder=2)

            ax.axvline(x=dp, color=COLORS["drift"], linewidth=1.0, linestyle="--", alpha=0.8)
            ax.set_title(DRIFT_LABELS.get(dtype, dtype), fontweight="bold")
            ax.set_xlabel("Step")
            ax.set_ylim(0.3, 1.05)
            ax.set_xlim(0, n)

        axes[0].set_ylabel(f"Rolling accuracy (w={window})")
        axes[0].legend(handles=_legend_handles(), loc="lower left", framealpha=0.85)
        fig.suptitle("Rolling Accuracy — Stability and Recovery Comparison",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(save_dir, "rolling_accuracy.png")
        fig.savefig(path, format="png")
        plt.close(fig)
    print(f"    saved: {path}")


# ── Figure 3: Post-drift zoom (recovery curves) ───────────────────────────────

def plot_recovery_zoom(
    runs_by_dtype: dict,
    save_dir: str = "figures",
    post_window: int = 200,
    smooth_window: int = 20,
):
    """Zoom into the first `post_window` steps after each drift.

    Shows: pre-drift accuracy as a horizontal target line, baseline collapse,
    homeostatic recovery trajectory, and the recovery target (80% of pre).
    """
    _ensure_dir(save_dir)
    dtypes  = [d for d in DRIFT_ORDER if d in runs_by_dtype
               if d != "regular"]   # no drift to zoom on for regular
    if not dtypes:
        return
    n_plots = len(dtypes)
    fig, axes = plt.subplots(1, n_plots, figsize=(4.2 * n_plots, 3.6), sharey=False)
    if n_plots == 1:
        axes = [axes]

    with plt.rc_context(STYLE):
        for ax, dtype in zip(axes, dtypes):
            data = runs_by_dtype[dtype]
            runs = data["runs"]
            dp   = runs[0]["drift_point"]

            # Build recovery curves per seed, then average
            def _curve(errors):
                n   = len(errors)
                seg = errors[dp:min(n, dp + post_window)]
                return _rolling(seg, smooth_window)

            base_curves  = [_curve(r["base_errors"])  for r in runs]
            homeo_curves = [_curve(r["homeo_errors"]) for r in runs]
            min_len = min(min(len(c) for c in base_curves),
                          min(len(c) for c in homeo_curves))
            base_curves  = np.array([c[:min_len] for c in base_curves])
            homeo_curves = np.array([c[:min_len] for c in homeo_curves])

            bm = np.mean(base_curves,  axis=0)
            hm = np.mean(homeo_curves, axis=0)
            bs = np.std(base_curves,   axis=0)
            hs = np.std(homeo_curves,  axis=0)

            x = np.arange(min_len)
            ax.plot(x, bm, color=COLORS["baseline"],    alpha=0.85, zorder=3)
            ax.plot(x, hm, color=COLORS["homeostatic"], alpha=0.85, zorder=4)
            ax.fill_between(x, bm - bs, bm + bs, color=COLORS["baseline"],    alpha=0.15)
            ax.fill_between(x, hm - hs, hm + hs, color=COLORS["homeostatic"], alpha=0.15)

            # Pre-drift reference line
            pre_acc_vals = []
            for r in runs:
                pre = r["base_errors"][max(0, dp - 50):dp]
                if len(pre) > 0:
                    pre_acc_vals.append(1.0 - float(np.mean(pre)))
            if pre_acc_vals:
                pre_acc = float(np.mean(pre_acc_vals))
                ax.axhline(y=pre_acc, color=COLORS["pre_target"], linewidth=1.0,
                           linestyle=":", alpha=0.8, label=f"Pre-drift acc ({pre_acc:.2f})")
                ax.axhline(y=pre_acc * 0.80, color=COLORS["pre_target"], linewidth=0.8,
                           linestyle="-.", alpha=0.6, label="Recovery target (80%)")

            ax.set_title(DRIFT_LABELS.get(dtype, dtype), fontweight="bold")
            ax.set_xlabel("Steps after drift")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.3, 1.05)
            ax.set_xlim(0, min_len)
            ax.legend(handles=_legend_handles()[:2] + [
                Line2D([0], [0], color=COLORS["pre_target"], linewidth=1.0,
                       linestyle=":", label="Pre-drift level"),
                Line2D([0], [0], color=COLORS["pre_target"], linewidth=0.8,
                       linestyle="-.", label="80% recovery target"),
            ], loc="lower right", fontsize=8, framealpha=0.85)

        fig.suptitle("Post-Drift Recovery Zoom — Baseline Collapse vs Homeostatic Recovery",
                     fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        path = os.path.join(save_dir, "recovery_zoom.png")
        fig.savefig(path, format="png")
        plt.close(fig)
    print(f"    saved: {path}")


# ── Figure 4: Benchmark comparison bar chart ──────────────────────────────────

def plot_benchmark_comparison(
    runs_by_dtype: dict,
    save_dir: str = "figures",
    metric: str = "post",   # "post" | "overall" | "final" | "auc"
):
    """Grouped bar chart: post-drift accuracy mean ± std for all drift types."""
    _ensure_dir(save_dir)
    dtypes = [d for d in DRIFT_ORDER if d in runs_by_dtype]
    if not dtypes:
        return

    x     = np.arange(len(dtypes))
    width = 0.32

    key_base  = f"base_{metric}"
    key_homeo = f"homeo_{metric}"
    metric_label = {"post": "Post-drift accuracy", "overall": "Overall accuracy",
                    "final": "Final-window accuracy", "auc": "AUC accuracy"}.get(metric, metric)

    base_means, base_stds   = [], []
    homeo_means, homeo_stds = [], []
    pvals = []

    for dtype in dtypes:
        s = runs_by_dtype[dtype]["stats"]
        bm = s.get(key_base,  {}).get("mean",  0.0)
        bs = s.get(key_base,  {}).get("std",   0.0)
        hm = s.get(key_homeo, {}).get("mean",  0.0)
        hs = s.get(key_homeo, {}).get("std",   0.0)
        base_means.append(bm);  base_stds.append(bs)
        homeo_means.append(hm); homeo_stds.append(hs)

        sig = runs_by_dtype[dtype]["stats"].get("significance_post", {})
        pvals.append(sig.get("p_value", None))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))

        bars_b = ax.bar(x - width / 2, base_means,  width, yerr=base_stds,
                        color=COLORS["baseline"],    alpha=0.80,
                        error_kw={"elinewidth": 1.2, "capsize": 4},
                        label="Baseline", zorder=3)
        bars_h = ax.bar(x + width / 2, homeo_means, width, yerr=homeo_stds,
                        color=COLORS["homeostatic"], alpha=0.80,
                        error_kw={"elinewidth": 1.2, "capsize": 4},
                        label="Homeostatic", zorder=3)

        # Significance annotations
        for i, (bm, hm, pv) in enumerate(zip(base_means, homeo_means, pvals)):
            if pv is not None and pv < 0.05:
                top = max(bm, hm) + max(base_stds[i], homeo_stds[i]) + 0.015
                ax.text(i, top, "★", ha="center", va="bottom", fontsize=12,
                        color="#333333", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([DRIFT_LABELS.get(d, d) for d in dtypes], rotation=10, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_ylim(0.4, 1.05)
        ax.legend(loc="upper left", framealpha=0.90)
        ax.set_title(f"Benchmark Comparison — {metric_label} (mean ± std)",
                     fontweight="bold")

        note = "★ p < 0.05 (Wilcoxon signed-rank, one-sided, paired)"
        ax.annotate(note, xy=(0.01, 0.01), xycoords="axes fraction",
                    fontsize=7.5, color="#555555", va="bottom")

        plt.tight_layout()
        path = os.path.join(save_dir, "benchmark_comparison.png")
        fig.savefig(path, format="png")
        plt.close(fig)
    print(f"    saved: {path}")


# ── Figure 5: Homeostasis score + controller actions ─────────────────────────

def plot_controller_trace(
    single_run: dict,
    save_dir: str = "figures",
):
    """Internal controller trace for a single run (useful for debugging/paper supplement).

    Panels:
      Top    — rolling accuracy (baseline vs homeostatic)
      Middle — homeostasis score H over time
      Bottom — per-step action as a categorical raster
    """
    _ensure_dir(save_dir)
    runs      = single_run if isinstance(single_run, list) else [single_run]
    r         = runs[0]
    n         = r["n_steps"]
    dp        = r["drift_point"]
    dtype     = r["drift_type"]

    roll_b    = _rolling(r["base_errors"],  50)
    roll_h    = _rolling(r["homeo_errors"], 50)
    H_scores  = r["homeo_scores"]

    ACTION_ORDER  = ["NO_ACTION", "FAST_ADAPT", "HORMONAL_ADAPT", "STRUCTURAL_ADAPT"]
    ACTION_COLORS = {"NO_ACTION": "#AAAAAA", "FAST_ADAPT": "#5BA4CF",
                     "HORMONAL_ADAPT": "#F6A623", "STRUCTURAL_ADAPT": "#D0021B"}
    action_map    = {a: i for i, a in enumerate(ACTION_ORDER)}

    # Reconstruct per-step action from decision_logs
    logs = r.get("decision_logs", [])
    action_idx = np.zeros(n, dtype=int)
    for i, log in enumerate(logs):
        a = log.get("action", "NO_ACTION")
        action_idx[i] = action_map.get(a, 0)

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.0),
                                 gridspec_kw={"height_ratios": [2, 1.5, 0.8]},
                                 sharex=True)
        x = np.arange(n)

        # Panel 1: accuracy
        axes[0].plot(x, roll_b, color=COLORS["baseline"],    alpha=0.85)
        axes[0].plot(x, roll_h, color=COLORS["homeostatic"], alpha=0.85)
        axes[0].axvline(dp, color=COLORS["drift"], linewidth=1.0, linestyle="--", alpha=0.8)
        axes[0].set_ylabel("Rolling accuracy")
        axes[0].set_ylim(0.3, 1.05)
        axes[0].legend(handles=_legend_handles(), loc="lower left", fontsize=8,
                       framealpha=0.85)
        axes[0].set_title(f"Controller Trace — {DRIFT_LABELS.get(dtype, dtype)}",
                          fontweight="bold")

        # Panel 2: homeostasis score
        axes[1].plot(x, H_scores, color="#4A90D9", linewidth=1.2, alpha=0.85)
        axes[1].axhline(0.70, color="#999999", linewidth=0.8, linestyle=":",
                        label="NO_ACTION threshold (0.70)")
        axes[1].axvline(dp, color=COLORS["drift"], linewidth=1.0, linestyle="--", alpha=0.8)
        axes[1].set_ylabel("Homeostasis H")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend(loc="lower left", fontsize=8, framealpha=0.85)

        # Panel 3: action raster
        cmap_colors = [ACTION_COLORS[a] for a in ACTION_ORDER]
        action_arr  = action_idx.reshape(1, -1)
        import matplotlib.colors as mcolors
        cmap  = mcolors.ListedColormap(cmap_colors)
        norm  = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(ACTION_ORDER)), ncolors=len(ACTION_ORDER))
        axes[2].imshow(action_arr, aspect="auto", cmap=cmap, norm=norm,
                       extent=[0, n, -0.5, 0.5], interpolation="nearest")
        axes[2].axvline(dp, color="white", linewidth=1.0, linestyle="--", alpha=0.9)
        axes[2].set_yticks([])
        axes[2].set_ylabel("Action")
        axes[2].set_xlabel("Step")

        patches = [mpatches.Patch(color=ACTION_COLORS[a], label=a)
                   for a in ACTION_ORDER]
        axes[2].legend(handles=patches, loc="upper right", fontsize=7,
                       ncol=2, framealpha=0.9)

        plt.tight_layout()
        path = os.path.join(save_dir, f"controller_trace_{dtype}.png")
        fig.savefig(path, format="png")
        plt.close(fig)
    print(f"    saved: {path}")


# ── Convenience: plot everything ──────────────────────────────────────────────

def plot_all(runs_by_dtype: dict, save_dir: str = "figures"):
    """Generate all paper figures from multi-seed results.

    Args:
        runs_by_dtype: Output of experiments.run_experiments.run_multi_seed().
        save_dir:      Directory to write PDF files.
    """
    _ensure_dir(save_dir)
    print(f"\n  Generating figures → {save_dir}/")
    plot_accuracy_over_time(runs_by_dtype, save_dir=save_dir)
    plot_rolling_accuracy(runs_by_dtype, save_dir=save_dir)
    plot_recovery_zoom(runs_by_dtype, save_dir=save_dir)
    plot_benchmark_comparison(runs_by_dtype, save_dir=save_dir)
    # Controller trace for one representative run per drift type
    for dtype, data in runs_by_dtype.items():
        if data["runs"]:
            plot_controller_trace(data["runs"][0], save_dir=save_dir)


def plot_single_run(run: dict, save_dir: str = "figures"):
    """Quick plot for a single experiment run (e.g. from test_all.run_experiment)."""
    _ensure_dir(save_dir)
    dtype = run.get("drift_type", "unknown")
    print(f"\n  Single-run figures for {dtype} → {save_dir}/")
    plot_controller_trace(run, save_dir=save_dir)
