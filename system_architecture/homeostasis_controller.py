import numpy as np
from collections import deque


class HomeostasisController:
    """Bio-inspired homeostasis controller with explicit plasticity mode.

    Key fixes vs previous version:
        1. Drift override path: when drift_detected=True is passed by the monitor,
           the controller forces HORMONAL_ADAPT (intensive rapid learning),
           resets the PID integral (prevents pre-drift health from biasing
           the response), and enters PLASTICITY_MODE.
        2. Plasticity mode: active for PLASTICITY_DURATION steps after each drift.
           - Target self-tuning is suppressed (prevents accepting high error as normal).
           - Reduced cooldowns (base: 10 structural / 2 hormonal vs 40 / 6).
           - Once enough post-drift samples accumulate, allows STRUCTURAL_ADAPT.
        3. Cooldowns halved (base_structural: 40→20) so a failed first rebuild
           can be retried within a useful time window.
        4. Kd raised 0.15→0.25 for sharper derivative response to error spikes.

    Stability/plasticity separation:
        - Stable (H > 0.85, no plasticity): slow target update, conservative LR,
          cooldowns at max, integral decay.
        - Plasticity (drift event → next 100 steps): aggressive LR, no target
          drift, short cooldowns, epoch-filtered replay in AdaptiveMechanisms.
    """

    METRIC_KEYS = [
        "error_rate",
        "prediction_confidence",
        "model_entropy",
        "data_shift",
        "memory_load",
    ]

    # Directional deviation: only penalise deviations in the harmful direction.
    # e.g. error_rate below target = "too accurate" = not stressful.
    # This prevents the controller from treating high accuracy as homeostatic stress.
    HIGHER_IS_WORSE = {"error_rate", "model_entropy", "data_shift"}
    LOWER_IS_WORSE  = {"prediction_confidence"}
    # memory_load: bidirectional — too empty or too full both matter

    # ── Duration for which plasticity mode stays active after each drift ─
    PLASTICITY_DURATION = 100   # steps

    def __init__(self):
        # ── Adaptive targets (self-tune only during stable operation) ────
        # Calibrated for realistic SGD online learning performance on 10D
        # random linear data.  Previous values (error_rate=0.15, entropy=0.50)
        # were too optimistic: a well-trained SGD achieves ~75–80% accuracy
        # (error_rate ≈ 0.20–0.25) and binary log-loss entropy at inference is
        # typically 0.7–0.9, not 0.5.  With the old targets, H on *stable*
        # data was structurally ~0.59 — permanently below the NO_ACTION
        # threshold (0.70) — so the controller over-adapted even when the
        # model was healthy.  Targets may be further refined by calling
        # calibrate(warmup_metrics_list) after initial training.
        self.targets = {
            "error_rate":            0.22,   # was 0.15 → ~78% acc is "healthy"
            "prediction_confidence": 0.70,
            "model_entropy":         0.75,   # was 0.50 → binary SGD is naturally less sharp
            "data_shift":            0.05,
            "memory_load":           0.90,
        }
        self.scales = {
            "error_rate":            0.10,   # wider tolerance (was 0.05)
            "prediction_confidence": 0.15,
            "model_entropy":         0.25,
            "data_shift":            0.15,
            "memory_load":           0.20,
        }
        self.weights = {
            "error_rate":            0.50,
            "prediction_confidence": 0.18,
            "model_entropy":         0.07,
            "data_shift":            0.18,
            "memory_load":           0.07,
        }

        # ── PID gains ────────────────────────────────────────────────────
        self.Kp = 0.85
        self.Ki = 0.008
        self.Kd = 0.25     # raised from 0.15 for sharper spike detection

        self.integral_term    = 0.0
        self.last_error       = 0.0
        self.last_homeostasis = None

        # ── Multi-timescale hormones ──────────────────────────────────────
        self.hormone_fast   = 0.0
        self.hormone_medium = 0.0
        self.hormone_slow   = 0.0

        # ── History ───────────────────────────────────────────────────────
        self._history_len   = 80
        self.metric_history = {k: deque(maxlen=self._history_len) for k in self.METRIC_KEYS}
        self._target_update_rate = 0.003
        self._trend_window  = 12
        self._last_values   = {}

        # ── Cooldowns (reduced vs original) ──────────────────────────────
        self.base_structural_cooldown = 20    # was 40
        self.base_hormonal_cooldown   = 4     # was 6
        self.structural_cooldown      = self.base_structural_cooldown
        self.hormonal_cooldown        = self.base_hormonal_cooldown
        self.last_structural_step     = -1000
        self.last_hormonal_step       = -1000

        self.step = 0

        # ── Anti-deterioration ────────────────────────────────────────────
        self._recent_H   = deque(maxlen=80)
        self._best_H_avg = 0.0

        # ── Plasticity mode ───────────────────────────────────────────────
        self.plasticity_mode             = False
        self.plasticity_steps_remaining  = 0
        self._post_drift_steps           = 0   # steps since last drift signal

    # ── Core computations ────────────────────────────────────────────────

    def normalize_deviation(self, metric, value):
        mu    = self.targets.get(metric, 0.0)
        sigma = self.scales.get(metric, 1.0)
        return (value - mu) / (sigma + 1e-8)

    def compute_homeostasis(self, metrics):
        deviations   = {}
        weighted_abs = 0.0

        for key in self.METRIC_KEYS:
            value = metrics.get(key)
            if value is None:
                continue
            dev = self.normalize_deviation(key, value)
            deviations[key] = dev

            # Asymmetric: only penalise harmful deviations, not beneficial ones.
            if key in self.HIGHER_IS_WORSE:
                effective = max(0.0, dev)     # only penalise when above target
            elif key in self.LOWER_IS_WORSE:
                effective = max(0.0, -dev)    # only penalise when below target
            else:
                effective = abs(dev)          # bidirectional (memory_load)

            weighted_abs += self.weights.get(key, 0.0) * effective

        # Use effective deviations for synergy too
        eff_vals = []
        for key in self.METRIC_KEYS:
            if key not in deviations:
                continue
            dev = deviations[key]
            if key in self.HIGHER_IS_WORSE:    eff_vals.append(max(0.0, dev))
            elif key in self.LOWER_IS_WORSE:   eff_vals.append(max(0.0, -dev))
            else:                              eff_vals.append(abs(dev))

        if len(eff_vals) >= 2:
            sorted_eff = sorted(eff_vals, reverse=True)
            synergy    = np.sqrt(sorted_eff[0] * sorted_eff[1]) * 0.12
        else:
            synergy = 0.0

        imbalance = weighted_abs + synergy
        H = float(np.exp(-imbalance))
        return H, deviations

    def compute_control_signal(self, H):
        error            = 1.0 - H
        derivative       = error - self.last_error
        self.integral_term += error
        self.integral_term = float(np.clip(self.integral_term, -2.5, 2.5))
        U = self.Kp * error + self.Ki * self.integral_term + self.Kd * derivative
        U = float(np.clip(U, 0.0, 1.0))
        self.last_error = error
        return U

    # ── Trend / hormone / target helpers ────────────────────────────────

    def _compute_trends(self):
        trends = {}
        for key in self.METRIC_KEYS:
            hist = self.metric_history[key]
            n    = min(len(hist), self._trend_window)
            if n < 4:
                trends[key] = 0.0
                continue
            y  = np.array(list(hist)[-n:])
            x  = np.arange(n, dtype=float)
            xm, ym = x.mean(), y.mean()
            slope = float(np.sum((x - xm) * (y - ym)) / (np.sum((x - xm) ** 2) + 1e-8))
            trends[key] = slope
        return trends

    def _allostatic_boost(self, trends):
        boost  = 0.0
        boost += max(0.0,  trends.get("error_rate",            0.0)) * 5.0
        boost += max(0.0, -trends.get("prediction_confidence", 0.0)) * 3.0
        boost += max(0.0,  trends.get("data_shift",            0.0)) * 4.0
        return float(np.clip(boost, 0.0, 0.5))

    def _update_hormones(self, metrics):
        error_rate  = metrics.get("error_rate", 0.0)
        target_err  = self.targets["error_rate"]
        spike       = max(0.0, error_rate - target_err)

        error_delta = 0.0
        if "error_rate" in self._last_values:
            error_delta = error_rate - self._last_values["error_rate"]

        drift       = metrics.get("data_shift", 0.0)
        drift_delta = 0.0
        if "data_shift" in self._last_values:
            drift_delta = abs(drift - self._last_values["data_shift"])

        stress = spike + max(0.0, error_delta) * 2.0 + drift_delta * 0.5

        self.hormone_fast   = float(np.clip(0.40 * self.hormone_fast   + 0.60 * stress, 0.0, 1.0))
        self.hormone_medium = float(np.clip(0.93 * self.hormone_medium + 0.07 * stress, 0.0, 1.0))
        self.hormone_slow   = float(np.clip(0.988 * self.hormone_slow  + 0.012 * stress, 0.0, 1.0))

    def _combined_hormone_scale(self):
        return 1.0 + 0.6 * self.hormone_fast + 0.3 * self.hormone_medium + 0.15 * self.hormone_slow

    def calibrate(self, warmup_metrics_list):
        """Set initial targets from warm-up phase metrics.

        Call this once after initial training, passing the list of metric dicts
        observed during warm-up.  Sets each target to the median of the warm-up
        distribution so the controller's 'healthy' setpoint matches reality.

        Args:
            warmup_metrics_list: List of dicts from PerformanceMonitor.get_metrics()
                                 collected during the initial training phase.
        """
        if not warmup_metrics_list:
            return
        for key in self.METRIC_KEYS:
            vals = [m[key] for m in warmup_metrics_list if key in m and m[key] is not None]
            if len(vals) >= 5:
                self.targets[key] = float(np.median(vals))
                self.metric_history[key].extend(vals)

    def _update_targets(self, metrics):
        """Adaptive set-point tuning — disabled during plasticity to prevent
        the controller from accepting high post-drift errors as 'normal'.

        FIX: removed the H > 0.60 guard — it prevented targets from ever
        adapting when the initial targets were too optimistic (H stayed at
        ~0.59 permanently, blocking all calibration).  Targets now adapt
        continuously during stable (non-plasticity) operation using a slow
        rate so they converge without obscuring genuine drift signals.
        """
        if self.plasticity_mode:
            return
        rate = self._target_update_rate
        for key in self.METRIC_KEYS:
            hist = self.metric_history[key]
            if len(hist) < 25:
                continue
            median_val = float(np.median(list(hist)))
            self.targets[key] += rate * (median_val - self.targets[key])

    def _update_cooldowns(self):
        stress = (self.hormone_fast + self.hormone_medium) / 2.0
        if stress > 0.30:
            factor = max(0.3, 1.0 - stress)
        else:
            factor = min(1.6, 1.0 + (0.30 - stress))
        self.structural_cooldown = max(5,  int(self.base_structural_cooldown * factor))
        self.hormonal_cooldown   = max(2,  int(self.base_hormonal_cooldown   * factor))

    def _feedback_type(self, H):
        if self.last_homeostasis is None:
            return "neutral"
        delta = H - self.last_homeostasis
        if delta >  0.012: return "positive"
        if delta < -0.012: return "negative"
        return "neutral"

    def decide_action(self, U):
        if U < 0.08:  return "NO_ACTION"
        if U < 0.22:  return "FAST_ADAPT"
        if U < 0.45:  return "HORMONAL_ADAPT"
        return "STRUCTURAL_ADAPT"

    def _check_deterioration(self, H):
        self._recent_H.append(H)
        if len(self._recent_H) < 25:
            return False
        current_avg = float(np.mean(list(self._recent_H)[-25:]))
        if current_avg > self._best_H_avg:
            self._best_H_avg = current_avg
        return current_avg < self._best_H_avg * 0.88

    # ── Main update ──────────────────────────────────────────────────────

    def update(self, metrics, drift_detected=False):
        """
        Args:
            metrics:        Dict from PerformanceMonitor.get_metrics().
            drift_detected: Bool forwarded from monitor's dual-window shock
                            detector.  When True, overrides normal PID path.
        """
        self.step += 1

        for key in self.METRIC_KEYS:
            val = metrics.get(key)
            if val is not None:
                self.metric_history[key].append(val)

        self._update_hormones(metrics)
        H, deviations = self.compute_homeostasis(metrics)
        feedback       = self._feedback_type(H)

        # ── DRIFT OVERRIDE PATH ──────────────────────────────────────────
        if drift_detected:
            # Enter plasticity mode
            self.plasticity_mode            = True
            self.plasticity_steps_remaining = self.PLASTICITY_DURATION
            self._post_drift_steps          = 0

            # Reset PID: don't let pre-drift integral bias the response
            self.integral_term = 0.0
            self.last_error    = 0.0

            # Use HORMONAL_ADAPT immediately (not STRUCTURAL: too few samples yet)
            action         = "HORMONAL_ADAPT"
            control_signal = 1.0
            self.last_hormonal_step = self.step

        # ── NORMAL / PLASTICITY PATH ─────────────────────────────────────
        else:
            if self.plasticity_mode:
                self.plasticity_steps_remaining -= 1
                self._post_drift_steps          += 1
                if self.plasticity_steps_remaining <= 0:
                    self.plasticity_mode = False

            deteriorating = self._check_deterioration(H)

            # ── Early NO_ACTION for healthy stable state ─────────────────
            # Check H BEFORE computing the PID signal to prevent integral
            # accumulation during stable operation.  Threshold 0.70 means
            # "system more than 70% healthy → no adaptation needed".
            if H > 0.70 and not deteriorating and not self.plasticity_mode:
                action = "NO_ACTION"
                # Decay integral so it doesn't poison the next drift response
                self.integral_term *= 0.95
                control_signal = 0.0
                self.last_error = 1.0 - H   # keep derivative anchored

            else:
                trends           = self._compute_trends()
                allostatic_boost = self._allostatic_boost(trends)
                base_signal      = self.compute_control_signal(H)
                hormone_scale    = self._combined_hormone_scale()
                control_signal   = float(np.clip(
                    base_signal * hormone_scale + allostatic_boost, 0.0, 1.0
                ))

                if deteriorating:
                    control_signal = float(np.clip(control_signal + 0.20, 0.0, 1.0))

                bias   = -0.03 if feedback == "positive" else (0.07 if feedback == "negative" else 0.0)
                action = self.decide_action(control_signal + bias)

                self._update_cooldowns()

                if action == "STRUCTURAL_ADAPT":
                    if self.step - self.last_structural_step < self.structural_cooldown:
                        action = "HORMONAL_ADAPT"
                    else:
                        self.last_structural_step = self.step

                if action == "HORMONAL_ADAPT":
                    if self.step - self.last_hormonal_step < self.hormonal_cooldown:
                        action = "FAST_ADAPT"
                    else:
                        self.last_hormonal_step = self.step

        self._update_targets(metrics)

        for key in self.METRIC_KEYS:
            val = metrics.get(key)
            if val is not None:
                self._last_values[key] = val

        self.last_homeostasis = H

        # ── Decision log (interpretable trace) ──────────────────────────
        # Identify the metric contributing most to homeostatic stress.
        eff_devs = {}
        for key in self.METRIC_KEYS:
            dev = deviations.get(key, 0.0)
            if key in self.HIGHER_IS_WORSE:    eff_devs[key] = max(0.0, dev)
            elif key in self.LOWER_IS_WORSE:   eff_devs[key] = max(0.0, -dev)
            else:                              eff_devs[key] = abs(dev)
        dominant = max(eff_devs, key=eff_devs.get) if eff_devs else "none"

        if drift_detected:
            mode   = "PLASTICITY_ENTRY"
            reason = f"Drift detected → forced HORMONAL_ADAPT, reset PID, enter plasticity"
        elif self.plasticity_mode:
            mode   = f"PLASTICITY ({self.plasticity_steps_remaining} steps left)"
            reason = f"Post-drift phase, dominant stressor: {dominant} (dev={eff_devs.get(dominant,0):.3f})"
        elif H > 0.70 and action == "NO_ACTION":
            mode   = "STABLE"
            reason = f"H={H:.3f} > 0.70 → system healthy, no adaptation needed"
        else:
            mode   = "NORMAL_PID"
            reason = (f"H={H:.3f}, U={control_signal:.3f}, "
                      f"dominant stressor: {dominant} (dev={eff_devs.get(dominant,0):.3f})")

        decision_log = {
            "step":      self.step,
            "mode":      mode,
            "H":         H,
            "U":         control_signal,
            "action":    action,
            "dominant":  dominant,
            "reason":    reason,
            "targets":   dict(self.targets),
        }

        return {
            "homeostasis_score":  H,
            "control_signal":     control_signal,
            "deviations":         deviations,
            "feedback":           feedback,
            "action":             action,
            "hormone_fast":       self.hormone_fast,
            "hormone_medium":     self.hormone_medium,
            "hormone_slow":       self.hormone_slow,
            "allostatic_boost":   0.0,
            "trends":             {},
            "deteriorating":      False,
            "adaptive_targets":   dict(self.targets),
            "plasticity_mode":    self.plasticity_mode,
            "post_drift_steps":   self._post_drift_steps,
            "decision_log":       decision_log,
        }
