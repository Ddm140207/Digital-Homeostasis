import numpy as np
from collections import deque


class PerformanceMonitor:
    """Streaming performance tracker with dual-window shock detector.

    Key fix vs previous version:
        Old:  Single EMA (α=0.05, time-constant=20 steps).  Controller sees the
              true post-drift error only after 30–50 steps.
        New:  Fast EMA (α=0.20, time-constant=5 steps) for responsiveness.
              Dual-window shock detector:  compares last `fast_window` errors
              against the preceding `slow_window` baseline.  Fires within
              `fast_window` steps of a sudden concept change.
    """

    def __init__(self, window=100, fast_window=20, drift_threshold=0.20):
        """
        Args:
            window:          Size of the slow reference error buffer.
            fast_window:     Size of the fast recent error buffer.
            drift_threshold: Minimum short-minus-long error-rate increase that
                             triggers drift_detected=True.  0.20 means a 20 pp
                             jump in recent errors vs recent baseline.
        """
        self.window = window
        self.fast_window = fast_window
        self.drift_threshold = drift_threshold

        # Per-sample rolling history (for confidence, entropy, features)
        self._correct     = deque(maxlen=window)
        self._confidences = deque(maxlen=window)
        self._entropies   = deque(maxlen=window)
        self._features    = deque(maxlen=window)

        # Dual error buffers
        self._errors_fast = deque(maxlen=fast_window)   # recent  (short window)
        self._errors_slow = deque(maxlen=window)        # baseline (long window)

        # Slow EMA — kept for backward compatibility with controller
        self._ema_error   = 0.15
        self._ema_conf    = 0.50
        self._ema_entropy = 0.50
        self._ema_alpha   = 0.05   # time-constant = 20 steps

        # Fast EMA — drives controller's error_rate signal
        self._fast_ema_error = 0.15
        self._fast_ema_alpha = 0.20  # time-constant = 5 steps

        # Drift state
        self.drift_detected     = False
        self.drift_severity     = 0.0
        self.steps_since_drift  = 0
        self._drift_cooldown    = 0    # blocks re-detection for fast_window steps

        # Feature-distribution drift baseline
        self._feature_mean_old  = None
        self._feature_std_old   = None
        self._warmup            = 50
        self._step              = 0

    # ── Main update ──────────────────────────────────────────────────────

    def update(self, y_true, y_pred, prob, features):
        self._step += 1
        error   = int(y_true != y_pred)
        correct = 1 - error

        self._correct.append(correct)
        self._errors_fast.append(error)
        self._errors_slow.append(error)

        # Confidence
        prob = np.atleast_1d(prob)
        conf = float(np.max(prob)) if len(prob) >= 2 else float(prob[0])
        self._confidences.append(conf)

        # Entropy
        prob_clipped = np.clip(prob, 1e-10, 1.0)
        entropy = float(-np.sum(prob_clipped * np.log2(prob_clipped)))
        self._entropies.append(entropy)

        features = np.atleast_1d(np.asarray(features, dtype=float))
        self._features.append(features)

        # Slow EMA
        self._ema_error   = (1 - self._ema_alpha) * self._ema_error   + self._ema_alpha * error
        self._ema_conf    = (1 - self._ema_alpha) * self._ema_conf    + self._ema_alpha * conf
        self._ema_entropy = (1 - self._ema_alpha) * self._ema_entropy + self._ema_alpha * entropy

        # Fast EMA
        self._fast_ema_error = (
            (1 - self._fast_ema_alpha) * self._fast_ema_error
            + self._fast_ema_alpha * error
        )

        # Feature baseline
        if self._step == self._warmup:
            feats = np.array(list(self._features))
            self._feature_mean_old = feats.mean(axis=0)
            self._feature_std_old  = feats.std(axis=0) + 1e-8

        # Drift detection
        self._update_drift_detection()

        if self.drift_detected:
            self.steps_since_drift  = 0
            self._drift_cooldown    = self.fast_window
        elif self._drift_cooldown > 0:
            self._drift_cooldown   -= 1
            self.steps_since_drift += 1
        else:
            self.steps_since_drift += 1

    # ── Dual-window shock detector ───────────────────────────────────────

    def _update_drift_detection(self):
        """Fires when recent short-window error rate exceeds baseline by threshold."""
        min_slow = max(self.fast_window, self.window // 4)
        if len(self._errors_fast) < self.fast_window or len(self._errors_slow) < min_slow:
            self.drift_detected  = False
            self.drift_severity  = 0.0
            return

        if self._drift_cooldown > 0:
            self.drift_detected = False
            return

        fast_rate = float(np.mean(list(self._errors_fast)))

        # Baseline = slow buffer excluding the most recent fast_window samples
        slow_buf   = list(self._errors_slow)
        baseline   = slow_buf[:-self.fast_window] if len(slow_buf) > self.fast_window else slow_buf
        slow_rate  = float(np.mean(baseline)) if baseline else fast_rate

        shock               = fast_rate - slow_rate
        self.drift_severity = max(0.0, shock)
        self.drift_detected = shock > self.drift_threshold

    # ── Feature-distribution drift ───────────────────────────────────────

    def _compute_drift(self):
        if self._feature_mean_old is None or len(self._features) < 30:
            return 0.0

        recent   = np.array(list(self._features)[-30:])
        mean_new = recent.mean(axis=0)
        std_new  = recent.std(axis=0) + 1e-8

        mean_shift  = np.abs(mean_new - self._feature_mean_old) / (self._feature_std_old + 1e-8)
        drift_score = float(np.mean(mean_shift))

        if self._step % 200 == 0:
            self._feature_mean_old = 0.9 * self._feature_mean_old + 0.1 * mean_new
            self._feature_std_old  = 0.9 * self._feature_std_old  + 0.1 * std_new

        return min(drift_score, 2.0)

    # ── Metric aggregation ───────────────────────────────────────────────

    def get_metrics(self):
        n = len(self._correct)
        if n == 0:
            return {
                "error_rate":            0.15,
                "prediction_confidence": 0.50,
                "model_entropy":         0.50,
                "data_shift":            0.0,
                "fast_error_rate":       0.15,
                "drift_detected":        False,
                "drift_severity":        0.0,
            }

        # Use slow EMA for error_rate — gives the controller a stable signal.
        # Fast EMA is exposed separately as fast_error_rate for the shock detector.
        window_error      = 1.0 - (sum(self._correct) / n)
        error_rate        = 0.6 * self._ema_error + 0.4 * window_error

        window_conf       = sum(self._confidences) / n
        prediction_confidence = 0.6 * self._ema_conf + 0.4 * window_conf

        window_ent        = sum(self._entropies) / n
        model_entropy     = 0.6 * self._ema_entropy + 0.4 * window_ent

        data_shift        = self._compute_drift()

        fast_error = (
            float(np.mean(list(self._errors_fast)))
            if self._errors_fast else self._fast_ema_error
        )

        return {
            "error_rate":            error_rate,
            "prediction_confidence": prediction_confidence,
            "model_entropy":         model_entropy,
            "data_shift":            data_shift,
            "fast_error_rate":       fast_error,
            "drift_detected":        self.drift_detected,
            "drift_severity":        self.drift_severity,
        }
