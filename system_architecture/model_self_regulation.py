import numpy as np


class ModelSelfRegulation:
    """Integrates all homeostatic components into one streaming step.

    Change vs previous version:
        drift_detected is extracted from the monitor and threaded to both the
        controller and the adaptive layer so that the plasticity response is
        triggered consistently in all three components.
    """

    def __init__(self, model, monitor, controller, adaptive_layer):
        self.model      = model
        self.monitor    = monitor
        self.controller = controller
        self.adaptive   = adaptive_layer

    def calibrate_on_warmup(self, X_arr, y_arr):
        """Run warm-up data through the monitor (predict-only) and calibrate
        controller targets to match actual stable-state performance.

        Call this once after model.train_initial() and before the streaming
        loop.  It does NOT retrain the model — it only observes metrics so
        the controller's 'healthy' setpoints reflect reality.
        """
        metrics_history = []
        for X, y in zip(X_arr, y_arr):
            X = np.atleast_2d(np.asarray(X, dtype=float))
            y_val = int(y) if np.isscalar(y) else int(y.flat[0])
            pred = self.model.predict(X)[0]
            prob = self.model.predict_proba(X)[0]
            self.monitor.update(y_true=y_val, y_pred=pred, prob=prob, features=X.ravel())
            m = self.monitor.get_metrics()
            m["memory_load"] = self.adaptive.memory_load
            metrics_history.append(m)
        self.controller.calibrate(metrics_history)

    def step(self, X, y):
        """One streaming time-step.

        Pipeline order (critical — predict before update):
            1. Predict        (uses current weights, no label leakage)
            2. Monitor update (records outcome, fires drift detector)
            3. Controller     (receives drift_detected from monitor)
            4. Adaptive       (executes action, manages epochs and replay)
        """
        X     = np.atleast_2d(np.asarray(X, dtype=float))
        y_val = int(y) if np.isscalar(y) else y

        # ── 1. Predict ───────────────────────────────────────────────────
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]

        # ── 2. Monitor ───────────────────────────────────────────────────
        self.monitor.update(
            y_true=y_val,
            y_pred=pred,
            prob=prob,
            features=X.ravel(),
        )
        metrics = self.monitor.get_metrics()
        metrics["memory_load"] = self.adaptive.memory_load

        # ── 3. Controller ────────────────────────────────────────────────
        drift_detected   = self.monitor.drift_detected
        controller_state = self.controller.update(metrics, drift_detected=drift_detected)
        action           = controller_state["action"]
        feedback         = controller_state.get("feedback", "neutral")

        # ── 4. Adaptive ──────────────────────────────────────────────────
        adaptive_state = self.adaptive.execute(
            action,
            metrics,
            feedback=feedback,
            X=X,
            y=np.atleast_1d(y_val),
            drift_detected=drift_detected,
        )

        return {
            "prediction":     pred,
            "metrics":        metrics,
            "controller":     controller_state,
            "adaptive":       adaptive_state,
            "drift_detected": drift_detected,
        }
