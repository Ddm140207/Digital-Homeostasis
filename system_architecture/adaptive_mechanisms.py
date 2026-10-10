import numpy as np


class AdaptiveMechanisms:
    """Executes adaptation actions with drift-aware, epoch-filtered memory.

    Key fixes vs previous version:
        1. Epoch management: on each drift event, model.notify_drift() is called
           to tag future samples with a new epoch.
        2. Epoch-filtered replay: during plasticity mode (first 100 steps after
           drift), replay is restricted to min_epoch=current_epoch.  This
           prevents old-concept samples from cancelling out post-drift learning.
        3. Deferred rebuild: STRUCTURAL_ADAPT only executes the memory rebuild
           once at least POST_DRIFT_REBUILD_THRESHOLD post-drift samples are
           available.  Until then, it falls back to HORMONAL_ADAPT intensity.
        4. Post-drift learning rate is at its maximum immediately (15× base),
           not ramped — the model needs to unlearn the old boundary fast.
    """

    POST_DRIFT_REBUILD_THRESHOLD = 30   # min post-drift samples before rebuild

    def __init__(self, model, controller):
        self.model      = model
        self.controller = controller
        self.base_lr    = getattr(model, "base_lr", 0.01)

        self._consecutive_adapts = 0
        self._last_action        = "NO_ACTION"

    @property
    def memory_load(self):
        return getattr(self.model, "memory_load", 0.5)

    def execute(self, action, metrics, feedback="neutral", X=None, y=None,
                drift_detected=False):
        """
        Args:
            action:         Action string from HomeostasisController.
            metrics:        Current performance metrics dict.
            feedback:       'positive' / 'negative' / 'neutral'.
            X, y:           Current sample (pre-shaped by model_self_regulation).
            drift_detected: Bool forwarded from the monitor shock detector.
                            When True, epoch is incremented before any learning.
        """
        info = {"action_executed": action, "lr_used": self.base_lr, "rebuild": False}

        if X is None or y is None:
            return info

        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(y)

        # ── Epoch management ─────────────────────────────────────────────
        if drift_detected:
            self.model.notify_drift()   # new epoch tag for all subsequent samples

        # ── Escalation counter ───────────────────────────────────────────
        if action != "NO_ACTION":
            self._consecutive_adapts += 1
        else:
            self._consecutive_adapts = max(0, self._consecutive_adapts - 1)

        in_plasticity  = self.controller.plasticity_mode
        current_epoch  = self.model.current_epoch
        n_post_drift   = self.model.post_drift_sample_count()

        # After any drift event, only replay from the current concept epoch.
        # This prevents pre-drift samples from cancelling post-drift learning
        # both during plasticity AND during subsequent NO_ACTION / FAST_ADAPT.
        replay_epoch = current_epoch if current_epoch > 0 else None

        # ── NO_ACTION ─────────────────────────────────────────────────────
        if action == "NO_ACTION":
            lr = self.base_lr   # full base LR; no reduction for stability
            self.model.update(X, y, learning_rate=lr)
            info["lr_used"] = lr
            if np.random.random() < 0.05:
                self.model.replay(n_samples=10, prioritized=False,
                                  min_epoch=replay_epoch)

        # ── FAST_ADAPT ────────────────────────────────────────────────────
        elif action == "FAST_ADAPT":
            lr = self.base_lr * (5.0 if feedback == "negative" else 3.0)
            if in_plasticity:
                lr = self.base_lr * 10.0
            self.model.update(X, y, learning_rate=lr)
            self.model.mark_errors(X, y)
            info["lr_used"] = lr
            self.model.replay(n_samples=15, prioritized=True, min_epoch=replay_epoch)

        # ── HORMONAL_ADAPT ────────────────────────────────────────────────
        elif action == "HORMONAL_ADAPT":
            lr = self.base_lr * (12.0 if feedback == "negative" else 8.0)
            if in_plasticity or drift_detected:
                lr = self.base_lr * 15.0

            for _ in range(3):
                self.model.update(X, y, learning_rate=lr)
            self.model.mark_errors(X, y)
            info["lr_used"] = lr

            self.model.replay(n_samples=40, prioritized=True, min_epoch=replay_epoch)
            if not in_plasticity and self._consecutive_adapts > 10:
                self.model.replay(n_samples=60, prioritized=True, min_epoch=replay_epoch)

        # ── STRUCTURAL_ADAPT ──────────────────────────────────────────────
        elif action == "STRUCTURAL_ADAPT":
            self.model.update(X, y, learning_rate=self.base_lr)

            if in_plasticity or drift_detected:
                # Never rebuild during plasticity: a rebuild on only 30 post-drift
                # samples produces a weaker model than continued online learning.
                # After plasticity ends (100 steps), the PID can trigger a proper
                # rebuild with 100+ clean post-drift samples.
                lr = self.base_lr * 15.0
                for _ in range(4):
                    self.model.update(X, y, learning_rate=lr)
                self.model.replay(n_samples=min(max(n_post_drift, 2), 20),
                                  prioritized=True, min_epoch=current_epoch)
                info["lr_used"] = lr
            else:
                # Post-plasticity or stable-data rebuild: use epoch-filtered data
                self.model.rebuild(keep_recent=300,
                                   min_epoch=current_epoch if current_epoch > 0 else None)
                self._consecutive_adapts = 0
                info["rebuild"] = True
                info["lr_used"] = self.base_lr

        self._last_action = action
        return info
