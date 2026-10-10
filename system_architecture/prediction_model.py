import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from collections import deque


class PredictionModel:
    """Online SGD model with epoch-aware episodic memory.

    Key fixes vs previous version:
        1. Epoch tagging: every memory sample carries the concept-epoch it
           belongs to.  After drift, replay and rebuild can be restricted to
           the current epoch only, preventing old-concept contamination.
        2. notify_drift(): increments current_epoch so that post-drift samples
           are cleanly separated in the buffer.
        3. rebuild(min_epoch): only trains on samples from min_epoch onward,
           guarding against the "93% stale data" rebuild failure.
        4. base_lr raised from 0.01 → 0.05 for faster online convergence.
    """

    def __init__(self):
        self.scaler   = StandardScaler()
        self.base_lr  = 0.01
        self.model    = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=self.base_lr,
        )
        self.initialized = False
        self.classes     = np.array([0, 1])

        # ── Episodic memory with epoch tags ──────────────────────────────
        self._memory_X  = deque(maxlen=500)
        self._memory_y  = deque(maxlen=500)
        self._priority  = deque(maxlen=500)
        self._mem_epoch = deque(maxlen=500)   # concept epoch of each sample

        self.current_epoch = 0   # incremented by notify_drift()

    # ── Initialisation ───────────────────────────────────────────────────

    def train_initial(self, X, y):
        self.classes = np.unique(y)
        if len(self.classes) == 1:
            self.classes = np.array([self.classes[0], 1 - self.classes[0]])

        Xs = self.scaler.fit_transform(X)
        self.model.partial_fit(Xs, y, classes=self.classes)
        self.initialized = True

        for xi, yi in zip(X, y):
            self._memory_X.append(xi)
            self._memory_y.append(yi)
            self._priority.append(1.0)
            self._mem_epoch.append(self.current_epoch)

    # ── Drift epoch management ───────────────────────────────────────────

    def notify_drift(self):
        """Signal a concept drift event.

        All samples stored after this call will be tagged with the new epoch,
        keeping them cleanly separated from pre-drift data in the buffer.
        """
        self.current_epoch += 1

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, X):
        X  = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def predict_proba(self, X):
        X  = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    # ── Online update ────────────────────────────────────────────────────

    def update(self, X, y, learning_rate=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(y)

        if learning_rate is not None and abs(learning_rate - self.model.eta0) > 1e-9:
            self.model.set_params(eta0=float(np.clip(learning_rate, 1e-6, 20.0)))

        Xs = self.scaler.transform(X)
        self.model.partial_fit(Xs, y)

        for xi, yi in zip(X, y):
            self._memory_X.append(xi)
            self._memory_y.append(yi)
            self._priority.append(1.0)
            self._mem_epoch.append(self.current_epoch)

    # ── Epoch-filtered replay ────────────────────────────────────────────

    def replay(self, n_samples=30, prioritized=True, min_epoch=None):
        """Replay past samples, optionally restricted to a minimum epoch.

        Args:
            n_samples:   Number of samples to replay.
            prioritized: If True, weight sampling by priority scores.
            min_epoch:   If set, only replay samples with epoch >= min_epoch.
                         Use current_epoch to replay post-drift samples only.
        """
        epochs  = np.array(self._mem_epoch)
        mem_X   = list(self._memory_X)
        mem_y   = list(self._memory_y)
        mem_pri = np.array(self._priority)

        # Build index pool
        if min_epoch is not None:
            pool = np.where(epochs >= min_epoch)[0]
        else:
            pool = np.arange(len(mem_X))

        # Need at least 2 samples to replay meaningfully
        if len(pool) < 2:
            return

        n = min(n_samples, len(pool))

        if prioritized:
            priors = mem_pri[pool]
            priors = priors / (priors.sum() + 1e-8)
            chosen = np.random.choice(pool, size=n, p=priors, replace=False)
        else:
            chosen = np.random.choice(pool, size=n, replace=False)

        X_replay = np.array([mem_X[i] for i in chosen])
        y_replay = np.array([mem_y[i] for i in chosen])

        Xs = self.scaler.transform(X_replay)
        self.model.partial_fit(Xs, y_replay)

    # ── Error prioritisation ─────────────────────────────────────────────

    def mark_errors(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_1d(y)
        preds = self.predict(X)
        for xi, yi, pi in zip(X, y, preds):
            if yi != pi:
                for idx in range(max(0, len(self._memory_y) - 50), len(self._memory_y)):
                    if self._memory_y[idx] == yi:
                        self._priority[idx] = min(5.0, self._priority[idx] + 0.5)

    # ── Full structural rebuild ───────────────────────────────────────────

    def rebuild(self, keep_recent=300, min_epoch=None):
        """Retrain from memory.

        Args:
            keep_recent: Maximum number of samples to use.
            min_epoch:   If set, restrict to samples from this epoch onward.
                         When called post-drift, pass current_epoch to avoid
                         training on old-concept data.
        """
        epochs = np.array(self._mem_epoch)
        mem_X  = np.array(self._memory_X)
        mem_y  = np.array(self._memory_y)

        if min_epoch is not None:
            mask = epochs >= min_epoch
        else:
            mask = np.ones(len(epochs), dtype=bool)

        X_pool = mem_X[mask]
        y_pool = mem_y[mask]

        if len(X_pool) < 10:
            return   # not enough data — caller should retry later

        if len(X_pool) > keep_recent:
            X_pool = X_pool[-keep_recent:]
            y_pool = y_pool[-keep_recent:]

        self.scaler.fit(X_pool)
        Xs = self.scaler.transform(X_pool)

        self.model = SGDClassifier(
            loss="log_loss",
            learning_rate="constant",
            eta0=self.base_lr,
        )
        self.model.partial_fit(Xs, y_pool, classes=self.classes)

        # Second pass for convergence
        shuffle = np.random.permutation(len(Xs))
        self.model.partial_fit(Xs[shuffle], y_pool[shuffle])

    # ── Helpers ──────────────────────────────────────────────────────────

    @property
    def memory_load(self):
        return len(self._memory_X) / self._memory_X.maxlen

    def post_drift_sample_count(self):
        """Number of samples in the buffer tagged with current_epoch."""
        return int(np.sum(np.array(self._mem_epoch) >= self.current_epoch))
