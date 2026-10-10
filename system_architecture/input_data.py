import numpy as np


class StreamingDataGenerator:
    """Streaming data generator with configurable concept drift.

    Concepts w1 and w2 are separated by a configurable angle (default π/2,
    i.e. orthogonal). This guarantees a sharp, unambiguous drift signal:
    a model trained on w1 achieves ~50% accuracy on w2 labels immediately
    after sudden drift, making adaptation necessary and measurable.

    Key fix vs previous version:
        Old:  w2 = w1 + N(0, 0.5)  →  cosine_similarity ≈ 0.89  →  ~27° rotation
        New:  Gram-Schmidt orthogonalisation  →  cosine_similarity = 0  →  90° rotation
    """

    def __init__(self, n_features=10, drift_point=500, drift_type="regular",
                 concept_angle=None, seed=None):
        """
        Args:
            n_features:     Dimensionality of each sample.
            drift_point:    Step index at which drift occurs (or period for recurring).
            drift_type:     One of 'regular', 'sudden', 'gradual', 'recurring'.
            concept_angle:  Angular separation between w1 and w2 in radians.
                            None → π/2 (orthogonal, maximum unambiguous drift).
                            np.pi → full reversal (labels completely flip).
            seed:           Optional numpy random seed for reproducibility.
        """
        if seed is not None:
            np.random.seed(seed)

        self.n_features = n_features
        self.drift_point = drift_point
        self.drift_type = drift_type
        self.t = 0

        # ── Base concept: unit-norm weight vector ────────────────────────
        w1_raw = np.random.randn(n_features)
        self.w1 = w1_raw / (np.linalg.norm(w1_raw) + 1e-10)

        # ── Orthogonal component via Gram-Schmidt ────────────────────────
        perp_raw = np.random.randn(n_features)
        perp_raw -= perp_raw.dot(self.w1) * self.w1   # remove w1 component
        norm = np.linalg.norm(perp_raw)
        if norm < 1e-8:                                # numerical edge case
            perp_raw = np.roll(self.w1, 1)
            perp_raw -= perp_raw.dot(self.w1) * self.w1
            norm = np.linalg.norm(perp_raw)
        perp = perp_raw / norm

        # ── Drifted concept: rotate w1 by theta in the (w1, perp) plane ─
        theta = concept_angle if concept_angle is not None else (np.pi / 2)
        self.w2 = np.cos(theta) * self.w1 + np.sin(theta) * perp

        # ── Diagnostics (useful for experiment validation) ───────────────
        self.cosine_similarity = float(np.dot(self.w1, self.w2))
        # Chord distance ∈ [0, 2]; 0 = identical, √2 ≈ orthogonal, 2 = reversed
        self.drift_magnitude = float(np.sqrt(max(0.0, 2.0 * (1.0 - self.cosine_similarity))))

    # ── Sample generation ────────────────────────────────────────────────

    def generate_sample(self):
        x = np.random.normal(0, 1, self.n_features)

        if self.drift_type == "regular":
            y = int(np.dot(self.w1, x) > 0)

        elif self.drift_type == "sudden":
            w = self.w1 if self.t < self.drift_point else self.w2
            y = int(np.dot(w, x) > 0)

        elif self.drift_type == "gradual":
            p = min(1.0, self.t / self.drift_point)
            w = self.w2 if np.random.rand() < p else self.w1
            y = int(np.dot(w, x) > 0)

        elif self.drift_type == "recurring":
            # Concept alternates every drift_point steps: w1 → w2 → w1 → …
            cycle = (self.t // self.drift_point) % 2
            w = self.w2 if cycle == 1 else self.w1
            y = int(np.dot(w, x) > 0)

        else:
            y = int(np.dot(self.w1, x) > 0)

        self.t += 1
        return x, y

    # ── Introspection helpers ────────────────────────────────────────────

    @property
    def active_concept(self):
        """Returns 0 (w1) or 1 (w2) for the concept active at the current step."""
        if self.drift_type == "regular":
            return 0
        elif self.drift_type == "sudden":
            return 0 if self.t < self.drift_point else 1
        elif self.drift_type == "gradual":
            return 1 if self.t >= self.drift_point else 0
        elif self.drift_type == "recurring":
            return (self.t // self.drift_point) % 2
        return 0
