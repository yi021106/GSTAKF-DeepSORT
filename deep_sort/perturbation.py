import numpy as np


class ArtificialPerturbation:
    CONFIGS = {
        "none": (0.0, 0.0, 0.0),
        "medium": (15.0, 0.25, 0.3),
        "high": (25.0, 0.40, 0.5),
    }

    def __init__(self, level="none", rng=None):
        if level not in self.CONFIGS:
            raise ValueError(level)
        self.sigma_uv, self.sigma_s, self.probability = self.CONFIGS[level]
        self.rng = np.random.default_rng() if rng is None else rng

    def apply(self, state_vector):
        state_vector = np.asarray(state_vector, dtype=float)
        if self.rng.random() > self.probability:
            return state_vector.copy(), False
        perturbed = state_vector.copy()
        perturbed[0] += self.rng.uniform(
            -self.sigma_uv, self.sigma_uv
        )
        perturbed[1] += self.rng.uniform(
            -self.sigma_uv, self.sigma_uv
        )
        scale = self.rng.uniform(
            1.0 - self.sigma_s, 1.0 + self.sigma_s
        )
        perturbed[2] *= scale
        perturbed[3] *= scale
        return perturbed, True

    def adjust_confidence(
        self, original_state, perturbed_state, confidence
    ):
        original_box = self._xyah_to_tlwh(original_state[:4])
        perturbed_box = self._xyah_to_tlwh(perturbed_state[:4])
        overlap = self._iou(original_box, perturbed_box)
        return float(np.clip(confidence * overlap, 0.1, 1.0))

    @staticmethod
    def _xyah_to_tlwh(state):
        x, y, aspect, height = state
        width = aspect * height
        return np.array(
            [x - width / 2, y - height / 2, width, height]
        )

    @staticmethod
    def _iou(first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[0] + first[2], second[0] + second[2])
        bottom = min(first[1] + first[3], second[1] + second[3])
        intersection = max(0.0, right - left) * max(
            0.0, bottom - top
        )
        union = (
            first[2] * first[3]
            + second[2] * second[3]
            - intersection
        )
        return 0.0 if union <= 0 else intersection / union
