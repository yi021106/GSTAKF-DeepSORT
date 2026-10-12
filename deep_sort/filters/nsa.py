import numpy as np

from .base import BaseKalmanFilter


class NSAOnlyFilter(BaseKalmanFilter):
    def project(self, mean, covariance, confidence=1.0):
        confidence = float(np.clip(confidence, 0.0, 1.0))
        projected_mean = self._update_mat @ mean
        measurement_covariance = (
            self._measurement_covariance(mean) * (1.0 - confidence)
        )
        projected_covariance = (
            self._update_mat @ covariance @ self._update_mat.T
            + measurement_covariance
        )
        return projected_mean, projected_covariance
