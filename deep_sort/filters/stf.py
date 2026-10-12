import numpy as np
import scipy.linalg

from .base import BaseKalmanFilter


class STFOnlyFilter(BaseKalmanFilter):
    def __init__(self, rho=0.82, beta=1.0):
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.V = None
        self.lambda_value = 1.0
        self.P_post = None
        self._gating_epsilon = 1e-6

    def initiate(self, measurement):
        mean, covariance = super().initiate(measurement)
        self.V = None
        self.lambda_value = 1.0
        self.P_post = covariance.copy()
        return mean, covariance

    def update(self, mean, covariance, measurement, confidence=1.0):
        process_covariance = self._process_covariance(mean)
        projected_mean = self._update_mat @ mean
        innovation = measurement - projected_mean
        innovation_outer = np.outer(innovation, innovation)
        if self.V is None:
            self.V = innovation_outer
        else:
            self.V = (
                self.rho * self.V + innovation_outer
            ) / (1.0 + self.rho)
        if self.P_post is None:
            propagated_covariance = covariance - process_covariance
        else:
            propagated_covariance = (
                self._motion_mat @ self.P_post @ self._motion_mat.T
            )
        projected_propagated = (
            self._update_mat
            @ propagated_covariance
            @ self._update_mat.T
        )
        measurement_covariance = self._measurement_covariance(mean)
        projected_process = (
            self._update_mat
            @ process_covariance
            @ self._update_mat.T
        )
        numerator = (
            self.V
            - projected_process
            - self.beta * measurement_covariance
        )
        denominator = np.trace(projected_propagated)
        if denominator > 1e-6:
            self.lambda_value = max(
                1.0, np.trace(numerator) / denominator
            )
        else:
            self.lambda_value = 1.0
        adjusted_covariance = (
            self.lambda_value * propagated_covariance
            + process_covariance
        )
        projected_covariance = (
            self._update_mat
            @ adjusted_covariance
            @ self._update_mat.T
            + measurement_covariance
        )
        projected_covariance = (
            projected_covariance + projected_covariance.T
        ) / 2
        projected_covariance += 1e-6 * np.eye(4)
        factor, lower = scipy.linalg.cho_factor(
            projected_covariance, lower=True, check_finite=False
        )
        gain = scipy.linalg.cho_solve(
            (factor, lower),
            (adjusted_covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T
        new_mean = mean + gain @ innovation
        new_covariance = (
            adjusted_covariance
            - gain @ projected_covariance @ gain.T
        )
        new_covariance = (new_covariance + new_covariance.T) / 2
        new_covariance += 1e-6 * np.eye(8)
        self.P_post = new_covariance.copy()
        return new_mean, new_covariance
