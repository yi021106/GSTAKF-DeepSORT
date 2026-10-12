import numpy as np
import scipy.linalg

from .base import BaseKalmanFilter


class GaussianSTAKFFilter(BaseKalmanFilter):
    def __init__(
        self,
        rho=0.82,
        beta=1.0,
        lambda_center=0.0,
        conf_center=0.5,
        sigma_lambda=0.5,
        sigma_conf=0.3,
        sigmoid_k=8.0,
        sigmoid_x0=0.75,
    ):
        super().__init__()
        self.rho = rho
        self.beta = beta
        self.lambda_center = lambda_center
        self.conf_center = conf_center
        self.sigma_lambda = sigma_lambda
        self.sigma_conf = sigma_conf
        self.sigmoid_k = sigmoid_k
        self.sigmoid_x0 = sigmoid_x0
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

    def _min_nsa(self, confidence):
        value = 1.0 / (
            1.0
            + np.exp(
                -self.sigmoid_k
                * (confidence - self.sigmoid_x0)
            )
        )
        return 0.02 + 0.98 * float(value)

    def _nsa_factor(self, lambda_value, confidence):
        minimum = self._min_nsa(confidence)
        lambda_norm = max(0.0, lambda_value - 1.0)
        lambda_distance = (
            lambda_norm - self.lambda_center
        ) / self.sigma_lambda
        confidence_distance = (
            confidence - self.conf_center
        ) / self.sigma_conf
        gaussian = np.exp(
            -(lambda_distance**2 + confidence_distance**2) / 2
        )
        factor = minimum + (1.0 - minimum) * gaussian
        return float(np.clip(factor, minimum, 1.0))

    def update(self, mean, covariance, measurement, confidence=1.0):
        confidence = float(np.clip(confidence, 0.0, 1.0))
        stf_weight = 1.0 - self._min_nsa(confidence)
        process_covariance = self._process_covariance(mean)
        projected_mean = self._update_mat @ mean
        innovation = measurement - projected_mean
        innovation_outer = np.outer(innovation, innovation)
        if self.V is None:
            self.V = innovation_outer
        else:
            residual_memory = (
                self.rho * self.V + innovation_outer
            ) / (1.0 + self.rho)
            self.V = (
                (1.0 - stf_weight) * innovation_outer
                + stf_weight * residual_memory
            )
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
        base_measurement_covariance = self._measurement_covariance(mean)
        nsa_factor = self._nsa_factor(
            self.lambda_value, confidence
        )
        measurement_covariance = (
            base_measurement_covariance
            * (1.0 - confidence * nsa_factor)
        )
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
            lambda_stf = max(
                1.0, np.trace(numerator) / denominator
            )
        else:
            lambda_stf = 1.0
        self.lambda_value = (
            1.0 + stf_weight * (lambda_stf - 1.0)
        )
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
        if self.P_post is None:
            self.P_post = new_covariance.copy()
        else:
            self.P_post = (
                (1.0 - stf_weight) * self.P_post
                + stf_weight * new_covariance
            )
        return new_mean, new_covariance
