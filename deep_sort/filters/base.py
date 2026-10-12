import numpy as np
import scipy.linalg


CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class BaseKalmanFilter:
    def __init__(self):
        ndim = 4
        self._motion_mat = np.eye(2 * ndim)
        for index in range(ndim):
            self._motion_mat[index, ndim + index] = 1.0
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._gating_epsilon = 0.0

    def _initial_covariance(self, measurement):
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return np.diag(np.square(std))

    def _process_covariance(self, mean):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        return np.diag(np.square(np.r_[std_pos, std_vel]))

    def _measurement_covariance(self, mean):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        return np.diag(np.square(std))

    def initiate(self, measurement):
        measurement = np.asarray(measurement, dtype=float)
        mean = np.r_[measurement, np.zeros_like(measurement)]
        return mean, self._initial_covariance(measurement)

    def predict(self, mean, covariance):
        process_covariance = self._process_covariance(mean)
        mean = self._motion_mat @ mean
        covariance = (
            self._motion_mat @ covariance @ self._motion_mat.T
            + process_covariance
        )
        return mean, covariance

    def project(self, mean, covariance, confidence=1.0):
        projected_mean = self._update_mat @ mean
        projected_covariance = (
            self._update_mat @ covariance @ self._update_mat.T
            + self._measurement_covariance(mean)
        )
        return projected_mean, projected_covariance

    def update(self, mean, covariance, measurement, confidence=1.0):
        projected_mean, projected_covariance = self.project(
            mean, covariance, confidence
        )
        factor, lower = scipy.linalg.cho_factor(
            projected_covariance, lower=True, check_finite=False
        )
        gain = scipy.linalg.cho_solve(
            (factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + gain @ innovation
        new_covariance = (
            covariance - gain @ projected_covariance @ gain.T
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean,
        covariance,
        measurements,
        only_position=False,
        confidence=1.0,
    ):
        projected_mean, projected_covariance = self.project(
            mean, covariance, confidence
        )
        if only_position:
            projected_mean = projected_mean[:2]
            projected_covariance = projected_covariance[:2, :2]
            measurements = measurements[:, :2]
        if self._gating_epsilon > 0:
            projected_covariance = (
                projected_covariance
                + self._gating_epsilon
                * np.eye(projected_covariance.shape[0])
            )
        factor = np.linalg.cholesky(projected_covariance)
        residual = measurements - projected_mean
        solved = scipy.linalg.solve_triangular(
            factor,
            residual.T,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )
        return np.sum(solved * solved, axis=0)
