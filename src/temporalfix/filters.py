"""Box stabilization filters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BoxFilter:
    """No-op or exponential box filter."""

    value: NDArray[np.float64]
    alpha: float | None = None

    def predict(self, _delta: float) -> NDArray[np.float64]:
        """Return the most recent box."""
        return self.value.copy()

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update from a direct observation."""
        if self.alpha is None:
            self.value = measurement.copy()
        else:
            self.value = self.alpha * measurement + (1.0 - self.alpha) * self.value
        return self.value.copy()


@dataclass(slots=True)
class KalmanBoxFilter:
    """Constant-velocity Kalman filter over four XYXY coordinates."""

    measurement: NDArray[np.float64]
    process_noise: float
    measurement_noise: float
    state: NDArray[np.float64] = field(init=False)
    covariance: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize position from the first observation and velocity at zero."""
        self.state = np.concatenate(
            [self.measurement.astype(np.float64, copy=True), np.zeros(4)]
        )
        self.covariance = np.eye(8, dtype=np.float64) * 10.0

    def predict(self, delta: float) -> NDArray[np.float64]:
        """Advance the state by a positive time delta."""
        transition = np.eye(8, dtype=np.float64)
        transition[:4, 4:] = np.eye(4, dtype=np.float64) * delta
        process = np.eye(8, dtype=np.float64) * self.process_noise * max(delta, 1.0)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process
        return self._valid_box(self.state[:4])

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Correct the predicted state with an observed box."""
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[:, :4] = np.eye(4, dtype=np.float64)
        residual = measurement - observation @ self.state
        innovation = (
            observation @ self.covariance @ observation.T
            + np.eye(4, dtype=np.float64) * self.measurement_noise
        )
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation)
        self.state = self.state + gain @ residual
        self.covariance = (np.eye(8) - gain @ observation) @ self.covariance
        self.state[:4] = self._valid_box(self.state[:4])
        return self.state[:4].copy()

    @staticmethod
    def _valid_box(box: NDArray[np.float64]) -> NDArray[np.float64]:
        result = box.copy()
        result[2] = max(result[2], result[0])
        result[3] = max(result[3], result[1])
        return result
