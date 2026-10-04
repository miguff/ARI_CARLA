"""VRU Awareness Message (VAM) modelling.

Loosely follows ETSI TS 103 300-3: the cyclist's ITS station periodically
broadcasts its kinematic state together with a ``vruMotionPredictionContainer``
holding a path prediction.  Generation is event-triggered with a rate floor
and ceiling, as the specification prescribes, and the transmitted state is
degraded by the cyclist's own positioning error (a slowly drifting GNSS bias
plus white noise) — not by the channel, which is modelled separately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

VRU_PROFILE_BICYCLIST = 2  # TS 103 300-3 VRU profile 2


@dataclass(frozen=True)
class VAM:
    """A single received-or-transmitted VRU Awareness Message."""

    station_id: int
    vru_profile: int
    generation_time_s: float
    position: np.ndarray            # (2,) reported world x/y
    speed_ms: float
    heading_deg: float
    # vruMotionPredictionContainer / pathPrediction
    path_prediction: np.ndarray     # (K, 2) reported future positions
    path_prediction_dt_s: float
    # Per-point 1-sigma confidence in metres, growing with the horizon.
    path_confidence_m: np.ndarray

    def horizon_s(self) -> float:
        return len(self.path_prediction) * self.path_prediction_dt_s


@dataclass
class VAMGenerator:
    """Event-triggered VAM generation with sender-side positioning error."""

    cfg: object
    rng: np.random.Generator
    station_id: int = 1

    _last_tx_time: Optional[float] = field(default=None, init=False)
    _last_position: Optional[np.ndarray] = field(default=None, init=False)
    _last_speed: Optional[float] = field(default=None, init=False)
    _last_heading: Optional[float] = field(default=None, init=False)
    _bias: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)

    def reset(self) -> None:
        self._last_tx_time = None
        self._last_position = None
        self._last_speed = None
        self._last_heading = None
        self._bias = self.rng.normal(0.0, self.cfg.gnss_bias_std_m, size=2)

    # ------------------------------------------------------------------ #
    def _update_bias(self, dt: float) -> None:
        """Ornstein-Uhlenbeck drift, so the GNSS error is correlated in time."""
        tau = max(1e-3, self.cfg.gnss_bias_tau_s)
        decay = math.exp(-dt / tau)
        sigma = self.cfg.gnss_bias_std_m * math.sqrt(max(0.0, 1.0 - decay ** 2))
        self._bias = decay * self._bias + self.rng.normal(0.0, sigma, size=2)

    def _should_generate(self, now_s: float, position: np.ndarray,
                         speed_ms: float, heading_deg: float) -> bool:
        if self._last_tx_time is None:
            return True
        elapsed = now_s - self._last_tx_time
        if elapsed < 1.0 / self.cfg.max_rate_hz - 1e-9:
            return False
        if elapsed >= 1.0 / self.cfg.min_rate_hz - 1e-9:
            return True
        moved = float(np.linalg.norm(position - self._last_position))
        if moved >= self.cfg.trigger_position_delta_m:
            return True
        if abs(speed_ms - self._last_speed) >= self.cfg.trigger_speed_delta_ms:
            return True
        heading_change = abs((heading_deg - self._last_heading + 180.0) % 360.0 - 180.0)
        return heading_change >= self.cfg.trigger_heading_delta_deg

    def generate(self, now_s: float, position: np.ndarray, speed_ms: float,
                 heading_deg: float, true_path: np.ndarray) -> Optional[VAM]:
        """Produce a VAM if the triggering conditions are met, else ``None``.

        ``true_path`` is the cyclist's actual future path, which gets noised
        with an error that grows with the prediction horizon.
        """
        position = np.asarray(position, dtype=np.float64)
        if not self._should_generate(now_s, position, speed_ms, heading_deg):
            return None

        dt = 0.0 if self._last_tx_time is None else now_s - self._last_tx_time
        self._update_bias(dt)

        noisy_position = (position + self._bias
                          + self.rng.normal(0.0, self.cfg.gnss_white_std_m, size=2))

        k = min(len(true_path), self.cfg.path_prediction_points)
        if k > 0:
            horizons = np.arange(1, k + 1) * self.cfg.path_prediction_dt_s
            sigma = (self.cfg.gnss_white_std_m
                     + self.cfg.path_prediction_noise_std_m_per_s * horizons)
            noise = self.rng.normal(0.0, 1.0, size=(k, 2)) * sigma[:, None]
            noisy_path = np.asarray(true_path[:k], dtype=np.float64) + self._bias + noise
            confidence = sigma
        else:
            noisy_path = np.zeros((0, 2))
            confidence = np.zeros(0)

        self._last_tx_time = now_s
        self._last_position = position.copy()
        self._last_speed = speed_ms
        self._last_heading = heading_deg

        return VAM(
            station_id=self.station_id,
            vru_profile=VRU_PROFILE_BICYCLIST,
            generation_time_s=now_s,
            position=noisy_position,
            speed_ms=float(max(0.0, speed_ms + self.rng.normal(
                0.0, self.cfg.speed_noise_std_ms))),
            heading_deg=float(heading_deg + self.rng.normal(
                0.0, self.cfg.heading_noise_std_deg)),
            path_prediction=noisy_path,
            path_prediction_dt_s=float(self.cfg.path_prediction_dt_s),
            path_confidence_m=confidence,
        )
