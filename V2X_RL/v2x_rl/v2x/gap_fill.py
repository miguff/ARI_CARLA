"""Bridging short V2X reception gaps with a motion-model prediction.

``V2XReceiver`` normally goes blind (``valid=0``) the moment its latest VAM
passes ``max_message_age_s`` -- deliberately, so the policy sees an honest "I
know nothing" signal rather than a silently stale one.  A gap filler is an
opt-in relaxation of that: it keeps a running belief about the cyclist's
position/speed/heading from past messages and can be asked to extrapolate it
forward, so a brief drop-out (a lost packet, a message arriving late) doesn't
immediately blank the observation.  It still expires -- past
``gap_fill_max_age_s`` ``predict()`` returns ``None`` and the receiver falls
back to the same "no information" state as when no filler is configured at
all, so a genuine prolonged blackout is never masked indefinitely.

Two implementations, both behind the same three-method interface so
``V2XReceiver`` doesn't care which one is active:

- ``DeadReckoningFiller`` -- constant-velocity extrapolation of the last
  received message. Cheap, has no tuning beyond the shared max age.
- ``KalmanFiller`` -- a constant-velocity Kalman filter over
  ``[x, y, vx, vy]``. Also smooths the GNSS noise on ordinary (non-gap)
  messages, not just the extrapolation through a gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple

import numpy as np

from .message import VAM

GAP_FILL_MODES = ("none", "dead_reckoning", "kalman")

# Estimated (position_xy, speed_ms, heading_deg).
Estimate = Tuple[np.ndarray, float, float]


class GapFiller(Protocol):
    def reset(self) -> None: ...
    def observe(self, message: VAM) -> None: ...
    def predict(self, now_s: float) -> Optional[Estimate]: ...


def build_gap_filler(cfg) -> Optional[GapFiller]:
    """Construct the filler named by ``cfg.gap_fill``, or ``None`` for "none"."""
    if cfg.gap_fill == "none":
        return None
    if cfg.gap_fill == "dead_reckoning":
        return DeadReckoningFiller(cfg)
    if cfg.gap_fill == "kalman":
        return KalmanFiller(cfg)
    raise ValueError(f"unknown v2x.gap_fill {cfg.gap_fill!r}, "
                      f"choose from {GAP_FILL_MODES}")


@dataclass
class DeadReckoningFiller:
    """Walks the last received state forward at constant velocity."""

    cfg: object
    _position: Optional[np.ndarray] = field(default=None, init=False)
    _velocity: Optional[np.ndarray] = field(default=None, init=False)
    _speed_ms: float = field(default=0.0, init=False)
    _obs_time_s: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._position = None
        self._velocity = None
        self._speed_ms = 0.0
        self._obs_time_s = 0.0

    def observe(self, message: VAM) -> None:
        heading = math.radians(message.heading_deg)
        self._position = np.asarray(message.position, dtype=np.float64).copy()
        self._velocity = message.speed_ms * np.array(
            [math.cos(heading), math.sin(heading)])
        self._speed_ms = message.speed_ms
        self._obs_time_s = message.generation_time_s

    def predict(self, now_s: float) -> Optional[Estimate]:
        if self._position is None:
            return None
        dt = now_s - self._obs_time_s
        if dt < 0.0 or dt > self.cfg.gap_fill_max_age_s:
            return None
        position = self._position + self._velocity * dt
        heading_deg = (math.degrees(math.atan2(self._velocity[1], self._velocity[0]))
                       if self._speed_ms > 1e-3 else 0.0)
        return position, self._speed_ms, heading_deg


@dataclass
class KalmanFiller:
    """Constant-velocity Kalman filter over the cyclist's ``[x, y, vx, vy]``.

    Measurement noise comes straight from the VAM's own reported GNSS white
    noise (``cfg.gnss_white_std_m``); process noise (``cfg.
    kf_process_accel_std_ms2``) models how much the cyclist's velocity can
    wander between updates, using the standard discretised white-noise-
    acceleration model so it scales sensibly with the irregular, event-
    triggered spacing of real VAMs.
    """

    cfg: object
    _x: Optional[np.ndarray] = field(default=None, init=False)  # [x, y, vx, vy]
    _P: Optional[np.ndarray] = field(default=None, init=False)  # (4, 4)
    _obs_time_s: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._obs_time_s = 0.0

    def _process_noise(self, dt: float) -> np.ndarray:
        q = self.cfg.kf_process_accel_std_ms2 ** 2
        block = np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                          [dt ** 3 / 2.0, dt ** 2]])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = block
        Q[np.ix_([1, 3], [1, 3])] = block
        return q * Q

    def _extrapolate(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        return F @ self._x, F @ self._P @ F.T + self._process_noise(dt)

    def observe(self, message: VAM) -> None:
        t = message.generation_time_s
        z = np.asarray(message.position, dtype=np.float64)
        if self._x is None:
            heading = math.radians(message.heading_deg)
            vx = message.speed_ms * math.cos(heading)
            vy = message.speed_ms * math.sin(heading)
            self._x = np.array([z[0], z[1], vx, vy])
            # Generous initial velocity uncertainty; position comes straight
            # from a real measurement so it starts as trustworthy as any other.
            r = self.cfg.gnss_white_std_m ** 2
            self._P = np.diag([r, r, 4.0, 4.0])
            self._obs_time_s = t
            return
        if t <= self._obs_time_s:
            return  # stale/out-of-order arrival: ignore rather than run time backwards
        x_pred, P_pred = self._extrapolate(t - self._obs_time_s)

        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2) * self.cfg.gnss_white_std_m ** 2
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self._x = x_pred + K @ y
        self._P = (np.eye(4) - K @ H) @ P_pred
        self._obs_time_s = t

    def predict(self, now_s: float) -> Optional[Estimate]:
        if self._x is None:
            return None
        dt = now_s - self._obs_time_s
        if dt < 0.0 or dt > self.cfg.gap_fill_max_age_s:
            return None
        x, _ = self._extrapolate(dt)
        speed_ms = float(np.hypot(x[2], x[3]))
        heading_deg = math.degrees(math.atan2(x[3], x[2])) if speed_ms > 1e-3 else 0.0
        return x[:2], speed_ms, heading_deg
