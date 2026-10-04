"""Non-learned controllers.

Steering for the ego and full control of the cyclist are handled here.  So is
the longitudinal *baseline*: :class:`ACCController` is the adaptive-cruise
scaffold the residual RL policy corrects, and :class:`OracleYieldController`
adds a ground-truth yield on top of it to generate the behaviour-cloning
dataset.  The RL policy itself lives elsewhere.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import geometry as geom
from .carla_utils import carla
from .scenario import Path


class PathFollower:
    """Pure-pursuit style lateral controller over a polyline.

    The lookahead grows with speed, which keeps the ego stable through a
    45 km/h straight while still tracking a tight turn radius.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._prev_error = 0.0

    def reset(self) -> None:
        self._prev_error = 0.0

    def steer(self, path: Path, xy: np.ndarray, yaw_deg: float,
              speed_ms: float, s: Optional[float] = None) -> float:
        """Return a steering command in [-1, 1]."""
        if s is None:
            s, _, _ = path.project(xy)
        lookahead = self.cfg.lookahead_min_m + self.cfg.lookahead_speed_gain * speed_ms
        target_xy, _, _ = path.pose_at(s + lookahead)

        error_deg = geom.bearing_deg(xy, yaw_deg, target_xy)
        derivative = geom.wrap_deg(error_deg - self._prev_error)
        self._prev_error = error_deg

        command_deg = self.cfg.kp * error_deg + self.cfg.kd * derivative
        command_deg = float(np.clip(command_deg, -self.cfg.max_steer_degrees,
                                    self.cfg.max_steer_degrees))
        return command_deg / self.cfg.max_steer_degrees


class CyclistController:
    """Drives the cyclist along its path at a target speed.

    Uses velocity/transform overrides rather than throttle: CARLA's bicycle
    physics are twitchy and we need the cyclist's motion to be repeatable so
    that episodes are comparable and its broadcast intent is truthful.

    Hold-and-release: on a conflicting episode the cyclist is spawned a fixed
    ``meet_horizon_s`` of travel from the conflict point and held stationary
    until :meth:`maybe_release` is told the ego is that same time away.  Both
    then cover their remaining distance and reach the conflict point together,
    regardless of how fast the ego actually drove.  While held it still
    broadcasts its intended crossing path (a cyclist creeping to the line),
    which is the V2X warning the ego is meant to act on.

    That release condition has a feedback-loop failure mode: a policy that
    brakes hard as soon as it sees the pre-release V2X broadcast can slow down
    just enough that its own time-to-arrival estimate never drops to
    ``meet_horizon_s`` -- an equilibrium where it never finishes closing in, so
    the cyclist never starts and the episode idles out.  Two safety nets guard
    against that: releasing anyway once the ego is close in plain distance
    (judged at a low, crawling-pace reference speed so a genuinely stopped ego
    cannot hold if off indefinitely), and an absolute cap on how long it can be
    held at all.
    """

    def __init__(self, actor, path: Path, target_speed_ms: float,
                 start_s: float, dt: float, hold_until_release: bool = False,
                 meet_horizon_s: float = 3.5, release_min_speed_ms: float = 3.0,
                 max_hold_s: float = 20.0) -> None:
        self.actor = actor
        self.path = path
        self.target_speed_ms = target_speed_ms
        self.dt = dt
        self.s = start_s
        self._speed = target_speed_ms
        self.meet_horizon_s = meet_horizon_s
        self.release_min_speed_ms = release_min_speed_ms
        self.max_hold_s = max_hold_s
        self._released = not hold_until_release
        self._held_steps = 0
        self.finished = False

    @property
    def released(self) -> bool:
        return self._released

    def maybe_release(self, ego_tta_to_conflict_s: float,
                      ego_gap_m: Optional[float] = None) -> None:
        if self._released:
            return
        self._held_steps += 1
        if ego_tta_to_conflict_s <= self.meet_horizon_s:
            self._released = True
        elif (ego_gap_m is not None
              and ego_gap_m <= self.meet_horizon_s * self.release_min_speed_ms):
            self._released = True
        elif self._held_steps * self.dt >= self.max_hold_s:
            self._released = True

    def step(self) -> None:
        if self.finished:
            return
        if self._released:
            self.s += self._speed * self.dt
            if self.s >= self.path.length - 1.0:
                self.s = self.path.length - 1.0
                self.finished = True

        xy, z, yaw = self.path.pose_at(self.s)
        transform = carla.Transform(
            carla.Location(float(xy[0]), float(xy[1]), float(z) + 0.05),
            carla.Rotation(pitch=0.0, yaw=float(yaw), roll=0.0))
        self.actor.set_transform(transform)
        # Keep the reported velocity consistent with the kinematic motion so
        # the lidar tracker and the VAM see the same thing (zero while held).
        speed = self._speed if self._released else 0.0
        yaw_rad = math.radians(yaw)
        self.actor.set_target_velocity(carla.Vector3D(
            x=speed * math.cos(yaw_rad),
            y=speed * math.sin(yaw_rad),
            z=0.0))

    @property
    def speed_ms(self) -> float:
        return 0.0 if (self.finished or not self._released) else self._speed

    def predicted_path(self, horizon_s: float, n_points: int) -> np.ndarray:
        """Ground-truth future path points, used to build the VAM."""
        if n_points <= 0:
            return np.zeros((0, 2))
        times = np.linspace(horizon_s / n_points, horizon_s, n_points)
        return np.stack([self.path.pose_at(self.s + self._speed * t)[0]
                         for t in times])


class ACCController:
    """Intelligent-Driver-Model adaptive cruise, reactive to onboard sensing.

    Given the ego speed, a target speed and — when the perception layer has a
    credible obstacle ahead — its range and closing rate, produces a
    throttle/brake command in ``[-1, 1]``.  Deliberately blind to V2X and to
    ground truth: proactive yielding for an occluded cyclist is the residual
    RL policy's job, not the baseline's.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def reset(self) -> None:  # stateless; kept for API symmetry
        pass

    def desired_accel(self, speed_ms: float, target_speed_ms: float,
                      lead_gap_m: Optional[float] = None,
                      lead_closing_ms: Optional[float] = None) -> float:
        """IDM longitudinal acceleration, m/s**2 (negative = braking)."""
        cfg = self.cfg
        v = max(0.0, speed_ms)
        v0 = max(0.1, target_speed_ms * cfg.target_speed_frac)
        free_term = (v / v0) ** cfg.accel_exponent
        accel = cfg.max_accel_ms2 * (1.0 - free_term)

        if lead_gap_m is not None:
            s = max(0.1, lead_gap_m)
            dv = max(0.0, lead_closing_ms or 0.0)   # >0 means we are closing
            s_star = cfg.min_gap_m + max(
                0.0,
                v * cfg.time_headway_s
                + v * dv / (2.0 * math.sqrt(cfg.max_accel_ms2 * cfg.comfort_decel_ms2)))
            follow_accel = cfg.max_accel_ms2 * (
                1.0 - free_term - (s_star / s) ** 2)
            accel = min(accel, follow_accel)

        return float(np.clip(accel, -cfg.emergency_decel_ms2, cfg.max_accel_ms2))

    def command(self, speed_ms: float, target_speed_ms: float,
                lead_gap_m: Optional[float] = None,
                lead_closing_ms: Optional[float] = None) -> float:
        return self._accel_to_command(
            self.desired_accel(speed_ms, target_speed_ms, lead_gap_m, lead_closing_ms))

    def _accel_to_command(self, accel_ms2: float) -> float:
        if accel_ms2 >= 0.0:
            return float(np.clip(accel_ms2 / self.cfg.accel_to_throttle_ms2, 0.0, 1.0))
        return float(np.clip(accel_ms2 / self.cfg.decel_to_brake_ms2, -1.0, 0.0))


class OracleYieldController:
    """ACC plus a ground-truth yield, used only to build the BC dataset.

    While a yield is required the conflict line is treated as a stationary
    lead, so the IDM brings the ego to a smooth stop a few metres short of it;
    once the cyclist has cleared, the plain ACC command resumes.
    """

    STOP_MARGIN_M = 1.0

    def __init__(self, cfg) -> None:
        self.acc = ACCController(cfg)
        self.cfg = cfg

    def reset(self) -> None:
        self.acc.reset()

    def command(self, speed_ms: float, target_speed_ms: float, *,
                yield_now: bool, dist_to_conflict_m: float,
                lead_gap_m: Optional[float] = None,
                lead_closing_ms: Optional[float] = None) -> float:
        base = self.acc.command(speed_ms, target_speed_ms, lead_gap_m, lead_closing_ms)
        if not yield_now:
            return base
        stop_gap = max(0.0, dist_to_conflict_m - self.STOP_MARGIN_M)
        via_line = self.acc.command(speed_ms, target_speed_ms,
                                    lead_gap_m=stop_gap,
                                    lead_closing_ms=max(0.0, speed_ms))
        return float(min(base, via_line))
