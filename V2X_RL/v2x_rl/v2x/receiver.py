"""Ego-side VAM reception and feature extraction.

Turns the most recent valid VAM into a fixed-length, normalised feature
vector.  Everything here is derived from *received* data only — never from
CARLA ground truth — so the policy is trained on exactly the information a
real vehicle would have.  When nothing has been received (and, unless a gap
filler is configured via ``cfg.gap_fill``, whenever the latest message has
gone stale) the features are zeroed and the validity flag drops to 0, giving
the network an explicit "I am blind" signal rather than a silently stale
value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .. import geometry as geom
from .gap_fill import GapFiller, build_gap_filler
from .message import VAM

V2X_FEATURE_NAMES = [
    "v2x_valid",
    "v2x_age",
    "v2x_range",
    "v2x_bearing_sin",
    "v2x_bearing_cos",
    "v2x_speed",
    "v2x_cyclist_dist_to_conflict",
    "v2x_cyclist_tta",
    "v2x_ego_tta",
    "v2x_arrival_gap",
    "v2x_interception",
]
V2X_FEATURE_DIM = len(V2X_FEATURE_NAMES)

RANGE_NORM_M = 100.0
SPEED_NORM_MS = 12.0
DIST_NORM_M = 60.0


@dataclass
class V2XDerived:
    """Human-readable version of the derived quantities, for logging."""

    valid: bool = False
    age_s: float = 0.0
    range_m: float = 0.0
    bearing_deg: float = 0.0
    speed_ms: float = 0.0
    interception: bool = False
    cyclist_dist_to_conflict_m: float = 0.0
    cyclist_tta_s: float = 0.0
    ego_dist_to_conflict_m: float = 0.0
    ego_tta_s: float = 0.0
    arrival_gap_s: float = 0.0
    conflict_xy: Optional[np.ndarray] = None


class V2XReceiver:
    """Holds the freshest VAM and derives interception features from it."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.latest: Optional[VAM] = None
        self.received_count = 0
        self.filler: Optional[GapFiller] = build_gap_filler(cfg)

    def reset(self) -> None:
        self.latest = None
        self.received_count = 0
        if self.filler is not None:
            self.filler.reset()

    def update(self, messages: Iterable[VAM]) -> None:
        """Accept delivered messages, keeping only the newest one.

        Latency jitter means messages can arrive out of order, so an older
        generation time never overwrites a newer one. The gap filler (if any)
        only ever sees messages in that same newest-first order.
        """
        for message in messages:
            self.received_count += 1
            if (self.latest is None
                    or message.generation_time_s > self.latest.generation_time_s):
                self.latest = message
                if self.filler is not None:
                    self.filler.observe(message)

    def age(self, now_s: float) -> float:
        if self.latest is None:
            return float("inf")
        return max(0.0, now_s - self.latest.generation_time_s)

    def is_valid(self, now_s: float) -> bool:
        return (self.cfg.enabled and self.latest is not None
                and self.age(now_s) <= self.cfg.max_message_age_s)

    def _resolve_message(self, now_s: float) -> Tuple[Optional[VAM], float]:
        """The message ``derive`` should treat as current, and its true age.

        A fresh real message is used as-is. Once it goes stale, a configured
        gap filler gets a chance to extrapolate one; past its own
        ``gap_fill_max_age_s`` (or with no filler at all) this returns
        ``(None, inf)``, the same "no information" outcome as always.
        """
        if not self.cfg.enabled or self.latest is None:
            return None, float("inf")
        age_s = self.age(now_s)
        if age_s <= self.cfg.max_message_age_s:
            return self.latest, age_s
        if self.filler is None or age_s > self.cfg.gap_fill_max_age_s:
            return None, float("inf")
        estimate = self.filler.predict(now_s)
        if estimate is None:
            return None, float("inf")
        position, speed_ms, heading_deg = estimate
        filled = VAM(
            station_id=self.latest.station_id,
            vru_profile=self.latest.vru_profile,
            generation_time_s=now_s,
            position=np.asarray(position, dtype=np.float64),
            speed_ms=float(speed_ms),
            heading_deg=float(heading_deg),
            path_prediction=np.zeros((0, 2)),
            path_prediction_dt_s=self.latest.path_prediction_dt_s,
            path_confidence_m=np.zeros(0),
        )
        return filled, age_s

    # ------------------------------------------------------------------ #
    def _extended_path(self, message: VAM) -> np.ndarray:
        """The cyclist's reported trajectory, as the receiver reconstructs it.

        Starts at the reported *current* position — so arc lengths along it are
        distances from where the cyclist is now — followed by the reported path
        prediction, followed by a constant-heading extrapolation.  The
        extrapolation matters because a VAM path prediction only spans a few
        seconds, which is shorter than the ego's decision horizon when
        approaching a junction at 50 km/h.
        """
        prediction = np.asarray(message.path_prediction, dtype=np.float64)
        position = np.asarray(message.position, dtype=np.float64).reshape(1, 2)
        path = np.vstack([position, prediction]) if len(prediction) else position

        extra = self.cfg.receiver_extrapolation_m
        if extra <= 0 or len(path) < 2:
            if extra <= 0:
                return path
            heading = np.radians(message.heading_deg)
            direction = np.array([np.cos(heading), np.sin(heading)])
        else:
            direction = path[-1] - path[-2]

        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return path
        direction = direction / norm
        steps = np.arange(1, max(1, int(extra // 2.0)) + 1) * 2.0
        return np.vstack([path, path[-1] + np.outer(steps, direction)])

    def derive(self, now_s: float, ego_xy: np.ndarray, ego_yaw_deg: float,
               ego_speed_ms: float, ego_path_ahead: np.ndarray) -> V2XDerived:
        """Compute the derived state from the latest message.

        ``ego_path_ahead`` is the ego's remaining route as a polyline starting
        at its current position.
        """
        message, age_s = self._resolve_message(now_s)
        if message is None:
            return V2XDerived()

        derived = V2XDerived(
            valid=True,
            age_s=age_s,
            range_m=float(np.linalg.norm(np.asarray(message.position) - ego_xy)),
            bearing_deg=geom.bearing_deg(ego_xy, ego_yaw_deg, message.position),
            speed_ms=float(message.speed_ms),
        )

        cyclist_path = self._extended_path(message)
        conflict = geom.paths_conflict_point(
            ego_path_ahead, cyclist_path,
            closest_approach_threshold_m=3.0)
        if conflict is None:
            return derived

        point, s_ego, s_cyclist = conflict
        cap = self.cfg.receiver_tta_norm_s
        derived.interception = True
        derived.conflict_xy = point
        derived.ego_dist_to_conflict_m = s_ego
        derived.cyclist_dist_to_conflict_m = s_cyclist
        derived.ego_tta_s = geom.time_to_arrival(s_ego, ego_speed_ms, cap)
        derived.cyclist_tta_s = geom.time_to_arrival(s_cyclist, message.speed_ms, cap)
        derived.arrival_gap_s = derived.ego_tta_s - derived.cyclist_tta_s
        return derived

    # ------------------------------------------------------------------ #
    def features(self, derived: V2XDerived) -> np.ndarray:
        """Normalised, bounded feature vector matching ``V2X_FEATURE_NAMES``."""
        if not derived.valid:
            return np.zeros(V2X_FEATURE_DIM, dtype=np.float32)

        cap = self.cfg.receiver_tta_norm_s
        bearing = np.radians(derived.bearing_deg)
        # Clip rather than let a gap-filled estimate's true age (which can
        # exceed max_message_age_s) push this feature outside its normal
        # [0, 1] range -- a filled reading reads as "maximally stale", never
        # as a magnitude the network hasn't seen from real messages.
        age_frac = min(derived.age_s, self.cfg.max_message_age_s)
        return np.array([
            1.0,
            age_frac / max(1e-6, self.cfg.max_message_age_s),
            min(derived.range_m, RANGE_NORM_M) / RANGE_NORM_M,
            np.sin(bearing),
            np.cos(bearing),
            min(derived.speed_ms, SPEED_NORM_MS) / SPEED_NORM_MS,
            min(derived.cyclist_dist_to_conflict_m, DIST_NORM_M) / DIST_NORM_M,
            derived.cyclist_tta_s / cap,
            derived.ego_tta_s / cap,
            float(np.clip(derived.arrival_gap_s / cap, -1.0, 1.0)),
            float(derived.interception),
        ], dtype=np.float32)
