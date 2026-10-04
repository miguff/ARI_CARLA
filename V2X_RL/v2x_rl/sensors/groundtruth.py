"""Noisy ground-truth perception, used as an upper-bound ablation.

Reads the cyclist's true pose from CARLA and degrades it with a field of
view, a range limit, occlusion, Gaussian noise and random dropout.  Useful to
separate "the policy cannot learn the task" from "the sensors are too weak".
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import geometry as geom
from ..carla_utils import actor_xy, actor_yaw_deg, has_line_of_sight
from .base import ObstacleTrack, RangeRateTracker


class GroundTruthPerception:
    def __init__(self, cfg, world, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self._tracker = RangeRateTracker(gate_m=6.0, lost_after_steps=4,
                                         ema_alpha=0.5)

    def reset(self) -> None:
        self._tracker.reset()

    def track(self, ego_actor, cyclist_actor, dt: float) -> ObstacleTrack:
        if cyclist_actor is None:
            return self._tracker.update(None, None, 0.0, dt)

        ego_xy = actor_xy(ego_actor)
        ego_yaw = actor_yaw_deg(ego_actor)
        cyclist_xy = actor_xy(cyclist_actor)

        range_m = float(np.linalg.norm(cyclist_xy - ego_xy))
        bearing_deg = geom.bearing_deg(ego_xy, ego_yaw, cyclist_xy)

        visible = (range_m <= self.cfg.max_range_m
                   and abs(bearing_deg) <= self.cfg.fov_degrees / 2.0
                   and self.rng.random() >= self.cfg.dropout_prob)
        if visible and self.cfg.require_line_of_sight:
            visible = has_line_of_sight(
                self.world,
                ego_actor.get_transform().location,
                cyclist_actor.get_transform().location)
        if not visible:
            return self._tracker.update(None, None, 0.0, dt)

        range_m += float(self.rng.normal(0.0, self.cfg.range_noise_std_m))
        bearing_deg += float(self.rng.normal(0.0, self.cfg.bearing_noise_std_deg))
        return self._tracker.update(max(0.0, range_m), bearing_deg, 1.7, dt)
