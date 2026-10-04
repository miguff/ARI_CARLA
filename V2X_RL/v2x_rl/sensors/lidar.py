"""Lidar with geometric clustering, used instead of a neural detector.

Pipeline per frame: crop to a region of interest, drop ground and overhead
returns, cluster with DBSCAN, keep clusters whose bounding box could plausibly
be a cyclist, discard clusters that sit where a *static* object was seen
before, then associate the nearest survivor across frames for a
range / bearing / range-rate track.

The static filter matters at a junction: lamp posts and sign poles are thin
and tall, so they pass the cyclist shape gates and — being closer than the
cyclist — win "nearest cluster".  A cluster whose world-frame position barely
moves for a few frames is written off and its location excluded for a while.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .. import geometry as geom
from ..carla_utils import carla
from .base import FrameSync, ObstacleTrack, RangeRateTracker


class LidarSensor:
    """Wraps a ``sensor.lidar.ray_cast`` and caches the latest point cloud."""

    def __init__(self, world, cfg, attach_to, delta_seconds: float) -> None:
        self.cfg = cfg
        blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")
        blueprint.set_attribute("channels", str(cfg.channels))
        blueprint.set_attribute("range", str(cfg.range_m))
        blueprint.set_attribute("points_per_second", str(cfg.points_per_second))
        # One full revolution per simulation tick keeps the cloud complete.
        blueprint.set_attribute("rotation_frequency", str(1.0 / delta_seconds))
        blueprint.set_attribute("upper_fov", str(cfg.upper_fov))
        blueprint.set_attribute("lower_fov", str(cfg.lower_fov))
        blueprint.set_attribute("dropoff_general_rate", str(cfg.dropoff_general_rate))
        blueprint.set_attribute("noise_stddev", str(cfg.noise_stddev))

        transform = carla.Transform(carla.Location(x=cfg.pos_x, z=cfg.pos_z))
        self.actor = world.spawn_actor(blueprint, transform, attach_to=attach_to)

        self._points = np.zeros((0, 3), dtype=np.float32)
        self.frames_received = 0
        self.sync = FrameSync()
        self.actor.listen(self._on_scan)

        self._tracker = RangeRateTracker(
            gate_m=4.0, lost_after_steps=cfg.track_lost_after_steps,
            ema_alpha=cfg.range_rate_ema_alpha)
        self._reset_static_state()

    def _on_scan(self, measurement) -> None:
        data = np.frombuffer(measurement.raw_data, dtype=np.float32)
        self._points = np.reshape(data, (-1, 4))[:, :3].copy()
        self.frames_received += 1
        self.sync.note(measurement.frame)

    @property
    def points(self) -> np.ndarray:
        return self._points

    def _reset_static_state(self) -> None:
        # world_xy of clusters judged static, each with a countdown ttl.
        self._static_blobs: List[Tuple[np.ndarray, int]] = []
        self._committed_world: Optional[np.ndarray] = None
        self._world_speed_ema: float = 0.0
        self._static_streak: int = 0

    def reset_track(self) -> None:
        self._tracker.reset()
        self._reset_static_state()

    # ------------------------------------------------------------------ #
    def _filter_points(self) -> np.ndarray:
        """Keep candidate obstacle returns in the sensor frame."""
        cfg = self.cfg
        points = self._points
        if len(points) == 0:
            return points
        planar = np.linalg.norm(points[:, :2], axis=1)
        mask = ((points[:, 2] > cfg.ground_z_threshold_m)
                & (points[:, 2] < cfg.max_z_m)
                & (planar < cfg.roi_radius_m)
                & (planar > 1.5))          # ignore returns off the ego body
        return points[mask]

    def filtered_points(self) -> np.ndarray:
        """ROI-cropped, ground/overhead-removed returns (sensor frame).

        This is exactly the point set the clusterer operates on; exposed for
        visualisation.
        """
        return self._filter_points()

    def clusters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(centroid_xy, extents_xyz)`` for cyclist-like clusters."""
        cfg = self.cfg
        points = self._filter_points()
        if len(points) < cfg.dbscan_min_samples:
            return []

        labels = DBSCAN(eps=cfg.dbscan_eps_m,
                        min_samples=cfg.dbscan_min_samples).fit_predict(points[:, :2])
        found = []
        for label in np.unique(labels):
            if label < 0:
                continue
            cluster = points[labels == label]
            extents = cluster.max(axis=0) - cluster.min(axis=0)
            footprint = float(max(extents[0], extents[1]))
            height = float(extents[2])
            if not (cfg.cluster_min_extent_m <= footprint <= cfg.cluster_max_extent_m):
                continue
            if not (cfg.cluster_min_height_m <= height <= cfg.cluster_max_height_m):
                continue
            found.append((cluster[:, :2].mean(axis=0), extents))
        return found

    def track(self, dt: float, ego_xy: Optional[np.ndarray] = None,
              ego_yaw_deg: Optional[float] = None) -> ObstacleTrack:
        """Nearest *dynamic* cyclist-like cluster, tracked across frames.

        The lidar sits at the vehicle origin looking forward, so cluster
        coordinates are already ego-relative: x forward, y right.  When the ego
        pose is supplied, clusters that stay put in the world frame (poles,
        signs) are filtered out.
        """
        candidates = self.clusters()
        static_filter = ego_xy is not None and ego_yaw_deg is not None

        if not static_filter:
            if not candidates:
                return self._tracker.update(None, None, 0.0, dt)
            centroid, extents = min(candidates,
                                    key=lambda c: float(np.linalg.norm(c[0])))
            return self._tracker.update(
                float(np.linalg.norm(centroid)),
                math.degrees(math.atan2(centroid[1], centroid[0])),
                float(extents[2]), dt)

        # Age the exclusion list.
        self._static_blobs = [(xy, ttl - 1) for xy, ttl in self._static_blobs
                              if ttl - 1 > 0]

        cfg = self.cfg
        usable = []
        for centroid, extents in candidates:
            world = geom.to_world_frame(ego_xy, ego_yaw_deg, centroid)
            if any(float(np.linalg.norm(world - blob)) < cfg.static_exclusion_radius_m
                   for blob, _ in self._static_blobs):
                continue
            usable.append((centroid, extents, world))
        if not usable:
            self._committed_world = None
            self._world_speed_ema = 0.0
            self._static_streak = 0
            return self._tracker.update(None, None, 0.0, dt)

        centroid, extents, world = min(
            usable, key=lambda c: float(np.linalg.norm(c[0])))

        # World-frame speed of whatever we are locked onto.
        if self._committed_world is not None:
            step = float(np.linalg.norm(world - self._committed_world))
            inst = step / max(dt, 1e-6)
            a = cfg.range_rate_ema_alpha
            self._world_speed_ema = a * inst + (1.0 - a) * self._world_speed_ema
            if self._world_speed_ema < cfg.min_dynamic_speed_ms:
                self._static_streak += 1
            else:
                self._static_streak = 0
        self._committed_world = world

        if self._static_streak >= cfg.static_reject_after_steps:
            self._static_blobs.append((world.copy(), cfg.static_blob_ttl_steps))
            self._committed_world = None
            self._world_speed_ema = 0.0
            self._static_streak = 0
            # Drop it outright rather than let the range-rate tracker coast on
            # a spot we have just decided is a lamp post.
            self._tracker.reset()
            return ObstacleTrack()

        return self._tracker.update(
            float(np.linalg.norm(centroid)),
            math.degrees(math.atan2(centroid[1], centroid[0])),
            float(extents[2]), dt)

    def destroy(self) -> None:
        try:
            if self.actor.is_listening:
                self.actor.stop()
            self.actor.destroy()
        except Exception:
            pass
