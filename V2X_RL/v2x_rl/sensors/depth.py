"""Depth camera and its reduction to angular min-range sectors.

The depth image is deliberately *not* fed to a detector.  Instead it is
reduced to the nearest obstacle range in each of ``n_sectors`` angular slices
of a horizontal band of the image.  That gives the policy dense, cheap
geometric awareness of what is close in front of it, without pretending the
car has a semantic understanding of the scene.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from ..carla_utils import carla
from .base import FrameSync


class DepthCamera:
    """Wraps a ``sensor.camera.depth`` and caches the latest frame."""

    def __init__(self, world, cfg, attach_to) -> None:
        self.cfg = cfg
        blueprint = world.get_blueprint_library().find("sensor.camera.depth")
        blueprint.set_attribute("image_size_x", str(cfg.width))
        blueprint.set_attribute("image_size_y", str(cfg.height))
        blueprint.set_attribute("fov", str(cfg.fov))
        transform = carla.Transform(
            carla.Location(x=cfg.pos_x, z=cfg.pos_z))
        self.actor = world.spawn_actor(blueprint, transform, attach_to=attach_to)

        self._depth_m = np.full((cfg.height, cfg.width), cfg.max_range_m,
                               dtype=np.float32)
        self.frames_received = 0
        self.sync = FrameSync()
        self.actor.listen(self._on_image)

    # ------------------------------------------------------------------ #
    def _on_image(self, image) -> None:
        """Decode CARLA's 24-bit depth encoding into metres."""
        raw = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = raw.reshape((image.height, image.width, 4)).astype(np.float32)
        # CARLA packs depth as R + G*256 + B*256^2 normalised over 2^24-1,
        # scaled to a 1000 m far plane.  raw_data is BGRA.
        normalised = (bgra[:, :, 2] + bgra[:, :, 1] * 256.0
                      + bgra[:, :, 0] * 65536.0) / (16777215.0)
        self._depth_m = normalised * 1000.0
        self.frames_received += 1
        self.sync.note(image.frame)

    @property
    def depth_m(self) -> np.ndarray:
        return self._depth_m

    # ------------------------------------------------------------------ #
    def _ground_mask(self, height: int, width: int) -> np.ndarray:
        """Boolean mask of pixels that look like flat ground.

        CARLA's depth sensor returns the **Euclidean** (slant) distance from
        the camera centre to the surface point, not the forward z-depth.

        For a pinhole camera at height ``H = pos_z`` with focal length ``f``
        and principal row ``cy``, a flat road at row ``v`` (below the horizon)
        lies at angle ``theta = atan(dv / f)`` below the optical axis, where
        ``dv = v - cy``.  The Euclidean distance to that ground point is::

            d_road = H / sin(theta) = H * sqrt(f^2 + dv^2) / dv

        Pixels whose measured depth is within ``ground_margin`` of that value
        are ground and must not be reported as obstacles — otherwise the
        nearest "obstacle" is always the tarmac a few metres ahead.
        """
        cfg = self.cfg
        f = width / (2.0 * math.tan(math.radians(cfg.fov / 2.0)))
        cy = height / 2.0
        rows = np.arange(height, dtype=np.float32)[:, None]
        dv = rows - cy
        # Only rows below the horizon can hit the ground; above it the
        # expected depth is infinite / undefined.
        safe_dv = np.where(dv > 0.5, dv, 1.0)
        ground_depth = np.where(
            dv > 0.5,
            cfg.pos_z * np.sqrt(f * f + safe_dv * safe_dv) / safe_dv,
            np.inf,
        )
        ground_depth = np.broadcast_to(ground_depth, (height, width))
        measured = self._depth_m
        is_ground = (measured >= ground_depth * (1.0 - cfg.ground_margin)) & \
                    (measured <= ground_depth * (1.0 + cfg.ground_margin)) & \
                    np.isfinite(ground_depth)
        return is_ground

    def sector_ranges(self) -> np.ndarray:
        """Nearest obstacle range per angular sector, in metres.

        Ground-plane pixels are removed first (see ``_ground_mask``); the
        remaining pixels in each angular sector are reduced with a robust low
        percentile so a handful of noisy pixels cannot fabricate an obstacle.
        """
        cfg = self.cfg
        height, width = self._depth_m.shape
        top = int(cfg.v_band[0] * height)
        bottom = int(cfg.v_band[1] * height)
        band = self._depth_m[top:bottom, :]
        if band.size == 0:
            return np.full(cfg.n_sectors, cfg.max_range_m, dtype=np.float32)

        ground = self._ground_mask(height, width)[top:bottom, :]
        obs = np.where(ground, cfg.max_range_m, band)

        columns = np.array_split(obs, cfg.n_sectors, axis=1)
        ranges = np.array([np.percentile(part, 2.0) if part.size else cfg.max_range_m
                           for part in columns], dtype=np.float32)
        return np.clip(ranges, 0.0, cfg.max_range_m)

    def sector_features(self) -> np.ndarray:
        """Sector ranges normalised to [0, 1]."""
        return (self.sector_ranges() / self.cfg.max_range_m).astype(np.float32)

    def sector_bearings_deg(self) -> np.ndarray:
        """Centre bearing of each sector, useful for debugging/plots."""
        cfg = self.cfg
        half = math.radians(cfg.fov) / 2.0
        edges = np.linspace(-half, half, cfg.n_sectors + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return np.degrees(centres).astype(np.float32)

    def cnn_image(self) -> np.ndarray:
        """Depth as a uint8 image of shape (1, H, W) for a CNN policy."""
        cfg = self.cfg
        clipped = np.clip(self._depth_m, 0.0, cfg.max_range_m) / cfg.max_range_m
        resized = cv2.resize(clipped, (cfg.cnn_size[1], cfg.cnn_size[0]),
                             interpolation=cv2.INTER_AREA)
        return (resized * 255.0).astype(np.uint8)[None, :, :]

    def destroy(self) -> None:
        try:
            if self.actor.is_listening:
                self.actor.stop()
            self.actor.destroy()
        except Exception:
            pass
