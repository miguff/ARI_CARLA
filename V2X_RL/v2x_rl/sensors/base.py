"""Shared types for the perception backends."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


class FrameSync:
    """Blocks until a sensor has delivered data for a given simulation frame.

    In synchronous mode ``world.tick()`` returns as soon as the frame is
    simulated, but sensor callbacks are dispatched on a separate thread, so
    reading the cached frame straight away can yield the previous tick's data.
    """

    def __init__(self) -> None:
        self.last_frame = -1

    def note(self, frame: int) -> None:
        self.last_frame = int(frame)

    def wait(self, frame: int, timeout_s: float = 2.0) -> bool:
        deadline = time.time() + timeout_s
        while self.last_frame < frame:
            if time.time() > deadline:
                return False
            time.sleep(0.0005)
        return True

TRACK_FEATURE_NAMES = [
    "track_valid",
    "track_range",
    "track_bearing_sin",
    "track_bearing_cos",
    "track_range_rate",
]
TRACK_FEATURE_DIM = len(TRACK_FEATURE_NAMES)

TRACK_RANGE_NORM_M = 50.0
TRACK_RANGE_RATE_NORM_MS = 20.0


@dataclass
class ObstacleTrack:
    """A tracked dynamic obstacle in the ego frame."""

    valid: bool = False
    range_m: float = 0.0
    bearing_deg: float = 0.0
    range_rate_ms: float = 0.0     # negative when closing
    height_m: float = 0.0

    def features(self) -> np.ndarray:
        if not self.valid:
            return np.zeros(TRACK_FEATURE_DIM, dtype=np.float32)
        bearing = np.radians(self.bearing_deg)
        return np.array([
            1.0,
            min(self.range_m, TRACK_RANGE_NORM_M) / TRACK_RANGE_NORM_M,
            np.sin(bearing),
            np.cos(bearing),
            float(np.clip(self.range_rate_ms / TRACK_RANGE_RATE_NORM_MS, -1.0, 1.0)),
        ], dtype=np.float32)


class RangeRateTracker:
    """Associates detections across frames and smooths the closing rate."""

    def __init__(self, gate_m: float, lost_after_steps: int,
                 ema_alpha: float) -> None:
        self.gate_m = gate_m
        self.lost_after_steps = lost_after_steps
        self.ema_alpha = ema_alpha
        self.reset()

    def reset(self) -> None:
        self._range: Optional[float] = None
        self._bearing: Optional[float] = None
        self._range_rate = 0.0
        self._misses = 0

    def update(self, range_m: Optional[float], bearing_deg: Optional[float],
               height_m: float, dt: float) -> ObstacleTrack:
        if range_m is None:
            self._misses += 1
            if self._misses > self.lost_after_steps or self._range is None:
                self.reset()
                return ObstacleTrack()
            # Coast on the last estimate for a few frames.
            self._range = max(0.0, self._range + self._range_rate * dt)
            return ObstacleTrack(True, self._range, self._bearing or 0.0,
                                 self._range_rate, height_m)

        if self._range is not None and abs(range_m - self._range) <= self.gate_m:
            raw_rate = (range_m - self._range) / max(dt, 1e-6)
            self._range_rate = (self.ema_alpha * raw_rate
                                + (1.0 - self.ema_alpha) * self._range_rate)
        else:
            self._range_rate = 0.0
        self._range = range_m
        self._bearing = bearing_deg
        self._misses = 0
        return ObstacleTrack(True, range_m, bearing_deg or 0.0,
                             self._range_rate, height_m)


def depth_sector_feature_names(n_sectors: int) -> List[str]:
    return [f"depth_sector_{i}" for i in range(n_sectors)]
