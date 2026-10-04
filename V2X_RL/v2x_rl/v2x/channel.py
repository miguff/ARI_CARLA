"""Wireless channel model for the cyclist -> vehicle link.

Three effects are modelled, all independently configurable so that a trained
policy can be re-evaluated under degraded communication:

* **range**: a hard cutoff beyond ``max_range_m``;
* **packet error rate**: rises with distance from ``per_near`` to ``per_far``,
  with an extra penalty when the line of sight is blocked.  Non-line-of-sight
  loss is the interesting case at an intersection, because that is exactly
  where an occluded cyclist is invisible to the ego's own sensors;
* **latency**: a uniformly random delay, so messages arrive stale and can
  arrive out of order.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .message import VAM


@dataclass
class ChannelStats:
    sent: int = 0
    dropped_range: int = 0
    dropped_error: int = 0
    delivered: int = 0

    def reset(self) -> None:
        self.sent = 0
        self.dropped_range = 0
        self.dropped_error = 0
        self.delivered = 0

    @property
    def loss_rate(self) -> float:
        return 0.0 if self.sent == 0 else 1.0 - self.delivered / self.sent


class V2XChannel:
    """Delivers VAMs to a single receiver with loss and latency."""

    def __init__(self, cfg, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.stats = ChannelStats()
        self._queue: List[Tuple[float, int, VAM]] = []
        self._counter = itertools.count()

    def reset(self) -> None:
        self._queue.clear()
        self.stats.reset()

    # ------------------------------------------------------------------ #
    def packet_error_rate(self, distance_m: float, line_of_sight: bool) -> float:
        """Packet error rate for a given geometry."""
        cfg = self.cfg
        if distance_m > cfg.max_range_m:
            return 1.0
        ratio = max(0.0, distance_m) / max(1e-6, cfg.max_range_m)
        per = cfg.per_near + (cfg.per_far - cfg.per_near) * ratio ** cfg.per_exponent
        if not line_of_sight:
            per += cfg.nlos_extra_per
        return float(np.clip(per, 0.0, 1.0))

    def transmit(self, message: Optional[VAM], now_s: float, distance_m: float,
                 line_of_sight: bool) -> None:
        """Offer a message to the channel; it may be dropped or delayed."""
        if message is None or not self.cfg.enabled:
            return
        self.stats.sent += 1

        if distance_m > self.cfg.max_range_m:
            self.stats.dropped_range += 1
            return
        if self.rng.random() < self.packet_error_rate(distance_m, line_of_sight):
            self.stats.dropped_error += 1
            return

        low, high = self.cfg.latency_ms
        latency_s = float(self.rng.uniform(low, high)) / 1000.0
        heapq.heappush(self._queue,
                       (now_s + latency_s, next(self._counter), message))

    def poll(self, now_s: float) -> List[VAM]:
        """Return every message whose delivery time has arrived."""
        delivered: List[VAM] = []
        while self._queue and self._queue[0][0] <= now_s + 1e-9:
            _, _, message = heapq.heappop(self._queue)
            delivered.append(message)
        self.stats.delivered += len(delivered)
        return delivered
