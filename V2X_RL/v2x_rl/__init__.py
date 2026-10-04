"""RL for V2X-assisted intersection negotiation with a cyclist in CARLA."""
from __future__ import annotations

from typing import Any

from .config import (CarlaCfg, DepthCfg, EnvCfg, LidarCfg, RewardCfg,
                     ScenarioCfg, SensorCfg, V2XCfg)

__all__ = [
    "EnvCfg",
    "CarlaCfg",
    "ScenarioCfg",
    "SensorCfg",
    "DepthCfg",
    "LidarCfg",
    "V2XCfg",
    "RewardCfg",
    "IntersectionV2XEnv",
    "make_env",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Import the environment lazily.

    ``v2x_rl.config``, ``v2x_rl.reward`` and the V2X model are usable without
    a CARLA installation; only the environment itself needs it.
    """
    if name in ("IntersectionV2XEnv", "make_env"):
        from .envs import IntersectionV2XEnv
        if name == "IntersectionV2XEnv":
            return IntersectionV2XEnv

        def make_env(cfg: EnvCfg | None = None, **overrides) -> IntersectionV2XEnv:
            return IntersectionV2XEnv(cfg, **overrides)

        return make_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
