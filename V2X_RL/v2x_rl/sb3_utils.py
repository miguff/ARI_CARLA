"""Helpers for building the vectorised environment and the SB3 algorithms."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv,
                                              VecFrameStack, VecNormalize,
                                              VecTransposeImage)

from .callbacks import METRIC_KEYS
from .config import EnvCfg

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3}

# Episode length is 600 steps at 20 Hz (30 s).  gamma=0.995 gives an effective
# horizon of ~200 steps (10 s), which comfortably spans the approach-and-yield
# decision; the default 0.99 (5 s) is too myopic for a 50 km/h approach.
_COMMON = dict(gamma=0.995, verbose=1)

DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "ppo": dict(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        **_COMMON,
    ),
    "sac": dict(
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        # A longer random warm-up seeds the replay buffer with real
        # forward-driving experience (and the occasional random goal reach)
        # before the policy starts exploiting.
        learning_starts=10_000,
        # Auto entropy tuning.  It was pinned to 0.1 while raw RL needed a held
        # entropy bonus to escape the "creep and time out" trap; residual mode
        # removes that trap (the ACC baseline always drives), so automatic
        # tuning -- explore early, converge later -- is the better choice and
        # helps escape local optima like the plateau at a fixed collision rate.
        ent_coef="auto",
        **_COMMON,
    ),
    "td3": dict(
        learning_rate=1e-3,
        buffer_size=300_000,
        batch_size=256,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        learning_starts=2_000,
        policy_delay=2,
        **_COMMON,
    ),
}


def make_vec_env(cfg: EnvCfg, log_dir: Optional[str] = None,
                 frame_stack: int = 1, normalize: bool = True,
                 norm_reward: bool = True) -> VecEnv:
    """Build the single-worker vectorised environment used for training.

    Only one worker is possible: a CARLA server hosts one world, and two ego
    vehicles in the same world would interfere with each other.
    """
    from .envs import IntersectionV2XEnv

    def _factory():
        env = IntersectionV2XEnv(cfg)
        monitor_path = os.path.join(log_dir, "monitor.csv") if log_dir else None
        return Monitor(env, filename=monitor_path,
                       info_keywords=tuple(["maneuver", *METRIC_KEYS]))

    venv: VecEnv = DummyVecEnv([_factory])
    if cfg.obs_mode == "vector_depth":
        venv = VecTransposeImage(venv)
    if frame_stack > 1:
        venv = VecFrameStack(venv, n_stack=frame_stack)
    if normalize:
        # Only the vector part is normalised; the depth image is already
        # scaled to [0, 255] and handled by the CNN extractor.
        norm_keys = ["vec"] if cfg.obs_mode == "vector_depth" else None
        venv = VecNormalize(venv, norm_obs=True, norm_reward=norm_reward,
                            clip_obs=10.0, gamma=_COMMON["gamma"],
                            norm_obs_keys=norm_keys)
    return venv


def policy_for(cfg: EnvCfg) -> str:
    return "MultiInputPolicy" if cfg.obs_mode == "vector_depth" else "MlpPolicy"


def build_model(algo: str, venv: VecEnv, cfg: EnvCfg, seed: int,
                tensorboard_log: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None):
    """Instantiate an SB3 algorithm with sensible defaults for this task."""
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}; choose from {sorted(ALGOS)}")

    kwargs = dict(DEFAULT_HYPERPARAMS[algo])
    kwargs.update(overrides or {})
    kwargs.setdefault("policy_kwargs", {})
    kwargs["policy_kwargs"].setdefault("net_arch", [256, 256])

    if algo == "td3":
        n_actions = venv.action_space.shape[0]
        kwargs.setdefault("action_noise", NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions)))

    # Training is CPU-only on this machine, and MLP policies of this size are
    # faster on CPU than on a GPU anyway.
    return ALGOS[algo](policy_for(cfg), venv, seed=seed, device="auto",
                       tensorboard_log=tensorboard_log, **kwargs)


def load_model(algo: str, path: str, venv: Optional[VecEnv] = None):
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}")
    return ALGOS[algo].load(path, env=venv, device="auto")


def infer_algo_from_path(path: str) -> Optional[str]:
    """Guess the algorithm from a checkpoint path, e.g. ``runs/sac_.../x.zip``."""
    lowered = path.lower()
    for name in ALGOS:
        if f"/{name}_" in lowered or f"{os.sep}{name}_" in lowered:
            return name
    for name in ALGOS:
        if name in lowered:
            return name
    return None
