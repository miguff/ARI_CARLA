#!/usr/bin/env python
"""Behaviour cloning: fit a policy to the oracle (ACC + ground-truth yield).

Rolls the ``oracle`` control mode to collect ``(observation, command)`` pairs,
fits the SAC MLP actor to them by regression, and saves it as a normal
Stable-Baselines3 checkpoint (+ VecNormalize stats).  It is then evaluated
exactly like an RL checkpoint, with ``--control-mode bc``:

    python scripts/bc_pretrain.py --episodes 150 --epochs 60 --run-name bc
    python scripts/evaluate.py runs/bc/models/bc_model.zip \
        --control-mode bc --episodes 40 --csv results/bc.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch as th
from torch.nn import functional as F

import _bootstrap  # noqa: F401
from common import ROOT, add_config_args, build_cfg, setup_logging

from v2x_rl.sb3_utils import build_model, make_vec_env

LOGGER = logging.getLogger("bc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--episodes", type=int, default=150,
                        help="Oracle rollout episodes to collect")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "runs"))
    return parser.parse_args()


def collect(venv, episodes: int):
    """Roll the oracle, returning (obs, command) arrays on the normalised space."""
    obs_buf, act_buf = [], []
    zero = np.zeros((venv.num_envs, venv.action_space.shape[0]), dtype=np.float32)
    for ep in range(episodes):
        venv.seed(1_000 + ep)
        obs = venv.reset()
        done = False
        while not done:
            obs_buf.append(np.asarray(obs[0], dtype=np.float32))
            obs, _, dones, infos = venv.step(zero)
            act_buf.append(np.float32(infos[0]["applied_command"]))
            done = bool(dones[0])
        if (ep + 1) % 20 == 0:
            LOGGER.info("collected %d/%d episodes (%d samples)",
                        ep + 1, episodes, len(obs_buf))
    return np.stack(obs_buf), np.asarray(act_buf, dtype=np.float32).reshape(-1, 1)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    cfg = build_cfg(args).merge({"control_mode": "oracle"})
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, args.run_name or f"bc_{stamp}")
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "env_config.yaml"))

    venv = make_vec_env(cfg, log_dir=run_dir, normalize=True, norm_reward=False)
    try:
        LOGGER.info("Collecting oracle demonstrations...")
        started = time.time()
        obs, act = collect(venv, args.episodes)
        LOGGER.info("Collected %d samples in %.1f min",
                    len(obs), (time.time() - started) / 60.0)
        venv.training = False   # freeze the observation statistics

        # The SAC model gives us the exact MlpPolicy actor to imitate into.
        model = build_model("sac", venv, cfg, seed=cfg.seed)
        actor = model.policy.actor
        device = model.device

        idx = np.random.default_rng(cfg.seed).permutation(len(obs))
        n_val = max(1, int(args.val_frac * len(obs)))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        x = th.as_tensor(obs, device=device)
        y = th.as_tensor(act, device=device)
        opt = th.optim.Adam(actor.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            actor.train()
            perm = tr_idx[np.random.permutation(len(tr_idx))]
            losses = []
            for start in range(0, len(perm), args.batch_size):
                b = perm[start:start + args.batch_size]
                pred = actor(x[b], deterministic=True)
                loss = F.mse_loss(pred, y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            actor.eval()
            with th.no_grad():
                val = float(F.mse_loss(actor(x[val_idx], deterministic=True),
                                       y[val_idx]))
            if epoch % 5 == 0 or epoch == args.epochs - 1:
                LOGGER.info("epoch %3d | train mse %.4f | val mse %.4f",
                            epoch, float(np.mean(losses)), val)

        model.save(os.path.join(model_dir, "bc_model.zip"))
        venv.save(os.path.join(model_dir, "bc_model_vecnormalize.pkl"))
        LOGGER.info("Saved BC policy to %s", os.path.join(model_dir, "bc_model.zip"))
        LOGGER.info("Evaluate with: python scripts/evaluate.py %s "
                    "--control-mode bc --episodes 40",
                    os.path.join(model_dir, "bc_model.zip"))
    finally:
        venv.close()


if __name__ == "__main__":
    main()
