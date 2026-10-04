#!/usr/bin/env python
"""Train a longitudinal-control policy for the V2X intersection scenario.

Examples
--------
    # Default: SAC, vector observations, depth + lidar, V2X on.
    python scripts/train.py --algo sac --timesteps 300000

    # PPO baseline matching the earlier experiments.
    python scripts/train.py --algo ppo --timesteps 500000

    # Ablation: no V2X at all, sensors only.
    python scripts/train.py --algo sac --no-v2x --run-name sac_no_v2x

    # Depth-image CNN policy (slow on CPU).
    python scripts/train.py --algo sac --obs-mode vector_depth
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime

import _bootstrap  # noqa: F401
from common import ROOT, add_config_args, build_cfg, setup_logging

from v2x_rl.callbacks import EpisodeMetricsCallback, PeriodicValidationCallback
from v2x_rl.sb3_utils import build_model, make_vec_env

LOGGER = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--algo", choices=["ppo", "sac", "td3"], default="sac")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "runs"))
    parser.add_argument("--val-freq", type=int, default=10_000,
                        help="Validation period in environment steps")
    parser.add_argument("--val-episodes", type=int, default=8)
    parser.add_argument("--val-seed", type=int, default=10_000,
                        help="Base seed for the fixed validation episodes")
    parser.add_argument("--frame-stack", type=int, default=1,
                        help="Stack N consecutive observations")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Disable VecNormalize")
    parser.add_argument("--resume", default=None,
                        help="Path to a .zip checkpoint to continue from")
    parser.add_argument("--render", action="store_true",
                        help="Draw debug overlays during training (slow)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Ramp scenario difficulty (conflict_episode_prob) as "
                             "validation success improves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = build_cfg(args)

    if cfg.control_mode not in ("raw", "residual"):
        raise SystemExit(
            f"control_mode={cfg.control_mode!r} is not trained with SB3. Use "
            "'raw' or 'residual' here; score 'acc'/'oracle' with evaluate.py "
            "and build 'bc' with bc_pretrain.py.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.algo}_{stamp}"
    run_dir = os.path.join(args.out_dir, run_name)
    model_dir = os.path.join(run_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    cfg.save(os.path.join(run_dir, "env_config.yaml"))
    with open(os.path.join(run_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    LOGGER.info("Run directory: %s", run_dir)
    LOGGER.info("Algorithm=%s obs_mode=%s depth=%s lidar=%s gt=%s v2x=%s",
                args.algo, cfg.obs_mode, cfg.sensors.depth.enabled,
                cfg.sensors.lidar.enabled, cfg.sensors.groundtruth.enabled,
                cfg.v2x.enabled)

    venv = make_vec_env(cfg, log_dir=run_dir, frame_stack=args.frame_stack,
                        normalize=not args.no_normalize)
    LOGGER.info("Observation space: %s", venv.observation_space)

    try:
        if args.resume:
            from v2x_rl.sb3_utils import load_model
            LOGGER.info("Resuming from %s", args.resume)
            model = load_model(args.algo, args.resume, venv)
            model.tensorboard_log = run_dir
        else:
            model = build_model(args.algo, venv, cfg, seed=cfg.seed,
                                tensorboard_log=run_dir)

        validation = PeriodicValidationCallback(
            venv, save_dir=model_dir, n_episodes=args.val_episodes,
            eval_freq=args.val_freq, base_seed=args.val_seed)
        callbacks = [EpisodeMetricsCallback(window=20), validation]

        if args.curriculum or cfg.curriculum.enabled:
            from v2x_rl.callbacks import CurriculumCallback
            cfg.curriculum.enabled = True
            cfg.save(os.path.join(run_dir, "env_config.yaml"))  # record the change
            callbacks.append(CurriculumCallback(validation, cfg.curriculum))
            LOGGER.info("Curriculum on: %d stages, advance at success>=%.2f "
                        "x%d, min %d steps/stage", len(cfg.curriculum.stages),
                        cfg.curriculum.advance_success,
                        cfg.curriculum.advance_patience,
                        cfg.curriculum.min_stage_steps)

        started = time.time()
        model.learn(total_timesteps=args.timesteps, callback=callbacks,
                    reset_num_timesteps=not bool(args.resume),
                    progress_bar=False)
        elapsed = time.time() - started

        model.save(os.path.join(model_dir, "final_model.zip"))
        if not args.no_normalize:
            venv.save(os.path.join(model_dir, "final_vecnormalize.pkl"))

        LOGGER.info("Finished %d steps in %.1f min (%.1f steps/s)",
                    args.timesteps, elapsed / 60.0, args.timesteps / max(elapsed, 1e-9))
        if validation.history:
            best = max(validation.history, key=lambda h: h["mean_reward"])
            LOGGER.info("Best validation: reward %.2f at step %d -> %s",
                        best["mean_reward"], best["timesteps"], best["path"])
        LOGGER.info("Validation log: %s", validation.csv_path)
        LOGGER.info("TensorBoard: tensorboard --logdir %s", args.out_dir)
    finally:
        venv.close()


if __name__ == "__main__":
    main()
