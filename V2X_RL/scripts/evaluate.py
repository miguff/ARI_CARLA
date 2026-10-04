#!/usr/bin/env python
"""Evaluate a saved checkpoint, or a non-learned baseline.

Any of the ``val_step*.zip`` snapshots written during training can be replayed,
not just the best one.  Pass no checkpoint and ``--control-mode acc`` (or
``oracle``) to score the non-learned baseline instead.

    python scripts/evaluate.py runs/sac_.../models/best_model.zip --render
    python scripts/evaluate.py --control-mode acc --episodes 40 --csv results/acc.csv
    python scripts/evaluate.py runs/residual_v2x/models/best_model.zip \
        --episodes 40 --csv results/residual_v2x.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

import _bootstrap  # noqa: F401
from common import add_config_args, build_cfg, setup_logging

from v2x_rl.callbacks import METRIC_KEYS
from v2x_rl.config import EnvCfg
from v2x_rl.sb3_utils import infer_algo_from_path, load_model, make_vec_env

LOGGER = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to a saved .zip model; omit to run a "
                             "non-learned baseline (needs --control-mode acc|oracle)")
    add_config_args(parser)
    parser.add_argument("--algo", default=None,
                        help="Algorithm of the checkpoint (inferred if omitted)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--render", action="store_true",
                        help="Follow the ego with the spectator and draw overlays")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of acting deterministically")
    parser.add_argument("--vecnormalize", default=None,
                        help="Path to the matching *_vecnormalize.pkl "
                             "(auto-detected if omitted)")
    parser.add_argument("--csv", default=None, help="Write per-episode rows here")
    parser.add_argument("--frame-stack", type=int, default=1)
    return parser.parse_args()


def find_vecnormalize(checkpoint: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    stem, _ = os.path.splitext(checkpoint)
    for candidate in (f"{stem}_vecnormalize.pkl",
                      os.path.join(os.path.dirname(checkpoint), "best_vecnormalize.pkl"),
                      os.path.join(os.path.dirname(checkpoint), "final_vecnormalize.pkl")):
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve_cfg(args: argparse.Namespace) -> EnvCfg:
    """Prefer the config the run was trained with, then apply CLI overrides."""
    if args.config is None and args.checkpoint:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
        trained = os.path.join(run_dir, "env_config.yaml")
        if os.path.isfile(trained):
            LOGGER.info("Using the training config %s", trained)
            args.config = trained
    return build_cfg(args)


def run(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = resolve_cfg(args)
    baseline = args.checkpoint is None
    if baseline and cfg.control_mode not in ("acc", "oracle"):
        raise SystemExit("No checkpoint given: pass --control-mode acc or oracle "
                         "to score a non-learned baseline.")

    algo = None
    if not baseline:
        algo = args.algo or infer_algo_from_path(args.checkpoint)
        if algo is None:
            raise SystemExit("Could not infer the algorithm; pass --algo ppo|sac|td3")

    venv = make_vec_env(cfg, frame_stack=args.frame_stack, normalize=False)
    try:
        model = None
        if baseline:
            LOGGER.info("Scoring the non-learned '%s' baseline (no model)",
                        cfg.control_mode)
        else:
            stats = find_vecnormalize(args.checkpoint, args.vecnormalize)
            if stats:
                from stable_baselines3.common.vec_env import VecNormalize
                LOGGER.info("Loading observation statistics from %s", stats)
                venv = VecNormalize.load(stats, venv)
                venv.training = False      # never update the statistics at eval time
                venv.norm_reward = False   # report raw, comparable returns
            else:
                LOGGER.warning("No VecNormalize statistics found; the policy will "
                               "see unnormalised observations and behave badly.")
            model = load_model(algo, args.checkpoint, venv)

        rows: List[Dict[str, Any]] = []
        zero_action = np.zeros((venv.num_envs, venv.action_space.shape[0]),
                               dtype=np.float32)

        for episode in range(args.episodes):
            venv.seed(args.eval_seed + episode)
            obs = venv.reset()
            total, steps, done, state = 0.0, 0, False, None
            info: Dict[str, Any] = {}
            while not done:
                if model is None:
                    action = zero_action
                else:
                    action, state = model.predict(
                        obs, state=state, deterministic=not args.stochastic)
                obs, reward, dones, infos = venv.step(action)
                total += float(reward[0])
                steps += 1
                done = bool(dones[0])
                info = infos[0]

            row = {"episode": episode, "reward": total, "steps": steps,
                   "maneuver": info.get("maneuver", "?"),
                   "control_mode": info.get("control_mode", cfg.control_mode),
                   "has_conflict": info.get("has_conflict", False)}
            row.update({key: info.get(key) for key in METRIC_KEYS})
            rows.append(row)
            LOGGER.info("ep %2d | %-8s conflict=%-5s | reward %+8.2f | %3d steps | "
                        "%s | speed %.1f km/h | minTTC %.1f s",
                        episode, row["maneuver"], row["has_conflict"], total, steps,
                        _outcome(row), row.get("mean_speed_kmh") or 0.0,
                        row.get("min_ttc") or 0.0)
        return rows
    finally:
        venv.close()


def _outcome(row: Dict[str, Any]) -> str:
    if row.get("collision"):
        return "COLLISION"
    if row.get("off_route"):
        return "OFF-ROUTE"
    if row.get("stuck"):
        return "STUCK"
    if row.get("success"):
        return "success"
    if row.get("timeout"):
        return "timeout"
    return "?"


def summarise(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    def mean(key: str, subset=None) -> float:
        values = [float(r[key]) for r in (subset or rows)
                  if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")

    modes = {r.get("control_mode") for r in rows if r.get("control_mode")}
    print("\n================ evaluation summary ================")
    print(f"control mode       : {', '.join(sorted(m for m in modes if m)) or '?'}")
    print(f"episodes           : {len(rows)}")
    print(f"mean reward        : {mean('reward'):+.2f}")
    print(f"success rate       : {100 * mean('success'):.1f}%")
    print(f"collision rate     : {100 * mean('collision'):.1f}%")
    print(f"cyclist collisions : {100 * mean('collision_with_cyclist'):.1f}%")
    print(f"timeout rate       : {100 * mean('timeout'):.1f}%")
    print(f"mean |residual|    : {mean('mean_residual'):.3f}")
    print(f"mean speed         : {mean('mean_speed_kmh'):.1f} km/h")
    print(f"mean speed error   : {mean('mean_speed_error_kmh'):.1f} km/h")
    print(f"mean |jerk|        : {mean('mean_jerk'):.3f}")
    print(f"min TTC            : {mean('min_ttc'):.2f} s")
    print(f"min cyclist dist   : {mean('min_cyclist_distance'):.2f} m")
    print(f"unnecessary brake  : {100 * mean('unnecessary_brake_frac'):.1f}% of steps")
    print(f"V2X available      : {100 * mean('v2x_valid_frac'):.1f}% of steps")
    print(f"V2X loss rate      : {100 * mean('v2x_loss_rate'):.1f}%")

    conflicts = [r for r in rows if r.get("has_conflict")]
    if conflicts:
        print(f"\n-- {len(conflicts)} episode(s) with a genuine conflict --")
        print(f"yield handled ok   : {100 * mean('yield_ok', conflicts):.1f}%")
        print(f"collision rate     : {100 * mean('collision', conflicts):.1f}%")
        print(f"min cyclist dist   : {mean('min_cyclist_distance', conflicts):.2f} m")
    clear = [r for r in rows if not r.get("has_conflict")]
    if clear:
        print(f"\n-- {len(clear)} episode(s) with no conflict --")
        print(f"mean speed         : {mean('mean_speed_kmh', clear):.1f} km/h")
        print(f"unnecessary brake  : "
              f"{100 * mean('unnecessary_brake_frac', clear):.1f}% of steps")
    print("====================================================\n")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    rows = run(args)
    summarise(rows)
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        LOGGER.info("Wrote %s", args.csv)


if __name__ == "__main__":
    main()
