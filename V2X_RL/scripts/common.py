"""Shared CLI plumbing for the scripts."""
from __future__ import annotations

import argparse
import ast
import logging
import os
from typing import Any, Dict, List, Optional

import _bootstrap  # noqa: F401

from v2x_rl.config import EnvCfg
from v2x_rl.v2x import GAP_FILL_MODES

ROOT = _bootstrap.ROOT


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="YAML config file (see configs/)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override any config field, e.g. --set v2x.per_far=1.0 "
                             "scenario.target_speed_kmh=30")
    parser.add_argument("--town", default=None, help="Override carla.town")
    parser.add_argument("--port", type=int, default=None, help="CARLA RPC port")
    parser.add_argument("--obs-mode", choices=["vector", "vector_depth"],
                        default=None)
    parser.add_argument("--no-depth", action="store_true",
                        help="Disable the depth camera")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Disable the lidar")
    parser.add_argument("--gt-perception", action="store_true",
                        help="Use the noisy ground-truth perception ablation "
                             "instead of the lidar tracker")
    parser.add_argument("--no-v2x", action="store_true",
                        help="Disable V2X entirely (sensors-only baseline)")
    parser.add_argument("--v2x-gap-fill",
                        choices=[m for m in GAP_FILL_MODES if m != "none"],
                        default=None,
                        help="Bridge brief V2X reception gaps with a motion "
                             "prediction instead of going straight to "
                             "'no information' (default: off)")
    parser.add_argument("--control-mode",
                        choices=["raw", "residual", "acc", "oracle", "bc"],
                        default=None,
                        help="How the policy action becomes throttle/brake "
                             "(default from config: residual)")
    parser.add_argument("--residual-scale", type=float, default=None,
                        help="Bound on the residual correction in residual mode")
    parser.add_argument("--no-rendering", action="store_true",
                        help="Run CARLA without rendering (much faster, but "
                             "incompatible with the depth camera)")
    parser.add_argument("--render-lidar", action="store_true",
                        help="Show a bird's-eye view of the lidar cloud and the "
                             "tracked cluster in an OpenCV window (slow)")
    parser.add_argument("--render-depth", action="store_true",
                        help="Show the depth camera frame with the observation "
                             "sectors overlaid in an OpenCV window (slow)")
    parser.add_argument("--live-view", action="store_true",
                        help="Periodically save a third-person frame to "
                             "--live-view-path (safe to leave on; no extra "
                             "CARLA client, unlike a live snapshot tool)")
    parser.add_argument("--live-view-path", default=None,
                        help="Where to save the live-view frame "
                             "(default: live_view.png)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def build_cfg(args: argparse.Namespace) -> EnvCfg:
    cfg = EnvCfg.load(args.config) if args.config else EnvCfg()

    overrides: Dict[str, Any] = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        overrides[key.strip()] = _parse_value(value.strip())

    if args.town:
        overrides["carla.town"] = args.town
    if args.port:
        overrides["carla.port"] = args.port
    if args.obs_mode:
        overrides["obs_mode"] = args.obs_mode
    if args.no_depth:
        overrides["sensors.depth.enabled"] = False
    if args.no_lidar:
        overrides["sensors.lidar.enabled"] = False
    if args.gt_perception:
        overrides["sensors.groundtruth.enabled"] = True
    if args.no_v2x:
        overrides["v2x.enabled"] = False
    if getattr(args, "v2x_gap_fill", None):
        overrides["v2x.gap_fill"] = args.v2x_gap_fill
    if getattr(args, "control_mode", None):
        overrides["control_mode"] = args.control_mode
    if getattr(args, "residual_scale", None) is not None:
        overrides["residual_scale"] = args.residual_scale
    if args.no_rendering:
        overrides["carla.no_rendering"] = True
        overrides["sensors.depth.enabled"] = False
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    if getattr(args, "render", False):
        overrides["render"] = True
    if getattr(args, "render_lidar", False):
        overrides["render_lidar"] = True
    if getattr(args, "render_depth", False):
        overrides["render_depth"] = True
    if getattr(args, "live_view", False):
        overrides["live_view"] = True
    if getattr(args, "live_view_path", None):
        overrides["live_view_path"] = args.live_view_path

    cfg = cfg.merge(overrides)

    if cfg.carla.no_rendering and cfg.sensors.depth.enabled:
        raise SystemExit("A depth camera requires rendering; drop --no-rendering "
                         "or add --no-depth.")
    if cfg.carla.no_rendering and cfg.live_view:
        raise SystemExit("--live-view needs a rendered world; drop --no-rendering.")
    if not (cfg.sensors.lidar.enabled or cfg.sensors.depth.enabled
            or cfg.sensors.groundtruth.enabled or cfg.v2x.enabled):
        raise SystemExit("Enable at least one information source: a sensor "
                         "(lidar / depth / ground-truth) or V2X.")
    return cfg


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S")
