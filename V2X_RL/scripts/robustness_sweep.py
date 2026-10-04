#!/usr/bin/env python
"""Re-evaluate one fixed policy under degraded V2X conditions.

This produces the headline result of the study: how much the V2X link actually
buys, and how gracefully the policy degrades as it becomes unreliable.  The
same checkpoint and the same episode seeds are used for every setting, so the
only thing that changes is the communication quality.

    python scripts/robustness_sweep.py runs/sac_.../models/best_model.zip \
        --episodes 20 --sweep packet-loss

Sweeps
------
packet-loss  raise ``v2x.per_far`` from perfect to total loss
range        shrink ``v2x.max_range_m``
latency      increase the delivery delay
nlos         increase the extra loss when the line of sight is blocked
off          a single run with V2X disabled (sensors only)
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import Any, Dict, List

import numpy as np

import _bootstrap  # noqa: F401
import evaluate as evaluate_script
from common import add_config_args, setup_logging

LOGGER = logging.getLogger("robustness")

SWEEPS: Dict[str, List[Dict[str, Any]]] = {
    "packet-loss": [
        {"label": "per_far=0.0", "v2x.per_near": 0.0, "v2x.per_far": 0.0},
        {"label": "per_far=0.3", "v2x.per_far": 0.3},
        {"label": "per_far=0.6", "v2x.per_far": 0.6},
        {"label": "per_far=0.9", "v2x.per_far": 0.9},
        {"label": "per_far=1.0", "v2x.per_near": 1.0, "v2x.per_far": 1.0},
    ],
    "range": [
        {"label": "range=200m", "v2x.max_range_m": 200.0},
        {"label": "range=120m", "v2x.max_range_m": 120.0},
        {"label": "range=60m", "v2x.max_range_m": 60.0},
        {"label": "range=30m", "v2x.max_range_m": 30.0},
        {"label": "range=10m", "v2x.max_range_m": 10.0},
    ],
    "latency": [
        {"label": "latency=0-20ms", "v2x.latency_ms": (0.0, 20.0)},
        {"label": "latency=20-120ms", "v2x.latency_ms": (20.0, 120.0)},
        {"label": "latency=100-300ms", "v2x.latency_ms": (100.0, 300.0)},
        {"label": "latency=300-800ms", "v2x.latency_ms": (300.0, 800.0)},
    ],
    "nlos": [
        {"label": "nlos=0.0", "v2x.nlos_extra_per": 0.0},
        {"label": "nlos=0.45", "v2x.nlos_extra_per": 0.45},
        {"label": "nlos=1.0", "v2x.nlos_extra_per": 1.0},
    ],
    "off": [
        {"label": "v2x_on", "v2x.enabled": True},
        {"label": "v2x_off", "v2x.enabled": False},
    ],
}

REPORT_KEYS = ["reward", "success", "collision", "yield_ok", "mean_speed_kmh",
               "mean_speed_error_kmh", "mean_jerk", "min_ttc",
               "min_cyclist_distance", "unnecessary_brake_frac",
               "v2x_valid_frac", "v2x_loss_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    add_config_args(parser)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="packet-loss")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=20_000)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--out", default=None, help="CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    settings = SWEEPS[args.sweep]
    results: List[Dict[str, Any]] = []

    for setting in settings:
        label = setting["label"]
        overrides = [f"{k}={v!r}" for k, v in setting.items() if k != "label"]
        LOGGER.info("=== %s (%s) ===", label, ", ".join(overrides) or "defaults")

        # Reuse evaluate.py so the two scripts can never diverge.
        sub = argparse.Namespace(**vars(args))
        sub.set = list(args.set) + overrides
        sub.render = False
        sub.stochastic = False
        sub.csv = None
        rows = evaluate_script.run(sub)

        summary: Dict[str, Any] = {"setting": label, "episodes": len(rows)}
        for key in REPORT_KEYS:
            values = [float(r[key]) for r in rows if r.get(key) is not None]
            summary[key] = float(np.mean(values)) if values else float("nan")
        conflicts = [r for r in rows if r.get("has_conflict")]
        summary["conflict_episodes"] = len(conflicts)
        summary["conflict_collision"] = (
            float(np.mean([float(r["collision"]) for r in conflicts]))
            if conflicts else float("nan"))
        results.append(summary)

    print(f"\n=========== robustness sweep: {args.sweep} ===========")
    header = (f"{'setting':<18}{'reward':>9}{'success':>9}{'collis.':>9}"
              f"{'yieldOK':>9}{'speed':>8}{'minTTC':>8}{'v2x%':>7}")
    print(header)
    print("-" * len(header))
    for row in results:
        print(f"{row['setting']:<18}{row['reward']:>+9.2f}"
              f"{100 * row['success']:>8.0f}%{100 * row['collision']:>8.0f}%"
              f"{100 * row['yield_ok']:>8.0f}%{row['mean_speed_kmh']:>8.1f}"
              f"{row['min_ttc']:>8.2f}{100 * row['v2x_valid_frac']:>6.0f}%")
    print("=" * len(header) + "\n")

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        f"robustness_{args.sweep}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":
    main()
