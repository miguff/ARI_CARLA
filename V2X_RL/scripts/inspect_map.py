#!/usr/bin/env python
"""Discover junctions and visualise sampled episode layouts in CARLA.

Run this first, before training: it caches the junction sites for the town and
lets you confirm that the left / right / straight routes and the cyclist
conflict actually look right in the simulator.

    python scripts/inspect_map.py --list
    python scripts/inspect_map.py --junction-id 1148 --episodes 6 --hold 8
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

import _bootstrap  # noqa: F401
from common import add_config_args, build_cfg, setup_logging

from v2x_rl.carla_utils import CarlaSession, carla, set_spectator_behind
from v2x_rl.scenario import ScenarioBuilder

LOGGER = logging.getLogger("inspect_map")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--list", action="store_true",
                        help="List the discovered junction sites and exit")
    parser.add_argument("--junction-id", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5,
                        help="How many episode layouts to sample and draw")
    parser.add_argument("--hold", type=float, default=6.0,
                        help="Seconds to display each layout")
    return parser.parse_args()


def draw_path(world, path, colour, life_time: float, z_offset: float = 0.4) -> None:
    for i in range(len(path.points) - 1):
        a = carla.Location(float(path.points[i][0]), float(path.points[i][1]),
                           float(path.z[i]) + z_offset)
        b = carla.Location(float(path.points[i + 1][0]), float(path.points[i + 1][1]),
                           float(path.z[i + 1]) + z_offset)
        world.debug.draw_line(a, b, thickness=0.12, color=colour,
                              life_time=life_time)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = build_cfg(args)
    if args.junction_id is not None:
        cfg = cfg.merge({"scenario.junction_id": args.junction_id})

    session = CarlaSession(cfg.carla)
    try:
        builder = ScenarioBuilder(session, cfg.scenario)
        print(f"\nMap: {session.map.name}")
        print(f"{len(builder.plans)} usable junction plan(s):")
        for plan in builder.plans:
            conflicts = {m: len(plan.conflicting(m)) for m in plan.maneuvers}
            print(f"  junction_id={plan.site.junction_id:<6} "
                  f"maneuvers={','.join(plan.maneuvers):<22} "
                  f"cyclist_options={len(plan.options):<3} "
                  f"conflicting={conflicts}")
        print(f"Manoeuvres available overall: {sorted(builder.available)}")
        if args.list:
            return

        rng = np.random.default_rng(cfg.seed)
        for episode in range(args.episodes):
            layout = builder.sample(rng)
            conflict = ("none" if layout.conflict_xy is None else
                        f"({layout.conflict_xy[0]:.1f}, {layout.conflict_xy[1]:.1f})")
            print(f"\n[{episode + 1}/{args.episodes}] {layout.describe()} "
                  f"conflict_at={conflict}")
            if layout.has_conflict:
                print(f"      ego reaches conflict at s={layout.ego_conflict_s:.1f} m, "
                      f"cyclist at s={layout.cyclist_conflict_s:.1f} m "
                      f"(starts at s={layout.cyclist_start_s:.1f} m, "
                      f"{layout.cyclist_conflict_s - layout.cyclist_start_s:.1f} m away)")

            # Park the spectator above the junction so the whole layout is visible.
            focus = (layout.conflict_xy if layout.conflict_xy is not None
                     else layout.ego_path.pose_at(layout.ego_path.junction_s)[0])
            session.spectator.set_transform(carla.Transform(
                carla.Location(float(focus[0]), float(focus[1]), 45.0),
                carla.Rotation(pitch=-89.0)))

            deadline = time.time() + args.hold
            while time.time() < deadline:
                draw_path(session.world, layout.ego_path,
                          carla.Color(0, 160, 255), life_time=0.3)
                if layout.cyclist_path is not None:
                    draw_path(session.world, layout.cyclist_path,
                              carla.Color(255, 150, 0), life_time=0.3)
                if layout.conflict_xy is not None:
                    session.world.debug.draw_point(
                        carla.Location(float(layout.conflict_xy[0]),
                                       float(layout.conflict_xy[1]),
                                       float(layout.ego_spawn.location.z) + 0.6),
                        size=0.35, color=carla.Color(255, 0, 0), life_time=0.3)
                session.world.debug.draw_string(
                    layout.ego_spawn.location + carla.Location(z=2.0),
                    f"EGO {layout.maneuver}", draw_shadow=False,
                    color=carla.Color(0, 160, 255), life_time=0.3)
                if layout.cyclist_spawn is not None:
                    session.world.debug.draw_string(
                        layout.cyclist_spawn.location + carla.Location(z=2.0),
                        "CYCLIST", draw_shadow=False,
                        color=carla.Color(255, 150, 0), life_time=0.3)
                session.world.tick()
                time.sleep(0.05)
    finally:
        session.restore()
        LOGGER.info("Restored asynchronous mode")


if __name__ == "__main__":
    main()
