"""Intersection scenario construction.

Finds junctions on the loaded map, builds an ego route for each available
manoeuvre, enumerates every plausible cyclist approach, and works out which
combinations actually conflict.

Two realities of CARLA's road graph drive the design:

* junction approaches usually have **dedicated turn lanes**, so no single lane
  offers left *and* right *and* straight.  Manoeuvres are therefore keyed to
  their own approach lane on the same approach road;
* which cyclist approach conflicts with which ego manoeuvre depends on the
  junction geometry and cannot be hardcoded.  So all candidate cyclist paths
  are enumerated once per junction and their conflicts with each ego route are
  precomputed; sampling an episode is then a lookup, which also guarantees
  that "I wanted a conflict" and "there is a conflict" never disagree.

Paths are polylines rather than CARLA waypoint lists so the cyclist can ride
near the kerb, as a real cyclist would, on maps without bicycle lanes.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as geom
from .carla_utils import carla

LOGGER = logging.getLogger(__name__)

MANEUVER_TARGET_YAW = {"straight": 0.0, "right": 90.0, "left": -90.0}
ALL_MANEUVERS = tuple(MANEUVER_TARGET_YAW)

# How far a cyclist rides from the centre of its lane, towards the kerb.
CYCLIST_KERB_OFFSET_M = 0.9
# Minimum clearance between the ego and cyclist spawn points.
MIN_SPAWN_SEPARATION_M = 7.0
# How far ahead paths are traced.
PATH_LENGTH_M = 160.0
# A non-conflicting cyclist further than this from the ego route is irrelevant
# to the episode, so such candidates are not used.
RELEVANCE_DISTANCE_M = 30.0


# --------------------------------------------------------------------------- #
#  Paths
# --------------------------------------------------------------------------- #
@dataclass
class Path:
    """A driveable polyline with per-point elevation and heading."""

    points: np.ndarray          # (N, 2) world x/y
    z: np.ndarray               # (N,) world z
    yaw: np.ndarray             # (N,) heading in degrees
    cum: np.ndarray             # (N,) arc length
    junction_s: float           # arc length at which the junction starts

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def pose_at(self, s: float) -> Tuple[np.ndarray, float, float]:
        """Interpolated (xy, z, yaw_deg) at arc length ``s``."""
        s = float(np.clip(s, 0.0, self.length))
        idx = int(np.searchsorted(self.cum, s, side="right")) - 1
        idx = max(0, min(idx, len(self.points) - 2))
        seg = self.cum[idx + 1] - self.cum[idx]
        t = 0.0 if seg < 1e-9 else (s - self.cum[idx]) / seg
        xy = self.points[idx] + t * (self.points[idx + 1] - self.points[idx])
        z = float(self.z[idx] + t * (self.z[idx + 1] - self.z[idx]))
        yaw = float(self.yaw[idx] + t * geom.wrap_deg(self.yaw[idx + 1] - self.yaw[idx]))
        return xy, z, yaw

    def project(self, xy: np.ndarray) -> Tuple[float, float, int]:
        return geom.project_on_polyline(self.points, xy, self.cum)

    @property
    def start_yaw(self) -> float:
        return float(self.yaw[0])


def path_from_waypoints(waypoints: Sequence["carla.Waypoint"],
                        junction_index: Optional[int],
                        lateral_offset_m: float = 0.0) -> Path:
    """Build a :class:`Path` from CARLA waypoints, optionally offset sideways.

    ``lateral_offset_m`` is positive to the right of the travel direction.
    """
    pts, zs = [], []
    for wp in waypoints:
        transform = wp.transform
        loc = transform.location
        if abs(lateral_offset_m) > 1e-6:
            right = transform.get_right_vector()
            loc = carla.Location(loc.x + right.x * lateral_offset_m,
                                 loc.y + right.y * lateral_offset_m,
                                 loc.z)
        pts.append([loc.x, loc.y])
        zs.append(loc.z)

    points = np.asarray(pts, dtype=np.float64)
    cum = geom.arc_lengths(points)

    diffs = np.diff(points, axis=0)
    yaw = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
    yaw = np.concatenate([yaw, yaw[-1:]]) if len(yaw) else np.zeros(len(points))

    junction_s = float(cum[junction_index]) if junction_index is not None else 0.0
    return Path(points=points, z=np.asarray(zs, dtype=np.float64),
                yaw=yaw, cum=cum, junction_s=junction_s)


# --------------------------------------------------------------------------- #
#  Lane walking
# --------------------------------------------------------------------------- #
def _probe_yaw(wp: "carla.Waypoint", distance: float, step: float) -> float:
    """Heading after following ``wp`` forward for ``distance`` metres."""
    cur = wp
    travelled = 0.0
    while travelled < distance:
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        travelled += step
    return float(cur.transform.rotation.yaw)


def classify_branches(wp: "carla.Waypoint", step: float = 2.0,
                      probe_m: float = 28.0) -> Dict[str, "carla.Waypoint"]:
    """Map each manoeuvre available from ``wp`` to its first junction waypoint.

    CARLA's yaw grows clockwise seen from above, so a right turn produces a
    positive heading change.
    """
    ref_yaw = float(wp.transform.rotation.yaw)
    branches: Dict[str, Tuple[float, "carla.Waypoint"]] = {}
    for candidate in wp.next(step):
        delta = geom.wrap_deg(_probe_yaw(candidate, probe_m, step) - ref_yaw)
        for maneuver, target in MANEUVER_TARGET_YAW.items():
            error = abs(geom.wrap_deg(delta - target))
            if error > 45.0:
                continue
            best = branches.get(maneuver)
            if best is None or error < best[0]:
                branches[maneuver] = (error, candidate)
    return {m: wp_ for m, (_, wp_) in branches.items()}


def walk_path_waypoints(start_wp: "carla.Waypoint", maneuver: str,
                        step: float, forward_m: float = PATH_LENGTH_M,
                        probe_m: float = 28.0,
                        ) -> Tuple[List["carla.Waypoint"], Optional[int]]:
    """Follow the lane from ``start_wp``, taking ``maneuver`` at the junction.

    Returns the waypoint list and the index of the first junction waypoint.
    """
    waypoints = [start_wp]
    junction_index: Optional[int] = None
    maneuver_taken = False
    cur = start_wp
    travelled = 0.0

    while travelled < forward_m:
        candidates = cur.next(step)
        if not candidates:
            break
        entering_junction = any(c.is_junction for c in candidates)
        decide_maneuver = entering_junction and not maneuver_taken
        if decide_maneuver:
            chosen = classify_branches(cur, step, probe_m).get(maneuver)
            if chosen is None:
                break
        elif len(candidates) == 1:
            chosen = candidates[0]
        else:
            # Past the junction (or a plain lane split): keep going straight.
            ref_yaw = float(cur.transform.rotation.yaw)
            chosen = min(candidates, key=lambda c: abs(geom.wrap_deg(
                _probe_yaw(c, probe_m, step) - ref_yaw)))
        maneuver_taken = maneuver_taken or entering_junction
        if junction_index is None and chosen.is_junction:
            junction_index = len(waypoints)
        waypoints.append(chosen)
        travelled += step
        cur = chosen

    return waypoints, junction_index


def _back_up(wp: "carla.Waypoint", distance: float, step: float = 2.0,
             ) -> Optional["carla.Waypoint"]:
    """Walk backwards along the lane, refusing to enter a junction."""
    cur = wp
    travelled = 0.0
    while travelled < distance:
        prev = cur.previous(step)
        if not prev:
            return None
        cur = prev[0]
        if cur.is_junction:
            return None
        travelled += step
    return cur


# OpenDRIVE calls the bicycle lane type "Biking".
CYCLIST_LANE_TYPES = (carla.LaneType.Driving, carla.LaneType.Biking)


def _kerb_side_lane(wp: "carla.Waypoint") -> Optional["carla.Waypoint"]:
    """The nearest lane to the right that a cyclist could legitimately use."""
    candidate = wp.get_right_lane()
    if candidate is None:
        return None
    same_direction = candidate.lane_id * wp.lane_id > 0
    usable = candidate.lane_type in CYCLIST_LANE_TYPES
    return candidate if same_direction and usable else None


# --------------------------------------------------------------------------- #
#  Junction discovery
# --------------------------------------------------------------------------- #
@dataclass
class JunctionSite:
    """A junction and its usable approaches, cached to disk after discovery."""

    junction_id: int
    # Manoeuvre -> the ego's start waypoint location for that manoeuvre.  They
    # all sit on the same approach road but may be in different turn lanes.
    ego_starts: Dict[str, Tuple[float, float, float]]
    # Every pre-junction waypoint of the junction, used to enumerate cyclist
    # approaches (including the ego's own road, for the right-hook case).
    approaches: List[Tuple[float, float, float]]

    @property
    def maneuvers(self) -> List[str]:
        return sorted(self.ego_starts)

    def to_json(self) -> dict:
        return {
            "junction_id": self.junction_id,
            "ego_starts": {k: list(v) for k, v in self.ego_starts.items()},
            "approaches": [list(a) for a in self.approaches],
        }

    @classmethod
    def from_json(cls, data: dict) -> "JunctionSite":
        return cls(
            junction_id=data["junction_id"],
            ego_starts={k: tuple(v) for k, v in data["ego_starts"].items()},
            approaches=[tuple(a) for a in data["approaches"]],
        )


def _as_xyz(wp: "carla.Waypoint") -> Tuple[float, float, float]:
    loc = wp.transform.location
    return (float(loc.x), float(loc.y), float(loc.z))


def _waypoint_at(carla_map, xyz: Sequence[float]) -> "carla.Waypoint":
    return carla_map.get_waypoint(carla.Location(*xyz), project_to_road=True,
                                  lane_type=carla.LaneType.Driving)


def _collect_approaches(carla_map, step: float, sample_resolution: float,
                        ) -> Dict[int, List["carla.Waypoint"]]:
    """Group the last pre-junction waypoint of every lane by junction id."""
    approaches: Dict[int, List["carla.Waypoint"]] = {}
    seen: set = set()
    for wp in carla_map.generate_waypoints(sample_resolution):
        if wp.is_junction:
            continue
        nxt = wp.next(step)
        if not nxt or not any(c.is_junction for c in nxt):
            continue
        junction = next((c.get_junction() for c in nxt if c.is_junction), None)
        if junction is None:
            continue
        key = (junction.id, wp.road_id, wp.lane_id)
        if key in seen:
            continue
        seen.add(key)
        approaches.setdefault(junction.id, []).append(wp)
    return approaches


def discover_junction_sites(carla_map, required_maneuvers: Sequence[str] = ALL_MANEUVERS,
                            min_maneuvers: int = 2, runway_m: float = 45.0,
                            step: float = 2.0, sample_resolution: float = 4.0,
                            ) -> List[JunctionSite]:
    """Scan the map for junctions usable by the scenario.

    For each junction the approach *road* covering the most of the required
    manoeuvres is selected; individual manoeuvres may use different turn lanes
    of that road, which is how real junctions are laid out.
    """
    required = [m for m in required_maneuvers if m in MANEUVER_TARGET_YAW]
    sites: List[JunctionSite] = []

    for junction_id, entries in _collect_approaches(
            carla_map, step, sample_resolution).items():
        # Bucket the approach lanes by road, then keep the best road.
        by_road: Dict[int, Dict[str, "carla.Waypoint"]] = {}
        for wp in entries:
            branches = classify_branches(wp, step)
            if not any(m in branches for m in required):
                continue
            start = _back_up(wp, runway_m, step)
            if start is None:
                continue
            bucket = by_road.setdefault(wp.road_id, {})
            for maneuver in required:
                if maneuver in branches and maneuver not in bucket:
                    bucket[maneuver] = start
        if not by_road:
            continue

        road_id, best = max(by_road.items(), key=lambda kv: len(kv[1]))
        if len(best) < min_maneuvers:
            continue
        sites.append(JunctionSite(
            junction_id=junction_id,
            ego_starts={m: _as_xyz(wp) for m, wp in best.items()},
            approaches=[_as_xyz(wp) for wp in entries],
        ))

    # Prefer junctions covering the most manoeuvres.
    sites.sort(key=lambda s: (-len(s.ego_starts), s.junction_id))
    return sites


def load_or_discover_sites(carla_map, cache_path: str, town: str,
                           **kwargs) -> List[JunctionSite]:
    """Load cached junction sites for ``town`` or discover and cache them."""
    cache: Dict[str, list] = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path) as fh:
                cache = json.load(fh)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring unreadable junction cache %s", cache_path)
    key = town.split("/")[-1]
    if key in cache and cache[key]:
        return [JunctionSite.from_json(d) for d in cache[key]]

    LOGGER.info("Discovering junction sites on %s (one-off, takes a minute)...", key)
    sites = discover_junction_sites(carla_map, **kwargs)
    if not sites:
        raise RuntimeError(
            f"No usable junction found on {key}. Try lowering "
            "min_maneuvers or runway_m.")
    cache[key] = [s.to_json() for s in sites]
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as fh:
        json.dump(cache, fh, indent=2)
    LOGGER.info("Cached %d junction site(s) to %s", len(sites), cache_path)
    return sites


# --------------------------------------------------------------------------- #
#  Precomputed per-junction plan
# --------------------------------------------------------------------------- #
@dataclass
class CyclistOption:
    """One candidate cyclist approach through the junction."""

    path: Path
    maneuver: str
    origin: str            # "same-approach" | "oncoming" | "crossing"


@dataclass
class Conflict:
    xy: np.ndarray
    ego_s: float
    cyclist_s: float


@dataclass
class SitePlan:
    """Everything about a junction that does not change between episodes."""

    site: JunctionSite
    ego_paths: Dict[str, Path]
    options: List[CyclistOption]
    # (ego manoeuvre, option index) -> conflict, or None when they never meet.
    conflicts: Dict[Tuple[str, int], Optional[Conflict]]
    # (ego manoeuvre, option index) -> closest approach between the two paths.
    proximity: Dict[Tuple[str, int], float]
    # (ego manoeuvre, option index) -> True when ``conflicts[...]`` came from an
    # actual path crossing rather than the closest-approach fallback (merging
    # or near-parallel lanes that pass close without ever crossing).
    true_crossings: Dict[Tuple[str, int], bool] = field(default_factory=dict)

    @property
    def maneuvers(self) -> List[str]:
        return sorted(self.ego_paths)

    def conflicting(self, maneuver: str) -> List[int]:
        return [i for i in range(len(self.options))
                if self.conflicts.get((maneuver, i)) is not None]

    def true_crossing(self, maneuver: str) -> List[int]:
        """Candidates whose paths genuinely cross -- not just pass close by.

        Used wherever an episode must force an unambiguous yield decision: the
        closest-approach fallback in ``conflicting()`` can be satisfied by two
        paths that pass within a few metres without ever crossing, which reads
        as "not really conflicting" even though it technically clears the
        geometric threshold.
        """
        return [i for i in range(len(self.options))
                if self.conflicts.get((maneuver, i)) is not None
                and self.true_crossings.get((maneuver, i), False)]

    def non_conflicting(self, maneuver: str, max_distance_m: float,
                        min_distance_m: float = 0.0) -> List[int]:
        """Candidates whose paths never cross and stay >= ``min_distance_m`` away.

        Without the lower bound, "non-conflicting" only meant "paths do not
        mathematically cross" -- a candidate could still pass within a few
        metres of the ego's path, and combined with being placed near the
        junction at the same time as the ego, that produced genuine collisions
        with no yield mechanism engaged.
        """
        return [i for i in range(len(self.options))
                if self.conflicts.get((maneuver, i)) is None
                and min_distance_m <= self.proximity.get((maneuver, i), 1e9) <= max_distance_m]


def _z_filtered_conflict(
        path_a: Path, path_b: Path,
        found: Optional[Tuple[np.ndarray, float, float]],
        max_z_gap_m: float) -> Optional[Tuple[np.ndarray, float, float]]:
    """Reject a 2D conflict hit if the paths are too far apart in elevation.

    ``geom.paths_conflict_point`` works in the top-down (x, y) projection only,
    so it flags a "conflict" between paths that cross in projection but are
    actually on a ramp/bridge over or under each other -- not a physically
    reachable conflict.  Checks the elevation of each path at its own arc
    length at the hit, not a single shared point (the two paths need not agree
    on where "the same place" is once they diverge in height).
    """
    if found is None:
        return None
    _, s_a, s_b = found
    z_a = path_a.pose_at(s_a)[1]
    z_b = path_b.pose_at(s_b)[1]
    return None if abs(z_a - z_b) > max_z_gap_m else found


def _classify_origin(ego_yaw: float, cyclist_yaw: float) -> str:
    delta = abs(geom.wrap_deg(cyclist_yaw - ego_yaw))
    if delta < 45.0:
        return "same-approach"
    if delta > 135.0:
        return "oncoming"
    return "crossing"


def build_site_plan(carla_map, site: JunctionSite, cfg) -> Optional[SitePlan]:
    """Trace every ego route and cyclist candidate, and precompute conflicts."""
    step = cfg.route_sampling_resolution

    ego_len = getattr(cfg, "route_length_m", PATH_LENGTH_M)
    ego_paths: Dict[str, Path] = {}
    for maneuver, xyz in site.ego_starts.items():
        if maneuver not in cfg.maneuvers:
            continue
        wps, junction_idx = walk_path_waypoints(
            _waypoint_at(carla_map, xyz), maneuver, step, forward_m=ego_len)
        if len(wps) >= 10 and junction_idx is not None:
            ego_paths[maneuver] = path_from_waypoints(wps, junction_idx)
    if not ego_paths:
        return None

    ego_road_lanes = {(_waypoint_at(carla_map, xyz).road_id,
                       _waypoint_at(carla_map, xyz).lane_id)
                      for xyz in site.ego_starts.values()}

    options: List[CyclistOption] = []
    for xyz in site.approaches:
        approach = _waypoint_at(carla_map, xyz)
        # A cyclist sharing the ego's own lane centre would be run over at
        # spawn, so on the ego's road require a genuine adjacent lane.
        lane = approach
        offset = min(CYCLIST_KERB_OFFSET_M, approach.lane_width / 2.0 - 0.6)
        if (approach.road_id, approach.lane_id) in ego_road_lanes:
            kerb = _kerb_side_lane(approach)
            if kerb is None:
                continue
            lane = kerb
            offset = min(CYCLIST_KERB_OFFSET_M, kerb.lane_width / 2.0 - 0.6)

        start = _back_up(lane, 45.0, step) or lane
        for maneuver in classify_branches(start, step):
            wps, junction_idx = walk_path_waypoints(start, maneuver, step)
            if len(wps) < 10 or junction_idx is None:
                continue
            path = path_from_waypoints(wps, junction_idx,
                                       lateral_offset_m=max(0.0, offset))
            options.append(CyclistOption(path=path, maneuver=maneuver,
                                         origin="pending"))

    if not options:
        return None

    reference_yaw = next(iter(ego_paths.values())).start_yaw
    for option in options:
        option.origin = _classify_origin(reference_yaw, option.path.start_yaw)

    conflicts: Dict[Tuple[str, int], Optional[Conflict]] = {}
    true_crossings: Dict[Tuple[str, int], bool] = {}
    proximity: Dict[Tuple[str, int], float] = {}
    for maneuver, ego_path in ego_paths.items():
        for index, option in enumerate(options):
            s_min_a = max(0.0, ego_path.junction_s - 10.0)
            s_min_b = max(0.0, option.path.junction_s - 10.0)
            crossing_threshold = cfg.conflict_zone_radius_m * 0.6
            found = geom.paths_conflict_point(
                ego_path.points, option.path.points,
                closest_approach_threshold_m=crossing_threshold,
                s_min_a=s_min_a, s_min_b=s_min_b)
            found = _z_filtered_conflict(ego_path, option.path, found,
                                        cfg.conflict_max_z_gap_m)
            conflicts[(maneuver, index)] = (
                None if found is None else Conflict(found[0], found[1], found[2]))
            is_crossing = found is not None and geom.paths_conflict_point(
                ego_path.points, option.path.points,
                closest_approach_threshold_m=crossing_threshold,
                s_min_a=s_min_a, s_min_b=s_min_b, require_crossing=True) is not None
            true_crossings[(maneuver, index)] = is_crossing
            distances = np.linalg.norm(
                ego_path.points[:, None, :] - option.path.points[None, :, :], axis=2)
            proximity[(maneuver, index)] = float(distances.min())

    return SitePlan(site=site, ego_paths=ego_paths, options=options,
                    conflicts=conflicts, proximity=proximity,
                    true_crossings=true_crossings)


# --------------------------------------------------------------------------- #
#  Episode layout
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeLayout:
    maneuver: str
    ego_path: Path
    ego_spawn: "carla.Transform"
    junction_id: int
    cyclist_present: bool
    cyclist_path: Optional[Path] = None
    cyclist_spawn: Optional["carla.Transform"] = None
    cyclist_speed_ms: float = 0.0
    cyclist_maneuver: str = "straight"
    cyclist_origin: str = "none"
    cyclist_start_s: float = 0.0
    conflict_xy: Optional[np.ndarray] = None
    ego_conflict_s: Optional[float] = None
    cyclist_conflict_s: Optional[float] = None

    @property
    def has_conflict(self) -> bool:
        return self.conflict_xy is not None and self.cyclist_present

    def describe(self) -> str:
        cyclist = "none"
        if self.cyclist_present:
            cyclist = (f"{self.cyclist_origin}/{self.cyclist_maneuver} "
                       f"@{self.cyclist_speed_ms:.1f}m/s")
        return (f"junction={self.junction_id} ego={self.maneuver} "
                f"cyclist={cyclist} conflict={self.has_conflict}")


class ScenarioBuilder:
    """Samples episode layouts from the precomputed junction plans."""

    def __init__(self, session, cfg) -> None:
        self.session = session
        self.cfg = cfg
        self.map = session.map
        self.sites = load_or_discover_sites(
            self.map,
            cache_path=self._resolve_cache(cfg.junction_cache),
            town=session.map.name,
            required_maneuvers=cfg.maneuvers,
        )
        if cfg.junction_id is not None:
            matching = [s for s in self.sites if s.junction_id == cfg.junction_id]
            if not matching:
                raise ValueError(
                    f"junction_id={cfg.junction_id} not among the discovered "
                    f"sites {[s.junction_id for s in self.sites]}")
            self.sites = matching

        self.plans: List[SitePlan] = []
        for site in self.sites:
            plan = build_site_plan(self.map, site, cfg)
            if plan is None:
                LOGGER.debug("Junction %d produced no usable plan", site.junction_id)
                continue
            self.plans.append(plan)
            LOGGER.info(
                "Junction %d: manoeuvres=%s cyclist options=%d "
                "(conflicting: %s)", site.junction_id, ",".join(plan.maneuvers),
                len(plan.options),
                {m: len(plan.conflicting(m)) for m in plan.maneuvers})
        if not self.plans:
            raise RuntimeError("No junction produced a usable scenario plan")

        if cfg.require_full_junctions and cfg.junction_id is None:
            full = [p for p in self.plans if self._is_full(p)]
            if full:
                LOGGER.info("Restricting to %d junction(s) supporting every "
                            "manoeuvre with real conflicts: %s", len(full),
                            [p.site.junction_id for p in full])
                self.plans = full
            else:
                LOGGER.warning("No junction supports every manoeuvre with a "
                               "conflicting cyclist; using all %d",
                               len(self.plans))

        # Manoeuvres actually available across all junctions.
        self.available: Dict[str, List[SitePlan]] = {}
        for plan in self.plans:
            for maneuver in plan.maneuvers:
                self.available.setdefault(maneuver, []).append(plan)
        missing = [m for m in cfg.maneuvers if m not in self.available]
        if missing:
            LOGGER.warning("Manoeuvre(s) %s unavailable on this map; "
                           "sampling from %s", missing, sorted(self.available))
        # Only conflicting manoeuvres need a yielding decision.  Warn if none
        # of them can ever produce a conflict, which would make the task
        # trivial and is almost certainly a geometry bug.
        if not any(plan.conflicting(m) for m, plans in self.available.items()
                   for plan in plans):
            LOGGER.warning("No cyclist candidate conflicts with any ego route; "
                           "the yielding task would be vacuous")

        # (maneuver, plan) pairs that can actually produce a conflict -- used to
        # sample "conflict" episodes directly instead of picking a maneuver
        # first and hoping a conflicting candidate happens to exist for it.
        # Restricted to left/right: straight is deliberately the conflict-free
        # case (see _is_full above) even though a straight-through path can
        # geometrically cross a cyclist's -- that geometric crossing is only
        # ever used to judge "how close is close" (via ``proximity``), never
        # promoted to a ground-truth conflict.
        # Restricted to genuine crossings (see SitePlan.true_crossing): a
        # "conflict" episode must force an unambiguous yield decision, not one
        # satisfied by two paths merely passing close without ever crossing.
        self._conflict_pairs: List[Tuple[str, SitePlan]] = [
            (m, plan) for m, plans in self.available.items() for plan in plans
            if m in ("left", "right") and plan.true_crossing(m)]
        self._conflict_maneuvers = sorted({m for m, _ in self._conflict_pairs})
        if cfg.conflict_episode_prob > 0.0 and not self._conflict_pairs:
            LOGGER.warning("conflict_episode_prob=%.2f but no (maneuver, "
                           "junction) pair can produce a conflict; conflict "
                           "episodes will fall back to a plain drive",
                           cfg.conflict_episode_prob)

    def _is_full(self, plan: SitePlan) -> bool:
        """Supports every requested manoeuvre, with conflicts where relevant.

        A conflicting cyclist is only meaningful for the turning manoeuvres;
        ``straight`` is deliberately the conflict-free case.
        """
        wanted = set(self.cfg.maneuvers)
        if not wanted.issubset(plan.maneuvers):
            return False
        turning = wanted.intersection({"left", "right"})
        return all(plan.conflicting(m) for m in turning)

    @staticmethod
    def _resolve_cache(path: str) -> str:
        if os.path.isabs(path):
            return path
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, path)

    # -------------------------------------------------------------- #
    def sample(self, rng: np.random.Generator) -> EpisodeLayout:
        """Sample an episode: decide the type first, then the manoeuvre/plan.

        Deciding "conflict / non-conflicting cyclist / no cyclist" before the
        manoeuvre (rather than picking a manoeuvre and hoping it can host the
        wanted kind) is what makes ``conflict_episode_prob`` an accurate,
        directly-settable fraction of episodes -- since only left/right turns
        can host a genuine conflict, a high value concentrates the manoeuvre
        mix on turns rather than silently falling back to "no cyclist".
        """
        cfg = self.cfg
        roll = rng.random()
        want_conflict = roll < cfg.conflict_episode_prob and bool(self._conflict_pairs)
        want_cyclist = want_conflict or roll < cfg.conflict_episode_prob + cfg.nonconflicting_cyclist_prob

        if want_conflict:
            maneuver, plan = self._conflict_pairs[int(rng.integers(len(self._conflict_pairs)))]
        else:
            maneuver = self._sample_maneuver(rng)
            plan = self.available[maneuver][int(rng.integers(len(self.available[maneuver])))]
        ego_path = plan.ego_paths[maneuver]

        layout = EpisodeLayout(
            maneuver=maneuver, ego_path=ego_path,
            ego_spawn=_pose_to_transform(ego_path, 0.0),
            junction_id=plan.site.junction_id, cyclist_present=False)

        if not want_cyclist:
            return layout

        min_clear = cfg.non_conflicting_min_clearance_m
        candidates = (plan.true_crossing(maneuver) if want_conflict
                      else plan.non_conflicting(maneuver, RELEVANCE_DISTANCE_M, min_clear))
        if not candidates:
            if want_conflict:
                # No safely-clear non-conflicting candidate either: fall back
                # to it anyway rather than dropping the cyclist -- keeping the
                # yield mechanism engaged is still better than forcing a
                # marginal "clear" placement that is not actually safe.
                candidates = plan.non_conflicting(maneuver, RELEVANCE_DISTANCE_M, min_clear)
            elif maneuver in ("left", "right"):
                # Only a turn may fall back to a real (crossing) conflict here
                # -- straight stays conflict-free by design (see _is_full) even
                # when no safe non-conflicting candidate exists for it; such a
                # draw simply goes cyclist-free below.
                candidates = plan.true_crossing(maneuver)
        if not candidates:
            return layout

        for index in rng.permutation(np.asarray(candidates))[:5]:
            index = int(index)
            option = plan.options[index]
            conflict = plan.conflicts[(maneuver, index)]
            speed = float(rng.uniform(*cfg.cyclist_speed_ms))
            start_s = self._cyclist_start_s(ego_path, option.path, conflict,
                                            speed, rng)
            spawn = _pose_to_transform(option.path, start_s)
            if spawn.location.distance(layout.ego_spawn.location) < MIN_SPAWN_SEPARATION_M:
                continue

            layout.cyclist_present = True
            layout.cyclist_path = option.path
            layout.cyclist_maneuver = option.maneuver
            layout.cyclist_origin = option.origin
            layout.cyclist_speed_ms = speed
            layout.cyclist_start_s = start_s
            layout.cyclist_spawn = spawn
            if conflict is not None:
                layout.conflict_xy = conflict.xy
                layout.ego_conflict_s = conflict.ego_s
                layout.cyclist_conflict_s = conflict.cyclist_s
            return layout
        return layout

    # -------------------------------------------------------------- #
    def _sample_maneuver(self, rng: np.random.Generator) -> str:
        cfg = self.cfg
        weights = {m: w for m, w in zip(cfg.maneuvers, cfg.maneuver_weights)}
        options = [m for m in cfg.maneuvers if m in self.available]
        probabilities = np.array([weights[m] for m in options], dtype=np.float64)
        probabilities /= probabilities.sum()
        return str(rng.choice(options, p=probabilities))

    def _cyclist_start_s(self, ego_path: Path, cyclist_path: Path,
                         conflict: Optional[Conflict], cyclist_speed: float,
                         rng: np.random.Generator) -> float:
        """Where along its own path the cyclist starts.

        For a conflicting cyclist the position is chosen so that both would
        reach the conflict point at roughly the same time, then displaced by a
        random offset.  That offset is what decides who genuinely has priority
        and is therefore the core of the decision the agent must learn.
        """
        # The ego's realistic speed on the way to the conflict point -- it
        # decelerates for the turn, so this is well below target.  Overestimating
        # it places the cyclist early and it clears before the ego arrives.
        ego_speed = max(1.0, self.cfg.ego_conflict_speed_kmh / 3.6)
        if conflict is None:
            # No conflict: still arrange for the cyclist to be near the
            # junction while the ego is there, so it is worth perceiving.
            ego_tta = ego_path.junction_s / ego_speed
            distance = cyclist_speed * ego_tta
            target = cyclist_path.junction_s - distance
            return float(np.clip(target, 0.0, max(0.0, cyclist_path.length - 10.0)))

        # Hold-and-release: spawn the cyclist a fixed time of travel from the
        # conflict point; the env holds it until the ego is the same time away.
        # A guessed ego arrival time is no longer used -- it was the source of
        # the "cyclist clears before the ego arrives" looseness.
        horizon = getattr(self.cfg, "cyclist_meet_horizon_s", 3.5)
        offset = float(rng.uniform(*self.cfg.cyclist_offset_m))
        distance_to_conflict = max(4.0, cyclist_speed * horizon - offset)
        return float(np.clip(conflict.cyclist_s - distance_to_conflict,
                             0.0, max(0.0, conflict.cyclist_s - 3.0)))


def _pose_to_transform(path: Path, s: float, z_offset: float = 0.4,
                       ) -> "carla.Transform":
    xy, z, yaw = path.pose_at(s)
    return carla.Transform(carla.Location(float(xy[0]), float(xy[1]), z + z_offset),
                           carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
