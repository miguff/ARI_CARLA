"""Pure-numpy geometry helpers.

Deliberately free of any CARLA import so that the conflict-detection logic
can be unit tested without a running simulator.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def wrap_deg(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def wrap_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def arc_lengths(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length of a polyline, shape (N,) for points (N, 2)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_on_polyline(points: np.ndarray, query: np.ndarray,
                        cumulative: Optional[np.ndarray] = None,
                        ) -> Tuple[float, float, int]:
    """Project ``query`` onto a polyline.

    Returns ``(s, lateral, index)`` where ``s`` is the arc length of the
    closest point, ``lateral`` is the signed offset (positive to the left of
    the travel direction) and ``index`` is the index of the segment start.
    """
    points = np.asarray(points, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if len(points) < 2:
        return 0.0, 0.0, 0
    if cumulative is None:
        cumulative = arc_lengths(points)

    starts = points[:-1]
    vecs = points[1:] - starts
    lengths_sq = np.einsum("ij,ij->i", vecs, vecs)
    lengths_sq = np.maximum(lengths_sq, 1e-12)
    t = np.einsum("ij,ij->i", query - starts, vecs) / lengths_sq
    t = np.clip(t, 0.0, 1.0)
    closest = starts + t[:, None] * vecs
    dists = np.linalg.norm(closest - query, axis=1)
    idx = int(np.argmin(dists))

    seg_len = math.sqrt(lengths_sq[idx])
    s = float(cumulative[idx] + t[idx] * seg_len)
    tangent = vecs[idx] / seg_len
    delta = query - closest[idx]
    # 2D cross product gives the signed lateral offset.
    lateral = float(tangent[0] * delta[1] - tangent[1] * delta[0])
    return s, lateral, idx


def point_at_arc_length(points: np.ndarray, s: float,
                        cumulative: Optional[np.ndarray] = None) -> np.ndarray:
    """Interpolate the polyline point at arc length ``s``."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros(2)
    if len(points) == 1:
        return points[0].copy()
    if cumulative is None:
        cumulative = arc_lengths(points)
    s = float(np.clip(s, cumulative[0], cumulative[-1]))
    idx = int(np.searchsorted(cumulative, s, side="right")) - 1
    idx = max(0, min(idx, len(points) - 2))
    seg_len = cumulative[idx + 1] - cumulative[idx]
    if seg_len < 1e-9:
        return points[idx].copy()
    t = (s - cumulative[idx]) / seg_len
    return points[idx] + t * (points[idx + 1] - points[idx])


def _segment_intersection(p1: np.ndarray, p2: np.ndarray,
                          q1: np.ndarray, q2: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return ``(t, u)`` parameters of a true segment intersection, if any."""
    r = p2 - p1
    s = q2 - q1
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-12:
        return None
    diff = q1 - p1
    t = (diff[0] * s[1] - diff[1] * s[0]) / denom
    u = (diff[0] * r[1] - diff[1] * r[0]) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return float(t), float(u)
    return None


def _tangent_deg(points: np.ndarray, index: int) -> float:
    """Heading of the polyline at vertex ``index``, in degrees."""
    i = min(max(index, 0), len(points) - 1)
    j = i + 1 if i + 1 < len(points) else i - 1
    delta = points[j] - points[i] if j > i else points[i] - points[j]
    return math.degrees(math.atan2(delta[1], delta[0]))


def paths_conflict_point(path_a: np.ndarray, path_b: np.ndarray,
                         closest_approach_threshold_m: float = 3.0,
                         s_min_a: float = 0.0, s_min_b: float = 0.0,
                         min_crossing_angle_deg: float = 20.0,
                         require_crossing: bool = False,
                         ) -> Optional[Tuple[np.ndarray, float, float]]:
    """Find where two paths conflict.

    Prefers a true crossing.  If the paths never cross but come within
    ``closest_approach_threshold_m`` of each other (e.g. merging or
    near-parallel lanes) the point of closest approach is returned instead --
    unless ``require_crossing`` is set, in which case a non-crossing pair
    returns ``None`` instead of that fallback.  Use this when a caller needs a
    guaranteed, visually-unambiguous crossing (e.g. picking the cyclist for an
    episode that is meant to force a yield decision) rather than "passes
    within a few metres without ever actually crossing," which the closest-
    approach case can be.

    ``s_min_a`` / ``s_min_b`` restrict the search to arc lengths beyond the
    given values.  This matters when the two paths share a stretch of lane
    before diverging: without it, the overlapping segments would register as
    a spurious crossing far upstream of the junction.

    The closest-approach fallback additionally requires the two headings to
    differ by at least ``min_crossing_angle_deg`` (from both 0 and 180
    degrees).  Two vehicles travelling along adjacent parallel lanes are
    permanently within a few metres of each other but never in conflict, and
    without this check every such episode would report a bogus conflict.

    Returns ``(point, s_a, s_b)`` or ``None``.
    """
    a = np.asarray(path_a, dtype=np.float64)
    b = np.asarray(path_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    cum_a = arc_lengths(a)
    cum_b = arc_lengths(b)
    # A segment is a candidate if any part of it lies beyond the guard, hence
    # the comparison against the segment *end*.
    valid_a = cum_a[1:] >= s_min_a
    valid_b = cum_b[1:] >= s_min_b

    for i in range(len(a) - 1):
        if not valid_a[i]:
            continue
        for j in range(len(b) - 1):
            if not valid_b[j]:
                continue
            hit = _segment_intersection(a[i], a[i + 1], b[j], b[j + 1])
            if hit is None:
                continue
            t, u = hit
            s_a = float(cum_a[i] + t * (cum_a[i + 1] - cum_a[i]))
            s_b = float(cum_b[j] + u * (cum_b[j + 1] - cum_b[j]))
            if s_a < s_min_a or s_b < s_min_b:
                continue
            return a[i] + t * (a[i + 1] - a[i]), s_a, s_b

    if require_crossing:
        return None

    # No crossing: fall back to the point of closest approach.  Vertices of
    # each path are projected onto the *segments* of the other, so the result
    # does not depend on how finely the polylines happen to be sampled.
    best: Optional[Tuple[float, np.ndarray, float, float]] = None
    for query_path, query_cum, other, other_cum, swapped in (
            (a, cum_a, b, cum_b, False), (b, cum_b, a, cum_a, True)):
        s_query_min = s_min_a if not swapped else s_min_b
        s_other_min = s_min_b if not swapped else s_min_a
        for index, vertex in enumerate(query_path):
            if query_cum[index] < s_query_min:
                continue
            s_other, lateral, _ = project_on_polyline(other, vertex, other_cum)
            if s_other < s_other_min:
                continue
            distance = abs(lateral)
            if best is None or distance < best[0]:
                closest = point_at_arc_length(other, s_other, other_cum)
                midpoint = 0.5 * (vertex + closest)
                s_a_hit = query_cum[index] if not swapped else s_other
                s_b_hit = s_other if not swapped else query_cum[index]
                best = (distance, midpoint, float(s_a_hit), float(s_b_hit))

    if best is None or best[0] > closest_approach_threshold_m:
        return None

    _, point, s_a, s_b = best
    heading_a = _tangent_deg(a, int(np.searchsorted(cum_a, s_a)))
    heading_b = _tangent_deg(b, int(np.searchsorted(cum_b, s_b)))
    heading_diff = abs(wrap_deg(heading_a - heading_b))
    if min(heading_diff, 180.0 - heading_diff) < min_crossing_angle_deg:
        return None
    return point, s_a, s_b


def to_local_frame(origin_xy: np.ndarray, yaw_deg: float,
                   point_xy: np.ndarray) -> np.ndarray:
    """Express ``point_xy`` in a frame at ``origin_xy`` rotated by ``yaw_deg``.

    The returned frame is x-forward, y-right, matching CARLA's left-handed
    world convention (``get_forward_vector()`` is ``(cos yaw, sin yaw)`` and
    ``get_right_vector()`` is ``(-sin yaw, cos yaw)``).  A positive y or a
    positive bearing therefore means "to the right of the heading", which is
    also what a CARLA lidar reports in its own sensor frame.
    """
    delta = np.asarray(point_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    return np.array([c * delta[0] + s * delta[1],
                     -s * delta[0] + c * delta[1]])


def to_world_frame(origin_xy: np.ndarray, yaw_deg: float,
                   local_xy: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_local_frame`: an x-forward, y-right point to world."""
    c = math.cos(math.radians(yaw_deg))
    s = math.sin(math.radians(yaw_deg))
    lx, ly = float(local_xy[0]), float(local_xy[1])
    origin = np.asarray(origin_xy, dtype=np.float64)
    return np.array([origin[0] + c * lx - s * ly,
                     origin[1] + s * lx + c * ly])


def bearing_deg(origin_xy: np.ndarray, yaw_deg: float,
                point_xy: np.ndarray) -> float:
    """Bearing of ``point_xy`` relative to the heading, in degrees."""
    local = to_local_frame(origin_xy, yaw_deg, point_xy)
    return math.degrees(math.atan2(local[1], local[0]))


def time_to_arrival(distance_m: float, speed_ms: float,
                    cap_s: float = 30.0) -> float:
    """Time to cover ``distance_m`` at ``speed_ms``, capped and non-negative."""
    if speed_ms <= 0.05:
        return cap_s
    return float(min(cap_s, max(0.0, distance_m) / speed_ms))


def time_to_collision(rel_position: np.ndarray, rel_velocity: np.ndarray,
                      radius_m: float, cap_s: float = 30.0) -> float:
    """Time until two constant-velocity discs of combined ``radius_m`` touch.

    ``rel_position`` and ``rel_velocity`` are the other object's position and
    velocity relative to the ego.  Returns ``cap_s`` when no contact occurs.
    """
    p = np.asarray(rel_position, dtype=np.float64)
    v = np.asarray(rel_velocity, dtype=np.float64)
    if np.linalg.norm(p) <= radius_m:
        return 0.0
    a = float(v @ v)
    if a < 1e-9:
        return cap_s
    b = 2.0 * float(p @ v)
    c = float(p @ p) - radius_m ** 2
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return cap_s
    root = math.sqrt(disc)
    t1 = (-b - root) / (2.0 * a)
    t2 = (-b + root) / (2.0 * a)
    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return cap_s
    return float(min(cap_s, min(candidates)))


def resample_polyline(points: np.ndarray, spacing_m: float) -> np.ndarray:
    """Resample a polyline at approximately uniform spacing."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or spacing_m <= 0:
        return points
    cum = arc_lengths(points)
    targets = np.arange(0.0, cum[-1] + 1e-9, spacing_m)
    return np.stack([point_at_arc_length(points, s, cum) for s in targets])
