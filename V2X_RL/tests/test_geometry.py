import numpy as np
import pytest

from v2x_rl import geometry as geom


def test_arc_lengths_and_interpolation():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    cum = geom.arc_lengths(points)
    assert cum == pytest.approx([0.0, 3.0, 7.0])
    assert geom.point_at_arc_length(points, 5.0) == pytest.approx([3.0, 2.0])
    # Out-of-range values clamp to the endpoints.
    assert geom.point_at_arc_length(points, -1.0) == pytest.approx([0.0, 0.0])
    assert geom.point_at_arc_length(points, 99.0) == pytest.approx([3.0, 4.0])


def test_project_on_polyline_signed_lateral():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    s, lateral, idx = geom.project_on_polyline(points, np.array([4.0, 2.0]))
    assert s == pytest.approx(4.0)
    # Travel is +x, so +y is to the right in CARLA's frame -> positive offset.
    assert lateral == pytest.approx(2.0)
    assert idx == 0

    _, lateral_neg, _ = geom.project_on_polyline(points, np.array([4.0, -2.0]))
    assert lateral_neg == pytest.approx(-2.0)


def test_bearing_uses_y_right_convention():
    # Heading +x; a point at +y must read as a positive (right) bearing.
    bearing = geom.bearing_deg(np.zeros(2), 0.0, np.array([1.0, 1.0]))
    assert bearing == pytest.approx(45.0)
    bearing = geom.bearing_deg(np.zeros(2), 0.0, np.array([1.0, -1.0]))
    assert bearing == pytest.approx(-45.0)


def test_paths_conflict_point_true_crossing():
    ego = np.array([[-10.0, 0.0], [10.0, 0.0]])
    cyclist = np.array([[0.0, -10.0], [0.0, 10.0]])
    point, s_ego, s_cyclist = geom.paths_conflict_point(ego, cyclist)
    assert point == pytest.approx([0.0, 0.0])
    assert s_ego == pytest.approx(10.0)
    assert s_cyclist == pytest.approx(10.0)


def test_parallel_adjacent_lanes_are_not_a_conflict():
    """Riding alongside the ego is close but never a conflict.

    This is the failure mode that matters in practice: the cyclist rides in the
    adjacent lane ~2 m away for the whole approach, and a naive
    closest-approach test would flag every such episode as a conflict.
    """
    ego = np.array([[0.0, 0.0], [20.0, 0.0], [30.0, -10.0]])
    cyclist = np.array([[0.0, 2.0], [20.0, 2.0], [40.0, 2.0]])
    assert geom.paths_conflict_point(
        ego, cyclist, closest_approach_threshold_m=4.0) is None


def test_oncoming_parallel_lanes_are_not_a_conflict():
    ego = np.array([[0.0, 0.0], [40.0, 0.0]])
    cyclist = np.array([[40.0, 3.0], [0.0, 3.0]])
    assert geom.paths_conflict_point(
        ego, cyclist, closest_approach_threshold_m=4.0) is None


def test_closest_approach_fallback_needs_a_crossing_angle():
    """Paths that nearly meet at a large angle do count as a conflict."""
    ego = np.array([[0.0, 0.0], [20.0, 0.0]])
    cyclist = np.array([[21.5, -10.0], [21.5, 10.0]])
    assert geom.paths_conflict_point(
        ego, cyclist, closest_approach_threshold_m=1.0) is None
    result = geom.paths_conflict_point(
        ego, cyclist, closest_approach_threshold_m=3.0)
    assert result is not None
    assert result[1] == pytest.approx(20.0)


def test_require_crossing_rejects_the_closest_approach_fallback():
    """A near-miss that never actually crosses must not count as a conflict
    when the caller needs an unambiguous, visually-real crossing."""
    ego = np.array([[0.0, 0.0], [20.0, 0.0]])
    cyclist = np.array([[21.5, -10.0], [21.5, 10.0]])
    assert geom.paths_conflict_point(
        ego, cyclist, closest_approach_threshold_m=3.0) is not None
    assert geom.paths_conflict_point(
        ego, cyclist, closest_approach_threshold_m=3.0,
        require_crossing=True) is None


def test_require_crossing_keeps_a_true_crossing():
    ego = np.array([[-10.0, 0.0], [10.0, 0.0]])
    cyclist = np.array([[0.0, -10.0], [0.0, 10.0]])
    result = geom.paths_conflict_point(ego, cyclist, require_crossing=True)
    assert result is not None
    assert result[0] == pytest.approx([0.0, 0.0])


def test_s_min_guard_skips_upstream_crossings():
    """The arc-length guard restricts the search to the junction region."""
    ego = np.array([[0.0, 0.0], [40.0, 0.0]])
    # Crosses the ego path twice: once at x=10, once at x=30.
    cyclist = np.array([[10.0, -5.0], [10.0, 5.0], [30.0, 5.0], [30.0, -5.0]])
    _, s_first, _ = geom.paths_conflict_point(ego, cyclist)
    assert s_first == pytest.approx(10.0)
    _, s_second, _ = geom.paths_conflict_point(ego, cyclist, s_min_a=20.0)
    assert s_second == pytest.approx(30.0)


def test_time_to_collision():
    # Head-on closing at 10 m/s from 20 m, contact radius 2 m -> 1.8 s.
    ttc = geom.time_to_collision(np.array([20.0, 0.0]), np.array([-10.0, 0.0]), 2.0)
    assert ttc == pytest.approx(1.8)
    # Diverging: no contact.
    assert geom.time_to_collision(np.array([20.0, 0.0]),
                                  np.array([10.0, 0.0]), 2.0) == 30.0
    # Stationary relative motion: no contact.
    assert geom.time_to_collision(np.array([20.0, 0.0]),
                                  np.zeros(2), 2.0) == 30.0
    # Already overlapping.
    assert geom.time_to_collision(np.array([1.0, 0.0]),
                                  np.array([-1.0, 0.0]), 2.0) == 0.0


def test_time_to_arrival_caps_at_standstill():
    assert geom.time_to_arrival(50.0, 0.0, cap_s=10.0) == 10.0
    assert geom.time_to_arrival(10.0, 5.0) == pytest.approx(2.0)


def test_wrap_deg():
    assert geom.wrap_deg(370.0) == pytest.approx(10.0)
    assert geom.wrap_deg(-190.0) == pytest.approx(170.0)
    # The range is [-180, 180), so +180 wraps to -180.
    assert geom.wrap_deg(180.0) == pytest.approx(-180.0)
    assert geom.wrap_deg(0.0) == pytest.approx(0.0)
