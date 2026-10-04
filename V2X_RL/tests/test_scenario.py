"""Unit tests for scenario-level candidate selection (no CARLA map needed)."""
import numpy as np

from v2x_rl.scenario import JunctionSite, Path, SitePlan, _z_filtered_conflict


def _plan(proximity):
    """A SitePlan with fake options/conflicts/proximity for the given clearances."""
    site = JunctionSite(junction_id=1, ego_starts={"left": (0.0, 0.0, 0.0)},
                        approaches=[])
    n = len(proximity)
    return SitePlan(
        site=site, ego_paths={}, options=[None] * n,
        conflicts={("left", i): None for i in range(n)},
        proximity={("left", i): p for i, p in enumerate(proximity)})


def test_non_conflicting_without_a_minimum_accepts_anything_close():
    plan = _plan([2.0, 8.0, 20.0])
    assert plan.non_conflicting("left", max_distance_m=30.0) == [0, 1, 2]


def test_non_conflicting_minimum_clearance_excludes_near_misses():
    # 2.0 m is "not a crossing" by the geometric test but is not safely clear;
    # this is exactly what produced a collision with no yield mechanism engaged.
    plan = _plan([2.0, 8.0, 20.0])
    assert plan.non_conflicting("left", max_distance_m=30.0, min_distance_m=10.0) == [2]


def test_non_conflicting_still_respects_the_upper_bound():
    plan = _plan([12.0, 40.0])
    assert plan.non_conflicting("left", max_distance_m=30.0, min_distance_m=10.0) == [0]


def test_conflicting_is_unaffected_by_the_clearance_change():
    site = JunctionSite(junction_id=1, ego_starts={"left": (0.0, 0.0, 0.0)},
                        approaches=[])
    plan = SitePlan(site=site, ego_paths={}, options=[None, None],
                    conflicts={("left", 0): "hit", ("left", 1): None},
                    proximity={("left", 0): 0.0, ("left", 1): 8.0})
    assert plan.conflicting("left") == [0]


# --------------------------------------------------------------------------- #
#  Elevation-aware conflict rejection (ramps / bridges)
# --------------------------------------------------------------------------- #
def _flat_path(z: float = 0.0, length: float = 50.0, n: int = 11) -> Path:
    xs = np.linspace(0.0, length, n)
    return Path(points=np.column_stack([xs, np.zeros(n)]), z=np.full(n, z),
               yaw=np.zeros(n), cum=xs.copy(), junction_s=20.0)


def test_z_filter_keeps_a_same_level_conflict():
    found = (np.array([10.0, 0.0]), 10.0, 10.0)
    assert _z_filtered_conflict(_flat_path(z=0.0), _flat_path(z=0.3),
                                found, max_z_gap_m=2.5) == found


def test_z_filter_rejects_a_ramp_over_the_other_path():
    # e.g. a cyclist path on a bridge well above the ego's road: the 2D test
    # sees a crossing, but the two can never physically touch.
    found = (np.array([10.0, 0.0]), 10.0, 10.0)
    assert _z_filtered_conflict(_flat_path(z=0.0), _flat_path(z=6.0),
                                found, max_z_gap_m=2.5) is None


def test_z_filter_passes_none_through():
    assert _z_filtered_conflict(_flat_path(), _flat_path(), None, 2.5) is None


# --------------------------------------------------------------------------- #
#  True-crossing vs. closest-approach-only conflicts
# --------------------------------------------------------------------------- #
def test_true_crossing_excludes_closest_approach_only():
    site = JunctionSite(junction_id=1, ego_starts={"left": (0.0, 0.0, 0.0)},
                        approaches=[])
    plan = SitePlan(site=site, ego_paths={}, options=[None, None],
                    conflicts={("left", 0): "hit", ("left", 1): "hit"},
                    proximity={("left", 0): 0.0, ("left", 1): 2.0},
                    true_crossings={("left", 0): True, ("left", 1): False})
    assert plan.true_crossing("left") == [0]
    assert plan.conflicting("left") == [0, 1]   # unaffected: both still "conflicting"


def test_true_crossing_defaults_to_false_when_unspecified():
    site = JunctionSite(junction_id=1, ego_starts={"left": (0.0, 0.0, 0.0)},
                        approaches=[])
    plan = SitePlan(site=site, ego_paths={}, options=[None],
                    conflicts={("left", 0): "hit"}, proximity={("left", 0): 0.0})
    assert plan.true_crossing("left") == []
