"""Unit tests for the non-learned controllers (ACC, oracle, cyclist)."""
import numpy as np
import pytest

from v2x_rl.config import AccCfg
from v2x_rl.controllers import (ACCController, CyclistController,
                               OracleYieldController)
from v2x_rl.scenario import Path

TARGET_MS = 50.0 / 3.6


def acc() -> ACCController:
    return ACCController(AccCfg())


# --------------------------------------------------------------------------- #
#  ACC baseline
# --------------------------------------------------------------------------- #
def test_free_flow_accelerates_from_standstill():
    cmd = acc().command(speed_ms=0.0, target_speed_ms=TARGET_MS)
    assert 0.0 < cmd <= 1.0


def test_at_target_speed_command_is_near_zero():
    cmd = acc().command(speed_ms=TARGET_MS, target_speed_ms=TARGET_MS)
    assert cmd == pytest.approx(0.0, abs=0.05)


def test_above_target_speed_brakes():
    cmd = acc().command(speed_ms=TARGET_MS * 1.4, target_speed_ms=TARGET_MS)
    assert cmd < -0.3


def test_close_closing_lead_triggers_hard_brake():
    cmd = acc().command(speed_ms=10.0, target_speed_ms=TARGET_MS,
                        lead_gap_m=3.0, lead_closing_ms=6.0)
    assert cmd <= -0.9


def test_far_lead_barely_affects_command():
    free = acc().command(speed_ms=10.0, target_speed_ms=TARGET_MS)
    with_far_lead = acc().command(speed_ms=10.0, target_speed_ms=TARGET_MS,
                                  lead_gap_m=120.0, lead_closing_ms=10.0)
    assert with_far_lead > 0.0
    assert with_far_lead <= free + 1e-6


def test_command_is_always_bounded():
    for v in (0.0, 5.0, 20.0, 40.0):
        for gap in (None, 1.0, 10.0, 200.0):
            cmd = acc().command(v, TARGET_MS, lead_gap_m=gap,
                                lead_closing_ms=(v if gap else None))
            assert -1.0 <= cmd <= 1.0


# --------------------------------------------------------------------------- #
#  Oracle yield (behaviour-cloning expert)
# --------------------------------------------------------------------------- #
def test_oracle_matches_acc_when_no_yield():
    cfg = AccCfg()
    base = ACCController(cfg).command(12.0, TARGET_MS)
    oracle = OracleYieldController(cfg).command(
        12.0, TARGET_MS, yield_now=False, dist_to_conflict_m=25.0)
    assert oracle == pytest.approx(base)


def test_oracle_brakes_harder_than_acc_when_yielding_and_close():
    cfg = AccCfg()
    base = ACCController(cfg).command(12.0, TARGET_MS)
    oracle = OracleYieldController(cfg).command(
        12.0, TARGET_MS, yield_now=True, dist_to_conflict_m=18.0)
    assert oracle < base
    assert oracle <= -0.5


def test_oracle_lets_a_far_stopped_ego_creep_toward_the_line():
    cfg = AccCfg()
    oracle = OracleYieldController(cfg).command(
        0.0, TARGET_MS, yield_now=True, dist_to_conflict_m=40.0)
    assert oracle > 0.0  # approach, then it will brake as the line nears


# --------------------------------------------------------------------------- #
#  Cyclist hold-and-release
# --------------------------------------------------------------------------- #
class _MockActor:
    def set_transform(self, _t):
        pass

    def set_target_velocity(self, _v):
        pass


def _straight_path(length: float = 50.0, n: int = 26) -> Path:
    xs = np.linspace(0.0, length, n)
    return Path(points=np.column_stack([xs, np.zeros(n)]),
               z=np.zeros(n), yaw=np.zeros(n), cum=xs.copy(), junction_s=20.0)


def test_cyclist_moves_immediately_without_hold():
    c = CyclistController(_MockActor(), _straight_path(), 5.0, 10.0, 0.05)
    assert c.speed_ms == 5.0
    c.step()
    assert c.s > 10.0


def test_held_cyclist_waits_then_releases_when_ego_is_close():
    c = CyclistController(_MockActor(), _straight_path(), 5.0, 15.0, 0.05,
                          hold_until_release=True, meet_horizon_s=3.5)
    assert c.speed_ms == 0.0 and not c.released

    c.maybe_release(ego_tta_to_conflict_s=6.0)   # still far
    c.step()
    assert c.speed_ms == 0.0 and c.s == 15.0     # held, not advanced

    c.maybe_release(ego_tta_to_conflict_s=3.0)   # within the horizon
    assert c.released
    c.step()
    assert c.speed_ms == 5.0 and c.s > 15.0      # now moving


def test_release_is_latched():
    c = CyclistController(_MockActor(), _straight_path(), 5.0, 15.0, 0.05,
                          hold_until_release=True, meet_horizon_s=3.5)
    c.maybe_release(1.0)
    c.maybe_release(99.0)   # a later large TTA must not re-hold it
    assert c.released


def test_distance_safety_net_releases_a_crawling_ego():
    # The ego is braking hard (in response to the pre-release V2X broadcast),
    # so its speed -- and therefore its own TTA estimate -- is tiny, keeping
    # ego_tta huge forever.  It is still close in plain distance, so the
    # distance safety net (judged at a low reference speed) must release it.
    c = CyclistController(_MockActor(), _straight_path(), 5.0, 15.0, 0.05,
                          hold_until_release=True, meet_horizon_s=3.5,
                          release_min_speed_ms=3.0)
    huge_tta = 500.0
    close_gap_m = 3.5 * 3.0 - 0.1   # just inside meet_horizon_s * release_min_speed_ms
    c.maybe_release(huge_tta, ego_gap_m=close_gap_m)
    assert c.released


def test_distance_safety_net_does_not_fire_while_still_far():
    c = CyclistController(_MockActor(), _straight_path(), 5.0, 15.0, 0.05,
                          hold_until_release=True, meet_horizon_s=3.5,
                          release_min_speed_ms=3.0)
    far_gap_m = 3.5 * 3.0 + 5.0
    c.maybe_release(500.0, ego_gap_m=far_gap_m)
    assert not c.released


def test_max_hold_caps_an_indefinite_standoff():
    # No ego_gap_m provided (the caller couldn't judge distance either) and the
    # TTA estimate never drops -- the absolute time cap is the last resort.
    dt = 0.05
    c = CyclistController(_MockActor(), _straight_path(), 5.0, 15.0, dt,
                          hold_until_release=True, meet_horizon_s=3.5,
                          max_hold_s=1.0)
    steps = int(1.0 / dt)
    for _ in range(steps - 1):
        c.maybe_release(500.0)
    assert not c.released
    c.maybe_release(500.0)
    assert c.released
