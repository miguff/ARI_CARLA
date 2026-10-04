import numpy as np
import pytest

from v2x_rl.config import RewardCfg
from v2x_rl.reward import StepContext, compute_reward, yield_is_required

TARGET = 50.0 / 3.6


def ctx(**kwargs) -> StepContext:
    base = dict(speed_ms=TARGET, target_speed_ms=TARGET, action=0.3,
                prev_action=0.3, throttle=0.3, brake=0.0)
    base.update(kwargs)
    return StepContext(**base)


# --------------------------------------------------------------------------- #
#  Yield logic
# --------------------------------------------------------------------------- #
def test_no_yield_without_conflict():
    assert not yield_is_required(ctx(has_conflict=False))


def test_yield_required_when_cyclist_arrives_first():
    # Ego 40 m away at 14 m/s (~2.9 s); cyclist 10 m away at 5 m/s (2.0 s).
    assert yield_is_required(ctx(has_conflict=True,
                                 ego_dist_to_conflict_m=40.0,
                                 cyclist_dist_to_conflict_m=10.0,
                                 cyclist_speed_ms=5.0))


def test_no_yield_when_ego_is_clearly_first():
    # Ego 10 m away at 14 m/s (0.7 s); cyclist 40 m away at 5 m/s (8 s).
    assert not yield_is_required(ctx(has_conflict=True,
                                     ego_dist_to_conflict_m=10.0,
                                     cyclist_dist_to_conflict_m=40.0,
                                     cyclist_speed_ms=5.0))


def test_no_yield_once_cyclist_has_cleared():
    assert not yield_is_required(ctx(has_conflict=True,
                                     ego_dist_to_conflict_m=20.0,
                                     cyclist_dist_to_conflict_m=-8.0,
                                     cyclist_speed_ms=5.0,
                                     cyclist_cleared=True))


def test_stopped_ego_yields_only_for_an_imminent_cyclist():
    """The yield decision is judged at a reference speed, so a braked ego
    cannot inflate its own time-to-arrival and make a distant cyclist "have
    priority" — that would reward sitting still (the SAC baseline collapse)."""
    distant = ctx(speed_ms=0.0, has_conflict=True, ego_dist_to_conflict_m=5.0,
                  cyclist_dist_to_conflict_m=25.0, cyclist_speed_ms=5.0)
    assert not yield_is_required(distant)

    imminent = ctx(speed_ms=0.0, has_conflict=True, ego_dist_to_conflict_m=5.0,
                   cyclist_dist_to_conflict_m=6.0, cyclist_speed_ms=5.0)
    assert yield_is_required(imminent)


# --------------------------------------------------------------------------- #
#  Reward shaping
# --------------------------------------------------------------------------- #
def test_speed_tracking_peaks_at_target():
    cfg = RewardCfg()
    at_target = compute_reward(ctx(speed_ms=TARGET), cfg)
    too_slow = compute_reward(ctx(speed_ms=TARGET * 0.5), cfg)
    too_fast = compute_reward(ctx(speed_ms=TARGET * 1.5), cfg)
    assert at_target.components["speed"] == pytest.approx(cfg.w_speed)
    assert at_target.total > too_slow.total
    assert at_target.total > too_fast.total


def test_holding_target_speed_beats_stopping_when_road_is_clear():
    """The core anti-degenerate check: doing nothing must not be optimal."""
    cfg = RewardCfg()
    cruising = compute_reward(ctx(speed_ms=TARGET, throttle=0.4, action=0.4,
                                  prev_action=0.4), cfg)
    stopped = compute_reward(ctx(speed_ms=0.0, throttle=0.0, brake=0.8,
                                 action=-0.8, prev_action=-0.8), cfg)
    assert cruising.total > 0.0
    assert stopped.total < 0.0


def test_parking_for_a_non_imminent_cyclist_is_not_rewarded():
    """The collapse mode from the SAC baseline: a conflicting cyclist exists
    but is seconds away, and the ego just stops.  Holding station must be
    net-negative, with no yield bonus."""
    cfg = RewardCfg()
    parked = ctx(speed_ms=0.0, throttle=0.0, brake=0.0, action=0.0,
                 prev_action=0.0, has_conflict=True, ego_dist_to_conflict_m=8.0,
                 cyclist_dist_to_conflict_m=35.0, cyclist_speed_ms=5.0,
                 ego_in_conflict_zone=False, slow_steps=200)
    result = compute_reward(parked, cfg)
    assert result.components.get("yield_correct", 0.0) == 0.0
    assert result.total < 0.0


def test_yield_bonus_decays_the_longer_the_ego_waits():
    cfg = RewardCfg()
    common = dict(speed_ms=0.0, has_conflict=True, ego_dist_to_conflict_m=10.0,
                  cyclist_dist_to_conflict_m=6.0, cyclist_speed_ms=5.0,
                  ego_in_conflict_zone=False)
    fresh = compute_reward(ctx(slow_steps=0, **common), cfg)
    stale = compute_reward(ctx(slow_steps=300, **common), cfg)
    assert fresh.components["yield_correct"] == pytest.approx(cfg.w_yield_correct)
    assert 0.0 < stale.components["yield_correct"] < fresh.components["yield_correct"]


def test_yielding_outside_the_zone_is_rewarded():
    cfg = RewardCfg()
    yielding = ctx(speed_ms=1.0, throttle=0.0, brake=0.5, action=-0.5,
                   prev_action=-0.5, has_conflict=True,
                   ego_dist_to_conflict_m=12.0,
                   cyclist_dist_to_conflict_m=6.0, cyclist_speed_ms=5.0,
                   ego_in_conflict_zone=False, cyclist_in_conflict_zone=False,
                   obstacle_perceived=True)
    result = compute_reward(yielding, cfg)
    assert result.yield_required
    assert not result.yield_violated
    assert result.components["yield_correct"] == pytest.approx(cfg.w_yield_correct)
    # Braking for a perceived cyclist must not be punished as unnecessary.
    assert "unnecessary_brake" not in result.components
    assert "stalling" not in result.components


def test_entering_the_zone_while_yielding_is_penalised():
    cfg = RewardCfg()
    violating = ctx(speed_ms=TARGET, throttle=0.8, action=0.8, prev_action=0.8,
                    has_conflict=True, ego_dist_to_conflict_m=1.0,
                    cyclist_dist_to_conflict_m=2.0, cyclist_speed_ms=5.0,
                    ego_in_conflict_zone=True, cyclist_in_conflict_zone=True,
                    obstacle_perceived=True)
    result = compute_reward(violating, cfg)
    assert result.yield_required and result.yield_violated
    assert result.components["yield_violation"] < 0.0
    assert result.total < 0.0


def test_yield_violation_scales_with_speed():
    cfg = RewardCfg()
    common = dict(has_conflict=True, ego_dist_to_conflict_m=1.0,
                  cyclist_dist_to_conflict_m=2.0, cyclist_speed_ms=5.0,
                  ego_in_conflict_zone=True, cyclist_in_conflict_zone=True,
                  obstacle_perceived=True)
    fast = compute_reward(ctx(speed_ms=TARGET, **common), cfg)
    slow = compute_reward(ctx(speed_ms=1.0, **common), cfg)
    assert fast.components["yield_violation"] < slow.components["yield_violation"]


def test_low_ttc_is_penalised_quadratically():
    cfg = RewardCfg()
    mild = compute_reward(ctx(ttc_s=cfg.ttc_threshold_s * 0.9), cfg)
    severe = compute_reward(ctx(ttc_s=0.2), cfg)
    safe = compute_reward(ctx(ttc_s=10.0), cfg)
    assert "ttc" not in safe.components
    assert severe.components["ttc"] < mild.components["ttc"] < 0.0


def test_jerk_and_hard_brake_penalties():
    cfg = RewardCfg()
    smooth = compute_reward(ctx(action=0.3, prev_action=0.3), cfg)
    jerky = compute_reward(ctx(action=1.0, prev_action=-1.0), cfg)
    assert "jerk" not in smooth.components
    assert jerky.components["jerk"] == pytest.approx(-cfg.w_jerk * 2.0)

    slam = compute_reward(ctx(brake=1.0, throttle=0.0, action=-1.0,
                              prev_action=-1.0), cfg)
    assert slam.components["hard_brake"] < 0.0


def test_unnecessary_brake_and_stalling_only_when_clear():
    cfg = RewardCfg()
    braking_for_nothing = compute_reward(
        ctx(speed_ms=0.0, throttle=0.0, brake=0.9, action=-0.9, prev_action=-0.9,
            obstacle_perceived=False), cfg)
    assert braking_for_nothing.components["unnecessary_brake"] < 0.0
    assert braking_for_nothing.components["stalling"] == pytest.approx(-cfg.w_stalling)

    # Same behaviour, but something is actually there -> no such penalties.
    justified = compute_reward(
        ctx(speed_ms=0.0, throttle=0.0, brake=0.9, action=-0.9, prev_action=-0.9,
            obstacle_perceived=True), cfg)
    assert "unnecessary_brake" not in justified.components
    assert "stalling" not in justified.components


def test_terminal_terms_bypass_the_step_clip():
    cfg = RewardCfg()
    crash = compute_reward(ctx(collision=True), cfg)
    assert crash.total < -cfg.collision_penalty + cfg.step_clip
    goal = compute_reward(ctx(goal_reached=True), cfg)
    assert goal.total > cfg.goal_bonus - cfg.step_clip
    assert compute_reward(ctx(timeout=True), cfg).total < 0.0


def test_step_reward_is_clipped():
    cfg = RewardCfg(step_clip=0.5)
    result = compute_reward(ctx(speed_ms=TARGET, ttc_s=0.01, brake=1.0,
                                action=1.0, prev_action=-1.0), cfg)
    assert result.total >= -cfg.step_clip - 1e-9
