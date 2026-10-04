"""Reward function.

Kept free of CARLA imports so it can be unit tested directly.  The reward is
computed from *ground truth* (the environment knows exactly where everybody
is); only the observation is restricted to what the ego could really sense.

The design balances four objectives that pull against each other:

1. hold the target speed (otherwise the safest policy is to never move);
2. yield when the cyclist genuinely has priority;
3. never collide and never cut it fine (time-to-collision margin);
4. behave smoothly — no bang-bang throttle, no braking for nothing.

The dense progress term (1), plus the fact that the yielding bonus (2) is paid
only for an *imminent* cyclist and decays while the ego sits still, are what
stop the classic degenerate solution of "brake always, collide never".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class StepContext:
    """Everything the reward needs about one simulation step."""

    speed_ms: float
    target_speed_ms: float
    action: float
    prev_action: float
    throttle: float
    brake: float
    # Forward progress along the route this step, in metres.
    progress_m: float = 0.0

    # Ground-truth conflict geometry.  ``has_conflict`` is False when there is
    # no cyclist, or its path never crosses the ego's.
    has_conflict: bool = False
    ego_dist_to_conflict_m: float = 0.0
    cyclist_dist_to_conflict_m: float = 0.0
    cyclist_speed_ms: float = 0.0
    ego_in_conflict_zone: bool = False
    cyclist_in_conflict_zone: bool = False
    cyclist_cleared: bool = False
    # Straight-line distance to the cyclist, whether or not paths conflict.
    cyclist_distance_m: float = float("inf")
    ttc_s: float = float("inf")

    # Whether any onboard sensor currently reports something close.  Used so
    # that braking for a genuinely perceived obstacle is never punished as
    # "unnecessary", even if ground truth says there is no path conflict.
    obstacle_perceived: bool = False

    collision: bool = False
    goal_reached: bool = False
    timeout: bool = False

    # Consecutive steps the ego has been essentially stationary.  Used to decay
    # the yielding bonus so that parking indefinitely is never a positive-reward
    # strategy.
    slow_steps: int = 0

    # Residual-RL only: the raw correction the policy applied on top of the ACC
    # baseline this step, in [-1, 1].  Zero in every other control mode.
    residual: float = 0.0


@dataclass
class RewardBreakdown:
    total: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    yield_required: bool = False
    yield_violated: bool = False


def yield_is_required(ctx: StepContext, margin_s: float = 1.5,
                      tta_cap_s: float = 30.0,
                      ref_speed_frac: float = 0.5) -> bool:
    """True when the cyclist has priority over the ego at the conflict point.

    The cyclist has priority while it has not yet cleared the conflict zone
    *and* it would reach the conflict point no later than the ego plus a
    safety margin.  Once it is through, the ego is free to proceed.

    The ego's time-to-arrival is evaluated at a *reference* speed — never
    slower than ``ref_speed_frac`` of the target — rather than at its current
    speed.  Judging it at the current speed lets the ego brake to a halt,
    drive its own time-to-arrival up to the cap, and thereby make yielding
    "required" for any cyclist however distant.  That turns "wait" into a
    positive-reward action and is exactly the degenerate policy this function
    must not invite.
    """
    if not ctx.has_conflict or ctx.cyclist_cleared:
        return False
    ref_speed = max(ctx.speed_ms, ref_speed_frac * ctx.target_speed_ms)
    ego_tta = _tta(ctx.ego_dist_to_conflict_m, ref_speed, tta_cap_s)
    cyclist_tta = _tta(ctx.cyclist_dist_to_conflict_m, ctx.cyclist_speed_ms, tta_cap_s)
    return cyclist_tta <= ego_tta + margin_s


def _tta(distance_m: float, speed_ms: float, cap_s: float = 30.0) -> float:
    if speed_ms <= 0.05:
        return cap_s
    return min(cap_s, max(0.0, distance_m) / speed_ms)


def _cyclist_is_imminent(ctx: StepContext, cfg) -> bool:
    """Whether the cyclist will actually reach the conflict point soon.

    A yield can be *nominally* required (the cyclist has priority) while the
    cyclist is still many seconds away.  Holding station for such a cyclist is
    not rewarded — only an imminent one earns the yielding bonus.
    """
    if not ctx.has_conflict or ctx.cyclist_cleared:
        return False
    return (_tta(ctx.cyclist_dist_to_conflict_m, ctx.cyclist_speed_ms)
            <= cfg.yield_imminent_tta_s)


def compute_reward(ctx: StepContext, cfg) -> RewardBreakdown:
    parts: Dict[str, float] = {}
    yield_required = yield_is_required(ctx)
    yield_violated = False

    # --- 1. speed tracking / yielding trade-off ------------------------ #
    if yield_required:
        # While yielding, being slow is correct.  Reward staying out of the
        # conflict zone, scaled down as the ego gets closer to it so that
        # creeping up to the line is fine but entering it is not.
        if not ctx.ego_in_conflict_zone:
            # Only an *imminent* cyclist earns the bonus, and it decays the
            # longer the ego sits still, so "park outside the zone forever" is
            # not a positive-reward strategy — the progress and living-cost
            # terms below then make crawling forward the better option.
            if _cyclist_is_imminent(ctx, cfg):
                decay = float(np.exp(-ctx.slow_steps
                                     / max(1.0, cfg.yield_correct_decay_steps)))
                parts["yield_correct"] = cfg.w_yield_correct * decay
        else:
            depth = 1.0 if ctx.cyclist_in_conflict_zone else 0.5
            speed_factor = min(1.0, ctx.speed_ms / max(1.0, ctx.target_speed_ms))
            parts["yield_violation"] = -cfg.w_yield_violation * depth * (0.4 + speed_factor)
            yield_violated = True
    else:
        # Asymmetric speed reward: being below target is penalised twice as
        # hard as being above it.  This prevents the "creep" local optimum
        # where the car moves just fast enough to avoid the binary stall
        # penalty but far below the target speed.
        diff = ctx.speed_ms - ctx.target_speed_ms
        target = max(1e-3, ctx.target_speed_ms)
        if diff < 0:
            error = -diff / target
            parts["speed"] = cfg.w_speed * float(np.clip(1.0 - 2.0 * error, -1.0, 1.0))
        else:
            error = diff / target
            parts["speed"] = cfg.w_speed * float(np.clip(1.0 - error, -1.0, 1.0))

    # Progress reward: a potential-style term proportional to the distance
    # gained along the route this step.  A stationary ego banks exactly zero of
    # it, which is the honest counterweight to the yielding bonus (see the
    # module docstring).  Applied whether or not a yield is required.
    parts["progress"] = cfg.w_progress * ctx.progress_m

    # --- 2. time-to-collision margin ----------------------------------- #
    if ctx.ttc_s < cfg.ttc_threshold_s:
        severity = 1.0 - ctx.ttc_s / cfg.ttc_threshold_s
        parts["ttc"] = -cfg.w_ttc * severity ** 2

    # --- 3. comfort / anti-reckless ------------------------------------ #
    jerk = abs(ctx.action - ctx.prev_action)
    if jerk > 0.0:
        parts["jerk"] = -cfg.w_jerk * jerk
    if ctx.brake > cfg.hard_brake_threshold:
        parts["hard_brake"] = -cfg.w_hard_brake * (ctx.brake - cfg.hard_brake_threshold)

    # --- 4. anti-overcautious ------------------------------------------ #
    # Braking "for nothing" is only punished when nothing is genuinely there.
    # Stalling is punished more broadly: also when a yield is nominally
    # required but the cyclist is not actually imminent, otherwise the agent
    # can sit still for a far-off cyclist at no cost.
    road_effectively_clear = (not ctx.obstacle_perceived
                              and not _cyclist_is_imminent(ctx, cfg))
    if not yield_required and not ctx.obstacle_perceived:
        if ctx.brake > 0.1:
            parts["unnecessary_brake"] = -cfg.w_unnecessary_brake * ctx.brake
    if road_effectively_clear and ctx.speed_ms * 3.6 < cfg.stall_speed_kmh:
        parts["stalling"] = -cfg.w_stalling

    # --- 4b. residual effort ----------------------------------------------- #
    # In residual mode the null correction is the prior: deviating from the ACC
    # baseline costs a little, so the policy only does it when it pays.
    if ctx.residual:
        parts["residual_effort"] = -cfg.w_residual_effort * abs(ctx.residual)

    parts["living_cost"] = -cfg.living_cost

    step_total = float(np.clip(sum(parts.values()), -cfg.step_clip, cfg.step_clip))

    # --- 5. terminal --------------------------------------------------- #
    terminal = 0.0
    if ctx.collision:
        terminal -= cfg.collision_penalty
        parts["collision"] = -cfg.collision_penalty
    if ctx.goal_reached:
        terminal += cfg.goal_bonus
        parts["goal"] = cfg.goal_bonus
    if ctx.timeout:
        terminal -= cfg.timeout_penalty
        parts["timeout"] = -cfg.timeout_penalty

    return RewardBreakdown(total=step_total + terminal, components=parts,
                           yield_required=yield_required,
                           yield_violated=yield_violated)
