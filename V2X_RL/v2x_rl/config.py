"""Configuration dataclasses for the V2X intersection RL environment.

Every tunable knob lives here so that experiments (and especially the V2X
robustness sweeps) can be described by a single serialisable object.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional, Tuple, get_type_hints

import yaml


@dataclass
class CarlaCfg:
    """Connection and simulation settings for the CARLA server."""

    host: str = "localhost"
    port: int = 2000
    timeout: float = 60.0
    town: str = "Town03"
    fixed_delta_seconds: float = 0.05
    # Lidar works without rendering; a depth camera does not.  Disabling
    # rendering is a large speedup, so it is opt-in per experiment.
    no_rendering: bool = False
    # Switching maps at runtime takes minutes on a modest machine and can take
    # the server down with it, so by default a mismatched town is a hard error
    # telling you to launch the server with the right map instead.
    allow_load_world: bool = False
    # Reload the current world on construction.  Slow, but guarantees a clean
    # state; only useful when debugging leaked actors.
    reload_world: bool = False
    traffic_manager_port: int = 8000


@dataclass
class ScenarioCfg:
    """Episode layout and randomisation."""

    # Which ego manoeuvres to sample from, and their relative weights.
    maneuvers: Tuple[str, ...] = ("left", "right", "straight")
    maneuver_weights: Tuple[float, ...] = (1.0, 1.0, 1.0)

    target_speed_kmh: float = 50.0
    # Used only to normalise observations; must exceed target_speed_kmh.
    max_speed_kmh: float = 70.0

    # Ego starts with a random rolling speed so the policy never sees a
    # standing start only.
    initial_speed_kmh: Tuple[float, float] = (20.0, 40.0)

    # Episode-type mix: probability of a genuine conflict, and of a present but
    # non-conflicting cyclist; the remainder (1 - both) is cyclist-free.  Only
    # left/right turns can host a genuine conflict (straight is deliberately
    # conflict-free -- see ScenarioBuilder._is_full), so a high
    # conflict_episode_prob concentrates episodes on turning manoeuvres.
    conflict_episode_prob: float = 0.85
    nonconflicting_cyclist_prob: float = 0.10
    cyclist_speed_ms: Tuple[float, float] = (3.0, 7.0)
    # Speed the ego is assumed to make on its way to the conflict point, used
    # only to place the cyclist for a simultaneous arrival.  Measured from the
    # ACC baseline (it decelerates for the turn), not the target speed -- a too
    # high estimate places the cyclist early and it clears before the ego gets
    # there, so encounters end up 25 m+ apart with no real decision.
    ego_conflict_speed_kmh: float = 28.0
    # Longitudinal offset of the cyclist along its own route relative to the
    # nominal "arrives at the same time as the ego" position, in metres.
    # Negative -> cyclist is behind (ego has priority in practice).  Kept tight
    # (about one car-length of timing slack) so a conflicting episode is a
    # genuine close encounter, not a comfortable pass.
    cyclist_offset_m: Tuple[float, float] = (-5.0, 7.0)
    # Hold-and-release: on a conflicting episode the cyclist is spawned this
    # many seconds of travel from the conflict point and held stationary until
    # the ego is the same time away, then released -- so the two arrive
    # together regardless of the ego's actual speed.  Also the V2X warning
    # window the RL gets before the cyclist starts moving.
    cyclist_meet_horizon_s: float = 3.5
    # Release safety net: a policy that brakes hard in response to the
    # pre-release V2X broadcast can slow down just enough to keep its own
    # estimated time-to-arrival above cyclist_meet_horizon_s forever -- an
    # equilibrium where the ego never finishes closing in, so the cyclist
    # never starts.  Judge closeness at this (low, crawling-pace) reference
    # speed as a fallback, and cap the hold outright as a last resort.
    cyclist_release_min_speed_ms: float = 3.0
    cyclist_max_hold_s: float = 20.0
    # A "non-conflicting" candidate must clear the ego path by at least this
    # much.  The crossing test alone only guarantees > ~3 m, which is not
    # safely clear -- combined with placing the cyclist near the junction at
    # the same time as the ego (so it is worth perceiving), that produced
    # genuine collisions with no yield mechanism engaged, since ground truth
    # says there is no conflict to react to.
    non_conflicting_min_clearance_m: float = 10.0

    route_sampling_resolution: float = 2.0
    # Total length of the ego route: ~45 m approach + the junction + a short
    # exit.  Kept short so the goal is reachable inside max_episode_steps even
    # at a modest speed -- a 160 m route was ~30 s of driving, i.e. the whole
    # step budget, so any hesitation made the +goal terminal unreachable.
    route_length_m: float = 95.0
    # Distance to the final route waypoint that counts as success.
    goal_tolerance_m: float = 6.0
    max_episode_steps: int = 600

    # Radius around the ego/cyclist path crossing that counts as the
    # conflict zone.
    conflict_zone_radius_m: float = 5.0
    # A candidate crossing is only a genuine conflict if the two paths are
    # within this much of each other in elevation at the crossing point.  The
    # 2D-only geometry test otherwise flags a "conflict" between paths that
    # cross in the top-down projection but are actually on a ramp/bridge over
    # or under the other, which can never collide.
    conflict_max_z_gap_m: float = 2.5

    junction_cache: str = "cache/junctions.json"
    # If set, always use this junction id from the cache instead of sampling.
    junction_id: Optional[int] = None
    # Keep only junctions that support every requested manoeuvre *and* offer a
    # conflicting cyclist for each turning manoeuvre.  Without this the sampler
    # picks up junctions where, say, a right turn can never conflict, which
    # dilutes the training signal.  Falls back to all junctions if none qualify.
    require_full_junctions: bool = True


@dataclass
class LateralCtrlCfg:
    """Waypoint-following lateral controller (not learned)."""

    max_steer_degrees: float = 40.0
    lookahead_min_m: float = 4.0
    lookahead_speed_gain: float = 0.4
    kp: float = 0.9
    kd: float = 0.15


@dataclass
class AccCfg:
    """Baseline adaptive-cruise controller — the scaffold under residual RL.

    Purely reactive to what the onboard perception (lidar track + depth
    sectors) reports; it does *not* look at V2X or ground truth.  Follows the
    Intelligent Driver Model in free-flow and car-following, then maps the
    desired acceleration to a throttle/brake command in [-1, 1].
    """

    target_speed_frac: float = 1.0     # cruise at this fraction of the target speed
    time_headway_s: float = 1.6        # IDM T
    min_gap_m: float = 5.0            # IDM s0 (standstill gap)
    max_accel_ms2: float = 2.5        # IDM a
    comfort_decel_ms2: float = 2.5    # IDM b
    accel_exponent: float = 4.0       # IDM delta
    emergency_decel_ms2: float = 6.0  # hard floor on the commanded deceleration
    # A candidate lead is only followed when its bearing is within this cone.
    lead_bearing_deg: float = 35.0
    # Depth returns only count as a lead when this close (something dead ahead
    # that must be braked for); further out the centre depth sectors are just
    # scene geometry across the junction and must not slow the ACC.
    depth_emergency_range_m: float = 10.0
    # m/s^2 that map to full throttle / full brake.  Kept aggressive so that a
    # clear-road ACC actually holds the target speed against CARLA's drag.
    accel_to_throttle_ms2: float = 1.5
    decel_to_brake_ms2: float = 6.0


@dataclass
class DepthCfg:
    enabled: bool = True
    width: int = 320
    height: int = 180
    fov: float = 90.0
    pos_x: float = 1.4
    pos_z: float = 1.6
    max_range_m: float = 60.0
    # Number of angular sectors the depth image is reduced to.
    n_sectors: int = 8
    # Vertical band of the image used for the sector reduction, as fractions
    # of the image height.  Trims the sky and the bonnet; the road surface is
    # removed geometrically instead (see ground_margin).
    v_band: Tuple[float, float] = (0.30, 0.95)
    # Pixels whose depth is within this fraction of the expected flat-road
    # depth for their image row are treated as ground and ignored.  Without
    # this the nearest "obstacle" is always the tarmac a few metres ahead.
    ground_margin: float = 0.2
    # Obstacles closer than this (after ground removal) count as "perceived",
    # which suppresses the unnecessary-braking penalty.
    obstacle_alert_range_m: float = 20.0
    # Downsampled image returned in the ``vector_depth`` observation mode.
    cnn_size: Tuple[int, int] = (64, 64)


@dataclass
class LidarCfg:
    enabled: bool = True
    channels: int = 32
    range_m: float = 50.0
    points_per_second: int = 200000
    rotation_frequency: float = 20.0
    upper_fov: float = 10.0
    lower_fov: float = -25.0
    pos_x: float = 0.0
    pos_z: float = 1.8
    dropoff_general_rate: float = 0.1
    noise_stddev: float = 0.02

    # --- clustering ---
    roi_radius_m: float = 40.0
    ground_z_threshold_m: float = -1.4      # sensor frame; below this = ground
    max_z_m: float = 1.2
    dbscan_eps_m: float = 0.8
    dbscan_min_samples: int = 5
    # Cyclist-like cluster gates (bounding box extents, metres).
    cluster_max_extent_m: float = 2.6
    # A thin, tall return (lamp post, sign pole) otherwise reads as a cyclist:
    # require a minimum horizontal footprint so poles are rejected.
    cluster_min_extent_m: float = 0.4
    cluster_min_height_m: float = 0.5
    cluster_max_height_m: float = 2.4
    # --- static-object rejection ---
    # A cyclist moves; roadside furniture does not.  A tracked cluster whose
    # world-frame speed stays below this for ``static_reject_after_steps``
    # frames is written off as static: its location is excluded (within
    # ``static_exclusion_radius_m``) for ``static_blob_ttl_steps`` frames and
    # the tracker looks past it to the next candidate.
    min_dynamic_speed_ms: float = 0.8
    static_reject_after_steps: int = 4
    static_exclusion_radius_m: float = 2.0
    static_blob_ttl_steps: int = 60
    # Tracker
    track_lost_after_steps: int = 10
    range_rate_ema_alpha: float = 0.4


@dataclass
class GroundTruthPerceptionCfg:
    """Noisy ground-truth backend, used as an upper-bound ablation."""

    enabled: bool = False
    max_range_m: float = 50.0
    fov_degrees: float = 180.0
    range_noise_std_m: float = 0.3
    bearing_noise_std_deg: float = 1.0
    dropout_prob: float = 0.05
    require_line_of_sight: bool = True


@dataclass
class SensorCfg:
    depth: DepthCfg = field(default_factory=DepthCfg)
    lidar: LidarCfg = field(default_factory=LidarCfg)
    groundtruth: GroundTruthPerceptionCfg = field(
        default_factory=GroundTruthPerceptionCfg)


@dataclass
class V2XCfg:
    """ETSI TS 103 300-3 (VAM) inspired cyclist-to-vehicle link.

    The cyclist broadcasts a VRU Awareness Message containing its kinematic
    state and a path prediction; the ego only receives.
    """

    enabled: bool = True

    # --- VAM generation (TS 103 300-3 clause 6.4) ---
    max_rate_hz: float = 10.0
    min_rate_hz: float = 1.0
    trigger_position_delta_m: float = 4.0
    trigger_speed_delta_ms: float = 0.5
    trigger_heading_delta_deg: float = 4.0
    # Number of predicted path points and their spacing in seconds.
    path_prediction_points: int = 6
    path_prediction_dt_s: float = 0.5

    # --- channel ---
    max_range_m: float = 120.0
    # Packet error rate at zero distance and at max_range_m; interpolated
    # with ``per_exponent`` in between.
    per_near: float = 0.02
    per_far: float = 0.9
    per_exponent: float = 2.5
    # Additional packet error rate when the line of sight is blocked.
    nlos_extra_per: float = 0.45
    check_line_of_sight: bool = True
    latency_ms: Tuple[float, float] = (20.0, 120.0)

    # --- GNSS / sensor error on the transmitted state ---
    # Correlated (random walk) position bias plus white noise.
    gnss_bias_std_m: float = 1.2
    gnss_bias_tau_s: float = 5.0
    gnss_white_std_m: float = 0.35
    speed_noise_std_ms: float = 0.2
    heading_noise_std_deg: float = 2.0
    # Extra noise growth applied along the predicted path, per second ahead.
    path_prediction_noise_std_m_per_s: float = 0.4

    # --- receiver ---
    # A message older than this is treated as unavailable.
    max_message_age_s: float = 2.0
    # The reported path prediction only spans a few seconds.  The receiver
    # extends it along its final heading by this many metres before looking
    # for a trajectory interception with the ego route.
    receiver_extrapolation_m: float = 40.0
    # Horizon used to normalise the derived time-to-arrival features.
    receiver_tta_norm_s: float = 10.0

    # --- gap filling (opt-in; "none" reproduces the receiver's original
    # behaviour exactly, going invalid the instant a message passes
    # max_message_age_s) ---
    # "none" | "dead_reckoning" | "kalman"; see v2x_rl.v2x.gap_fill.
    gap_fill: str = "none"
    # A gap-filled estimate keeps being offered up to this age instead of
    # max_message_age_s; past it the receiver goes invalid exactly as if no
    # filler were configured. Must be >= max_message_age_s to have any effect.
    gap_fill_max_age_s: float = 5.0
    # Kalman filler only: assumed 1-sigma acceleration noise driving the
    # cyclist's velocity between updates (process noise).
    kf_process_accel_std_ms2: float = 1.5


@dataclass
class RewardCfg:
    # Speed tracking toward the target speed (per step).
    w_speed: float = 0.8
    # Progress reward: per-metre reward for forward travel along the route.
    # This is the dense signal that a stationary ego earns nothing of; it must
    # dominate the per-step trade-off, or "stop on the approach and time out"
    # remains an attractor for the value function (the SAC baseline collapse).
    w_progress: float = 0.6
    # Living cost, encourages finishing the manoeuvre.
    living_cost: float = 0.05
    # Yielding: penalty for encroaching on the conflict zone before the cyclist
    # has cleared it.  Kept moderate so that early exploration through the
    # junction is a recoverable mistake, not a catastrophe that teaches the
    # policy to never approach.
    w_yield_violation: float = 1.5
    # Bonus for waiting outside the conflict zone while an *imminent* cyclist
    # has priority.  Decays over consecutive stationary steps.
    w_yield_correct: float = 0.4
    # A yield is only "live" (and only earns w_yield_correct) when the cyclist
    # would reach the conflict point within this horizon.
    yield_imminent_tta_s: float = 6.0
    # w_yield_correct decays as exp(-slow_steps / this).  At ~80 steps (4 s at
    # 20 Hz) the bonus is down to ~1/e, so indefinite parking trends negative.
    yield_correct_decay_steps: float = 80.0
    # Safety: penalty as time-to-collision drops below the threshold.
    ttc_threshold_s: float = 3.0
    w_ttc: float = 1.5
    # Comfort / anti-reckless.
    w_jerk: float = 0.25
    w_hard_brake: float = 0.3
    hard_brake_threshold: float = 0.6
    # Anti-overcautious: braking or crawling with no active conflict.
    w_unnecessary_brake: float = 0.4
    w_stalling: float = 0.1
    stall_speed_kmh: float = 8.0
    # Terminal.  Large enough that a collision dominates any episode's positive
    # reward -- otherwise the policy will accept a steady collision rate because
    # driving fast to the goal still nets positive.
    collision_penalty: float = 200.0
    goal_bonus: float = 60.0
    timeout_penalty: float = 40.0
    # Clip the per-step reward (excluding terminal terms) for stability.
    step_clip: float = 10.0
    # Residual-RL only: cost per unit of correction the policy applies on top of
    # the ACC baseline, so "leave ACC alone" is the prior and the policy only
    # deviates when it earns something (e.g. an early V2X-informed yield).
    w_residual_effort: float = 0.1


@dataclass
class CurriculumCfg:
    """Difficulty schedule for training raw RL from scratch.

    Starts with no cyclist (just "drive the route") and steps to harder stages
    as the validation success rate at the current stage holds above a target.
    Only consulted when training with ``--curriculum``.
    """

    enabled: bool = False
    # (conflict_episode_prob, nonconflicting_cyclist_prob) per stage, easiest
    # first.  Early stages are cyclist-free or mostly non-conflicting; the last
    # stage matches the default scenario mix.
    stages: Tuple[Tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.30, 0.35),
        (0.55, 0.25),
        (0.85, 0.10),
    )
    # Advance once validation success stays >= advance_success for
    # advance_patience consecutive validations, and >= min_stage_steps have
    # elapsed in the current stage.
    advance_success: float = 0.75
    advance_patience: int = 2
    min_stage_steps: int = 20_000


@dataclass
class EnvCfg:
    carla: CarlaCfg = field(default_factory=CarlaCfg)
    scenario: ScenarioCfg = field(default_factory=ScenarioCfg)
    lateral: LateralCtrlCfg = field(default_factory=LateralCtrlCfg)
    acc: AccCfg = field(default_factory=AccCfg)
    sensors: SensorCfg = field(default_factory=SensorCfg)
    v2x: V2XCfg = field(default_factory=V2XCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)

    # How the policy action becomes the applied throttle/brake command:
    #   "raw"      - action IS the command in [-1, 1] (the original setup)
    #   "residual" - command = clip(acc_baseline + residual_scale * action, -1, 1)
    #   "acc"      - ignore the action, apply the ACC baseline (non-learned
    #                baseline for the ablation table)
    #   "oracle"   - ignore the action, apply ACC + a ground-truth yield rule
    #                (used only to generate the behaviour-cloning dataset)
    #   "bc"       - action IS the command; the policy was trained by imitation
    control_mode: str = "residual"
    # Bound on the residual correction in "residual" mode.
    residual_scale: float = 0.4

    # "vector" -> flat Box observation.
    # "vector_depth" -> Dict{"vec", "depth"} for a MultiInputPolicy.
    obs_mode: str = "vector"
    # Limit the change in the action per step (0 disables).  Models actuator
    # dynamics and discourages bang-bang control.
    action_rate_limit: float = 0.4
    seed: int = 42
    # Render the spectator camera and debug overlays (evaluation only).
    render: bool = False
    # Extra perception visualisations, independent of ``render`` (both slow).
    # Each opens its own OpenCV window: ``render_lidar`` a bird's-eye view of
    # the point cloud with the filtered returns and tracked cluster picked out,
    # ``render_depth`` the depth frame with the observation sectors and their
    # reported nearest-range values overlaid.
    render_lidar: bool = False
    render_depth: bool = False
    # Periodically saves a third-person chase-cam frame to ``live_view_path``,
    # overwriting it in place.  Runs inside the training process's own single
    # CARLA connection (unlike a second client grabbing a frame on demand,
    # which is not safe against a live session), so it is safe to leave on.
    live_view: bool = False
    live_view_every_steps: int = 20
    live_view_path: str = "live_view.png"
    verbose: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EnvCfg":
        return _from_dict(cls, data or {})

    @classmethod
    def load(cls, path: str) -> "EnvCfg":
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def merge(self, overrides: Dict[str, Any]) -> "EnvCfg":
        """Return a copy with ``overrides`` applied.

        Supports dotted keys, e.g. ``{"v2x.per_far": 1.0}``.
        """
        nested: Dict[str, Any] = {}
        for key, value in overrides.items():
            target = nested
            parts = key.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        merged = _deep_update(self.to_dict(), nested)
        return EnvCfg.from_dict(merged)


def _deep_update(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _from_dict(cls, data: Dict[str, Any]):
    """Rebuild a (possibly nested) dataclass from a plain dict.

    ``from __future__ import annotations`` turns every field type into a
    string, so the annotations have to be resolved before nested dataclasses
    can be recognised.
    """
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data)}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        hint = hints.get(name)
        if isinstance(hint, type) and is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value)
        elif isinstance(value, list):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


MANEUVERS: List[str] = ["left", "right", "straight"]
