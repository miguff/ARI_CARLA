"""Gymnasium environment: negotiating an intersection with a V2X cyclist.

The agent controls longitudinal motion only (one continuous action mapping to
throttle or brake); steering is handled by a pure-pursuit path follower.  Each
episode the ego takes a left, right or straight manoeuvre through a junction
while a cyclist, which broadcasts VAMs but never listens, may or may not be on
a conflicting path.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .. import geometry as geom
from ..carla_utils import (ActorRegistry, CarlaSession, actor_velocity_xy,
                           actor_xy, actor_yaw_deg, carla, has_line_of_sight,
                           set_spectator_behind)
from ..config import EnvCfg
from ..controllers import (ACCController, CyclistController,
                           OracleYieldController, PathFollower)
from ..reward import StepContext, compute_reward, yield_is_required
from ..scenario import EpisodeLayout, ScenarioBuilder
from ..sensors import (TRACK_FEATURE_DIM, TRACK_FEATURE_NAMES, CollisionSensor,
                       DepthCamera, GroundTruthPerception, LidarSensor,
                       ObstacleTrack, depth_sector_feature_names)
from ..v2x import (V2X_FEATURE_DIM, V2X_FEATURE_NAMES, V2XChannel, V2XDerived,
                   V2XReceiver, VAMGenerator)

LOGGER = logging.getLogger(__name__)

MANEUVER_ORDER = ["left", "right", "straight"]
EGO_FEATURE_NAMES = [
    "ego_speed",
    "ego_target_speed",
    "ego_speed_error",
    "ego_prev_throttle",
    "ego_prev_brake",
    "ego_accel",
    "ego_steer_cmd",
    "ego_dist_to_junction",
    "ego_dist_to_goal",
    "ego_acc_cmd",
    "ego_maneuver_left",
    "ego_maneuver_right",
    "ego_maneuver_straight",
]

CONTROL_MODES = ("raw", "residual", "acc", "oracle", "bc")

DIST_NORM_M = 60.0
ACCEL_NORM = 10.0
# Combined radius used for the ground-truth time-to-collision computation.
COLLISION_RADIUS_M = 2.5


class _LiveViewCamera:
    """Third-person chase-cam that periodically saves its latest frame to disk.

    A human (or an agent with no live vision) can then inspect a recent frame
    by just reading that file -- no second CARLA client involved, so this
    carries none of the concurrency risk of grabbing a frame from an outside
    process against a live session.  Lives inside the training process's own
    single connection, exactly like the depth/lidar sensors.
    """

    def __init__(self, world, attach_to, path: str) -> None:
        self.path = path
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", "960")
        blueprint.set_attribute("image_size_y", "540")
        blueprint.set_attribute("fov", "100")
        transform = carla.Transform(carla.Location(x=-7.0, z=3.0),
                                    carla.Rotation(pitch=-12.0))
        self.actor = world.spawn_actor(blueprint, transform, attach_to=attach_to)
        self._frame: Optional[np.ndarray] = None
        self.actor.listen(self._on_image)

    def _on_image(self, image) -> None:
        self._frame = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4))

    def save(self) -> None:
        if self._frame is not None:
            cv2.imwrite(self.path, self._frame[:, :, :3])  # BGRA -> BGR

    def destroy(self) -> None:
        try:
            if self.actor.is_listening:
                self.actor.stop()
            self.actor.destroy()
        except Exception:
            pass


class IntersectionV2XEnv(gym.Env):
    """See module docstring."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: Optional[EnvCfg] = None, **overrides) -> None:
        super().__init__()
        self.cfg = (cfg or EnvCfg())
        if overrides:
            self.cfg = self.cfg.merge(overrides)
        cfg = self.cfg

        if cfg.obs_mode not in ("vector", "vector_depth"):
            raise ValueError(f"unknown obs_mode {cfg.obs_mode!r}")
        if cfg.obs_mode == "vector_depth" and not cfg.sensors.depth.enabled:
            raise ValueError("obs_mode='vector_depth' requires sensors.depth.enabled")
        if cfg.control_mode not in CONTROL_MODES:
            raise ValueError(f"unknown control_mode {cfg.control_mode!r}; "
                             f"choose from {CONTROL_MODES}")

        self.session = CarlaSession(cfg.carla)
        self.world = self.session.world
        self.dt = cfg.carla.fixed_delta_seconds
        self.registry = ActorRegistry(self.world)
        self.scenario = ScenarioBuilder(self.session, cfg.scenario)
        self.follower = PathFollower(cfg.lateral)
        self.acc = ACCController(cfg.acc)
        self.oracle = OracleYieldController(cfg.acc)

        # Independent random streams keep the scenario reproducible even when
        # the channel noise configuration changes between evaluations.
        seeds = np.random.SeedSequence(cfg.seed).spawn(3)
        self._scenario_rng = np.random.default_rng(seeds[0])
        self._v2x_rng = np.random.default_rng(seeds[1])
        self._sensor_rng = np.random.default_rng(seeds[2])

        self.channel = V2XChannel(cfg.v2x, self._v2x_rng)
        self.receiver = V2XReceiver(cfg.v2x)
        self.vam_generator = VAMGenerator(cfg.v2x, self._v2x_rng)
        self.gt_perception = (
            GroundTruthPerception(cfg.sensors.groundtruth, self.world, self._sensor_rng)
            if cfg.sensors.groundtruth.enabled else None)

        self.vector_feature_names = self._build_feature_names()
        vector_dim = len(self.vector_feature_names)
        vector_space = spaces.Box(low=-1.0, high=1.0, shape=(vector_dim,),
                                  dtype=np.float32)
        if cfg.obs_mode == "vector":
            self.observation_space = vector_space
        else:
            h, w = cfg.sensors.depth.cnn_size
            self.observation_space = spaces.Dict({
                "vec": vector_space,
                "depth": spaces.Box(low=0, high=255, shape=(1, h, w), dtype=np.uint8),
            })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,),
                                      dtype=np.float32)

        # Per-episode actors and state, populated in reset().
        self.ego = None
        self.cyclist = None
        self.depth_camera: Optional[DepthCamera] = None
        self.lidar: Optional[LidarSensor] = None
        self.live_view_camera: Optional[_LiveViewCamera] = None
        self.collision: Optional[CollisionSensor] = None
        self.cyclist_controller: Optional[CyclistController] = None
        self.layout: Optional[EpisodeLayout] = None
        self._closed = False
        self._episode = 0
        self._curriculum_stage = 0
        # Set False the first time an OpenCV window fails to open (e.g.
        # headless), so it is not retried every step.
        self._depth_window_ok = True
        self._lidar_window_ok = True
        self._depth_window_name = "v2x_rl depth  (per-sector nearest range, m)"
        self._lidar_window_name = "v2x_rl lidar  (BEV, up = forward)"

    # ------------------------------------------------------------------ #
    #  Spaces
    # ------------------------------------------------------------------ #
    def _build_feature_names(self) -> List[str]:
        names = list(EGO_FEATURE_NAMES)
        if self.cfg.sensors.depth.enabled:
            names += depth_sector_feature_names(self.cfg.sensors.depth.n_sectors)
        if self.cfg.sensors.lidar.enabled or self.cfg.sensors.groundtruth.enabled:
            names += list(TRACK_FEATURE_NAMES)
        names += list(V2X_FEATURE_NAMES)
        return names

    # ------------------------------------------------------------------ #
    #  Curriculum (live difficulty updates from a training callback)
    # ------------------------------------------------------------------ #
    def apply_curriculum_stage(self, stage_index: int, conflict_episode_prob: float,
                               nonconflicting_cyclist_prob: float) -> None:
        """Update the scenario difficulty; takes effect from the next reset().

        ``ScenarioBuilder`` reads these probabilities fresh each episode, and it
        shares this exact config object, so mutating it here is enough.
        """
        self._curriculum_stage = int(stage_index)
        self.cfg.scenario.conflict_episode_prob = float(conflict_episode_prob)
        self.cfg.scenario.nonconflicting_cyclist_prob = float(
            nonconflicting_cyclist_prob)

    # ------------------------------------------------------------------ #
    #  Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None,
              ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            seeds = np.random.SeedSequence(seed).spawn(3)
            self._scenario_rng = np.random.default_rng(seeds[0])
            self._v2x_rng = np.random.default_rng(seeds[1])
            self._sensor_rng = np.random.default_rng(seeds[2])
            self.channel.rng = self._v2x_rng
            self.vam_generator.rng = self._v2x_rng
            if self.gt_perception is not None:
                self.gt_perception.rng = self._sensor_rng

        self._destroy_actors()
        cfg = self.cfg

        for attempt in range(6):
            self.layout = self.scenario.sample(self._scenario_rng)
            self.ego = self._spawn_ego(self.layout)
            if self.ego is not None:
                break
            LOGGER.debug("Ego spawn failed (attempt %d), resampling", attempt + 1)
        if self.ego is None:
            raise RuntimeError("Could not spawn the ego vehicle after 6 attempts")

        self.cyclist = None
        self.cyclist_controller = None
        if self.layout.cyclist_present:
            self.cyclist = self._spawn_cyclist(self.layout)
            if self.cyclist is None:
                self.layout.cyclist_present = False
                self.layout.conflict_xy = None
            else:
                self.cyclist_controller = CyclistController(
                    self.cyclist, self.layout.cyclist_path,
                    self.layout.cyclist_speed_ms, self.layout.cyclist_start_s,
                    self.dt,
                    hold_until_release=self.layout.has_conflict,
                    meet_horizon_s=cfg.scenario.cyclist_meet_horizon_s,
                    release_min_speed_ms=cfg.scenario.cyclist_release_min_speed_ms,
                    max_hold_s=cfg.scenario.cyclist_max_hold_s)

        self._attach_sensors()

        # --- reset per-episode state ---
        self.follower.reset()
        self.acc.reset()
        self.oracle.reset()
        self.channel.reset()
        self.receiver.reset()
        self.vam_generator.reset()
        if self.gt_perception is not None:
            self.gt_perception.reset()
        if self.lidar is not None:
            self.lidar.reset_track()

        self._step_count = 0
        self._episode += 1
        self._prev_action = 0.0
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._prev_speed_ms = 0.0
        self._accel = 0.0
        self._steer_cmd = 0.0
        self._acc_cmd = 0.0
        self._last_residual = 0.0
        self._desired_command = 0.0
        self._stuck_steps = 0
        self._prev_ego_s = 0.0
        self._track = ObstacleTrack()
        self._derived = V2XDerived()
        self._metrics = _EpisodeMetrics()

        # Let physics settle, then give the ego a rolling start.
        for _ in range(int(0.4 / self.dt)):
            self.world.tick()
        self._apply_initial_speed()

        frame = self.world.tick()
        self._sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
        self._await_sensors(frame)
        self._update_perception()

        lead_gap, lead_closing = self._acc_lead()
        self._acc_cmd = self.acc.command(
            self._speed_ms(self.ego),
            self.cfg.scenario.target_speed_kmh / 3.6, lead_gap, lead_closing)

        return self._observation(), self._info(terminal=False)

    # ------------------------------------------------------------------ #
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        cfg = self.cfg
        raw_action = float(np.clip(
            np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))

        ego_xy = actor_xy(self.ego)
        ego_yaw = actor_yaw_deg(self.ego)
        speed_ms = self._speed_ms(self.ego)
        ego_s, _, _ = self.layout.ego_path.project(ego_xy)
        self._steer_cmd = self.follower.steer(
            self.layout.ego_path, ego_xy, ego_yaw, speed_ms, ego_s)

        # --- longitudinal command: baseline + (optionally) the policy ----- #
        target_ms = cfg.scenario.target_speed_kmh / 3.6
        lead_gap, lead_closing = self._acc_lead()
        self._acc_cmd = self.acc.command(speed_ms, target_ms, lead_gap, lead_closing)

        residual = 0.0
        if cfg.control_mode in ("raw", "bc"):
            desired = raw_action
        elif cfg.control_mode == "residual":
            residual = raw_action
            desired = float(np.clip(
                self._acc_cmd + cfg.residual_scale * raw_action, -1.0, 1.0))
        elif cfg.control_mode == "acc":
            desired = self._acc_cmd
        else:  # "oracle" - ACC + ground-truth yield, for the BC dataset only
            yield_now, dist_c = self._oracle_yield_state(ego_s)
            desired = self.oracle.command(
                speed_ms, target_ms, yield_now=yield_now, dist_to_conflict_m=dist_c,
                lead_gap_m=lead_gap, lead_closing_ms=lead_closing)
        self._last_residual = residual
        self._desired_command = desired

        if cfg.action_rate_limit > 0.0:
            delta = np.clip(desired - self._prev_action,
                            -cfg.action_rate_limit, cfg.action_rate_limit)
            command = float(self._prev_action + delta)
        else:
            command = desired
        throttle = max(0.0, command)
        brake = max(0.0, -command)

        self.ego.apply_control(carla.VehicleControl(
            throttle=throttle, brake=brake, steer=self._steer_cmd))
        if self.cyclist_controller is not None:
            if (self.layout.has_conflict
                    and self.layout.ego_conflict_s is not None):
                ego_gap = self.layout.ego_conflict_s - ego_s
                self.cyclist_controller.maybe_release(
                    ego_gap / max(speed_ms, 0.5), ego_gap_m=ego_gap)
            self.cyclist_controller.step()

        self._transmit_vam()

        frame = self.world.tick()
        self._sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
        self._step_count += 1
        self._await_sensors(frame)

        self.receiver.update(self.channel.poll(self._sim_time))
        self._update_perception()

        new_speed = self._speed_ms(self.ego)
        self._accel = (new_speed - self._prev_speed_ms) / self.dt
        self._prev_speed_ms = new_speed

        ctx, terminated, truncated = self._build_context(
            command, throttle, brake, residual)
        breakdown = compute_reward(ctx, cfg.reward)

        if (ctx.collision and self.collision is not None
                and self.collision.hit_cyclist):
            self._metrics.collision_with_cyclist = True

        self._metrics.update(ctx, breakdown, self._derived, abs(command - self._prev_action))
        self._prev_action = command
        self._prev_throttle = throttle
        self._prev_brake = brake
        self._prev_ego_s = self._cur_ego_s

        if cfg.render:
            self._render_debug(throttle, brake, ctx)
        if cfg.render_lidar and self.lidar is not None:
            self._render_lidar_window()
        if cfg.render_depth and self.depth_camera is not None:
            self._render_depth_window()
        if (self.live_view_camera is not None
                and self._step_count % max(1, cfg.live_view_every_steps) == 0):
            self.live_view_camera.save()

        observation = self._observation()
        info = self._info(terminal=terminated or truncated)
        return observation, breakdown.total, terminated, truncated, info

    # ------------------------------------------------------------------ #
    #  Context / termination
    # ------------------------------------------------------------------ #
    def _build_context(self, command: float, throttle: float, brake: float,
                       residual: float = 0.0) -> Tuple[StepContext, bool, bool]:
        cfg = self.cfg
        layout = self.layout
        path = layout.ego_path
        ego_xy = actor_xy(self.ego)
        speed_ms = self._speed_ms(self.ego)
        ego_s, lateral, _ = path.project(ego_xy)
        self._cur_ego_s = ego_s

        ctx = StepContext(
            speed_ms=speed_ms,
            target_speed_ms=cfg.scenario.target_speed_kmh / 3.6,
            action=command,
            prev_action=self._prev_action,
            throttle=throttle,
            brake=brake,
            progress_m=max(0.0, ego_s - self._prev_ego_s),
            obstacle_perceived=self._obstacle_perceived(),
            slow_steps=self._stuck_steps,
            residual=residual,
        )

        if layout.has_conflict and self.cyclist_controller is not None:
            radius = cfg.scenario.conflict_zone_radius_m
            cyclist_s = self.cyclist_controller.s
            ego_gap = layout.ego_conflict_s - ego_s
            cyclist_gap = layout.cyclist_conflict_s - cyclist_s
            ctx.has_conflict = True
            ctx.ego_dist_to_conflict_m = ego_gap
            ctx.cyclist_dist_to_conflict_m = cyclist_gap
            ctx.cyclist_speed_ms = self.cyclist_controller.speed_ms
            ctx.ego_in_conflict_zone = abs(ego_gap) < radius
            ctx.cyclist_in_conflict_zone = abs(cyclist_gap) < radius
            ctx.cyclist_cleared = cyclist_gap < -radius

        if self.cyclist is not None:
            cyclist_xy = actor_xy(self.cyclist)
            ctx.cyclist_distance_m = float(np.linalg.norm(cyclist_xy - ego_xy))
            ctx.ttc_s = geom.time_to_collision(
                cyclist_xy - ego_xy,
                actor_velocity_xy(self.cyclist) - actor_velocity_xy(self.ego),
                COLLISION_RADIUS_M)

        terminated = False
        truncated = False

        if self.collision is not None and self.collision.happened:
            ctx.collision = True
            terminated = True
        elif ego_s >= path.length - cfg.scenario.goal_tolerance_m:
            ctx.goal_reached = True
            terminated = True
        elif abs(lateral) > 6.0:
            # The lateral controller lost the path: treat as a failed episode
            # rather than letting the ego wander off the map.
            ctx.timeout = True
            truncated = True
            self._metrics.off_route = True
        elif self._step_count >= cfg.scenario.max_episode_steps:
            ctx.timeout = True
            truncated = True

        if speed_ms * 3.6 < 1.0:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        # A correct, patient yield for an approaching cyclist must not be scored
        # as "stuck"; let those episodes run to the step limit instead.
        if (not (terminated or truncated)
                and self._stuck_steps > int(20.0 / self.dt)
                and not yield_is_required(ctx)):
            ctx.timeout = True
            truncated = True
            self._metrics.stuck = True

        return ctx, terminated, truncated

    def _obstacle_perceived(self) -> bool:
        """Whether any onboard source currently reports something in the way.

        Only the **center** depth sectors are checked — the ones covering the
        path ahead.  The full-FOV sectors catch roadside buildings and walls
        that are close but not in the car's way, which would falsely suppress
        the anti-overcautious penalties.

        A genuine, *imminent* V2X-reported interception counts too.  Without
        this, a control mode with no lidar/depth (e.g. V2X-only) can never
        satisfy this check, so braking in direct response to what V2X reported
        would always be scored as "unnecessary" whenever the ground-truth
        scenario label happens to say there is no conflict.  ``interception``
        alone is a purely geometric fact (the reported path crosses the ego's
        *somewhere*), true regardless of who would actually arrive first --
        gating on it alone made braking free for the whole episode as soon as
        any crossing existed, which is most conflict episodes now that they
        are 85% of the mix, and the policy degenerated to "stop and wait out
        every V2X interception" instead of only the ones that matter.  The
        same imminence horizon the ground-truth yield logic uses keeps this
        consistent with what "the cyclist actually matters right now" means
        elsewhere in the reward.
        """
        if (self._derived.valid and self._derived.interception
                and self._derived.cyclist_tta_s <= self.cfg.reward.yield_imminent_tta_s):
            return True
        if (self._track.valid and self._track.range_m < 25.0
                and abs(self._track.bearing_deg) < 60.0):
            return True
        if self.depth_camera is not None:
            ranges = self.depth_camera.sector_ranges()
            alert = self.cfg.sensors.depth.obstacle_alert_range_m
            n = ranges.size
            # Use only the center quarter of sectors (the direct path ahead,
            # roughly +/- one sector width around the heading).
            quarter = max(1, n // 4)
            start = n // 2 - quarter // 2
            center = ranges[start:start + quarter]
            if center.size and float(center.min()) < alert:
                return True
        return False

    # ------------------------------------------------------------------ #
    #  ACC baseline support
    # ------------------------------------------------------------------ #
    def _acc_lead(self) -> Tuple[Optional[float], float]:
        """Nearest credible obstacle ahead for the ACC baseline.

        Returns ``(range_m, closing_speed_ms)`` from onboard sensing only.  The
        lidar/GT track (cyclist-shaped, dynamic-filtered) is the real lead; the
        centre depth sectors contribute only as a very-short-range emergency
        stop for something dead ahead.  Further-out depth returns are junction
        geometry, not a lead vehicle, and must not slow the ACC.
        ``(None, 0.0)`` when the road looks clear.
        """
        candidates = []
        if (self._track.valid
                and abs(self._track.bearing_deg) < self.cfg.acc.lead_bearing_deg):
            # range_rate_ms is negative when closing.
            candidates.append((float(self._track.range_m),
                               float(max(0.0, -self._track.range_rate_ms))))
        if self.depth_camera is not None:
            ranges = self.depth_camera.sector_ranges()
            n = ranges.size
            quarter = max(1, n // 4)
            start = n // 2 - quarter // 2
            centre = ranges[start:start + quarter]
            if centre.size:
                r = float(centre.min())
                if r < self.cfg.acc.depth_emergency_range_m:
                    candidates.append((r, float(self._speed_ms(self.ego))))
        if not candidates:
            return None, 0.0
        return min(candidates, key=lambda c: c[0])

    def _oracle_yield_state(self, ego_s: float) -> Tuple[bool, float]:
        """Ground-truth ``(yield_now, ego_dist_to_conflict_m)`` for the BC oracle."""
        layout = self.layout
        if not (layout.has_conflict and self.cyclist_controller is not None):
            return False, 0.0
        ego_gap = layout.ego_conflict_s - ego_s
        cyclist_gap = layout.cyclist_conflict_s - self.cyclist_controller.s
        radius = self.cfg.scenario.conflict_zone_radius_m
        probe = StepContext(
            speed_ms=self._speed_ms(self.ego),
            target_speed_ms=self.cfg.scenario.target_speed_kmh / 3.6,
            action=0.0, prev_action=0.0, throttle=0.0, brake=0.0,
            has_conflict=True,
            ego_dist_to_conflict_m=ego_gap,
            cyclist_dist_to_conflict_m=cyclist_gap,
            cyclist_speed_ms=self.cyclist_controller.speed_ms,
            cyclist_cleared=cyclist_gap < -radius,
        )
        return yield_is_required(probe), max(0.0, ego_gap)

    # ------------------------------------------------------------------ #
    #  Perception & V2X
    # ------------------------------------------------------------------ #
    def _update_perception(self) -> None:
        if self.gt_perception is not None:
            self._track = self.gt_perception.track(self.ego, self.cyclist, self.dt)
        elif self.lidar is not None:
            self._track = self.lidar.track(
                self.dt, actor_xy(self.ego), actor_yaw_deg(self.ego))
        else:
            self._track = ObstacleTrack()

        ego_xy = actor_xy(self.ego)
        self._derived = self.receiver.derive(
            self._sim_time, ego_xy, actor_yaw_deg(self.ego),
            self._speed_ms(self.ego), self._path_ahead(ego_xy))

    def _path_ahead(self, ego_xy: np.ndarray, horizon_m: float = 70.0) -> np.ndarray:
        """The remaining ego route, starting at the current position."""
        path = self.layout.ego_path
        s, _, _ = path.project(ego_xy)
        mask = (path.cum >= s) & (path.cum <= s + horizon_m)
        points = path.points[mask]
        return np.vstack([ego_xy[None, :], points]) if len(points) else ego_xy[None, :]

    def _transmit_vam(self) -> None:
        """Generate the cyclist's VAM and offer it to the channel."""
        if not self.cfg.v2x.enabled or self.cyclist_controller is None:
            return
        cyclist = self.cyclist
        true_xy = actor_xy(cyclist)
        message = self.vam_generator.generate(
            now_s=self._sim_time,
            position=true_xy,
            speed_ms=self.cyclist_controller.speed_ms,
            heading_deg=actor_yaw_deg(cyclist),
            true_path=self.cyclist_controller.predicted_path(
                self.cfg.v2x.path_prediction_points * self.cfg.v2x.path_prediction_dt_s,
                self.cfg.v2x.path_prediction_points),
        )
        if message is None:
            return
        ego_location = self.ego.get_transform().location
        cyclist_location = cyclist.get_transform().location
        distance = float(ego_location.distance(cyclist_location))
        los = True
        if self.cfg.v2x.check_line_of_sight:
            los = has_line_of_sight(self.world, cyclist_location, ego_location)
        self._metrics.note_transmission(los)
        self.channel.transmit(message, self._sim_time, distance, los)

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #
    def _observation(self):
        vector = self._vector_observation()
        if self.cfg.obs_mode == "vector":
            return vector
        depth = (self.depth_camera.cnn_image() if self.depth_camera is not None
                 else np.zeros((1, *self.cfg.sensors.depth.cnn_size), dtype=np.uint8))
        return {"vec": vector, "depth": depth}

    def _vector_observation(self) -> np.ndarray:
        cfg = self.cfg
        path = self.layout.ego_path
        ego_xy = actor_xy(self.ego)
        speed_ms = self._speed_ms(self.ego)
        target_ms = cfg.scenario.target_speed_kmh / 3.6
        max_ms = cfg.scenario.max_speed_kmh / 3.6
        s, _, _ = path.project(ego_xy)

        maneuver = [1.0 if self.layout.maneuver == m else 0.0 for m in MANEUVER_ORDER]
        ego = np.array([
            speed_ms / max_ms,
            target_ms / max_ms,
            (target_ms - speed_ms) / max_ms,
            self._prev_throttle,
            self._prev_brake,
            self._accel / ACCEL_NORM,
            self._steer_cmd,
            (path.junction_s - s) / DIST_NORM_M,
            (path.length - s) / (2.0 * DIST_NORM_M),
            self._acc_cmd,
            *maneuver,
        ], dtype=np.float32)

        parts = [ego]
        if cfg.sensors.depth.enabled and self.depth_camera is not None:
            parts.append(self.depth_camera.sector_features())
        elif cfg.sensors.depth.enabled:
            parts.append(np.ones(cfg.sensors.depth.n_sectors, dtype=np.float32))
        if cfg.sensors.lidar.enabled or cfg.sensors.groundtruth.enabled:
            parts.append(self._track.features())
        parts.append(self.receiver.features(self._derived))

        observation = np.concatenate(parts).astype(np.float32)
        return np.clip(observation, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Info / metrics
    # ------------------------------------------------------------------ #
    def _info(self, terminal: bool) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "maneuver": self.layout.maneuver,
            "junction_id": self.layout.junction_id,
            "cyclist_present": bool(self.layout.cyclist_present),
            "cyclist_origin": self.layout.cyclist_origin,
            "has_conflict": bool(self.layout.has_conflict),
            "v2x_valid": bool(self._derived.valid),
            "control_mode": self.cfg.control_mode,
            "applied_command": float(self._desired_command),
            "curriculum_stage": int(self._curriculum_stage),
        }
        if terminal:
            # The episode summary must not shadow the descriptive fields, so
            # it is merged first and the identifiers rewritten afterwards.
            summary = self._metrics.summary(self._step_count, self.channel.stats)
            summary.update(info)
            return summary
        return info

    # ------------------------------------------------------------------ #
    #  Spawning / sensors
    # ------------------------------------------------------------------ #
    def _spawn_ego(self, layout: EpisodeLayout):
        blueprint = self.world.get_blueprint_library().filter("vehicle.audi.a2")
        if not blueprint:
            blueprint = self.world.get_blueprint_library().filter("vehicle.*")
        bp = blueprint[0]
        if bp.has_attribute("color"):
            bp.set_attribute("color", "20,80,200")
        actor = self.world.try_spawn_actor(bp, layout.ego_spawn)
        return self.registry.add(actor)

    def _spawn_cyclist(self, layout: EpisodeLayout):
        library = self.world.get_blueprint_library()
        candidates = (library.filter("*crossbike*") or library.filter("*bike*")
                      or library.filter("vehicle.bh.*"))
        if not candidates:
            LOGGER.warning("No bicycle blueprint available on this map")
            return None
        actor = self.world.try_spawn_actor(candidates[0], layout.cyclist_spawn)
        if actor is None:
            return None
        actor.set_simulate_physics(False)
        return self.registry.add(actor)

    def _attach_sensors(self) -> None:
        cfg = self.cfg
        self.collision = CollisionSensor(self.world, self.ego)
        self.registry.add(self.collision.actor, is_sensor=True)

        self.depth_camera = None
        if cfg.sensors.depth.enabled:
            self.depth_camera = DepthCamera(self.world, cfg.sensors.depth, self.ego)
            self.registry.add(self.depth_camera.actor, is_sensor=True)

        self.lidar = None
        if cfg.sensors.lidar.enabled:
            self.lidar = LidarSensor(self.world, cfg.sensors.lidar, self.ego, self.dt)
            self.registry.add(self.lidar.actor, is_sensor=True)

        self.live_view_camera = None
        if cfg.live_view:
            self.live_view_camera = _LiveViewCamera(
                self.world, self.ego, cfg.live_view_path)
            self.registry.add(self.live_view_camera.actor, is_sensor=True)

    def _await_sensors(self, frame: int) -> None:
        for sensor in (self.depth_camera, self.lidar):
            if sensor is not None and not sensor.sync.wait(frame):
                LOGGER.debug("Sensor %s missed frame %d",
                             type(sensor).__name__, frame)

    def _apply_initial_speed(self) -> None:
        low, high = self.cfg.scenario.initial_speed_kmh
        speed_ms = float(self._scenario_rng.uniform(low, high)) / 3.6
        yaw = math.radians(actor_yaw_deg(self.ego))
        self.ego.set_target_velocity(carla.Vector3D(
            x=speed_ms * math.cos(yaw), y=speed_ms * math.sin(yaw), z=0.0))
        self._prev_speed_ms = speed_ms

    @staticmethod
    def _speed_ms(actor) -> float:
        v = actor.get_velocity()
        return float(math.sqrt(v.x * v.x + v.y * v.y))

    # ------------------------------------------------------------------ #
    #  Rendering / teardown
    # ------------------------------------------------------------------ #
    def _render_debug(self, throttle: float, brake: float, ctx: StepContext) -> None:
        debug = self.world.debug
        set_spectator_behind(self.session.spectator, self.ego)
        layout = self.layout

        def draw_path(path, colour):
            for i in range(0, len(path.points) - 1, 2):
                a = carla.Location(float(path.points[i][0]), float(path.points[i][1]),
                                   float(path.z[i]) + 0.3)
                b = carla.Location(float(path.points[i + 1][0]),
                                   float(path.points[i + 1][1]),
                                   float(path.z[i + 1]) + 0.3)
                debug.draw_line(a, b, thickness=0.08, color=colour, life_time=0.12)

        draw_path(layout.ego_path, carla.Color(0, 180, 255))
        if layout.cyclist_path is not None:
            draw_path(layout.cyclist_path, carla.Color(255, 160, 0))
        if layout.conflict_xy is not None:
            # Draw where each path actually reaches the conflict, not the raw
            # ``conflict_xy`` -- for a closest-approach match (e.g. a
            # right-hook, where the paths never literally cross) that point is
            # the midpoint between the two paths and can visibly float off
            # both of them, which reads as "the marker is in the wrong place"
            # even though the underlying arc-length distances are correct.
            z = float(self.ego.get_transform().location.z) + 0.5
            ego_pt = layout.ego_path.pose_at(layout.ego_conflict_s)[0]
            debug.draw_point(carla.Location(float(ego_pt[0]), float(ego_pt[1]), z),
                             size=0.25, color=carla.Color(255, 0, 0), life_time=0.12)
            if layout.cyclist_path is not None and layout.cyclist_conflict_s is not None:
                cyc_pt = layout.cyclist_path.pose_at(layout.cyclist_conflict_s)[0]
                debug.draw_point(carla.Location(float(cyc_pt[0]), float(cyc_pt[1]), z),
                                 size=0.25, color=carla.Color(255, 140, 0), life_time=0.12)
                debug.draw_line(carla.Location(float(ego_pt[0]), float(ego_pt[1]), z),
                                carla.Location(float(cyc_pt[0]), float(cyc_pt[1]), z),
                                thickness=0.04, color=carla.Color(255, 0, 0), life_time=0.12)

        text = (f"{layout.maneuver} T{throttle:.2f} B{brake:.2f} "
                f"{ctx.speed_ms * 3.6:.0f}km/h "
                f"V2X{'+' if self._derived.valid else '-'} "
                f"gap{self._derived.arrival_gap_s:+.1f}s")
        debug.draw_string(self.ego.get_transform().location + carla.Location(z=2.6),
                          text, draw_shadow=False,
                          color=carla.Color(255, 255, 0), life_time=0.12)

        if self.cyclist is not None:
            # Always mark the cyclist's actual current position, whether held
            # or moving and regardless of how visible the bike model itself is
            # from the spectator's angle -- otherwise "why can't I see it" is
            # only answerable by guessing.
            cyc_loc = self.cyclist.get_transform().location
            held = (self.cyclist_controller is not None
                    and not self.cyclist_controller.released)
            speed = self.cyclist_controller.speed_ms if self.cyclist_controller else 0.0
            colour = carla.Color(255, 220, 0) if held else carla.Color(0, 255, 0)
            debug.draw_point(cyc_loc + carla.Location(z=1.2), size=0.2,
                             color=colour, life_time=0.12)
            debug.draw_string(cyc_loc + carla.Location(z=2.2),
                              f"cyclist {'HELD' if held else 'moving'} {speed:.1f}m/s",
                              draw_shadow=False, color=colour, life_time=0.12)

    def _render_lidar_window(self) -> None:
        """Bird's-eye view of the lidar cloud in its own window.

        Grey = every return; coloured = the points that survive the ROI /
        ground / overhead filter, i.e. exactly what the clusterer sees, tinted
        by height.  The red circle is the tracked obstacle.  Up is forward.
        """
        if not self._lidar_window_ok:
            return
        try:
            import cv2

            lidar_cfg = self.cfg.sensors.lidar
            size = 520
            view_m = float(lidar_cfg.roi_radius_m)
            ppm = size / (2.0 * view_m)
            cx = cy = size // 2
            frame = np.zeros((size, size, 3), dtype=np.uint8)

            for ring_m in range(10, int(view_m) + 1, 10):
                cv2.circle(frame, (cx, cy), int(ring_m * ppm), (38, 38, 38), 1)
            cv2.line(frame, (cx, 0), (cx, size), (38, 38, 38), 1)
            cv2.line(frame, (0, cy), (size, cy), (38, 38, 38), 1)

            def to_px(x_fwd, y_right):
                return int(cx + y_right * ppm), int(cy - x_fwd * ppm)

            # Every return, dim — vectorised, it can be tens of thousands.
            raw = self.lidar.points
            if len(raw) > 6000:
                raw = raw[self._sensor_rng.choice(len(raw), 6000, replace=False)]
            if len(raw):
                uu = (cx + raw[:, 1] * ppm).astype(np.int32)
                vv = (cy - raw[:, 0] * ppm).astype(np.int32)
                keep = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
                frame[vv[keep], uu[keep]] = (85, 85, 85)

            # Obstacle-candidate returns, bright, tinted by height.
            z_lo, z_hi = lidar_cfg.ground_z_threshold_m, lidar_cfg.max_z_m
            for px, py, pz in self.lidar.filtered_points():
                u, v = to_px(px, py)
                if not (0 <= u < size and 0 <= v < size):
                    continue
                t = float(np.clip((pz - z_lo) / max(1e-3, z_hi - z_lo), 0.0, 1.0))
                cv2.circle(frame, (u, v), 2,
                           (int(255 * (1.0 - t)), 190, int(255 * t)), -1)

            cv2.drawMarker(frame, (cx, cy), (0, 255, 255),
                           cv2.MARKER_TRIANGLE_UP, 14, 2)

            if self._track.valid:
                bearing = math.radians(self._track.bearing_deg)
                u, v = to_px(self._track.range_m * math.cos(bearing),
                             self._track.range_m * math.sin(bearing))
                cv2.line(frame, (cx, cy), (u, v), (0, 0, 235), 1)
                cv2.circle(frame, (u, v), 9, (0, 0, 235), 2)
                cv2.putText(frame,
                            f"{self._track.range_m:.1f} m  "
                            f"{self._track.range_rate_ms:+.1f} m/s",
                            (min(u + 12, size - 150), max(v, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 235), 1,
                            cv2.LINE_AA)

            cv2.putText(frame, f"+/-{view_m:.0f} m   rings 10 m   up = forward",
                        (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (190, 190, 190), 1, cv2.LINE_AA)
            cv2.imshow(self._lidar_window_name, frame)
            cv2.waitKey(1)
        except Exception as exc:  # pragma: no cover - display not always present
            LOGGER.warning("Disabling lidar window (%s)", exc)
            self._lidar_window_ok = False

    def _render_depth_window(self) -> None:
        """Show the depth frame with the observation sectors overlaid."""
        if not self._depth_window_ok:
            return
        try:
            import cv2

            cfg = self.cfg.sensors.depth
            depth = np.clip(self.depth_camera.depth_m, 0.0, cfg.max_range_m)
            gray = (255.0 * (1.0 - depth / cfg.max_range_m)).astype(np.uint8)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            h, w = frame.shape[:2]

            top, bot = int(cfg.v_band[0] * h), int(cfg.v_band[1] * h)
            cv2.rectangle(frame, (0, top), (w - 1, bot), (255, 255, 255), 1)

            ranges = self.depth_camera.sector_ranges()
            n = len(ranges)
            for i, rng in enumerate(ranges):
                x0, x1 = int(w * i / n), int(w * (i + 1) / n)
                cv2.line(frame, (x1, 0), (x1, h), (50, 50, 50), 1)
                cv2.putText(frame, f"{rng:.0f}", ((x0 + x1) // 2 - 12, bot - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)

            cv2.imshow(self._depth_window_name, frame)
            cv2.waitKey(1)
        except Exception as exc:  # pragma: no cover - display not always present
            LOGGER.warning("Disabling depth window (%s)", exc)
            self._depth_window_ok = False

    def _destroy_actors(self) -> None:
        for sensor in (self.depth_camera, self.lidar, self.live_view_camera,
                      self.collision):
            if sensor is not None:
                sensor.destroy()
        self.depth_camera = None
        self.lidar = None
        self.live_view_camera = None
        self.collision = None
        self.registry.destroy_all()
        self.ego = None
        self.cyclist = None
        self.cyclist_controller = None
        try:
            self.world.tick()
        except RuntimeError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._destroy_actors()
        finally:
            self.session.restore()
        for enabled, name in ((self.cfg.render_depth, self._depth_window_name),
                              (self.cfg.render_lidar, self._lidar_window_name)):
            if not enabled:
                continue
            try:
                import cv2
                cv2.destroyWindow(name)
            except Exception:
                pass

    def __del__(self):  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


class _EpisodeMetrics:
    """Accumulates the per-episode diagnostics reported through ``info``."""

    def __init__(self) -> None:
        self.speed_sum = 0.0
        self.speed_error_sum = 0.0
        self.jerk_sum = 0.0
        self.residual_sum = 0.0
        self.steps = 0
        self.yield_steps = 0
        self.yield_violation_steps = 0
        self.unnecessary_brake_steps = 0
        self.min_ttc = float("inf")
        self.min_cyclist_distance = float("inf")
        self.had_conflict = False
        self.collision = False
        self.collision_with_cyclist = False
        self.success = False
        self.timeout = False
        self.off_route = False
        self.stuck = False
        self.vam_transmissions = 0
        self.vam_nlos = 0
        self.v2x_valid_steps = 0

    def note_transmission(self, line_of_sight: bool) -> None:
        self.vam_transmissions += 1
        if not line_of_sight:
            self.vam_nlos += 1

    def update(self, ctx: StepContext, breakdown, derived, jerk: float) -> None:
        self.steps += 1
        self.speed_sum += ctx.speed_ms
        self.speed_error_sum += abs(ctx.speed_ms - ctx.target_speed_ms)
        self.jerk_sum += jerk
        self.residual_sum += abs(ctx.residual)
        if breakdown.yield_required:
            self.yield_steps += 1
        if breakdown.yield_violated:
            self.yield_violation_steps += 1
        if "unnecessary_brake" in breakdown.components:
            self.unnecessary_brake_steps += 1
        self.min_ttc = min(self.min_ttc, ctx.ttc_s)
        self.min_cyclist_distance = min(self.min_cyclist_distance,
                                        ctx.cyclist_distance_m)
        self.had_conflict = self.had_conflict or ctx.has_conflict
        if derived.valid:
            self.v2x_valid_steps += 1
        self.collision = self.collision or ctx.collision
        self.success = self.success or ctx.goal_reached
        self.timeout = self.timeout or ctx.timeout

    def summary(self, steps: int, channel_stats) -> Dict[str, Any]:
        n = max(1, self.steps)
        return {
            "ep_steps": steps,
            "success": bool(self.success),
            "collision": bool(self.collision),
            "collision_with_cyclist": bool(self.collision_with_cyclist),
            "timeout": bool(self.timeout),
            "off_route": bool(self.off_route),
            "stuck": bool(self.stuck),
            "mean_speed_kmh": 3.6 * self.speed_sum / n,
            "mean_speed_error_kmh": 3.6 * self.speed_error_sum / n,
            "mean_jerk": self.jerk_sum / n,
            "mean_residual": self.residual_sum / n,
            "yield_steps": self.yield_steps,
            "yield_violation_steps": self.yield_violation_steps,
            "yield_ok": bool(self.yield_steps > 0 and self.yield_violation_steps == 0),
            "unnecessary_brake_frac": self.unnecessary_brake_steps / n,
            "min_ttc": (float(self.min_ttc) if math.isfinite(self.min_ttc) else 30.0),
            "min_cyclist_distance": (float(self.min_cyclist_distance)
                                     if math.isfinite(self.min_cyclist_distance) else 100.0),
            # Conflict-only versions: ``None`` when this episode had no genuine
            # conflict, so the callback averages them over conflict episodes
            # only instead of diluting with the 30 s / 100 m no-cyclist caps.
            "min_ttc_conflict": (float(self.min_ttc)
                                 if self.had_conflict and math.isfinite(self.min_ttc)
                                 else None),
            "min_cyclist_distance_conflict": (
                float(self.min_cyclist_distance)
                if self.had_conflict and math.isfinite(self.min_cyclist_distance)
                else None),
            "v2x_valid_frac": self.v2x_valid_steps / n,
            "v2x_sent": channel_stats.sent,
            "v2x_delivered": channel_stats.delivered,
            "v2x_loss_rate": channel_stats.loss_rate,
            "v2x_nlos_frac": (self.vam_nlos / self.vam_transmissions
                              if self.vam_transmissions else 0.0),
        }
