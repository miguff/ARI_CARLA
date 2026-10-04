import carla
import random
import time
import sys
import os

# Try to locate the CARLA route planner.  Fall back to common install paths
# or the CARLA_ROOT environment variable if it is not on PYTHONPATH.
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ModuleNotFoundError:
    carla_root = os.environ.get('CARLA_ROOT', '')
    candidates = [
        carla_root,
        os.path.join(carla_root, 'PythonAPI', 'carla'),
        '/opt/carla/PythonAPI/carla',
        os.path.expanduser('~/CARLA_0.9.15/PythonAPI/carla'),
    ]
    found = False
    for p in candidates:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
            if os.path.isdir(os.path.join(p, 'agents')):
                found = True
    if not found:
        raise RuntimeError(
            "Could not find CARLA's 'agents' package. "
            "Set CARLA_ROOT or add the CARLA PythonAPI to PYTHONPATH."
        )
    from agents.navigation.global_route_planner import GlobalRoutePlanner

import numpy as np
import math
from ultralytics import YOLO
import cv2
import torch
from typing import Optional
import torch.nn.functional as F
import torch as T


class EnvironmentClass:
    """CARLA environment for longitudinal RL control with a cyclist ahead.

    Design:
      - The RL agent controls throttle/brake for the *entire* episode.
      - Steering is handled by a waypoint-following controller (not learned).
      - The cyclist speed is randomised per episode.
      - Distance to the cyclist is estimated either from stereo vision
        (default) or from CARLA ground truth (``use_gt_distance=True``),
        the latter serving as an upper-bound ablation baseline.
    """

    def __init__(self, eval_mode=None, FIXED_DELTA_SECONDS=0.05,
                 MAX_STEER_DEGREES=40, SEED: int = 42,
                 safe_brake_distance: float = 6.0,
                 max_speed: int = 28, model_type: str = "PPO",
                 use_gt_distance: bool = False,
                 cyclist_speed_range=(0.5, 1.5),
                 initial_speed_range=(5, 15),
                 max_episode_steps: int = 600):
        self.seed(SEED)
        self.eval_mode = eval_mode
        self.FIXED_DELTA_SECONDS = FIXED_DELTA_SECONDS
        self.MAX_STEER_DEGREES = MAX_STEER_DEGREES

        self.SAFE_BRAKE_DISTANCE = safe_brake_distance
        self.TOO_CLOSE_BRAKE_DISTANCE = 3.5
        self.model_type = model_type
        self.max_speed = max_speed
        self.use_gt_distance = use_gt_distance
        self.cyclist_speed_range = cyclist_speed_range
        self.initial_speed_range = initial_speed_range
        self.max_episode_steps = max_episode_steps

        # --- CARLA connection ---
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)

        self.spawn_points = self.world.get_map().get_spawn_points()
        self.dt = self.settings.fixed_delta_seconds

        # --- Steering PID (lateral controller, not learned) ---
        self.Kp_steer = 0.8
        self.Kd_steer = 0.2

        # --- State ---
        self.speed = 0
        self.avg_distance = 30.0
        self.previousDistance = 30.0
        self.distance_ema = None          # temporal filter for stereo depth
        self.detection_valid = False      # whether we currently see the cyclist
        self.distance_front = 0.0
        self.distance_right = 0.0
        self.EPISODE_REWARD = 0
        self.step_counter = 0

        # --- YOLO detector ---
        self.model = YOLO("best.pt")
        self.CAMERA_POS_Z = 1.5
        self.CAMERA1_POS_X = 0
        self.CAMERA2_POS_X = 1
        self.CAMERA1_POS_Y = 0.5

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '640')
        self.camera_bp.set_attribute('image_size_y', '360')

        self.rightcamera1_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA1_POS_X, y=self.CAMERA1_POS_Y),
            carla.Rotation(yaw=90))
        self.rightcamera2_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA2_POS_X, y=self.CAMERA1_POS_Y),
            carla.Rotation(yaw=90))
        self.frontcamera1_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA1_POS_X, y=self.CAMERA1_POS_Y))
        self.frontcamera2_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA2_POS_X, y=self.CAMERA1_POS_Y))

        self.image_w = self.camera_bp.get_attribute('image_size_x').as_int()
        self.image_h = self.camera_bp.get_attribute('image_size_y').as_int()

        # Stereo parameters
        self.fov = 90
        self.baseline = 1.0
        self.focal_length = self.image_w / (2 * math.tan(math.radians(self.fov / 2)))
        self.y_threshold = 20
        self.ema_alpha = 0.3

        # Observation vector (6 dims)
        self.OBS_DIM = 6
        self.objectreturn = torch.zeros(self.OBS_DIM, dtype=torch.float32)

        self.spectator = self.world.get_spectator()

    # ------------------------------------------------------------------ #
    #  Seeding                                                            #
    # ------------------------------------------------------------------ #
    def seed(self, seed):
        self.seed_value = seed
        if seed is not None:
            random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ #
    #  Cleanup                                                            #
    # ------------------------------------------------------------------ #
    def cleanup(self):
        actors_to_cleanup = [
            getattr(self, name, None) for name in [
                'vehicle', 'bicycle',
                'rightcamera1', 'rightcamera2', 'frontcamera1', 'frontcamera2',
                'collision_sensor'
            ]
        ]
        for actor in actors_to_cleanup:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self.world.tick()
        self.rightcamera1 = None
        self.rightcamera2 = None
        self.frontcamera1 = None
        self.frontcamera2 = None
        self.collision_sensor = None
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------ #
    #  Reset                                                              #
    # ------------------------------------------------------------------ #
    def reset(self):
        print(f"EPISODE REWARD: {self.EPISODE_REWARD:.2f}")

        self.EPISODE_REWARD = 0
        self.speed = 0.0
        self.avg_distance = 30.0
        self.previousDistance = 30.0
        self.distance_ema = None
        self.detection_valid = False
        self.collision_happened = False
        self.steering_angle = 0.0
        self.step_counter = 0
        self.curr_wp = 5

        self.vehicle = None
        self.bicycle = None
        self.cleanup()
        self.bicycleorigin()
        self.carorigin()

        # Randomise cyclist speed each episode
        self.bicycle_speed = random.uniform(*self.cyclist_speed_range)

        # Collision sensor
        self.collision_detector_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            self.collision_detector_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self.process_collision(event))

        # Route
        self.targetid = 27
        self.targetPoint = self.spawn_points[self.targetid]
        self.point_A = self.vehicle_start_point.location
        self.point_B = self.targetPoint.location
        self.sampling_resolution = 3
        self.grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)
        self.route = self.grp.trace_route(self.point_A, self.point_B)

        # Cameras
        self.rightcamera1 = self.world.spawn_actor(self.camera_bp, self.rightcamera1_init_trans, attach_to=self.vehicle)
        self.rightcamera2 = self.world.spawn_actor(self.camera_bp, self.rightcamera2_init_trans, attach_to=self.vehicle)
        self.frontcamera1 = self.world.spawn_actor(self.camera_bp, self.frontcamera1_init_trans, attach_to=self.vehicle)
        self.frontcamera2 = self.world.spawn_actor(self.camera_bp, self.frontcamera2_init_trans, attach_to=self.vehicle)

        self.rightcamera1_data = {'image': np.zeros((self.image_h, self.image_w, 4), dtype=np.uint8)}
        self.rightcamera2_data = {'image': np.zeros((self.image_h, self.image_w, 4), dtype=np.uint8)}
        self.frontcamera1_data = {'image': np.zeros((self.image_h, self.image_w, 4), dtype=np.uint8)}
        self.frontcamera2_data = {'image': np.zeros((self.image_h, self.image_w, 4), dtype=np.uint8)}

        self.rightcamera1.listen(lambda image: self.camera_callback(image, self.rightcamera1_data))
        self.rightcamera2.listen(lambda image: self.camera_callback(image, self.rightcamera2_data))
        self.frontcamera1.listen(lambda image: self.camera_callback(image, self.frontcamera1_data))
        self.frontcamera2.listen(lambda image: self.camera_callback(image, self.frontcamera2_data))

        self.world.tick()
        self._update_observation()

        done = False
        terminated = False
        reward = 0.0
        return [self.objectreturn, reward, done, terminated]

    # ------------------------------------------------------------------ #
    #  Step                                                               #
    # ------------------------------------------------------------------ #
    def step(self, action: Optional[float] = None, training: bool = True):
        """Apply the RL action (longitudinal control) and advance the sim.

        ``action`` is a scalar in [-1, 1]:
            action > 0  ->  throttle = action,  brake = 0
            action < 0  ->  throttle = 0,       brake = |action|
        Steering is always controlled by the waypoint follower.
        """
        if action is None:
            action = 0.0

        action = float(action)
        action = max(-1.0, min(1.0, action))
        throttle = max(0.0, action)
        brake = max(0.0, -action)

        # Cyclist moves at its randomised speed
        if training:
            self.bicycle.apply_control(carla.VehicleControl(throttle=self.bicycle_speed))

        # Ego vehicle: RL longitudinal + waypoint steering
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, brake=brake,
                                 steer=float(self.steering_angle)))

        # Spectator (eval only)
        if not training:
            transform = self.vehicle.get_transform()
            location = transform.location
            rotation = transform.rotation
            offset = carla.Location(x=-6, z=3)
            camera_location = self.get_offset_location(location, rotation.yaw, offset)
            self.spectator.set_transform(
                carla.Transform(camera_location, carla.Rotation(pitch=-15, yaw=rotation.yaw)))
            loc = self.vehicle.get_location()
            self.world.debug.draw_string(
                loc + carla.Location(z=2.5), "Ego", draw_shadow=False,
                color=carla.Color(255, 255, 0), persistent_lines=False)
            self.world.debug.draw_string(
                loc + carla.Location(z=2.4), f"B:{brake:.2f} T:{throttle:.2f}",
                draw_shadow=False, color=carla.Color(255, 255, 0), persistent_lines=False)

        self.world.tick()
        self.step_counter += 1
        self.Detection()

        # Waypoint debug
        next_wp = self.route[min(self.curr_wp, len(self.route) - 1)][0].transform.location
        self.world.debug.draw_point(next_wp, size=0.3, color=carla.Color(0, 255, 0), life_time=2.0)

        # Speed
        v = self.vehicle.get_velocity()
        self.speed = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # --- Reward ---
        self._update_observation()
        reward, done, terminated = self._compute_reward(throttle, brake)

        self.EPISODE_REWARD += reward
        self.previousDistance = self.avg_distance

        return [self.objectreturn, reward, done, terminated]

    def _compute_reward(self, throttle, brake):
        """Reward designed for full RL longitudinal control.

        Components:
          1. Progress incentive: reward proportional to speed (encourage moving).
          2. Safety: penalise being too close to the cyclist.
          3. Distance keeping: penalise deviation from safe braking distance.
          4. Closing-rate penalty: penalise approaching the cyclist too fast.
          5. Control regularisation: small penalty on control magnitude.
          6. Terminal: large penalty for collision, large bonus for reaching goal.
        """
        reward = 0.0
        done = False
        terminated = False

        # 1. Progress incentive — reward forward motion strongly enough
        #    to overcome the living cost.  At 20 km/h this gives +0.7/step.
        speed_reward = self.speed / max(self.max_speed, 1)
        speed_reward = max(0.0, min(1.0, speed_reward))
        reward += 0.5 * speed_reward

        # Small living cost so the agent prefers shorter episodes
        reward -= 0.05

        # 2-4. Safety and distance shaping (only when we have a valid detection)
        if self.detection_valid and self.avg_distance < 50.0:
            e_d = self.avg_distance - self.SAFE_BRAKE_DISTANCE

            # Distance error: 0 at safe distance, negative away from it
            reward += -0.3 * abs(e_d) / self.SAFE_BRAKE_DISTANCE

            # Closing-rate penalty (deltat < 0 means approaching)
            deltat = self.avg_distance - self.previousDistance
            deltat = max(min(deltat, 5.0), -5.0)
            if deltat < 0:
                # Penalise closing, more so when already close
                closeness_factor = max(0.0, 1.0 - self.avg_distance / self.SAFE_BRAKE_DISTANCE)
                reward += 0.5 * deltat * (1.0 + closeness_factor)

            # Too close — large penalty
            if self.avg_distance < self.TOO_CLOSE_BRAKE_DISTANCE:
                reward -= 5.0

        # 5. Control regularisation (very small, just to avoid oscillation)
        reward -= 0.005 * (throttle + brake)

        # 6. Terminal conditions
        if self.collision_happened:
            reward -= 100.0
            done = True
            terminated = True
            self.cleanup()
            return reward, done, terminated

        # Reached goal
        if self.vehicle.get_transform().location.distance(
                self.route[-1][0].transform.location) < 6:
            reward += 100.0
            done = True
            self.cleanup()
            return reward, done, terminated

        # Timeout
        if self.step_counter >= self.max_episode_steps:
            done = True
            self.cleanup()
            return reward, done, terminated

        return reward, done, terminated

    def _update_observation(self):
        """Build the observation vector and update filtered distance."""
        # Get distance (GT or stereo)
        if self.use_gt_distance:
            self._compute_gt_distance()
        else:
            self._compute_stereo_distance()

        # Clamp inf/nan
        if (self.avg_distance == np.inf or self.avg_distance == -np.inf
                or math.isnan(self.avg_distance)):
            self.avg_distance = 30.0
            self.detection_valid = False

        deltat = self.avg_distance - self.previousDistance
        deltat = max(min(deltat, 10.0), -10.0)
        e_d = self.avg_distance - self.SAFE_BRAKE_DISTANCE

        # Normalised observation vector:
        #   [speed/max_speed, distance/30, delta_dist/10, dist_error/30,
        #    safe_brake/30, detection_flag]
        self.objectreturn = torch.tensor([
            self.speed / max(self.max_speed, 1),
            self.avg_distance / 30.0,
            deltat / 10.0,
            e_d / 30.0,
            self.SAFE_BRAKE_DISTANCE / 30.0,
            1.0 if self.detection_valid else 0.0
        ], dtype=torch.float32)

    def _compute_gt_distance(self):
        """Ground-truth distance from CARLA actor locations."""
        if self.bicycle is None or self.vehicle is None:
            self.avg_distance = 30.0
            self.detection_valid = False
            return
        d = self.vehicle.get_location().distance(self.bicycle.get_location())
        self.avg_distance = d
        self.detection_valid = True

    def _compute_stereo_distance(self):
        """Stereo depth with sub-pixel disparity and EMA temporal filter."""
        raw_distance = None

        if self.distance_front > 0 and self.distance_right > 0:
            min_d = min(self.distance_front, self.distance_right)
            max_d = max(self.distance_front, self.distance_right)
            raw_distance = min_d * 0.7 + max_d * 0.3
        elif self.distance_front > 0:
            raw_distance = self.distance_front
        elif self.distance_right > 0:
            raw_distance = self.distance_right

        if raw_distance is not None and raw_distance > 0 and raw_distance != float('inf'):
            # EMA temporal filter
            if self.distance_ema is None:
                self.distance_ema = raw_distance
            else:
                self.distance_ema = (self.ema_alpha * raw_distance
                                     + (1 - self.ema_alpha) * self.distance_ema)
            self.avg_distance = self.distance_ema
            self.detection_valid = True
        else:
            # No detection — keep last known distance but mark as invalid
            if self.distance_ema is not None:
                self.avg_distance = self.distance_ema
            else:
                self.avg_distance = 30.0
            self.detection_valid = False

    # ------------------------------------------------------------------ #
    #  Spawning                                                           #
    # ------------------------------------------------------------------ #
    def bicycleorigin(self):
        self.bicycle_bp = self.world.get_blueprint_library().filter('*crossbike*')
        # Try the preferred spawn point, then fall back
        preferred = [1, 2, 3, 4, 6, 7, 8]
        self.bicycle = None
        for sp_id in preferred:
            if sp_id >= len(self.spawn_points):
                continue
            self.bicycle_start_point = self.spawn_points[sp_id]
            self.bicycle = self.world.try_spawn_actor(
                self.bicycle_bp[0], self.bicycle_start_point)
            if self.bicycle is not None:
                print(f"Bicycle spawned at point {sp_id}")
                break
        if self.bicycle is None:
            raise RuntimeError("Failed to spawn the bicycle at any spawn point.")
        bicyclepos = carla.Transform(
            self.bicycle_start_point.location + carla.Location(x=-3, y=3.5))
        self.bicycle.set_transform(bicyclepos)
        for _ in range(40):
            self.world.tick()
            time.sleep(0.05)

    def carorigin(self):
        self.vehicle_bp = self.world.get_blueprint_library().filter('*mini*')
        # Try the preferred spawn point, then fall back to alternatives
        preferred = [94, 0, 5, 10, 50, 100, 55, 35]
        self.vehicle = None
        for sp_id in preferred:
            if sp_id >= len(self.spawn_points):
                continue
            self.vehicle_start_point = self.spawn_points[sp_id]
            self.vehicle = self.world.try_spawn_actor(
                self.vehicle_bp[0], self.vehicle_start_point)
            if self.vehicle is not None:
                print(f"Ego vehicle spawned at point {sp_id}")
                break
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn the ego vehicle at any spawn point.")
        for _ in range(40):
            self.world.tick()
            time.sleep(0.05)

        # Give the car an initial forward velocity so it doesn't get stuck
        # at a standstill with a random initial policy.
        initial_speed_kmh = random.uniform(*self.initial_speed_range)
        initial_speed_ms = initial_speed_kmh / 3.6  # m/s
        # Get the forward direction from the spawn point's rotation
        yaw = math.radians(self.vehicle_start_point.rotation.yaw)
        forward_vec = carla.Vector3D(x=math.cos(yaw), y=math.sin(yaw), z=0)
        velocity = carla.Vector3D(
            x=forward_vec.x * initial_speed_ms,
            y=forward_vec.y * initial_speed_ms,
            z=0)
        self.vehicle.set_target_velocity(velocity)
        self.world.tick()
        print(f"Ego vehicle initial speed: {initial_speed_kmh:.1f} km/h")

    # ------------------------------------------------------------------ #
    #  Utilities                                                          #
    # ------------------------------------------------------------------ #
    def get_offset_location(self, base_location, yaw, offset):
        rad = math.radians(yaw)
        x = base_location.x + offset.x * math.cos(rad) - offset.y * math.sin(rad)
        y = base_location.y + offset.x * math.sin(rad) + offset.y * math.cos(rad)
        z = base_location.z + offset.z
        return carla.Location(x=x, y=y, z=z)

    def process_collision(self, event):
        self.collision_happened = True

    def angle_between(self, v1, v2):
        return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

    def get_angle(self, car, wp):
        vp = car.get_transform()
        dx = wp.transform.location.x - vp.location.x
        dy = wp.transform.location.y - vp.location.y
        norm = math.sqrt(dx**2 + dy**2)
        if norm < 1e-6:
            return 0.0
        wx = dx / norm
        wy = dy / norm
        fv = vp.get_forward_vector()
        return self.angle_between((wx, wy), (fv.x, fv.y))

    # ------------------------------------------------------------------ #
    #  Detection & stereo                                                 #
    # ------------------------------------------------------------------ #
    def Detection(self):
        """Run YOLO on all 4 cameras, match stereo pairs, compute depth."""
        # Advance waypoint
        if self.vehicle.get_transform().location.distance(
                self.route[self.curr_wp][0].transform.location) < 3:
            self.curr_wp = min(self.curr_wp + 1, len(self.route) - 1)

        # Grab frames
        rf1 = cv2.cvtColor(self.rightcamera1_data['image'], cv2.COLOR_BGRA2BGR)
        rf2 = cv2.cvtColor(self.rightcamera2_data['image'], cv2.COLOR_BGRA2BGR)
        ff1 = cv2.cvtColor(self.frontcamera1_data['image'], cv2.COLOR_BGRA2BGR)
        ff2 = cv2.cvtColor(self.frontcamera2_data['image'], cv2.COLOR_BGRA2BGR)

        # YOLO detection (verbose=False for all to reduce log spam)
        det_r1 = self._detect_bicycles(rf1)
        det_r2 = self._detect_bicycles(rf2)
        det_f1 = self._detect_bicycles(ff1)
        det_f2 = self._detect_bicycles(ff2)

        # Stereo matching with sub-pixel disparity
        self.distance_right = self._stereo_depth(det_r1, det_r2)
        self.distance_front = self._stereo_depth(det_f1, det_f2)

        # Steering (waypoint follower — not learned)
        self.predicted_angle = self.get_angle(self.vehicle, self.route[self.curr_wp][0])
        if self.predicted_angle < -300:
            self.predicted_angle += 360
        elif self.predicted_angle > 300:
            self.predicted_angle -= 360

        self.steering_angle = max(-self.MAX_STEER_DEGREES,
                                  min(self.MAX_STEER_DEGREES, self.predicted_angle))
        self.steering_angle = self.steering_angle / self.MAX_STEER_DEGREES

    def _detect_bicycles(self, frame):
        """Run YOLO and return list of (center_x_float, center_y_float, conf)."""
        results = self.model(frame, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf[0] < 0.5:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                # Use float centres for sub-pixel disparity
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                conf = float(box.conf[0])
                detections.append((cx, cy, conf))
        return detections

    def _stereo_depth(self, det_left, det_right):
        """Match detections between stereo pair and return best depth.

        Uses sub-pixel disparity (float centre_x) and returns the
        minimum depth (closest detected cyclist).
        """
        if not det_left or not det_right:
            return 0.0

        best_depth = float('inf')
        for lx, ly, _ in det_left:
            closest_rx = None
            min_dy = self.y_threshold
            for rx, ry, _ in det_right:
                dy = abs(ly - ry)
                if dy < min_dy:
                    min_dy = dy
                    closest_rx = rx
            if closest_rx is not None:
                disparity = abs(lx - closest_rx)
                if disparity > 0.5:  # avoid division by near-zero
                    depth = (self.focal_length * self.baseline) / disparity
                    if depth < best_depth:
                        best_depth = depth

        return best_depth if best_depth != float('inf') else 0.0

    def camera_callback(self, image, data_dict):
        data_dict['image'] = np.reshape(
            np.copy(image.raw_data), (image.height, image.width, 4))

    def __str__(self):
        v = [round(x.item(), 3) for x in self.objectreturn]
        return (f"Speed_norm={v[0]}, Dist_norm={v[1]}, Delta={v[2]}, "
                f"Err={v[3]}, SafeBrake={v[4]}, Det={v[5]}")
