import numpy as np
import gymnasium
from gymnasium import spaces
import carla
import cv2
import math
from ultralytics import YOLO
import sys
from gymnasium.utils import seeding
import random
import time
sys.path.append('F:\CARLA\Windows\CARLA_0.9.15\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner


FIXED_DELTA_SECONDS = 0.2
MAX_STEER_DEGREES = 40

class CarEnv(gymnasium.Env):


    def __init__(self, eval_mode = None) -> None:
        super(CarEnv).__init__()
        self.eval_mode = eval_mode

         # Continuous action space: [throttle, brake]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),   # Minimum values: 0.0 throttle, 0.0 brake
            high=np.array([1.0, 1.0]),  # Maximum values: 1.0 throttle, 1.0 brake
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, 0]),  # min values for kmh, avg_distance, throttle, brake
            high=np.array([1, 1, 1, 1, 1, 1]),  # normalized max values
            dtype=np.float32
        )


        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        # self.settings.no_rendering_mode = False
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)

        self.vehicle_bp = self.world.get_blueprint_library().filter('*mini*')

        self.safe_brake_distance = 3
        self.too_close_brake_distance = 1.5 
        self.spawn_points = self.world.get_map().get_spawn_points()

        
        
        self.step_counter = 0

    def cleanup(self):
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        cv2.destroyAllWindows()

    def step(self, action):

        throttle = action[0]
        brake = action[1]

        print(f"Brake Value: {brake}")
        print(f"Throttle Value: {throttle}")
        # Ensure only one of throttle or brake is applied
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0

        

        self.bicycle.apply_control(carla.VehicleControl(throttle=1))
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer = float(self.steering_angle)))

        self.world.tick()

        self.movingandDetecting()

        next_waypoint_location = self.route[self.curr_wp][0].transform.location

        # Draw a sphere at the waypoint location
        self.world.debug.draw_string(
            next_waypoint_location,            # Location of the waypoint
            "Next WP",                         # Label to display
            draw_shadow=False,
            color=carla.Color(255, 0, 0),      # Red color
            life_time=2.0,                     # Duration the marker is visible (seconds)
            persistent_lines=False
        )

        # Optionally, draw a larger sphere as a visual marker
        self.world.debug.draw_point(
            next_waypoint_location,
            size=0.3,                         # Size of the sphere
            color=carla.Color(0, 255, 0),     # Green color
            life_time=2.0                     # Duration
        )

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        end_time = time.time()

        episode_run_time = end_time-self.episode_start
        print(episode_run_time)

        reward = 0
        done = False
        terminated = False
        distance_to_goal = self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location)

        print("That is my speed: {}".format(kmh))
        #Speeding Reward
        if 20 <= kmh <= 30:
            reward += 50
        elif 10 <= kmh < 20:
            reward += 20
        elif kmh < 10:
            reward -= 30  # Penalize being too slow
        elif kmh > 30:
            reward -= 300  # Penalize speeding

        #Distance to goal reward
        progress = self.previousDistance - distance_to_goal
        if progress > 0:
            # Progress reward increases over time, encouraging faster progress
            progress_reward = 100 * (progress / distance_to_goal)
        else:
            # Penalize stagnation or moving away from the goal
            progress_reward = -50 * abs(progress)

        #Reward reaching intermediate waypoints:
        if self.vehicle.get_transform().location.distance(self.route[self.curr_wp][0].transform.location) < 5:
            reward += 5  # Reward for reaching waypoint
            self.curr_wp += 1

        if brake > 0.8:
            if self.safe_brake_distance > self.avg_distance > self.too_close_brake_distance:
                reward += 15  # Proper braking
            elif self.avg_distance > self.safe_brake_distance:
                reward -= 10  # Unnecessary braking
            elif self.avg_distance < self.too_close_brake_distance:
                reward += 5  # Failure to brake in time

        #Collision and Out-of-Bounds Penalties
        if self.collision_happened:
            reward -= 1000
            done = True
            self.cleanup()
        if distance_to_goal > 50:
            reward -= 200  # Strayed too far from goal
            done = True
            self.cleanup()
        if episode_run_time > 40:
            done = True
            reward -= 1000
            self.cleanup()

        # Adjust for time: Scale progress reward by the remaining distance and time penalty
        time_varying_reward = progress_reward - 0.1 * episode_run_time
        reward += time_varying_reward

        #Reaching the end point            
        if self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location) < 6:
            reward += 50
            done = True
            if episode_run_time < 25:
                reward += 10
            else:
                reward -= 20
            self.cleanup()
            print(f"Point of the given episode: {self.episode_point+50}")

    
        progress_reward = 200*(1/distance_to_goal)
        time_penalty_reward = -1*episode_run_time
        timevaryreward = progress_reward+time_penalty_reward

        reward += timevaryreward

        # Update previous distance
        self.previousDistance = distance_to_goal

        #This is some test, to see how it works
        # Normalize features to create observation
        max_speed = 50.0  # Assume max speed is 50 km/h
        max_bicycle_distance = 30  # Assume max bicycle distance for normalization
        #max_endpoint_distance = 500.0  # Assume max endpoint distance for normalization

        obs = np.array([
            kmh / max_speed,
            self.avg_distance / max_bicycle_distance,
            throttle,
            brake,
            self.steering_angle,
            distance_to_goal/50,
        ], dtype=np.float32)

        self.episode_point += reward

        return obs, reward, done, terminated, {}
            

    def reset(self, seed=None, options=None):
        
        print(f"We are in this mode: {self.eval_mode}")
        self.episode_point = 0
        self.episode_start = time.time()
        self.seed(seed)
        self.bicycle_speed = random.uniform(0.1, 1)
        self.previousDistance = 100
        self.vehicle = None
        self.bicycle = None
        self.curr_wp = 5
        self.cleanup()
        self.bicycleorigin()
        self.carorigin()
        self.steering_angle = 0
        self.collision_happened = False
        self.avg_distance = 20
        self.steering_lock = False

        self.collision_detector_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
                self.collision_detector_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
        self.collision_sensor.listen(lambda event: self.process_collision(event))

        self.targetid = 27
        self.targetPoint = self.spawn_points[self.targetid]

        self.point_A = self.vehicle_start_point.location
        self.point_B = self.targetPoint.location


        self.sampling_resolution = 3
        self.grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)

        self.route = self.grp.trace_route(self.point_A, self.point_B)


        self.CAMERA_POS_Z = 1.5 
        self.CAMERA1_POS_X = 0
        self.CAMERA2_POS_X = 1
        self.CAMERA1_POS_Y = 0.5
        self.CAMERA2_POS_Y = 1.5

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.14 on Windows
        self.camera_bp.set_attribute('image_size_y', '360')


        self.rightcamera1_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA1_POS_Y), carla.Rotation(yaw=90))
        self.rightcamera1 = self.world.spawn_actor(self.camera_bp,self.rightcamera1_init_trans,attach_to=self.vehicle)

        self.rightcamera2_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA2_POS_X, y = self.CAMERA1_POS_Y), carla.Rotation(yaw=90))
        self.rightcamera2 = self.world.spawn_actor(self.camera_bp,self.rightcamera2_init_trans,attach_to=self.vehicle)

        self.frontcamera1_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA1_POS_Y))
        self.frontcamera1 = self.world.spawn_actor(self.camera_bp,self.frontcamera1_init_trans,attach_to=self.vehicle)

        self.frontcamera2_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA2_POS_Y))
        self.frontcamera2 = self.world.spawn_actor(self.camera_bp,self.frontcamera2_init_trans,attach_to=self.vehicle)

        self.image_w = self.camera_bp.get_attribute('image_size_x').as_int()
        self.image_h = self.camera_bp.get_attribute('image_size_y').as_int()

        self.rightcamera1_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.rightcamera2_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.frontcamera1_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.frontcamera2_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        # this actually opens a live stream from the camera
        self.rightcamera1.listen(lambda image: self.camera_callback(image,self.rightcamera1_data))
        self.rightcamera2.listen(lambda image: self.camera_callback(image,self.rightcamera2_data))
        self.frontcamera1.listen(lambda image: self.camera_callback(image,self.frontcamera1_data))
        self.frontcamera2.listen(lambda image: self.camera_callback(image,self.frontcamera2_data))
        self.model = YOLO("best.pt")

        self.movingandDetecting()

        self.world.tick()

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        obs = np.array([kmh / 50.0, self.avg_distance / 100.0, 0.0, 0.0, 0.0, 35.0,], dtype=np.float32)
        return obs, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def movingandDetecting(self):

        self.world.tick()

        if self.vehicle.get_transform().location.distance(self.route[self.curr_wp][0].transform.location) < 3:
            self.curr_wp += 1

        self.rightframe1 = self.rightcamera1_data['image']
        self.rightframe2 = self.rightcamera2_data['image']
        self.frontframe1 = self.frontcamera1_data['image']
        self.frontframe2 = self.frontcamera2_data['image']

        # Convert RGB image from BGRA to BGR
        self.rightframe1 = cv2.cvtColor(self.rightframe1, cv2.COLOR_BGRA2BGR)
        self.rightframe2 = cv2.cvtColor(self.rightframe2, cv2.COLOR_BGRA2BGR)
        self.frontframe1 = cv2.cvtColor(self.frontframe1, cv2.COLOR_BGRA2BGR)
        self.frontframe2 = cv2.cvtColor(self.frontframe2, cv2.COLOR_BGRA2BGR)

        self.results_right1 = self.model(self.rightframe1, verbose=False)
        self.results_right2 = self.model(self.rightframe2)
        
        self.results_front1 = self.model(self.frontframe1)
        self.results_front2 = self.model(self.frontframe2)
        
        self.bicycles_right1 = []
        self.bicycles_right2 = []
        self.bicycles_front1 = []
        self.bicycles_front2 = []


        for result in self.results_right1:
            for box in result.boxes:
                # Extract box coordinates and other details
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = int((x1 + x2) / 2)  # x-center of the bicycle
                center_y = int((y1 + y2) / 2)  # y-center of the bicycle
                self.bicycles_right1.append((center_x, center_y))
                conf = box.conf[0]            # Confidence score
                cls = box.cls[0]
                cv2.rectangle(self.rightframe1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(self.rightframe1, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                #break
                # Bounding box coordinates
        
        for result in self.results_right2:
            for box2 in result.boxes:
                # Extract box coordinates and other details
                x1, y1, x2, y2 = box2.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.bicycles_right2.append((center_x, center_y))
                conf = box2.conf[0]            # Confidence score
                cls = box2.cls[0]

                cv2.rectangle(self.rightframe2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(self.rightframe2, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for result in self.results_front1:
            for box2 in result.boxes:
                # Extract box coordinates and other details
                x1, y1, x2, y2 = box2.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.bicycles_front1.append((center_x, center_y))
                conf = box2.conf[0]            # Confidence score
                cls = box2.cls[0]

                cv2.rectangle(self.frontframe1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(self.frontframe1, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for result in self.results_front2:
            for box2 in result.boxes:
                # Extract box coordinates and other details
                x1, y1, x2, y2 = box2.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                self.bicycles_front2.append((center_x, center_y))
                conf = box2.conf[0]            # Confidence score
                cls = box2.cls[0]

                cv2.rectangle(self.frontframe2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(self.frontframe2, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self.matched_bicycles_with_distances_right = self.match_bicycles_between_left_right(self.bicycles_right1, self.bicycles_right2)
        self.matched_bicycles_with_distances_front = self.match_bicycles_between_left_right(self.bicycles_front1, self.bicycles_front2)
        # Display distance for each matched bicycle on the left frame
        
        self.distance_front = 0
        self.distance_right = 0


        for (left_bicycle, distance) in self.matched_bicycles_with_distances_right:
            left_x, left_y = left_bicycle
            distance_label = f"Distance (right camera): {distance:.2f}m"
            self.distance_right = distance
            cv2.putText(self.rightframe1, distance_label, (left_x, left_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        for (left_bicycle, distance) in self.matched_bicycles_with_distances_front:
            left_x, left_y = left_bicycle
            self.distance_front = distance
            distance_label = f"Distance (front camera): {distance:.2f}m"
            cv2.putText(self.frontframe1, distance_label, (left_x, left_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if self.distance_front > 0 and self.distance_right > 0:
            min_distance = np.min([self.distance_front, self.distance_right])
            max_distance = np.max([self.distance_front, self.distance_right])

            self.avg_distance = (min_distance*0.7 + max_distance*0.3)
        elif self.distance_front == 0 and self.distance_right > 0:
            self.avg_distance = self.distance_right
        elif self.distance_right == 0 and self.distance_front > 0:
            self.avg_distance = self.distance_front
        else:
            self.avg_distance = 20
        self.predicted_angle = self.get_angle(self.vehicle, self.route[self.curr_wp][0])


        if self.predicted_angle < -300:
            self.predicted_angle = self.predicted_angle+360
        elif self.predicted_angle > 300:
            self.predicted_angle = self.predicted_angle - 360
        self.steering_angle = self.predicted_angle

        if self.predicted_angle < -MAX_STEER_DEGREES:
            self.steering_angle = -MAX_STEER_DEGREES
        elif self.predicted_angle>MAX_STEER_DEGREES:
            self.steering_angle = MAX_STEER_DEGREES

    
        self.estimated_throttel = 0
        self.steering_angle = self.steering_angle/MAX_STEER_DEGREES
        



    def bicycleorigin(self):
        self.bicycle_bp = self.world.get_blueprint_library().filter('*crossbike*')
        self.bicycle_start_point = self.spawn_points[1]

        self.bicycle = self.world.try_spawn_actor(self.bicycle_bp[0], self.bicycle_start_point)
        bicyclepos = carla.Transform(self.bicycle_start_point.location + carla.Location(x=-3, y=3.5))
        self.bicycle.set_transform(bicyclepos)

        

    def camera_callback(self, image,data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))

    def carorigin(self):
        self.vehicle_bp = self.world.get_blueprint_library().filter('*mini*')
        self.vehicle_start_point = self.spawn_points[94]
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp[0], self.vehicle_start_point)
    
    def process_collision(self, event):
        # Extract collision data
        self.other_actor = event.other_actor
        self.impulse = event.normal_impulse
        self.collision_location = event.transform.location
        self.collision_happened = True

    def angle_between(self, v1, v2):
        return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

    def get_angle(self, car, wp):
        self.vehicle_pos = car.get_transform()
        self.car_x = self.vehicle_pos.location.x
        self.car_y = self.vehicle_pos.location.y
        self.wp_x = wp.transform.location.x
        self.wp_y = wp.transform.location.y


        #vector to waypoint
        self.x = (self.wp_x - self.car_x)/((self.wp_y - self.car_y)**2 + (self.wp_x - self.car_x)**2)**0.5
        self.y = (self.wp_y - self.car_y)/((self.wp_y - self.car_y)**2 + (self.wp_x - self.car_x)**2)**0.5


        #car vector
        self.car_vector = self.vehicle_pos.get_forward_vector()
        self.degrees = self.angle_between((self.x,self.y), (self.car_vector.x, self.car_vector.y))

        return self.degrees
    
    def match_bicycles_between_left_right(self, bicycles_left: list, bicycles_right: list):
        self.image_w = 640  # Image width
        self.fov = 90  # Field of view in degrees
        self.baseline = 1  # Baseline distance in meters
        self.focal_length = self.image_w / (2 * math.tan(math.radians(self.fov / 2)))  # Focal length in pixels

        
        self.y_threshold = 20  # pixels, adjust based on image scale
        self.matched_bicycles_with_distances = []

        for left_bicycle in bicycles_left:
            left_x, left_y = left_bicycle
            closest_bicycle = None
            min_dist = float('inf')

            for right_bicycle in bicycles_right:
                right_x, right_y = right_bicycle
                # Check if the y-coordinates are similar
                if abs(left_y - right_y) < self.y_threshold:
                    # Calculate the distance (disparity)
                    dist = abs(left_x - right_x)
                    if dist < min_dist:
                        min_dist = dist
                        closest_bicycle = right_bicycle

            # If a match was found, calculate depth and add to list
            if closest_bicycle:
                right_x, _ = closest_bicycle
                disparity = abs(left_x - right_x)
                depth = (self.focal_length * self.baseline) / disparity if disparity != 0 else float('inf')
                self.matched_bicycles_with_distances.append((left_bicycle, depth))

        return self.matched_bicycles_with_distances

    
