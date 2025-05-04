import carla
import random
import time
import sys
sys.path.append('F:\CARLA\Windows\CARLA_0.9.15\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner
import numpy as np
import math
from ultralytics import YOLO
import cv2
import torch
from typing import Optional

class EnvironmentClass:

    def __init__(self, eval_mode = None, FIXED_DELTA_SECONDS = 0.05, MAX_STEER_DEGREES = 40):

        self.eval_mode = eval_mode
        self.FIXED_DELTA_SECONDS = FIXED_DELTA_SECONDS
        self.MAX_STEER_DEGREES = MAX_STEER_DEGREES

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()

        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = self.FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)

        self.SAFE_BRAKE_DISTANCE = 5.5
        self.TOO_CLOSE_BRAKE_DISTANCE = 3.5
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle_bp = self.world.get_blueprint_library().filter('*mini*')
        self.Kp = 0.3
        self.Ki = 0.0
        self.Kd = 0.1
        self.dt = self.settings.fixed_delta_seconds
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_speed = 28

        self.step_counter = 0
        self.episode_point = 0

        #Car properties
        self.speed = 0


        #Braking properties
        self.goodbrake=0
        self.wrongbrake = 0
        self.emergencybrake = 0

        self.reallybadthrottle = 0
        self.badthrottle = 0
        self.goodthrottle = 0

        self.USEREINFORCEMENT = 7

        #camera setup
        self.model = YOLO("best.pt")
        self.CAMERA_POS_Z = 1.5 
        self.CAMERA1_POS_X = 0
        self.CAMERA2_POS_X = 1
        self.CAMERA1_POS_Y = 0.5
        self.CAMERA2_POS_Y = 1.5

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.14 on Windows
        self.camera_bp.set_attribute('image_size_y', '360')

        self.rightcamera1_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA1_POS_Y), carla.Rotation(yaw=90))
        self.rightcamera2_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA2_POS_X, y = self.CAMERA1_POS_Y), carla.Rotation(yaw=90))
        self.frontcamera1_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA1_POS_Y))
        self.frontcamera2_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA2_POS_Y))

        self.image_w = self.camera_bp.get_attribute('image_size_x').as_int()
        self.image_h = self.camera_bp.get_attribute('image_size_y').as_int()

        #return string
        self.objectreturn = torch.tensor([0, 0, 0, 0], dtype=torch.float16)
        #reward properties
        self.EPISODE_REWARD = 0


    def setup_PID_controller(self, Kp = 0.3, Ki = 0.0, Kd = 0.1, integral_error = 0.0, last_error = 0.0, max_speed = 28):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = self.settings.fixed_delta_seconds
        self.integral_error = integral_error
        self.last_error = last_error
        self.max_speed = max_speed


    def cleanup(self):
        print("Cleaning up environment...")

        actors_to_cleanup = [
            getattr(self, name, None) for name in [
                'vehicle', 'bicycle',
                'rightcamera1', 'rightcamera2', 'frontcamera1', 'frontcamera2',
                'collision_sensor'
            ]
        ]

        for actor in actors_to_cleanup:
            print(actor)
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Could not destroy actor: {e}")
        self.world.tick()
        self.rightcamera1 = None
        self.rightcamera2 = None
        self.frontcamera1 = None
        self.frontcamera2 = None
        self.collision_sensor = None

        cv2.destroyAllWindows()

    
    def reset(self):


        print(f"EPISODE REWARD: {self.EPISODE_REWARD}")
        print(f"Number of good brake in episode {self.goodbrake}")
        print(f"Number of wrong brake in a episode {self.wrongbrake}")
        print(f"Number of Emergency Brake in episode: {self.emergencybrake}")
        print(f"Number of good throttle in episode {self.goodthrottle}")
        print(f"Number of bad throttle in a episode {self.badthrottle}")
        print(f"Number of really bad throttle in episode: {self.reallybadthrottle}")

        self.EPISODE_REWARD = 0
        self.goodbrake=0
        self.wrongbrake = 0
        self.emergencybrake = 0
        self.goodthrottle = 0
        self.badthrottle = 0
        self.reallybadthrottle = 0

        self.give_points = False
        print(f"We are in this mode: {self.eval_mode}")
        self.episode_point = 0
        self.episode_run_time = 0
        self.speed = 0.0
        self.bicycle_speed = random.uniform(0.5, 1)
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
        self.previous_speed = 0

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

        #Setup camera image
        self.rightcamera1 = self.world.spawn_actor(self.camera_bp,self.rightcamera1_init_trans,attach_to=self.vehicle)
        self.rightcamera2 = self.world.spawn_actor(self.camera_bp,self.rightcamera2_init_trans,attach_to=self.vehicle)
        self.frontcamera1 = self.world.spawn_actor(self.camera_bp,self.frontcamera1_init_trans,attach_to=self.vehicle)
        self.frontcamera2 = self.world.spawn_actor(self.camera_bp,self.frontcamera2_init_trans,attach_to=self.vehicle)

        self.rightcamera1_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.rightcamera2_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.frontcamera1_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.frontcamera2_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        # this actually opens a live stream from the camera
        self.rightcamera1.listen(lambda image: self.camera_callback(image,self.rightcamera1_data))
        self.rightcamera2.listen(lambda image: self.camera_callback(image,self.rightcamera2_data))
        self.frontcamera1.listen(lambda image: self.camera_callback(image,self.frontcamera1_data))
        self.frontcamera2.listen(lambda image: self.camera_callback(image,self.frontcamera2_data))


        self.world.tick()
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        done = False
        terminated = False
        next_step = 0
        reward = 0

        return [self.objectreturn, reward, done, terminated, next_step]

    def step(self, controlValues: Optional[int] = None):

        if self.avg_distance < self.USEREINFORCEMENT:
            self.give_points = True
            if controlValues <= 0:
                brake = controlValues*-1
                throttle = 0
            else:
                brake = 0
                throttle = controlValues
                
        else:
            self.give_points = False
            throttle, brake = self.update_control(28)
        
        self.bicycle.apply_control(carla.VehicleControl(throttle=self.bicycle_speed))
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer = float(self.steering_angle)))

        self.world.tick()
        self.Detection()
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

        self.world.debug.draw_point(
            next_waypoint_location,
            size=0.3,                         # Size of the sphere
            color=carla.Color(0, 255, 0),     # Green color
            life_time=2.0                     # Duration
        )

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        self.speed = kmh


        #Giving reward
        reward = 0
        done = False
        terminated = False

        
        if self.give_points:
            if kmh >= self.previous_speed and kmh > 0:
                reward += 2
                self.previous_speed = kmh
            else:
                reward -= 5

            if kmh < 28:
                reward += 5
            else:
                reward -= 100

        #Reward reaching intermediate waypoints:
        if self.give_points:
            if self.vehicle.get_transform().location.distance(self.route[self.curr_wp][0].transform.location) < 5:
                reward += 5  # Reward for reaching waypoint
                self.curr_wp += 1

            if brake > 0.1:
                if self.SAFE_BRAKE_DISTANCE > self.avg_distance > self.TOO_CLOSE_BRAKE_DISTANCE:
                    self.goodbrake += 1
                    reward += 100  # Proper braking
                elif self.avg_distance > self.SAFE_BRAKE_DISTANCE:
                    self.wrongbrake += 1
                    reward -= 70  # Unnecessary braking
                elif self.avg_distance < self.TOO_CLOSE_BRAKE_DISTANCE:
                    self.emergencybrake += 1
                    reward += 5  # Failure to brake in tim

            if throttle > 0.1:
                if self.SAFE_BRAKE_DISTANCE > self.avg_distance > self.TOO_CLOSE_BRAKE_DISTANCE:
                    self.badthrottle += 1
                    reward += -70  # Proper braking
                elif self.avg_distance > self.SAFE_BRAKE_DISTANCE:
                    self.goodthrottle += 1
                    reward -= 100  # Unnecessary braking
                elif self.avg_distance < self.TOO_CLOSE_BRAKE_DISTANCE:
                    self.reallybadthrottle += 1
                    reward -= 120  # Failure to brake in time


        self.objectreturn = torch.tensor([
            self.speed,
            self.avg_distance,
            throttle,
            brake,

        ], dtype=torch.float32)

        if self.avg_distance < self.USEREINFORCEMENT:
            next_step = 1
        else:
            next_step = 0
           
        #Collision and Out-of-Bounds Penalties
        if self.collision_happened:
                reward -= 2000
                done = True
                terminated = True
                self.EPISODE_REWARD += reward
                self.cleanup()

                return [self.objectreturn, reward, done, terminated, next_step]


        #Reaching the end
        if self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location) < 6:
            if self.give_points:
                reward += 50
                done = True
                if self.episode_run_time < 8:
                    reward += 30
                else:
                    reward -= 40
            done = True
            self.cleanup()

            return [self.objectreturn, reward, done, terminated, next_step]


        self.EPISODE_REWARD += reward


        return [self.objectreturn, reward, done, terminated, next_step]


    #Origins
    def bicycleorigin(self):
        self.bicycle_bp = self.world.get_blueprint_library().filter('*crossbike*')
        self.bicycle_start_point = self.spawn_points[1]

        self.bicycle = self.world.try_spawn_actor(self.bicycle_bp[0], self.bicycle_start_point)
        bicyclepos = carla.Transform(self.bicycle_start_point.location + carla.Location(x=-3, y=3.5))
        self.bicycle.set_transform(bicyclepos)
        for _ in range(40):  # wait for half a second
            #"Still Falling - Cyclist")
            self.world.tick()
            time.sleep(0.05)

    def carorigin(self):
        self.vehicle_bp = self.world.get_blueprint_library().filter('*mini*')
        self.vehicle_start_point = self.spawn_points[94]
        self.vehicle = self.world.try_spawn_actor(self.vehicle_bp[0], self.vehicle_start_point)
        for _ in range(40):  # wait for half a second
            #print("Still Falling - Car")
            self.world.tick()
            time.sleep(0.05)


    #Utilities
    def update_control(self, desired_speed):
        current_speed = self.speed
        speed_error = desired_speed - current_speed
        self.integral_error += speed_error * self.dt
        derivative_error = (speed_error - self.last_error) / self.dt
        self.last_error = speed_error

        # PID computation
        control_output = self.Kp * speed_error + self.Ki * self.integral_error + self.Kd * derivative_error


        # Map control output to throttle and brake command
        if control_output > 0:
            throttle = min(control_output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(control_output), 1.0)
        
        return throttle, brake
    
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

    def Detection(self):


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
            self.avg_distance = 30
        self.predicted_angle = self.get_angle(self.vehicle, self.route[self.curr_wp][0])


        if self.predicted_angle < -300:
            self.predicted_angle = self.predicted_angle+360
        elif self.predicted_angle > 300:
            self.predicted_angle = self.predicted_angle - 360
        self.steering_angle = self.predicted_angle

        if self.predicted_angle < -self.MAX_STEER_DEGREES:
            self.steering_angle = -self.MAX_STEER_DEGREES
        elif self.predicted_angle>self.MAX_STEER_DEGREES:
            self.steering_angle = self.MAX_STEER_DEGREES

    
        self.estimated_throttel = 0
        self.steering_angle = self.steering_angle/self.MAX_STEER_DEGREES



    #Camera Properties
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
    
    def camera_callback(self, image,data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))


    
    def __str__(self):
        mylist = [round(x.item(), 2) for x in self.objectreturn]
        return f"These are the return values: Speed: %s, Distance %s, Throttle: %s, Brake: %s" % tuple(mylist)