import carla
import numpy as np
import cv2
import time
import sys
from gymnasium.utils import seeding
sys.path.append('F:\CARLA\Windows\CARLA_0.9.15\PythonAPI\carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
import math
import gymnasium
from gymnasium import spaces
from ultralytics import YOLO


MAX_STEER_DEGREES = 40




class CarEnv(gymnasium.Env):
    def __init__(self, eval_mode = None) -> None:
        super(CarEnv).__init__()
        self.eval_mode = eval_mode

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),   # Minimum values: 0.0 throttle, 0.0 brake
            high=np.array([1.0, 1.0]),  # Maximum values: 1.0 throttle, 1.0 brake
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -1]),  # min values for is_there_a_cyclist, distance_to_bicycle, speed, brake, throttle, steeeing angle
            high=np.array([1, 1, 1, 1, 1, 1]),  # normalized max values
            dtype=np.float32
        )


        
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

        self.settings = self.world.get_settings()
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.04  # 20 FPS at 0.05
        self.world.apply_settings(self.settings)


        self.vehicle_bp = self.world.get_blueprint_library().filter('*mini*')

        self.safe_brake_distance = 3
        self.too_close_brake_distance = 1.5 
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.step_counter = 0

        self.move_duration = 0.04  # seconds
        self.frame_time = self.settings.fixed_delta_seconds

        #For PID controller
        self.Kp = 0.3
        self.Ki = 0.0
        self.Kd = 0.1
        self.dt = self.settings.fixed_delta_seconds
        self.integral_error = 0.0
        self.last_error = 0.0
        self.max_speed = 28

    def carorigin(self):
        vehicle_bp = self.world.get_blueprint_library().filter('*mini*')
        self.vehicle_start_point = self.spawn_points[94]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp[0], self.vehicle_start_point)
        for _ in range(20):  # wait for half a second
            print("Még esek")
            self.world.tick()
            time.sleep(0.05)


    def bicycleorigin(self):
        bicycle_bp = self.world.get_blueprint_library().filter('*crossbike*')
        bicycle_start_point = self.spawn_points[99]

        self.bicycle = self.world.try_spawn_actor(bicycle_bp[0], bicycle_start_point)
        
        new_location = bicycle_start_point.location + carla.Location(y=25)
        new_rotation = carla.Rotation(pitch=0, yaw=bicycle_start_point.rotation.yaw + 0, roll=0)

        
        bicyclepos = carla.Transform(new_location, new_rotation)
        self.bicycle.set_transform(bicyclepos)


    def destroy(self):#Destroying the existing things
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def process_collision(self, event):
        # Extract collision data
        self.other_actor = event.other_actor
        self.impulse = event.normal_impulse
        self.collision_location = event.transform.location
        print(f"Collision with {self.other_actor.type_id}")
        print(f"Impulse: {self.impulse}")
        print(f"Location: {self.collision_location}")
        self.collision_happened = True



    def reset(self, seed = None, options=None):
        """
        This action will be called, to reset the environment, when a collision happend
        """
        #Spawn the cars and cyclist
        self.destroy()
        self.carorigin()
        self.bicycleorigin()
        self.seed(seed)
        self.curr_wp = 5

        #Setup Collision Detector
        self.collision_detector_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
                self.collision_detector_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
        self.collision_sensor.listen(lambda event: self.process_collision(event))
        self.collision_happened = False
        
        
        self.steering_angle = 0
        self.episode_reward = 0
        
        #Set the goal state
        self.targetid = 30 #To make a right and left turn
        self.targetPoint = self.spawn_points[self.targetid]
        self.point_A = self.vehicle_start_point.location
        self.point_B = self.targetPoint.location
        self.sampling_resolution = 3
        self.grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)
        self.route = self.grp.trace_route(self.point_A, self.point_B)

        #Setup Camera system
        self.CAMERA_POS_Z = 1.5 
        self.CAMERA1_POS_X = 0
        self.CAMERA1_POS_Y = 0.5
        self.CAMERA2_POS_X = 1
        self.CAMERA2_POS_Y = -0.5

        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '640') # this ratio works in CARLA 9.14 on Windows
        self.camera_bp.set_attribute('image_size_y', '360')

        self.frontcamera1_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA1_POS_Y))
        self.frontcamera1 = self.world.spawn_actor(self.camera_bp,self.frontcamera1_init_trans,attach_to=self.vehicle)

        self.frontcamera2_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA1_POS_X, y = self.CAMERA2_POS_Y))
        self.frontcamera2 = self.world.spawn_actor(self.camera_bp,self.frontcamera2_init_trans,attach_to=self.vehicle)

        self.image_w = self.camera_bp.get_attribute('image_size_x').as_int()
        self.image_h = self.camera_bp.get_attribute('image_size_y').as_int()

        self.frontcamera1_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.frontcamera2_data = {'image': np.zeros((self.image_h,self.image_w,4), dtype=np.uint8)}
        self.frontcamera1.listen(lambda image: self.camera_callback(image,self.frontcamera1_data))
        self.frontcamera2.listen(lambda image: self.camera_callback(image,self.frontcamera2_data))
        self.model = YOLO("best.pt")
        

        #return values
        self.is_there_a_cyclist = 0 #False
        self.distance_to_bycicle = 20
        v = self.vehicle.get_velocity()
        self.speed = round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2), 0)
        self.brake = 0
        self.throttle = 0
        self.observation_space = np.array([
            self.is_there_a_cyclist,
            self.distance_to_bycicle/30,
            self.speed/50,
            self.brake,
            self.throttle,
            self.steering_angle
        ], dtype=np.float32)

        return self.observation_space, {}




    def camera_callback(self, image,data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))


    def step(self, action):

        print(action)
        throttle = action[0]
        brake = action[1]  

        #Check if there is a cyclist, if yes, the Reinforcement Algorithm will be used
        if self.is_there_a_cyclist:
            print("there is a cyclist")
            if throttle >= brake:
                brake = 0.0
            else:
                throttle = 0
        else:
            throttle, brake = self.update_control(self.max_speed)
        print(f"Throttle: {throttle}")
        print(f"Brake: {brake}")
   

        #Move the car
        self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer = float(self.steering_angle)))
        self.bicycle.apply_control(carla.VehicleControl(throttle=1))
        next_waypoint_location = self.route[self.curr_wp][0].transform.location
        #Check if we are close to the next coordinate
        if self.vehicle.get_transform().location.distance(self.route[self.curr_wp][0].transform.location) < 3:
            self.curr_wp += 1
        
        #deal with steering
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

        self.steering_angle = self.steering_angle/MAX_STEER_DEGREES
 

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

        # Tick the simulation
        self.world.tick()
        distance =self.detect()
        #We do our action after the tick, so we calculate with the fresh value
        if self.is_there_a_cyclist:
            reward, done, terminated = self.rewardsystem(distance)
        else:
            reward = 0
            done = False
            terminated = False
        
        print(f"Done: {done}")
        print(f"Terminated: {terminated}")
        print(f"Reward: {reward}")
        print(f"Speed of vehicle: {self.speed}")
        
        #print("Pausing for 0.5 second")
        #time.sleep(0.5)

        #self.is_there_a_cyclist = 0 #Becuase We have not implemented the YOLO algorithm
        #self.distance_to_bycicle = 30

        max_bicycledistance = 30 #This is theoretical max distance to cyclist, assume the camera can only detect this far
        max_speed = 50

        self.observation_space = np.array([
            self.is_there_a_cyclist,
            distance/max_bicycledistance,
            self.speed/max_speed,
            self.brake,
            self.throttle,
            self.steering_angle
        ], dtype=np.float32)

        return self.observation_space, reward, done, terminated, {}
    
    def detect(self):
        front1 = self.frontcamera1_data['image']
        front2 = self.frontcamera2_data['image']
        front1 = cv2.cvtColor(front1, cv2.COLOR_BGRA2BGR)
        front2 = cv2.cvtColor(front2, cv2.COLOR_BGRA2BGR)

        results_front1 = self.model(front1, verbose=False)
        results_front2 = self.model(front2)


        bicycles_front1 = []
        bicycles_front2 = []

        for result in results_front1:
            for box2 in result.boxes:
                # Extract box coordinates and other details
                x1, y1, x2, y2 = box2.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                bicycles_front1.append((center_x, center_y))
                conf = box2.conf[0]            # Confidence score
                cls = box2.cls[0]

                cv2.rectangle(front1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(front1, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for result in results_front2:
            for box2 in result.boxes:
                # Extract box coordinates and other details
                x1, y1, x2, y2 = box2.xyxy[0]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                bicycles_front2.append((center_x, center_y))
                conf = box2.conf[0]            # Confidence score
                cls = box2.cls[0]

                cv2.rectangle(front2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(cls)]}: {conf:.2f}"
                cv2.putText(front2, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        matched_bicycles_with_distances_front = self.match_bicycles_between_left_right(bicycles_front1, bicycles_front2)
        distance_front = 30

        for (left_bicycle, distance) in matched_bicycles_with_distances_front:
            distance_front = distance
        # Display distance for each matched bicycle on the left frame
        

        # cv2.imshow('RGB Camera', front2)
        # cv2.waitKey(1)

        
        if distance_front < 30:
            print(distance_front)
            self.is_there_a_cyclist = True
        else:
            self.is_there_a_cyclist = False
        return distance_front

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
    

    def rewardsystem(self, distance):
        v = self.vehicle.get_velocity()
        self.speed = round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2), 0)
        print(f"That is my speed: {self.speed}")

        #reward system
        reward = 0
        done = False
        terminated = False

        if self.brake > self.throttle and distance > self.safe_brake_distance:
            #Too early brake
            reward -= 100
        elif self.throttle > self.brake and distance > self.safe_brake_distance:
            #If no need to brake and we go
            if self.speed > 30:
                #If we go faster than speed limit
                reward -= 50
            else:
                #If going slover than speed limit
                reward += 50
        elif self.brake > self.throttle and self.safe_brake_distance > distance and self.too_close_brake_distance < distance:
            #If we brake in a good distance
            reward += 100
        elif self.brake > self.throttle and distance < self.too_close_brake_distance:
            #Too close to speed
            reward -= 100
        elif self.brake > self.throttle and distance < self.too_close_brake_distance:
            #Emergency braking, slight reward because not idea
            reward += 20



        #Check if there were an accident
        if self.collision_happened:
            terminated = True
            done = True
            reward -= 200
            self.destroy()
        distance_to_goal = self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location)
        #Check if it reached the goal
        if distance_to_goal < 4:
            done = True
            reward += 200
            self.destroy()

        self.episode_reward +=reward
        print(f"Episode reward: {self.episode_reward}")
        return reward, done, terminated

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
    
    def angle_between(self, v1, v2):
        return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))
    

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