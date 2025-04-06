from gymnasium.utils import seeding
import gymnasium
from gymnasium import spaces
import numpy as np
import carla
import math
import sys
import random
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
            low=np.array([0.0, 0.0, -1]),   # Minimum values: 0.0 throttle, 0.0 brake, turning_radius: -1
            high=np.array([1.0, 1.0, 1]),  # Maximum values: 1.0 throttle, 1.0 brake, turning radius: 1
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1]),  # min values for kmh, avg_distance, throttle, brake, steering angle
            high=np.array([1, 1, 1, 1, 1]),  # normalized max values
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


    def step(self, action):
            throttle = action[0]
            brake = action[1]
            self.steering_angle = action[2]
            self.steering_angle *= 40

            # Ensure only one of throttle or brake is applied
            if throttle >= brake:
                brake = 0.0
            else:
                throttle = 0.0

            

            #self.bicycle.apply_control(carla.VehicleControl(throttle=1))
            self.vehicle.apply_control(carla.VehicleControl(throttle=float(throttle), brake=float(brake), steer = float(self.steering_angle)))

            self.world.tick()
            self.episode_run_time += FIXED_DELTA_SECONDS

            reward = 0

            reward = self.movingandDetecting(reward)

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


            
            done = False
            terminated = False
            #distance_to_goal = self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location)
            distance_to_next_waypoint = self.vehicle.get_transform().location.distance(next_waypoint_location)
            

            print("That is my speed: {}".format(kmh))
            #Speeding Reward
            if 20 <= kmh <= 30:
                reward += 5*kmh
            elif 10 <= kmh < 19:
                reward += 4
            elif kmh < 10:
                reward -= 0.10  # Penalize being too slow
            if kmh > 25:
                reward -= 100  # Penalize speeding


            #Distance to goal reward
            progress = self.previousDistance - distance_to_next_waypoint
            #reward += 100*(progress/FIXED_DELTA_SECONDS)
            print(f"Ez volt a progress: {round(progress, 2)}")
            #Collision and Out-of-Bounds Penalties
            if self.collision_happened:
                reward -= 100
                done = True
                self.cleanup()
            
            if progress < 0.1:
                reward -= 5
                if self.episode_run_time > 20:
                    reward -= 100
                    done = True
                    self.cleanup()
            elif progress > 0.1 and self.episode_run_time > 0.8: #Becuase, When it falls to the ground it makes a lot of progres
                reward += progress*10
            

            #Reaching the end point            
            if self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location) < 6:
                reward += 50
                done = True
                self.cleanup()
                print(f"Point of the given episode: {self.episode_point+50}")

            print(f"Episode Run Time: {self.episode_run_time}")


            # Update previous distance
            self.previousDistance = distance_to_next_waypoint

            #This is some test, to see how it works
            # Normalize features to create observation
            max_speed = 30  # Assume max speed is 30 km/h
            max_bicycle_distance = 30  # Assume max bicycle distance for normalization
            #max_endpoint_distance = 500.0  # Assume max endpoint distance for normalization
            self.steering_angle  /= 40

            obs = np.array([
                kmh / max_speed,
                self.avg_distance / max_bicycle_distance,
                throttle,
                brake,
                self.steering_angle,
            ], dtype=np.float32)

            self.episode_point += reward
            print(f" Episode Point: {self.episode_point}")

            return obs, reward, done, terminated, {}


    def reset(self, seed=None, options=None):
        
            print(f"We are in this mode: {self.eval_mode}")
            self.episode_point = 0
            self.previous_speed = 0
            self.episode_run_time = 0
            self.seed(seed)
            self.bicycle_speed = random.uniform(0.5, 1)
            self.previousDistance = 100
            self.vehicle = None
            self.bicycle = None
            self.curr_wp = 2
            self.cleanup()
            #self.bicycleorigin()
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

            self.targetid = 4
            self.targetPoint = self.spawn_points[self.targetid]

            self.point_A = self.vehicle_start_point.location
            self.point_B = self.targetPoint.location


            self.sampling_resolution = 3
            self.grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)

            self.route = self.grp.trace_route(self.point_A, self.point_B)

            self.world.tick()
            self.movingandDetecting()


            v = self.vehicle.get_velocity()
            kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            obs = np.array([kmh / 50.0, self.avg_distance / 100.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return obs, {}

    def seed(self, seed=None):
                self.np_random, seed = seeding.np_random(seed)
                return [seed]
            

    def movingandDetecting(self, reward = 0):
            if self.vehicle.get_transform().location.distance(self.route[self.curr_wp][0].transform.location) < 3:
                self.curr_wp += 1
                reward += 1000
            return reward


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