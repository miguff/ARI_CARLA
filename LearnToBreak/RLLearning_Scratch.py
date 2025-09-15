import carla
import numpy as np
import cv2
import time
import sys
sys.path.append('F:\CARLA\Windows\CARLA_0.9.15\PythonAPI\carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d
import math
from collections import deque
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt

FIXED_DELTA_SECONDS = 1
SPEED_THRESHOLD = 2
PREFERRED_SPEED= 30
REPLAY_MEMORY_SIZE =5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5 #Update every 5 episodes

MIN_REWARD = -200 
GAMMA = 0.99
EPISODES = 20
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001
AGGREAGATE_STATS_EVERY = 10
MODEL_NAME = "Braking"



class CarEnv():
    def __init__(self) -> None:
        super(CarEnv).__init__()
        self.state_dim = 6 #Is there a cyclist, Distance to bicycle, speed, brake value, throttle value, inntend to turn
        
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

    def carorigin(self):
        vehicle_bp = self.world.get_blueprint_library().filter('*mini*')
        self.vehicle_start_point = self.spawn_points[94]
        self.vehicle = self.world.try_spawn_actor(vehicle_bp[0], self.vehicle_start_point)


    def bicycleorigin(self):
        bicycle_bp = self.world.get_blueprint_library().filter('*crossbike*')
        bicycle_start_point = self.spawn_points[99]

        self.bicycle = self.world.try_spawn_actor(bicycle_bp[0], bicycle_start_point)
        
        new_location = bicycle_start_point.location + carla.Location(y=20)
        new_rotation = carla.Rotation(pitch=0, yaw=bicycle_start_point.rotation.yaw + 0, roll=0)

        
        bicyclepos = carla.Transform(new_location, new_rotation)
        self.bicycle.set_transform(bicyclepos)

    # def maintain_speed(self) -> float:
    
    #     if self.speed >= PREFERRED_SPEED:
    #         return 0
    #     elif self.speed < PREFERRED_SPEED - SPEED_THRESHOLD:
    #         return 0.8
    #     else:
    #         return 0.4



    def destroy(self):#Destroying the existing things
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()


    def process_collision(self, event):
        # Extract collision data
        self.other_actor = event.other_actor
        self.impulse = event.normal_impulse
        self.collision_location = event.transform.location
        print(f"Collision with {self.other_actor.type_id}")
        print(f"Impulse: {self.impulse}")
        print(f"Location: {self.collision_location}")
        self.collision_happened = True



    def reset(self):
        """
        This action will be called, to reset the environment, when a collision happend
        """
        self.destroy()
        #self.bicycleorigin()
        self.carorigin()

        self.collision_detector_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
                self.collision_detector_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
        self.collision_sensor.listen(lambda event: self.process_collision(event))
        self.collision_happened = False

        #Set the goal state
        self.targetid = 55
        self.targetPoint = self.spawn_points[self.targetid]
        self.point_A = self.vehicle_start_point.location
        self.point_B = self.targetPoint.location
        self.sampling_resolution = 3
        self.grp = GlobalRoutePlanner(self.world.get_map(), self.sampling_resolution)
        self.route = self.grp.trace_route(self.point_A, self.point_B)


        #setup the Front Cameras
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

        #return values
        self.is_there_a_cyclist = 0 #False
        self.distance_to_bycicle = 20
        v = self.vehicle.get_velocity()
        self.speed = round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2), 0)
        self.brake = 0
        self.throttle = 0
        self.intend_to_turn = 0 #False
        self.reward = 0
        self.done = False
        self.terminated = False
        self.observation_space = np.array([
            self.is_there_a_cyclist,
            self.distance_to_bycicle,
            self.speed,
            self.brake,
            self.throttle,
            self.intend_to_turn
        ], dtype=np.float32)

        return self.observation_space, self.reward, self.done, self.terminated, {}




    def camera_callback(self, image,data_dict):
        data_dict['image'] = np.reshape(np.copy(image.raw_data),(image.height,image.width,4))


    def step(self, action: float, action_index: int):

        move_frames = int(self.move_duration / self.frame_time)
        #print(f"Move frame: {move_frames}")
        
  
        for _ in range(move_frames):
            # Apply control if not using autopilot
            
            if action_index == 1:
                self.vehicle.apply_control(carla.VehicleControl(throttle = action))
                self.throttle = action
                self.brake = 0
            else:
                self.vehicle.apply_control(carla.VehicleControl(brake = action))
                self.throttle = 0
                self.brake = action
            #For training regulation, we take it out
            #self.bicycle.apply_control(carla.VehicleControl(throttle=1))
            # Tick the simulation
            self.world.tick()

        
        front1 = self.frontcamera1_data['image']
        front1 = cv2.cvtColor(front1, cv2.COLOR_BGRA2BGR)
        front2 = self.frontcamera2_data['image']
        front2 = cv2.cvtColor(front2, cv2.COLOR_BGRA2BGR)

        cv2.imshow('RGB Camera - Front 1', front1)
        cv2.imshow('RGB Camera - Front 2', front2)

        

        #print("Pausing for 1 second")
        #time.sleep(0.5)

        #Check if there were an accident
        if self.collision_happened:
            self.terminated = True
            self.done = True
            self.reward -= 200
        distance_to_goal = self.vehicle.get_transform().location.distance(self.route[-1][0].transform.location)
        #Check if it reached the goal
        if distance_to_goal < 8:
            self.done = True
            self.reward += 200
        



        self.is_there_a_cyclist = 0 #Becuase We have not implemented the YOLO algorithm
        self.distance_to_bycicle = 20

        v = self.vehicle.get_velocity()
        self.speed = round(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2), 0)
        print(f"That is my speed: {self.speed}")
        if self.speed > 30:
            self.reward -= (self.speed - 30)*5
        elif self.speed < 5:
            self.reward -= (25 - self.speed)*5
        elif self.speed >25 and self.speed < 30:
            self.reward += 40
        
        
        if self.speed > 50:
            self.reward -= 50
            self.terminated = True
            self.done = True

        self.intend_to_turn = 0

        self.observation_space = np.array([
            self.is_there_a_cyclist,
            self.distance_to_bycicle,
            self.speed,
            self.brake,
            self.throttle,
            self.intend_to_turn
        ], dtype=np.float32)

        return self.observation_space, self.reward, self.done, self.terminated, {}



    

# class DQNAgent:
#     def __init__(self):
#         self.model = self.create_model()
#         self.target_model = self.create_model()
#         self.target_model.set_weights(self.model.get_weights())

#         self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
#         self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

#         self.target_update_counter = 0

#         self.terminate = False

#         self.last_logged_episode = 0

#         self.training_initialized = False

#     def create_model(self):
#         model = nn.Sequential()



class DuelingPolicy(nn.Module):
    def __init__(self, n_observations, num_hidden=128):
        super(DuelingPolicy, self).__init__()

        self.BrakingNet = nn.Sequential(
            nn.Linear(n_observations, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 2),
            nn.Softmax()
        
        )

    def forward(self, x):   
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        print(self.BrakingNet(x))
        return  self.BrakingNet(x) 


class PPO:
    def __init__(self, policy_net, optimizer, epsilon=0.2, gamma=0.99, lam=0.95, lr=3e-4):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.epsilon = epsilon  # Clipping value for PPO
        self.gamma = gamma  # Discount factor
        self.lam = lam  # GAE lambda
        self.lr = lr  # Learning rate

    def update(self, states, actions, log_probs_old, rewards, values, next_value, done):
        advantages = self.compute_advantages(rewards, values, next_value, done)
        
        # Convert advantages to tensor
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.policy_net.device)
        
        # Convert states and actions to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.policy_net.device)
        
        # Get action probabilities from the policy network (throttle and brake probabilities)
        action_probs = self.policy_net(states)  # [throttle_prob, brake_prob]
        
        # Assuming actions is a list of tuples (throttle_value, brake_value) from the environment
        throttle_actions, brake_actions = zip(*actions)
        throttle_actions = torch.tensor(throttle_actions, dtype=torch.float32).to(self.policy_net.device)
        brake_actions = torch.tensor(brake_actions, dtype=torch.float32).to(self.policy_net.device)
        
        # Compute the probability for each action based on the network output
        throttle_prob = action_probs[:, 0]
        brake_prob = action_probs[:, 1]
        
        # Compute the log probabilities for throttle and brake actions
        log_probs_throttle = torch.log(throttle_prob)
        log_probs_brake = torch.log(brake_prob)
        
        # Select the appropriate log probability based on the action chosen
        log_probs_selected = log_probs_throttle * throttle_actions + log_probs_brake * brake_actions
        
        # Compute the ratio (new / old policy)
        ratio = torch.exp(log_probs_selected - log_probs_old)
        
        # Compute the surrogate loss (clipped version)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        loss = -torch.min(surr1, surr2).mean()
        
        # Compute the entropy loss to encourage exploration
        entropy_loss = -torch.mean(action_probs * torch.log(action_probs))
        
        # Combine the loss and entropy loss
        total_loss = loss - 0.01 * entropy_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


    def compute_advantages(self, rewards, values, next_value, done):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - done[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - done[t]) * gae
            advantages.insert(0, gae)
            next_value = values[t]
        return advantages  # This should now be a list of scalar values


    def select_action(self, state):
        action_probs = self.policy_net(state)
        action_max_prob, action_index = torch.max(action_probs, dim=1)
        return action_max_prob.item(), action_index.item()

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    Environment = CarEnv()
    policy = DuelingPolicy(Environment.state_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    ppo = PPO(policy, optimizer)

    states, actions, log_probs_old, rewards, values, next_value, done = [], [], [], [], [], [], []
    
    observation_space, reward, done_episode, terminated, info = Environment.reset()
    
    # Get the action probabilities and index from PPO
    action_prob, action_index = ppo.select_action(observation_space)
    
    # Store the log probabilities of the chosen action
    log_probs_old.append(torch.log(torch.tensor(action_prob)))  # Store the log probability for the chosen action
    
    for i in range(20):
        # Step through the environment with the selected action
        next_observation_space, reward, done_episode, terminated, info = Environment.step(action_prob, action_index)
        
        rewards.append(reward)
        values.append(policy(observation_space))  # Store the value (can use a separate value network)
        done.append(done_episode)
        states.append(observation_space)
        
        # Store the action (throttle or brake)
        actions.append((action_prob if action_index == 0 else 0, action_prob if action_index == 1 else 0))
        
        # Update the current observation for the next iteration
        observation_space = next_observation_space
        
        if done_episode:
            break
    
    # Get the final value prediction for the last observation
    next_value = policy(observation_space)
    
    # Update the PPO agent
    ppo.update(states, actions, log_probs_old, rewards, values, next_value, done)
    

    # all_losses = []

    # observation_space, reward, done, terminated, info = Environment.reset()
    # state_tensor = torch.FloatTensor(observation_space)
    # action, out = policy(state_tensor)
    # print(action)
    # print(out)
    # exit()

    # for episode in range(EPISODES):
    #     observation_space, reward, done, terminated, info = Environment.reset()
    #     log_probs = []
    #     rewards = []
    #     print(f"This is the {episode+1} episode")
    #     for time in range(1000): #10 step lookahead

    #         #Choose an action
    #         state_tensor = torch.FloatTensor(observation_space)
    #         action = policy(state_tensor)
            
    #         std = torch.ones_like(action) * 0.05  
    #         dist = torch.distributions.Normal(action, std)
    #         sampled_action = dist.rsample()
            
    #         squashed_action = torch.tanh(sampled_action)
    #         print(squashed_action)
    #         log_prob = dist.log_prob(sampled_action)
    #         log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-7)
    #         log_prob = log_prob.sum(-1, keepdim=True)

    #         env_action = squashed_action.item()
    #         netx_observation_space, reward, done, terminated, info = Environment.step(env_action)
    #         print("REward")
    #         print(reward)
    #         log_probs.append(log_prob)
    #         rewards.append(reward)

    #         observation_space = netx_observation_space

    #         if cv2.waitKey(1) == ord('q'):
    #             break
    #         if terminated:
    #             print("I crashed")
    #             break
    #         elif done:
    #             print("I reached the goal")
    #             break
        

    #     returns = []
    #     G = 0
    #     for r in reversed(rewards):
    #         G = r + GAMMA*G
    #         returns.insert(0, G)
        
    #     # print(rewards)
    #     # print(returns)

    #     returns = torch.tensor(returns)
    #     returns = (returns- returns.mean()) / (returns.std() + 1e-8)

    #     loss = +torch.stack(log_probs) * returns
    #     loss = loss.sum()
    #     all_losses.append(loss.detach().numpy())

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # plt.plot(all_losses)
    # plt.show()








if __name__ == "__main__":
    main()