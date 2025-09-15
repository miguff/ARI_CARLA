import os
import numpy as np
import torch as T
import torch.nn.functional as F
from .networks import ActorNetwork, CriticNetwork
from .Noise import OUActionNoise
from .buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from datetime import datetime

class DDPGAgent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=100, fc1_dims=400, fc2_dims=300, 
                 batch_size=64, writer: Optional[SummaryWriter] = None, seed: int = 42, FilenamePrefix: Optional[str] = None, MODELSAVE: Optional[str] = None):
        
        """
        Initialize the DDPG agent.

        Parameters:
        - alpha: learning rate for the actor network
        - beta: learning rate for the critic network
        - input_dims: dimensionality of the state space
        - tau: parameter for soft updates of target networks
        - n_actions: number of actions in the action space
        - gamma: discount factor for future rewards
        - max_size: max size of the replay buffer
        - fc1_dims, fc2_dims: dimensions of the first and second fully connected layers
        - batch_size: number of samples per learning update
        """


        self.gamma = gamma # Discount factor for future rewards
        self.tau = tau  # Soft update coefficient for target networks
        self.batch_size = batch_size
        self.alpha = alpha # Actor learning rate
        self.beta = beta # Critic learning rate
        self.writer = writer
        self.Filenameprefix = FilenamePrefix
        self.seed(seed)
        self.MODELSAVE = MODELSAVE

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                n_actions=n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def seed(self, seed):
        self.seed_value = seed

        # Seed numpy random generators
        self.np_random = np.random.RandomState(seed)
        
        # If using PyTorch, seed the torch RNG too
        if T:
            T.manual_seed(seed)
            T.cuda.manual_seed_all(seed)

    def choose_action(self, observation):
        """
        Select an action for the given state, adding noise for exploration.

        Parameters:
        - observation: current state of the environment

        Returns:
        - action: action chosen by the actor network + exploration noise
        """

        self.actor.eval()
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        # Add exploration noise sampled from OU noise process
        mu_prime = mu + T.tensor(self.noise(), 
                                    dtype=T.float).to(self.actor.device)
        self.actor.train()
        # Return the noisy action as a numpy array
        return mu_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, state_, done):
        """
        Store a transition (experience) in the replay buffer.

        Parameters:
        - state: current state
        - action: action taken
        - reward: reward received after action
        - state_: next state after action
        - done: boolean indicating if episode ended
        """
        self.memory.store_transition(state, action, reward, state_, done)
        print(f"Number of element in: {self.memory.mem_cntr}")
    def save_models(self, reward: str):
        """
        Save the weights of all the networks to disk.
        """
        actor_path = os.path.join(self.MODELSAVE, f'./{self.Filenameprefix}_actor_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{reward}.pth')
        critic_path = os.path.join(self.MODELSAVE, f'./{self.Filenameprefix}_critic_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{reward}.pth')
        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)
        target_actor_path = os.path.join(self.MODELSAVE, f'./{self.Filenameprefix}_target_actor_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{reward}.pth')
        target_critic_path = os.path.join(self.MODELSAVE, f'./{self.Filenameprefix}_target_critic_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{reward}.pth')
        T.save(self.target_actor.state_dict(), target_actor_path)
        T.save(self.target_critic.state_dict(), target_critic_path)

        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        """
        Load the weights of all the networks from disk.
        """
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self, observation, action, reward, observation_, done, timestamp):
        """
        Sample a batch of experiences from memory and update networks.

        Implements the DDPG learning algorithm:
        - Update critic by minimizing MSE between predicted Q-values and target Q-values
        - Update actor using policy gradient to maximize expected Q-value
        - Soft update the target networks
        """
        self.remember(observation, action, reward, observation_, done)

        if self.memory.mem_cntr < self.batch_size:
            return
        print("Now I am learning")
        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)


        # Compute target actions from target actor network for next states
        target_actions = self.target_actor.forward(states_) #Here we choose another action, that is in the future value
        # Compute target critic values for next states and target actions
        critic_value_ = self.target_critic.forward(states_, target_actions) #Calculate the corresponding critic
        # Compute current critic values for states and actions taken
        critic_value = self.critic.forward(states, actions)

        # If done (episode ended), zero out the target critic value
        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)


        # Calculate the target Q-values (rewards + discounted future Q-values)
        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1) # Reshape for loss computation


        if self.writer and timestamp == 0:
            dummy_input = observation.unsqueeze(0) if observation.dim() == 1 else observation
            self.writer.add_graph(self.actor, dummy_input)
            self.writer.add_graph(self.critic, dummy_input)


        # Update Critic network by minimizing MSE loss between predicted and target Q-values
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # Update Actor network by maximizing the expected Q-value (policy gradient)
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        #Add to summary
        if self.writer:
            self.writer.add_scalar("Actor Loss/Train", actor_loss, timestamp)
            self.writer.add_scalar("Critic Loss/Train", critic_loss, timestamp)
            self.writer.flush()


        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        """
        Perform soft update of the target networks using parameter tau.

        New target parameters = tau * current network parameters + (1 - tau) * old target parameters
        """
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        # Update target critic parameters with soft update rule
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
