import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional


class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Network, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation: T.tensor):
        state = observation.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class ActorCriticAgent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2, layer1_size=64, layer2_size=64, n_outputs=1, writer: Optional[SummaryWriter] = None):
        

        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = Network(alpha, input_dims, layer1_size, layer2_size, n_actions)
        self.critic = Network(beta, input_dims, layer1_size, layer2_size, 1)
        self.writer = writer


    def choose_action(self, observation):
        mu, sigma = self.actor.forward(observation)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma) #Calculate the probabilities of taking the action. The mu and sigma are the values, that the agent will try to maximize
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs])) #Single sample, of a normal distribution. chooese a random number from a normal distibution
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs) #We insert the value into tanh to get a value betwwwen -1 and 1

        return action.item()
    

    def learn(self, state, reward, new_state, done, timestamp):
                
        """
        Temporal Difference type, it learns after every timestamp.
        """

        #Replay memory can be used
        
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()


        critic_value_ = self.critic.forward(new_state) #We put through the new state, again, to get a state-value. How good is that state
        critic_value = self.critic.forward(state) #We put through the current state, to check the goodness of it

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        #Temporal Fifference loss
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        #We want to minimize the delta
        actor_loss = -self.log_probs * delta
        critic_loss = delta**2
        loss = actor_loss + critic_loss

        #Add to summary
        if self.writer:
            self.writer.add_scalar("Actor Loss/Train", actor_loss, timestamp)
            self.writer.add_scalar("Critic Loss/Train", critic_loss, timestamp)
            self.writer.add_scalar("Loss/Train", loss, timestamp)
            self.writer.flush()
        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

