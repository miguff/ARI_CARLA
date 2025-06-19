import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import os
from datetime import datetime
from torchviz import make_dot



class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, seed: int = 42):
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
        self.seed(seed)
        print("Network random")
        print(print(T.rand(1)))
        

    def forward(self, observation: T.tensor):
        state = observation.to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def seed(self, seed):
        self.seed_value = seed

        # Seed numpy random generators
        self.np_random = np.random.RandomState(seed)
        
        # If using PyTorch, seed the torch RNG too
        if T:
            T.manual_seed(seed)
            T.cuda.manual_seed_all(seed)
    

class ActorCriticAgent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2, layer1_size=64, layer2_size=64, n_outputs=1, 
                 writer: Optional[SummaryWriter] = None, MODELSAVE: Optional[str] = None, FilenamePrefix: Optional[str] = None,
                 SaveGraph: Optional[str] = "TorchVizImages", NumberofIterSave: Optional[int] = 500, seed: int = 42):
        
        self.seed(seed)
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = Network(alpha, input_dims, layer1_size, layer2_size, n_actions)
        self.critic = Network(beta, input_dims, layer1_size, layer2_size, 1)
        self.writer = writer
        self.MODELSAVE = MODELSAVE
        self.Filenameprefix = FilenamePrefix
        self.SaveGraph = SaveGraph
        self.NumberofIterSave = NumberofIterSave
        print("Actor Critic random")
        print(print(T.rand(1)))

    def seed(self, seed):
        self.seed_value = seed

        # Seed numpy random generators
        self.np_random = np.random.RandomState(seed)
        
        # If using PyTorch, seed the torch RNG too
        if T:
            T.manual_seed(seed)
            T.cuda.manual_seed_all(seed)

    def choose_action(self, observation):
        print("Observation")
        print(observation)
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

        
        critic_value_ = self.critic.forward(new_state) #We put through the new state, again, to get a state-value. How good is that state
        critic_value = self.critic.forward(state) #We put through the current state, to check the goodness of it

        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        #Temporal Fifference loss
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value



        #We want to minimize the delta
        actor_loss = -self.log_probs * delta.detach()
        critic_loss = delta**2
        loss = actor_loss + critic_loss

    #     if timestamp % self.NumberofIterSave == 0:
    #         save_dir = self.SaveGraph
    #         os.makedirs(save_dir, exist_ok=True)

    #         # Actor graph
    #         actor_graph_path = os.path.join(save_dir, f"actor_graph_{timestamp}.png")
    #         make_dot(actor_loss, params=dict(self.actor.named_parameters())).render(
    #             actor_graph_path[:-4], format="png"
    #         )

    #         # Critic graph
    #         critic_graph_path = os.path.join(save_dir, f"critic_graph_{timestamp}.png")
    #         make_dot(critic_loss, params=dict(self.critic.named_parameters())).render(
    #             critic_graph_path[:-4], format="png"
    # )


        if self.writer and timestamp == 0:
            dummy_input = state.unsqueeze(0) if state.dim() == 1 else state
            self.writer.add_graph(self.actor, dummy_input)
            self.writer.add_graph(self.critic, dummy_input)




        #Add to summary
        if self.writer:
            self.writer.add_scalar("Actor Loss/Train", actor_loss, timestamp)
            self.writer.add_scalar("Critic Loss/Train", critic_loss, timestamp)
            self.writer.add_scalar("Loss/Train", loss, timestamp)
            self.writer.flush()

        self.actor.optimizer.zero_grad()
        # actor_loss.backward()
        # self.actor.optimizer.step()
        
        
        self.critic.optimizer.zero_grad()
        # critic_loss.backward()
        # self.critic.optimizer.step()

        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        

    def save_models(self, reward: str):
        if not os.path.exists(self.MODELSAVE):
            os.makedirs(self.MODELSAVE)
        actor_path = os.path.join(self.MODELSAVE, f'./{self.Filenameprefix}_actor_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{reward}.pth')
        critic_path = os.path.join(self.MODELSAVE, f'./{self.Filenameprefix}_critic_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{reward}.pth')
        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)
        print(f"Saved models to {self.MODELSAVE}")
