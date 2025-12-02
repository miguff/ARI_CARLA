# import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import os
from datetime import datetime
# from torchviz import make_dot



class Network(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_outputs,
                 lstm_hidden_dims=128, seed: int = 42):
        super(Network, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_outputs = n_outputs
        self.lstm_hidden_dims = lstm_hidden_dims

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)

        self.lstm = nn.LSTM(
            input_size=self.fc1_dims,
            hidden_size=self.lstm_hidden_dims,
            num_layers=1,
            batch_first=True
        )

        self.fc2 = nn.Linear(self.lstm_hidden_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

        self.seed(seed)
        print("Network random")
        print(T.rand(1))  # no need for print(print(...))

    def forward(self, observation: T.Tensor, hidden=None):
        """
        observation:
          - (batch, input_dims)     OR
          - (batch, seq_len, input_dims)

        hidden: (h, c) or None
        """
        observation = observation.to(self.device)

        # If observation is (batch, input_dims) -> make it (batch, 1, input_dims)
        if observation.dim() == 2:
            observation = observation.unsqueeze(1)

        # (batch, seq_len, input_dims) -> (batch, seq_len, fc1_dims)
        x = F.relu(self.fc1(observation))

        # LSTM: (batch, seq_len, lstm_hidden_dims)
        x_out, hidden = self.lstm(x, hidden)

        # Take last timestep
        x_last = x_out[:, -1, :]

        x = F.relu(self.fc2(x_last))
        x = self.fc3(x)

        # x has shape (batch, n_outputs)
        return x, hidden

    def seed(self, seed):
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)
    

class ActorCriticAgent(object):
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        gamma=0.99,
        n_actions=1,  # continuous 1D action (e.g. acceleration) 
        layer1_size=64,
        layer2_size=64,
        writer: Optional[SummaryWriter] = None,
        MODELSAVE: Optional[str] = None,
        FilenamePrefix: Optional[str] = None,
        SaveGraph: Optional[str] = "TorchVizImages",
        NumberofIterSave: Optional[int] = 500,
        seed: int = 42
    ):
        self.seed(seed)
        self.gamma = gamma
        self.log_probs = None
        self.writer = writer
        self.MODELSAVE = MODELSAVE
        self.Filenameprefix = FilenamePrefix
        self.SaveGraph = SaveGraph
        self.NumberofIterSave = NumberofIterSave

        # Actor outputs [mu, log_sigma]
        self.actor = Network(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_outputs=2 * n_actions  # mu and log_sigma
        )

        # Critic outputs value
        self.critic = Network(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_outputs=1
        )

        # Recurrent hidden states (start as None, reset each episode)
        self.actor_hidden = None
        self.critic_hidden = None

        print("Actor Critic random")
        print(T.rand(1))

    def reset_hidden(self):
        """Call this at the beginning of each episode."""
        self.actor_hidden = None
        self.critic_hidden = None

    def seed(self, seed):
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)

    def choose_action(self, observation):
        # Expect observation as np.array or tensor shape (input_dims,)
        if not isinstance(observation, T.Tensor):
            observation = T.tensor(observation, dtype=T.float32)

        observation = observation.unsqueeze(0) if observation.dim() == 1 else observation
        # observation now (batch=1, input_dims)

        actor_out, self.actor_hidden = self.actor(observation, self.actor_hidden)
        self.actor_hidden = self._detach_hidden(self.actor_hidden)
        # actor_out shape: (1, 2 * n_actions)
        actor_out = actor_out.squeeze(0)  # (2 * n_actions,)

        # Split into mu and log_sigma
        n = actor_out.shape[0] // 2
        mu = actor_out[:n]
        log_sigma = actor_out[n:]
        sigma = T.exp(log_sigma)

        dist = T.distributions.Normal(mu, sigma)
        # Sample one action
        action = dist.sample()
        self.log_probs = dist.log_prob(action).sum()  # sum if multi-dim action

        # squash to [-1, 1]
        action = T.tanh(action)

        return action.detach().cpu().item()  # or .item() if scalar

    def _detach_hidden(self, hidden):
        if hidden is None:
            return None
        h, c = hidden
        return (h.detach(), c.detach())


    def learn(self, state, action, reward, new_state, done, timestamp, Train=True):
        if not Train:
            return

        # Convert states to tensors
        if not isinstance(state, T.Tensor):
            state = T.tensor(state, dtype=T.float32)
        if not isinstance(new_state, T.Tensor):
            new_state = T.tensor(new_state, dtype=T.float32)

        # Add batch dimension if needed: (features,) -> (1, features)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if new_state.dim() == 1:
            new_state = new_state.unsqueeze(0)

        # Detach recurrent history for critic
        self.critic_hidden = self._detach_hidden(self.critic_hidden)

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        # ---- Critic value for current state (with grad) ----
        critic_value, self.critic_hidden = self.critic(state, self.critic_hidden)
        critic_value = critic_value.squeeze(-1)  # (batch,) -> scalar for batch=1

        # ---- Critic value for next state (no grad) ----
        with T.no_grad():
            critic_value_next, _ = self.critic(new_state, self.critic_hidden)
            critic_value_next = critic_value_next.squeeze(-1)

        reward = T.tensor(reward, dtype=T.float32).to(self.actor.device)

        # TD(0) target
        target = reward + self.gamma * critic_value_next * (1 - int(done))
        delta = target - critic_value

        # Actor + critic losses
        actor_loss = -self.log_probs * delta.detach()
        critic_loss = delta.pow(2)
        loss = actor_loss + critic_loss

        # Optional graph logging
        if self.writer and timestamp == 0:
            dummy_input = state
            self.writer.add_graph(self.actor, dummy_input)
            self.writer.add_graph(self.critic, dummy_input)

        # Scalars to TensorBoard
        if self.writer:
            self.writer.add_scalar("Actor Loss/Train", actor_loss.item(), timestamp)
            self.writer.add_scalar("Critic Loss/Train", critic_loss.item(), timestamp)
            self.writer.add_scalar("Loss/Train", loss.item(), timestamp)
            self.writer.flush()

        loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

        if done:
            # reset LSTM states at episode end
            self.reset_hidden()

    def save_models(self, reward: str):
        if not os.path.exists(self.MODELSAVE):
            os.makedirs(self.MODELSAVE)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        actor_path = os.path.join(self.MODELSAVE, f'{self.Filenameprefix}_actor_{ts}_{reward}.pth')
        critic_path = os.path.join(self.MODELSAVE, f'{self.Filenameprefix}_critic_{ts}_{reward}.pth')
        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)
        print(f"Saved models to {self.MODELSAVE}")
