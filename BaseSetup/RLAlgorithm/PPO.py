from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from .networks import ActorNetworkPPO, CriticNetworkPPO, CarEncoder
import torch as T
device = T.device("cuda" if T.cuda.is_available() else "cpu")
import numpy as np
import torch.nn.functional as F



class PPOAgent():
    def __init__(self,
                writer: Optional[SummaryWriter] = None,
                clip_epsilon: float = 0.2,
                gamma: float = 0.99,
                lmbda: float = 0.9,
                entropy_eps: float = 1e-4,
                embed_dim: int = 64,
                input_dims: int = 4,
                epochs: int = 2,
                batch_size: int = 64,
                value_coef: float = 0.5,
                entropy_coef: float = 0.0,
                alpha: float = 3e-4,
                beta: float = 1e-3,
                seed: int = 42,
                FilenamePrefix: Optional[str] = None,
                MaxBufferSize: int  = 100):
        
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_eps = entropy_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.MaxBufferSize = MaxBufferSize

        #// To store the values
        self.buffer = {
            "states" : [],
            "actions" : [],
            "logprobs" : [],
            "values" : [],
            "rewards" : [],
            "dones" : [],
            "next_values" : []
        }
        self.encoder = CarEncoder(in_dim=input_dims[0],
                                  hidden_dim=64,
                                  embed_dim=embed_dim).to(device)
        self.actor = ActorNetworkPPO(input_dims=embed_dim, n_actions=1).to(device)
        self.critic = CriticNetworkPPO(input_dims=embed_dim).to(device)

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef


        self.actor_opt = T.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_opt = T.optim.Adam(self.critic.parameters(), lr=beta)

    def seed(self, seed):
        self.seed_value = seed

        # Seed numpy random generators
        self.np_random = np.random.RandomState(seed)
        
        # If using PyTorch, seed the torch RNG too
        if T:
            T.manual_seed(seed)
            T.cuda.manual_seed_all(seed)

    def _encode_state(self, state):
        state = T.tensor(state, dtype=T.float32, device=device)
        state = state.unsqueeze(0)
        with T.no_grad():
            emb = self.encoder(state)
        return emb


    def choose_action(self, state):
        emb = self._encode_state(state)

        #// Add the current state to the buffer
        self.buffer["states"].append(emb)


        with T.no_grad():
            action, logprob = self.actor.act(emb)
            value = self.critic(emb).item()
        value = T.tensor(value)
        self.buffer["actions"].append(T.tensor(action))
        self.buffer["logprobs"].append(logprob)
        self.buffer["values"].append(value)
        return action
    

    def reset_hidden(self):
        pass

    def _compute_gae(self, rewards, dones, values, next_values):
        """
        Compute GAE-Lambda advantages and returns.
        rewards: (T,)
        dones: (T,)  boolean or 0/1
        values: (T,)
        next_values: (T,) value at next state, for each step
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lmbda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns
    
    def next_value(self, state):
        emb = self._encode_state(state)
        #value = self.critic(emb).item()
        value = self.critic(emb)
        value = T.tensor(value)
        self.buffer["next_values"].append(value)

    def learn(self, objectssata, action, reward, objectssata_, done, TRAINING_STEP, Train):
        self.buffer["dones"].append(T.tensor(int(done)))
        self.buffer["rewards"].append(T.tensor(reward))
        self.next_value(objectssata_)


        #// Check if train or valid
        if Train == 0:
            return

        #// Check if we should train or just rollout
        if len(self.buffer["next_values"]) < self.MaxBufferSize:
            return
        
        states = self.buffer["states"]
        actions = self.buffer["actions"]
        old_logprobs = self.buffer["logprobs"]
        rewards = self.buffer["rewards"]
        sum_rewards = T.stack(rewards, dim=0).sum(dim=0).sum(dim=0)
        dones = self.buffer["dones"]
        values = self.buffer["values"]
        next_values = self.buffer["next_values"]


        advantages, returns = self._compute_gae(rewards, dones, values, next_values)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_t = T.tensor(advantages, dtype=T.float32, device=device)
        returns_t = T.tensor(returns, dtype=T.float32, device=device)
        states_t = T.stack(states)
        actions_t = T.stack(actions)
        old_logprobs_t = T.stack(old_logprobs)

        dataset_size = len(states)
        indices = np.arange(dataset_size)


        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_logprobs = old_logprobs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                new_logprobs, entropy = self.actor.evaluate_actions(b_states, b_actions)
                values_pred = self.critic(b_states)
                

                ratio = T.exp(new_logprobs - b_old_logprobs)

                surr1 = ratio * b_advantages
                surr2 = T.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -T.min(surr1, surr2).mean()


                value_loss = F.mse_loss(values_pred, b_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.actor.zero_grad()
                self.critic.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()


        self.buffer = {
            "states" : [],
            "actions" : [],
            "logprobs" : [],
            "values" : [],
            "rewards" : [],
            "dones" : [],
            "next_values" : []
        }

    def save_models(self, total_reward: str):
        pass
