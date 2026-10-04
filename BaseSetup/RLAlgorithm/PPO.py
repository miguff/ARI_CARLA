import os
from datetime import datetime
from typing import Optional

import numpy as np
import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .networks import ActorNetworkPPO, CriticNetworkPPO, CarEncoder

device = T.device("cuda" if T.cuda.is_available() else "cpu")


class PPOAgent:
    """PPO agent with Beta-distribution policy on [-1, 1].

    The action is a scalar in [-1, 1]:
        action > 0  ->  throttle
        action < 0  ->  brake

    Key fix vs. previous version: the rollout buffer is flushed and a PPO
    update is performed at *every* episode end, regardless of episode length.
    This ensures the policy actually learns from short episodes.
    """

    def __init__(
        self,
        writer: Optional[SummaryWriter] = None,
        clip_epsilon: float = 0.2,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        embed_dim: int = 64,
        input_dims: int = 6,
        n_actions: int = 1,
        epochs: int = 4,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        alpha: float = 3e-4,
        beta: float = 1e-3,
        seed: int = 42,
        FilenamePrefix: Optional[str] = None,
        model_dir: str = "models",
        min_buffer_size: int = 32,
    ):
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.model_dir = model_dir
        self.writer = writer
        self.Filenameprefix = FilenamePrefix

        self.seed(seed)

        self._clear_buffer()

        self.encoder = CarEncoder(
            in_dim=input_dims[0], hidden_dim=64, embed_dim=embed_dim
        ).to(device)
        self.actor = ActorNetworkPPO(
            input_dims=embed_dim, n_actions=n_actions
        ).to(device)
        self.critic = CriticNetworkPPO(input_dims=embed_dim).to(device)

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        self.actor_opt = T.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_opt = T.optim.Adam(self.critic.parameters(), lr=beta)
        self.encoder_opt = T.optim.Adam(self.encoder.parameters(), lr=alpha)

        self.update_count = 0

    def seed(self, seed):
        self.seed_value = seed
        self.np_random = np.random.RandomState(seed)
        T.manual_seed(seed)
        if T.cuda.is_available():
            T.cuda.manual_seed_all(seed)

    def _clear_buffer(self):
        self.buffer = {
            "states": [],
            "actions": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "next_values": [],
        }

    def _encode_state(self, state):
        state = T.as_tensor(state, dtype=T.float32, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with T.no_grad():
            emb = self.encoder(state)
        return emb

    def choose_action(self, state, store=True):
        emb = self._encode_state(state)

        with T.no_grad():
            action, logprob = self.actor.act(emb)
            value = self.critic(emb).item()

        if store:
            # Store raw observation (not embedding) so encoder can be trained
            self.buffer["states"].append(
                T.as_tensor(state, dtype=T.float32).cpu())
            self.buffer["actions"].append(action.squeeze(0).cpu())
            self.buffer["logprobs"].append(logprob.cpu())
            self.buffer["values"].append(float(value))

        return action.item()

    def reset_hidden(self):
        pass

    def _compute_gae(self, rewards, dones, values, next_values):
        """GAE-Lambda advantages and returns."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lmbda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def learn(self, state, action, reward, state_, done, timestamp, Train):
        """Store transition and perform PPO update at episode end."""
        if Train == 0:
            # Validation: still store next_value for completeness but don't train
            return

        self.buffer["rewards"].append(float(reward))
        self.buffer["dones"].append(int(done))

        # Compute next state value
        emb_ = self._encode_state(state_)
        with T.no_grad():
            next_val = self.critic(emb_).item()
        self.buffer["next_values"].append(float(next_val))

        # Only train at episode end
        if not done:
            return

        n = len(self.buffer["rewards"])
        if n < self.min_buffer_size:
            self._clear_buffer()
            return

        self._ppo_update(timestamp)
        self._clear_buffer()

    def _ppo_update(self, timestamp):
        states = self.buffer["states"]
        actions = self.buffer["actions"]
        old_logprobs = self.buffer["logprobs"]
        rewards = self.buffer["rewards"]
        dones = self.buffer["dones"]
        values = self.buffer["values"]
        next_values = self.buffer["next_values"]

        advantages, returns = self._compute_gae(rewards, dones, values, next_values)

        # Normalise advantages
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        advantages_t = T.tensor(advantages, dtype=T.float32, device=device)
        returns_t = T.tensor(returns, dtype=T.float32, device=device)
        states_t = T.stack(states).to(device)
        actions_t = T.stack(actions).to(device)
        old_logprobs_t = T.stack(old_logprobs).to(device)

        n = len(rewards)
        bs = min(self.batch_size, n)
        indices = np.arange(n)

        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, n, bs):
                end = min(start + bs, n)
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_logprobs = old_logprobs_t[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                # Re-encode states (with grad for encoder training)
                b_emb = self.encoder(b_states)

                new_logprobs, entropy = self.actor.evaluate_actions(b_emb, b_actions)
                values_pred = self.critic(b_emb)

                ratio = T.exp(new_logprobs - b_old_logprobs)

                surr1 = ratio * b_advantages
                surr2 = T.clamp(ratio, 1.0 - self.clip_epsilon,
                                1.0 + self.clip_epsilon) * b_advantages
                policy_loss = -T.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, b_returns)

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy.mean())

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                self.encoder_opt.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                T.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                self.actor_opt.step()
                self.critic_opt.step()
                self.encoder_opt.step()

        self.update_count += 1

        if self.writer:
            self.writer.add_scalar("PPO/Policy Loss", policy_loss.item(), self.update_count)
            self.writer.add_scalar("PPO/Value Loss", value_loss.item(), self.update_count)
            self.writer.add_scalar("PPO/Entropy", entropy.mean().item(), self.update_count)
            self.writer.add_scalar("PPO/Updates", self.update_count, timestamp)
            self.writer.flush()

    def save_models(self, total_reward: str):
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = f"{self.Filenameprefix}_rew{total_reward}_{ts}"
        T.save(self.actor.state_dict(),
               os.path.join(self.model_dir, f"{prefix}_actor.pth"))
        T.save(self.critic.state_dict(),
               os.path.join(self.model_dir, f"{prefix}_critic.pth"))
        T.save(self.encoder.state_dict(),
               os.path.join(self.model_dir, f"{prefix}_encoder.pth"))
        print(f"PPO models saved: {prefix}")
