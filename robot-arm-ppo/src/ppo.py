"""
PPO for continuous control (2-link arm reaching).

Actor: Gaussian policy (mean from MLP, learned log_std).
Critic: MLP value function.
"""

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor = mlp([obs_dim, hidden, hidden, act_dim])
        self.critic = mlp([obs_dim, hidden, hidden, 1])
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def dist(self, obs):
        mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std), value


class PPO:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, value = self.model.dist(obs_t)
            action = dist.sample()
            logp = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), float(value.cpu()), float(logp.cpu())

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        values = np.append(values, 0.0)
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + self.gamma * self.lam * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, batch):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["adv"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        losses = []

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]
                dist, values = self.model.dist(obs[mb])
                logp = dist.log_prob(act[mb]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(logp - old_logp[mb])
                clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                policy_loss = -torch.min(ratio * advantages[mb], clipped * advantages[mb]).mean()
                value_loss = ((values - returns[mb]) ** 2).mean()
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
