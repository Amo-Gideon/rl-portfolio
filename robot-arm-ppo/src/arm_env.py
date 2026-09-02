"""
2-link planar robot arm environment (Gymnasium-compatible).

State:  [cos(j1), sin(j1), cos(j2), sin(j2), target_x, target_y]
Action: continuous Box(2) in [-1, 1]; scaled by max_speed internally.
Reward: scaled negative distance - small action penalty; +1 bonus on success.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ArmEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        link1: float = 1.0,
        link2: float = 1.0,
        max_steps: int = 100,
        max_speed: float = 0.15,
        success_radius: float = 0.08,
        action_penalty: float = 0.01,
        reward_scale: float = 0.1,
        seed=None,
    ):
        super().__init__()
        self.link1 = link1
        self.link2 = link2
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.success_radius = success_radius
        self.action_penalty = action_penalty
        self.reward_scale = reward_scale
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.joints = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.joints = self.rng.uniform(-np.pi, np.pi, size=2).astype(np.float32)
        self.target = self._sample_target()
        self.step_count = 0
        return self._state(), {}

    def _sample_target(self):
        # Reachable target: within 90% of full reach, outside near-center dead zone
        r = self.rng.uniform(0.3 * (self.link1 + self.link2), 0.9 * (self.link1 + self.link2))
        theta = self.rng.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)

    def _forward_kinematics(self, joints=None):
        j = self.joints if joints is None else joints
        x1 = self.link1 * np.cos(j[0])
        y1 = self.link1 * np.sin(j[0])
        x2 = x1 + self.link2 * np.cos(j[0] + j[1])
        y2 = y1 + self.link2 * np.sin(j[0] + j[1])
        return np.array([x1, y1], dtype=np.float32), np.array([x2, y2], dtype=np.float32)

    def _state(self):
        return np.array(
            [
                np.cos(self.joints[0]),
                np.sin(self.joints[0]),
                np.cos(self.joints[1]),
                np.sin(self.joints[1]),
                self.target[0],
                self.target[1],
            ],
            dtype=np.float32,
        )

    def distance(self):
        _, ee = self._forward_kinematics()
        return float(np.linalg.norm(ee - self.target))

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.joints = self.joints + action * self.max_speed
        self.step_count += 1

        dist = self.distance()
        reward = self.reward_scale * (-dist) - self.action_penalty * float(np.sum(action ** 2))

        success = dist < self.success_radius
        terminated = bool(success)
        truncated = self.step_count >= self.max_steps
        if success:
            reward += 1.0

        info = {"distance": dist, "success": success}
        return self._state(), reward, terminated, truncated, info
