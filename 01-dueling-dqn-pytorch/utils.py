import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import cv2


class PreprocessFrame(gym.ObservationWrapper):
    """
    Convert Atari RGB frames to grayscale 84x84, normalized [0, 1].
    """
    def __init__(self, env):
        super().__init__(env)
        # New observation space: single grayscale 84x84 frame
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(84, 84), dtype=np.float32
        )

    def observation(self, obs):
        # Convert RGB to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1] and cast to float32
        normalized = resized.astype(np.float32) / 255.0
        return normalized


class FrameStack(gym.ObservationWrapper):
    """
    Stack last k frames into shape (k, 84, 84).
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        # Observation space is now k stacked frames
        low = np.repeat(self.observation_space.low[np.newaxis, ...], k, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], k, axis=0)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill buffer with k copies of the first observation
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Stack into (k, 84, 84)
        return np.stack(self.frames, axis=0).astype(np.float32)


class RepeatAction(gym.Wrapper):
    """
    Repeat action for 'repeat' frames (frame skipping).
    Paper uses 4-frame skip.
    """
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def make_env(env_name):
    """
    Build the full Atari environment with all wrappers.
    Order matters: raw env -> frame skip -> preprocess -> stack
    """
    env = gym.make(env_name, render_mode=None)
    env = RepeatAction(env, repeat=4)
    env = PreprocessFrame(env)
    env = FrameStack(env, k=4)
    return env


# ==================== TEST ====================
if __name__ == "__main__":
    env = make_env("ALE/Pong-v5")
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")      # Should be (4, 84, 84)
    print(f"Observation dtype: {obs.dtype}")      # Should be float32
    print(f"Min/Max: {obs.min():.3f}/{obs.max():.3f}")  # Should be 0.0/1.0

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step output shape: {next_obs.shape}")  # Should be (4, 84, 84)
    print(f"Reward: {reward}")
    print(f"Done: {terminated or truncated}")