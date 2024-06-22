import gymnasium as gym
from gymnasium.spaces import discrete
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")

env.reset()

observation, info = env.reset(seed=42)
for _ in range(1000):
   action = 2
   new_state, reward, terminated, truncated, info = env.step(action)
   if terminated or truncated:
      observation, info = env.reset()

env.close()
