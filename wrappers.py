import gymnasium as gymn
from config import Config
from gymnasium.wrappers import (RecordEpisodeStatistics, ResizeObservation, GrayscaleObservation, FrameStackObservation,)
import numpy as np

class MaxAndSkipEnv(gymn.Wrapper):
    def __init__(self,env, skip=4):
        super().__init__(env)
        self.skip = skip
        self.__obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        
    def step(self,action):
        total_reward = 0.0
        for step in range(0,self.skip):
            self.env.step(action)

def make_env(env_id, render_mode=None):
    env = gymn.make(env_id,render_mode=render_mode)

    return env