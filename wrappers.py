import gymnasium as gymn
from config import Config
from gymnasium.wrappers import (RecordEpisodeStatistics, ResizeObservation, GrayscaleObservation, FrameStackObservation,)
import numpy as np
import ale_py

class MaxAndSkipEnv(gymn.Wrapper):
    def __init__(self,env, skip=4):
        super().__init__(env)
        self._skip = skip

        self.__obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        
    def step(self,action):
        total_reward = 0.0

        for i in range(0,self._skip):
            obs, reward, terminated,truncated, info = self.env.step(action)
            total_reward += reward
            print(f"[MaxAndSkipEnv] frame {i} | reward: {reward} | donde: {terminated or truncated}")
            
            # Buffer
            if i ==  self._skip -2:
                self.__obs_buffer[0] = obs
            if i == self._skip -1:
                self.__obs_buffer[1] = obs

            if terminated or truncated:
                print(f"[MaxAndSkipEnv] the episode finished in frame {i}")
                break

        max_frame = self.__obs_buffer.max(axis=0)
        print(f"[MaxAndSkipEnv] max_frame shape: {max_frame.shape}")
        return max_frame, total_reward, terminated, truncated, info


def make_env(env_id, render_mode=None):
    env = gymn.make(env_id,render_mode=render_mode)
    env = MaxAndSkipEnv(env,skip=4)
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, (84,84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)

    return env


if __name__ == "__main__":
    env = make_env("ALE/Breakout-v5", render_mode="human")
    obs, _ = env.reset()
    print(obs.shape) # (4,84,84)
    print(obs.dtype) # uint8
    env.close()