import gymnasium as gymn
from config import Config
from gymnasium.wrappers import (RecordEpisodeStatistics, ResizeObservation, GrayscaleObservation, FrameStackObservation,)
import numpy as np
import ale_py

class MaxAndSkipEnv(gymn.Wrapper):
    def __init__(self, env, skip=4, debug=False):
        super().__init__(env)
        self._skip = skip
        self._debug = debug

        self.__obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(0, self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if self._debug:
                print(f"[MaxAndSkipEnv] frame {i} | reward: {reward} | done: {terminated or truncated}")

            if i == self._skip - 2:
                self.__obs_buffer[0] = obs
            if i == self._skip - 1:
                self.__obs_buffer[1] = obs

            if terminated or truncated:
                if self._debug:
                    print(f"[MaxAndSkipEnv] episode finished at frame {i}")
                break

        max_frame = self.__obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


def make_env(env_id, render_mode=None):
    # frameskip=1 desactiva el frame-skip nativo de v5 para que MaxAndSkipEnv(skip=4)
    # sea el único responsable del frame skipping. Sticky actions se dejan en su default
    # de v5 (repeat_action_probability=0.25) — recomendación moderna (Machado 2018).
    env = gymn.make(env_id, render_mode=render_mode, frameskip=1)
    env = MaxAndSkipEnv(env, skip=4)
    env = RecordEpisodeStatistics(env)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env)
    env = FrameStackObservation(env, stack_size=4)

    return env


if __name__ == "__main__":
    env = make_env("ALE/Breakout-v5")
    obs, _ = env.reset()
    print(f"obs.shape: {obs.shape}")  # (4, 84, 84)
    print(f"obs.dtype: {obs.dtype}")  # uint8

    # Random rollout — debe durar cientos de pasos, no ~10
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done = term or trunc
        steps += 1
    print(f"Random episode length: {steps} steps")
    env.close()