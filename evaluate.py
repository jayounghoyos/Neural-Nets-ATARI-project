from dataclasses import replace

import numpy as np
import torch

from agent import DQNAgent
from config import Config
from utils import resolve_device, set_seeds
from wrappers import make_env


def evaluate(cfg: Config, checkpoint_path: str, episodes: int = 10, epsilon: float = 0.05,
             render_mode: str | None = None) -> dict:
    device = resolve_device(cfg.DEVICE)
    set_seeds(cfg.SEED)

    env = make_env(cfg.ENV_ID, render_mode=render_mode)
    num_actions = env.action_space.n

    agent = DQNAgent(num_actions=num_actions, cfg=cfg, device=device)
    info = agent.load(checkpoint_path)
    print(f"[eval] loaded {checkpoint_path} (trained step={info['global_step']})")

    rewards = []
    lengths = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=cfg.SEED + 10000 + ep)
        obs = np.array(obs)
        ep_reward, ep_len = 0.0, 0
        done = False
        while not done:
            action = agent.select_action(obs, epsilon=epsilon)
            obs, reward, term, trunc, _ = env.step(action)
            obs = np.array(obs)
            ep_reward += float(reward)
            ep_len += 1
            done = term or trunc
        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"  episode {ep+1:>2d}: reward={ep_reward:6.1f}  len={ep_len:4d}")

    env.close()
    rewards = np.array(rewards)
    summary = {
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "min_reward": float(rewards.min()),
        "max_reward": float(rewards.max()),
        "mean_length": float(np.mean(lengths)),
        "episodes": episodes,
        "checkpoint": checkpoint_path,
    }
    print(
        f"[eval] mean={summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}  "
        f"min={summary['min_reward']:.0f} max={summary['max_reward']:.0f}"
    )
    return summary


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/dqn_latest.pt"
    evaluate(Config(), checkpoint_path=ckpt, episodes=5)
