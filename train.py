import signal
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent
from config import Config
from replay_buffer import ReplayBuffer
from utils import (
    Timer,
    linear_epsilon,
    resolve_device,
    set_seeds,
    update_latest_pointer,
)
from wrappers import make_env


def train(cfg: Config):
    device = resolve_device(cfg.DEVICE)
    set_seeds(cfg.SEED)

    env = make_env(cfg.ENV_ID)
    env.action_space.seed(cfg.SEED)
    num_actions = env.action_space.n

    agent = DQNAgent(num_actions=num_actions, cfg=cfg, device=device)
    buffer = ReplayBuffer(capacity=cfg.BUFFER_SIZE, device=device)

    writer = SummaryWriter(log_dir=str(Path(cfg.RUNS_DIR) / cfg.RUN_NAME))
    writer.add_text("config", "\n".join(f"{k}: {v}" for k, v in cfg.to_dict().items()))

    # Resume support
    global_step = 0
    episode_count = 0
    if cfg.RESUME_FROM:
        info = agent.load(cfg.RESUME_FROM)
        global_step = info["global_step"]
        episode_count = info["episode_count"]
        print(f"[resume] from {cfg.RESUME_FROM} | step={global_step} ep={episode_count}")

    obs, _ = env.reset(seed=cfg.SEED + episode_count)
    obs = np.array(obs)
    episode_reward = 0.0
    episode_len = 0

    # Graceful Ctrl+C: save and exit
    interrupted = {"flag": False}

    def _handle_signal(signum, frame):
        print(f"\n[signal] received {signum} — saving and exiting...")
        interrupted["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    timer = Timer()
    print(f"[train] device={device} run={cfg.RUN_NAME} starting at step={global_step}")

    try:
        while global_step < cfg.TOTAL_STEPS and not interrupted["flag"]:
            epsilon = linear_epsilon(
                global_step, cfg.EPSILON_START, cfg.EPSILON_END, cfg.EPSILON_DECAY
            )
            action = agent.select_action(obs, epsilon)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = np.array(next_obs)
            done = terminated or truncated
            clipped_reward = float(np.sign(reward))

            buffer.add(obs, action, clipped_reward, next_obs, terminated)

            obs = next_obs
            episode_reward += float(reward)
            episode_len += 1
            global_step += 1

            # Train
            if global_step >= cfg.LEARNING_STARTS and global_step % cfg.TRAIN_FREQUENCY == 0:
                batch = buffer.sample(cfg.BATCH_SIZE)
                loss = agent.update(batch)
                if global_step % 1000 == 0:
                    writer.add_scalar("train/loss", loss, global_step)
                    writer.add_scalar("train/epsilon", epsilon, global_step)

            # Target sync
            if global_step % cfg.TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target()

            # End of episode
            if done:
                writer.add_scalar("episode/reward", episode_reward, global_step)
                writer.add_scalar("episode/length", episode_len, global_step)
                if episode_count % 10 == 0:
                    print(
                        f"step={global_step:>8d} ep={episode_count:>5d} "
                        f"reward={episode_reward:6.1f} len={episode_len:4d} "
                        f"eps={epsilon:.3f} elapsed={timer.elapsed_str()}"
                    )
                episode_count += 1
                episode_reward = 0.0
                episode_len = 0
                obs, _ = env.reset(seed=cfg.SEED + episode_count)
                obs = np.array(obs)

            # Periodic save
            if global_step % cfg.SAVE_FREQUENCY == 0:
                ckpt_path = Path(cfg.CHECKPOINT_DIR) / f"dqn_step_{global_step}.pt"
                agent.save(str(ckpt_path), global_step, episode_count, epsilon)
                update_latest_pointer(str(ckpt_path))
                print(f"[save] {ckpt_path} (latest pointer updated)")
    finally:
        # Always persist on exit (graceful or crash)
        ckpt_path = Path(cfg.CHECKPOINT_DIR) / f"dqn_step_{global_step}.pt"
        agent.save(
            str(ckpt_path),
            global_step,
            episode_count,
            linear_epsilon(global_step, cfg.EPSILON_START, cfg.EPSILON_END, cfg.EPSILON_DECAY),
        )
        update_latest_pointer(str(ckpt_path))
        writer.close()
        env.close()
        print(f"[exit] saved final checkpoint at step={global_step}, latest pointer updated")
        if interrupted["flag"]:
            sys.exit(0)


if __name__ == "__main__":
    train(Config())
