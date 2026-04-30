import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model import QNetwork


class DQNAgent:
    def __init__(self, num_actions: int, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.num_actions = num_actions

        self.q_net = QNetwork(num_actions).to(device)
        self.target_net = QNetwork(num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.LEARNING_RATE)

    @torch.no_grad()
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        s = torch.from_numpy(np.asarray(state)).unsqueeze(0).to(self.device)
        q_values = self.q_net(s)
        return int(q_values.argmax(dim=1).item())

    def update(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch

        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.cfg.GAMMA * next_q * (1.0 - dones)

        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.GRAD_CLIP_NORM)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        if self.cfg.TAU_SOFT_UPDATE is None:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            tau = self.cfg.TAU_SOFT_UPDATE
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def save(self, path: str, global_step: int, episode_count: int, epsilon: float):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": global_step,
            "episode_count": episode_count,
            "epsilon": epsilon,
            "config": asdict(self.cfg),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(ckpt, path)

    def load(self, path: str) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        rng = ckpt.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"].to("cpu", dtype=torch.uint8))
        if rng.get("cuda") is not None and torch.cuda.is_available():
            cuda_states = [s.to("cpu", dtype=torch.uint8) for s in rng["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)

        return {
            "global_step": ckpt["global_step"],
            "episode_count": ckpt["episode_count"],
            "epsilon": ckpt["epsilon"],
            "config": ckpt.get("config"),
        }


if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(num_actions=4, cfg=cfg, device=device)
    print(f"Device: {device}")

    B = 32
    states = torch.randint(0, 256, (B, 4, 84, 84), dtype=torch.uint8, device=device)
    next_states = torch.randint(0, 256, (B, 4, 84, 84), dtype=torch.uint8, device=device)
    actions = torch.randint(0, 4, (B,), dtype=torch.int64, device=device)
    rewards = torch.randn(B, device=device)
    dones = torch.zeros(B, device=device)

    loss = agent.update((states, actions, rewards, next_states, dones))
    print(f"Loss tras 1 update: {loss:.4f}  (debe ser finito)")
    assert np.isfinite(loss)

    agent.update_target()
    a = agent.select_action(np.zeros((4, 84, 84), dtype=np.uint8), epsilon=0.0)
    print(f"select_action greedy: {a}")
    a = agent.select_action(np.zeros((4, 84, 84), dtype=np.uint8), epsilon=1.0)
    print(f"select_action random: {a}")

    agent.save("checkpoints/_smoke_test.pt", global_step=0, episode_count=0, epsilon=1.0)
    info = agent.load("checkpoints/_smoke_test.pt")
    print(f"Reloaded: {info['global_step']=} {info['episode_count']=} {info['epsilon']=}")
    Path("checkpoints/_smoke_test.pt").unlink()
    print("OK")
