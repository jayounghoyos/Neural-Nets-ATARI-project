import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    frac = step / decay_steps
    return eps_start + frac * (eps_end - eps_start)


def update_latest_pointer(checkpoint_path: str, latest_path: str = "checkpoints/dqn_latest.pt"):
    Path(latest_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_path, latest_path)


def format_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class Timer:
    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start

    def elapsed_str(self) -> str:
        return format_time(self.elapsed())
