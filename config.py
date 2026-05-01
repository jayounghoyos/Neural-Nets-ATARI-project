import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Config:
    # Environment
    ENV_ID: str = "ALE/Breakout-v5"
    SEED: int = 1

    # Training hyperparameters
    TOTAL_STEPS: int = 10_000_000
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-4
    GAMMA: float = 0.99
    TRAIN_FREQUENCY: int = 4
    GRAD_CLIP_NORM: float = 10.0

    # Replay buffer
    BUFFER_SIZE: int = 100_000
    LEARNING_STARTS: int = 10_000

    # Epsilon schedule (linear decay)
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.1
    EPSILON_DECAY: int = 500_000

    # Target network
    TARGET_UPDATE_FREQUENCY: int = 10_000
    TAU_SOFT_UPDATE: Optional[float] = None  # None = hard copy; float (e.g. 0.005) = Polyak soft update

    # Evaluation
    EVAL_FREQUENCY: int = 50_000
    EVAL_EPISODES: int = 5

    # Device
    DEVICE: str = "auto"  # "auto" | "cuda" | "cpu"

    # Run identity & resume
    RUN_NAME: str = "dqn_breakout"
    RESUME_FROM: Optional[str] = None

    # Logging and saving
    LOG_DIR: str = "logs/"
    CHECKPOINT_DIR: str = "checkpoints/"
    RUNS_DIR: str = "runs/"
    SAVE_FREQUENCY: int = 100_000

    def __post_init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RUNS_DIR, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


if __name__ == "__main__":
    cfg = Config()
    for k, v in cfg.to_dict().items():
        print(f"{k:28s} = {v}")
