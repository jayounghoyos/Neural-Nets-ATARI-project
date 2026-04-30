from pathlib import Path

from config import Config


def run_baseline(cfg: Config):
    """Entrena Stable-Baselines3 DQN con la misma configuración como referencia.

    Resultado: TensorBoard logs en runs/<RUN_NAME>_sb3/ — comparable lado a lado
    con la curva de la implementación manual usando `tensorboard --logdir runs/`.
    """
    try:
        from stable_baselines3 import DQN
        from stable_baselines3.common.atari_wrappers import AtariWrapper
        from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
        import gymnasium as gym
        import ale_py  # noqa: F401  ensures ALE namespace is registered
    except ImportError as e:
        raise RuntimeError(
            "stable-baselines3 no está instalado — corre `pip install stable-baselines3[extra]`"
        ) from e

    def _make():
        env = gym.make(cfg.ENV_ID, frameskip=1)
        return AtariWrapper(env)

    venv = DummyVecEnv([_make])
    venv = VecFrameStack(venv, n_stack=4)

    run_name = f"{cfg.RUN_NAME}_sb3"
    tb_log = str(Path(cfg.RUNS_DIR))

    model = DQN(
        policy="CnnPolicy",
        env=venv,
        learning_rate=cfg.LEARNING_RATE,
        buffer_size=cfg.BUFFER_SIZE,
        learning_starts=cfg.LEARNING_STARTS,
        batch_size=cfg.BATCH_SIZE,
        gamma=cfg.GAMMA,
        target_update_interval=cfg.TARGET_UPDATE_FREQUENCY,
        train_freq=cfg.TRAIN_FREQUENCY,
        exploration_initial_eps=cfg.EPSILON_START,
        exploration_final_eps=cfg.EPSILON_END,
        exploration_fraction=cfg.EPSILON_DECAY / cfg.TOTAL_STEPS,
        max_grad_norm=cfg.GRAD_CLIP_NORM,
        seed=cfg.SEED,
        verbose=1,
        tensorboard_log=tb_log,
    )

    print(f"[baseline] training SB3 DQN for {cfg.TOTAL_STEPS} steps — logs at runs/{run_name}")
    model.learn(total_timesteps=cfg.TOTAL_STEPS, tb_log_name=run_name)

    save_path = Path(cfg.CHECKPOINT_DIR) / f"sb3_{cfg.RUN_NAME}.zip"
    model.save(save_path)
    print(f"[baseline] saved to {save_path}")
