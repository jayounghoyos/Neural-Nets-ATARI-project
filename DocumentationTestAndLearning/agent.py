import torch
import gymnasium as gym
import ale_py

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

env = gym.make("ALE/Breakout-v5", render_mode="human")
obs, info = env.reset()

print(f"Obs shape: {obs.shape}")       # debe ser (210, 160, 3)
print(f"Actions: {env.action_space}")  # debe ser Discrete(4)

# Corre 5 pasos con acciones random
for _ in range(500):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"reward: {reward}, done: {terminated or truncated}")

env.close()
print("Setup OK")
