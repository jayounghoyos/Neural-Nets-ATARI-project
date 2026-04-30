import numpy as np
import torch


class ReplayBuffer:
    """Buffer FIFO con muestreo uniforme.

    Almacena observaciones en uint8 para ahorrar RAM. Convierte a tensores en sample().
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple = (4, 84, 84),
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.device = device
        self.idx = 0
        self.size = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action: int, reward: float, next_state, done: bool):
        self.states[self.idx] = state
        self.next_states[self.idx] = next_state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idxs]).to(self.device),
            torch.from_numpy(self.actions[idxs]).to(self.device),
            torch.from_numpy(self.rewards[idxs]).to(self.device),
            torch.from_numpy(self.next_states[idxs]).to(self.device),
            torch.from_numpy(self.dones[idxs]).to(self.device),
        )

    def __len__(self):
        return self.size


if __name__ == "__main__":
    buf = ReplayBuffer(capacity=1000)
    rng = np.random.default_rng(0)

    for _ in range(500):
        s = rng.integers(0, 256, (4, 84, 84), dtype=np.uint8)
        ns = rng.integers(0, 256, (4, 84, 84), dtype=np.uint8)
        buf.add(s, action=rng.integers(0, 4), reward=rng.random(), next_state=ns, done=False)

    print(f"len(buf) = {len(buf)} (esperado 500)")

    states, actions, rewards, next_states, dones = buf.sample(32)
    print(f"states:      {tuple(states.shape)} {states.dtype}")
    print(f"actions:     {tuple(actions.shape)} {actions.dtype}")
    print(f"rewards:     {tuple(rewards.shape)} {rewards.dtype}")
    print(f"next_states: {tuple(next_states.shape)} {next_states.dtype}")
    print(f"dones:       {tuple(dones.shape)} {dones.dtype}")
    assert states.shape == (32, 4, 84, 84)
    assert actions.shape == (32,)
    print("OK")
