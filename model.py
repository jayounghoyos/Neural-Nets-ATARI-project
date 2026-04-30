import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """CNN del paper Mnih et al. 2015 (Nature).

    Input:  (B, 4, 84, 84) uint8 — 4 frames apilados
    Output: (B, num_actions) float — Q-values
    """

    def __init__(self, num_actions: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Acepta uint8 o float; normaliza a [0, 1] internamente
        x = x.float() / 255.0
        return self.head(self.features(x))


if __name__ == "__main__":
    net = QNetwork(num_actions=4)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")  # esperado ~1,686,180

    dummy = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    out = net(dummy)
    print(f"Input:  {tuple(dummy.shape)} {dummy.dtype}")
    print(f"Output: {tuple(out.shape)} {out.dtype}")
    assert out.shape == (2, 4)
    print("OK")
