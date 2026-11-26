"""Model giả siêu nhỏ chỉ để test khung chung (không dùng để benchmark)."""
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size=64, stride=32)
        self.drop = nn.Dropout(0.3)  # để test khôi phục RNG khi resume
        self.fc = nn.Linear(channels, 2)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T] -> [B,2], cột 0 = thật, cột 1 = giả."""
        h = torch.relu(self.conv(x.unsqueeze(1))).mean(dim=-1)
        return self.fc(self.drop(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lg = self.logits(x)
        return lg[:, 1] - lg[:, 0]
