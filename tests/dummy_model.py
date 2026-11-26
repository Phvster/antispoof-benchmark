"""Model giả chỉ để test khung chung (không dùng để benchmark).

Quy ước: logits cột 0 = giả, cột 1 = thật; forward = logits[:,1] - logits[:,0] (cao = thật).
"""
import torch
import torch.nn as nn

from common.audio import SAMPLE_RATE

SPOOF_TONE_HZ = 3000  # tests/conftest.py thêm sóng sin 3 kHz vào file giả


class TinyModel(nn.Module):
    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(1, channels, kernel_size=64, stride=32)
        self.drop = nn.Dropout(0.3)  # để test khôi phục RNG khi resume
        self.fc = nn.Linear(channels, 2)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T] -> [B,2], cột 0 = giả, cột 1 = thật."""
        h = torch.relu(self.conv(x.unsqueeze(1))).mean(dim=-1)
        return self.fc(self.drop(h))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lg = self.logits(x)
        return lg[:, 1] - lg[:, 0]


class OracleModel(nn.Module):
    """Nhận ra file giả bằng năng lượng quanh 3 kHz -> kiểm tra chiều nhãn/dấu điểm end-to-end."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # để state_dict/optimizer không rỗng

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        spec = torch.fft.rfft(x).abs()
        k = round(SPOOF_TONE_HZ * n / SAMPLE_RATE)
        e = spec[:, max(k - 2, 0):k + 3].sum(dim=-1) / n * 100 + 0 * self.dummy
        return torch.stack([e, -e], dim=1)  # nhiều tone -> logit giả cao

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lg = self.logits(x)
        return lg[:, 1] - lg[:, 0]

    def load_pretrained(self, path: str) -> None:
        pass
