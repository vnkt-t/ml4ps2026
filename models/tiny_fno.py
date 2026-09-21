"""Minimal 2D Fourier Neural Operator (the project's primary surrogate).

Standard FNO: lift to `width` channels, a few spectral-conv + pointwise-conv blocks with
GELU, project back. Spectral conv truncates to `modes` low Fourier modes per dimension.
Input is (a, x, y) channels; output is the scalar field u.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_ch, out_ch, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = self.modes1, self.modes2
        out_ft[:, :, :m1, :m2] = self._mul(x_ft[:, :, :m1, :m2], self.w1)
        out_ft[:, :, -m1:, :m2] = self._mul(x_ft[:, :, -m1:, :m2], self.w2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, modes: int = 12, width: int = 24, n_layers: int = 3, in_ch: int = 3):
        super().__init__()
        self.fc0 = nn.Linear(in_ch, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):  # x: (B, H, W, in_ch)
        x = self.fc0(x).permute(0, 3, 1, 2)        # (B, width, H, W)
        for conv, w in zip(self.convs, self.ws):
            x = F.gelu(conv(x) + w(x))
        x = x.permute(0, 2, 3, 1)                  # (B, H, W, width)
        x = F.gelu(self.fc1(x))
        return self.fc2(x).squeeze(-1)             # (B, H, W)
