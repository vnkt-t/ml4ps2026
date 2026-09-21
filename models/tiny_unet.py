"""Tiny 2D U-Net surrogate for architecture-sensitivity checks.

The encoder/decoder uses convolutions, skip connections and GroupNorm, without
Fourier layers. GroupNorm does use spatial statistics. The I/O convention
matches FNO2d: input (B, H, W, in_ch), output (B, H, W),
so it is a drop-in surrogate for the residual-localization ladder. A replicated
localization gap provides evidence beyond the particular FNO architecture;
it does not establish architecture-independent behavior.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, ch), ch)


class _Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.n1 = _gn(out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n2 = _gn(out_ch)

    def forward(self, x):
        x = F.gelu(self.n1(self.c1(x)))
        return F.gelu(self.n2(self.c2(x)))


class UNet2d(nn.Module):
    """Two-level U-Net (H -> H/2 -> H/4 -> H/2 -> H). H must be divisible by 4 (32, 64 ok)."""

    def __init__(self, width: int = 32, in_ch: int = 3):
        super().__init__()
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("width must be a positive integer")
        if isinstance(in_ch, bool) or not isinstance(in_ch, int) or in_ch < 1:
            raise ValueError("in_ch must be a positive integer")
        self.in_ch = in_ch
        w = width
        self.enc1 = _Block(in_ch, w)
        self.enc2 = _Block(w, 2 * w)
        self.bott = _Block(2 * w, 4 * w)
        self.up2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
        self.dec2 = _Block(4 * w, 2 * w)
        self.up1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
        self.dec1 = _Block(2 * w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):  # x: (B, H, W, in_ch)
        if x.ndim != 4 or x.shape[-1] != self.in_ch:
            raise ValueError("input must have shape (batch, height, width, in_ch)")
        if x.shape[1] < 4 or x.shape[2] < 4 or x.shape[1] % 4 or x.shape[2] % 4:
            raise ValueError("spatial dimensions must be positive multiples of four")
        x = x.permute(0, 3, 1, 2)              # (B, in_ch, H, W)
        s1 = self.enc1(x)                      # (B, w, H, W)
        s2 = self.enc2(F.max_pool2d(s1, 2))    # (B, 2w, H/2, W/2)
        b = self.bott(F.max_pool2d(s2, 2))     # (B, 4w, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up2(b), s2], dim=1))   # (B, 2w, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))  # (B, w, H, W)
        return self.out(d1).squeeze(1)         # (B, H, W)
