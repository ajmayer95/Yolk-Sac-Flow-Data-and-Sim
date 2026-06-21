"""Lightweight U-Net with GroupNorm for vessel segmentation."""

from __future__ import annotations


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


class ConvBlock(__import__("torch").nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        _, nn = _torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(__import__("torch").nn.Module):
    """Four-level U-Net. Returns logits; apply sigmoid only for inference."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base: int = 32):
        _, nn = _torch()
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_channels, 1)

    @staticmethod
    def _crop_like(skip, x):
        if skip.shape[-2:] == x.shape[-2:]:
            return skip
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        return skip[..., dh // 2: dh // 2 + x.shape[-2], dw // 2: dw // 2 + x.shape[-1]]

    def forward(self, x):
        torch, _ = _torch()
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self._crop_like(e4, d4)], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self._crop_like(e3, d3)], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self._crop_like(e2, d2)], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self._crop_like(e1, d1)], dim=1))
        return self.out(d1)
