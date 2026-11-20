"""Advanced model: Conformer encoder (Gulati et al., Interspeech 2020) for clip-level
sound classification, implemented from scratch.

log-mel (B,1,128,431) -> BN -> SpecAugment -> Conv2d subsampling (x4 in time & freq)
-> Linear to d_model + sinusoidal positions -> N x ConformerBlock -> attention pooling
-> classifier.

ConformerBlock (macaron): x + 1/2 FFN -> x + MHSA -> x + ConvModule -> x + 1/2 FFN -> LayerNorm

Ablation switches (Table 5):
  use_mhsa=False         -> remove self-attention (global context)
  use_conv_module=False  -> remove the convolution module (local context)
  pooling='mean'         -> replace attention pooling with mean pooling
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import AttentionPool, SpecAugment


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):                       # (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0).to(x.dtype)


class ConvSubsampling(nn.Module):
    """Two stride-2 3x3 convs: (B,1,M,T) -> (B, T/4, d_model)."""

    def __init__(self, n_mels: int, d_model: int, channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.SiLU(),
        )
        m_out = ((n_mels + 1) // 2 + 1) // 2
        self.proj = nn.Linear(channels * m_out, d_model)

    def forward(self, x):
        x = self.conv(x)                        # (B, C, M', T')
        B, C, M, T = x.shape
        return self.proj(x.permute(0, 3, 1, 2).reshape(B, T, C * M))


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        return self.drop(h)


class ConvModule(nn.Module):
    """LN -> pointwise(2d) -> GLU -> depthwise(k) -> BN -> SiLU -> pointwise -> dropout."""

    def __init__(self, d: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel_size, padding=kernel_size // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B, T, D)
        h = self.norm(x).transpose(1, 2)        # (B, D, T)
        h = nn.functional.glu(self.pw1(h), dim=1)
        h = nn.functional.silu(self.bn(self.dw(h)))
        return self.drop(self.pw2(h).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff_mult: int, kernel_size: int, dropout: float,
                 use_mhsa: bool = True, use_conv_module: bool = True):
        super().__init__()
        self.ff1 = FeedForward(d, ff_mult, dropout)
        self.mhsa = MHSA(d, heads, dropout) if use_mhsa else None
        self.conv = ConvModule(d, kernel_size, dropout) if use_conv_module else None
        self.ff2 = FeedForward(d, ff_mult, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        if self.mhsa is not None:
            x = x + self.mhsa(x)
        if self.conv is not None:
            x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.norm(x)


class Conformer(nn.Module):
    def __init__(self, n_classes: int = 50, n_mels: int = 128, d_model: int = 144, n_layers: int = 4,
                 heads: int = 4, ff_mult: int = 4, kernel_size: int = 15, dropout: float = 0.1,
                 subsample_channels: int = 64, head_dropout: float = 0.3,
                 use_mhsa: bool = True, use_conv_module: bool = True, pooling: str = "attention",
                 spec_augment: dict | None = None):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        sa = dict(spec_augment or {})
        self.spec_augment = SpecAugment(**{k: v for k, v in sa.items() if k != "enabled"}) \
            if sa and sa.get("enabled", True) else nn.Identity()
        self.subsample = ConvSubsampling(n_mels, d_model, subsample_channels)
        self.pos = SinusoidalPositions(d_model)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, heads, ff_mult, kernel_size, dropout, use_mhsa, use_conv_module)
            for _ in range(n_layers)])
        self.pooling = pooling
        self.pool = AttentionPool(d_model, d_model) if pooling == "attention" else None
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, n_classes))

    def embed(self, x):
        x = self.spec_augment(self.bn0(x))
        x = self.in_drop(self.pos(self.subsample(x)))      # (B, T', D)
        for blk in self.blocks:
            x = blk(x)
        return self.pool(x) if self.pool is not None else x.mean(1)

    def forward(self, x):
        return self.head(self.embed(x))
