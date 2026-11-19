"""Classic deep models trained from scratch on log-mel spectrograms.

CNN  : VGG-style 2-D CNN (4 conv blocks) + global pooling + MLP head.
CRNN : 3 conv blocks (local time-frequency patterns) + BiGRU (temporal context)
       + attention pooling.

Both take a log-mel tensor (B, 1, n_mels, frames) and return logits (B, n_classes).
Input normalisation is learned (BatchNorm2d on the single input channel), so the
models don't depend on dataset-level mean/std statistics.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import AttentionPool, ConvBlock, SpecAugment


def _make_specaug(cfg: dict | None) -> nn.Module:
    if not cfg or not cfg.get("enabled", True):
        return nn.Identity()
    cfg = {k: v for k, v in cfg.items() if k != "enabled"}
    return SpecAugment(**cfg)


class CNN(nn.Module):
    def __init__(self, n_classes: int = 50, channels=(32, 64, 128, 256),
                 dropout: float = 0.3, block_dropout: float = 0.1,
                 head_hidden: int = 256, spec_augment: dict | None = None):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.spec_augment = _make_specaug(spec_augment)
        blocks, c_in = [], 1
        for c in channels:
            blocks.append(ConvBlock(c_in, c, pool=(2, 2), dropout=block_dropout))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * c_in, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spec_augment(self.bn0(x))
        x = self.features(x)                 # (B, C, M', T')
        x = x.mean(dim=2)                    # average over frequency -> (B, C, T')
        # mean + max over time: max captures short events (dog bark), mean long ones (rain)
        return torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)  # (B, 2C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


class CRNN(nn.Module):
    def __init__(self, n_classes: int = 50, channels=(64, 128, 256),
                 rnn_hidden: int = 128, rnn_layers: int = 1, dropout: float = 0.3,
                 block_dropout: float = 0.1, attn_hidden: int = 128,
                 spec_augment: dict | None = None):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(1)
        self.spec_augment = _make_specaug(spec_augment)
        blocks, c_in = [], 1
        for i, c in enumerate(channels):
            # pool frequency every block, time only in the first two -> keep temporal resolution
            pool = (2, 2) if i < 2 else (2, 1)
            blocks.append(ConvBlock(c_in, c, pool=pool, dropout=block_dropout))
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.rnn = nn.GRU(c_in, rnn_hidden, num_layers=rnn_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.pool = AttentionPool(2 * rnn_hidden, attn_hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * rnn_hidden, n_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spec_augment(self.bn0(x))
        x = self.features(x)                 # (B, C, M', T')
        x = x.mean(dim=2).transpose(1, 2)    # (B, T', C)
        x, _ = self.rnn(x)                   # (B, T', 2H)
        return self.pool(x)                  # (B, 2H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))
