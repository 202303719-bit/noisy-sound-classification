"""Building blocks shared by the deep models.

Written from scratch (requirement #5 "write at least one key component yourselves"):
  * SpecAugment  - frequency / time masking on log-mel (Park et al., 2019)
  * AttentionPool - learnable attention pooling over time (used by the CRNN)
  * LogMelFrontend - wraps the team's `datasets.esc50.LogMel` so every model
    sees exactly the same input representation (fair comparison).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Front-end
# --------------------------------------------------------------------------- #
class LogMelFrontend(nn.Module):
    """waveform (B, T) | (B, 1, T) -> log-mel (B, 1, n_mels, frames).

    Uses the shared `LogMel` transform from src/datasets/esc50.py when available;
    otherwise falls back to an equivalent torchaudio implementation.
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 128, n_fft: int = 2048,
                 hop_length: int = 512, use_dataset_logmel: bool = True):
        super().__init__()
        self.transform = None
        if use_dataset_logmel:
            try:
                from datasets.esc50 import LogMel  # team's shared preprocessing
                self.transform = LogMel()
                self.source = "datasets.esc50.LogMel"
            except Exception:  # pragma: no cover - fallback path
                self.transform = None
        if self.transform is None:
            import torchaudio

            self.transform = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                                     hop_length=hop_length, n_mels=n_mels),
                torchaudio.transforms.AmplitudeToDB(top_db=80.0),
            )
            self.source = "torchaudio (fallback)"

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 3 and wave.size(1) == 1:
            wave = wave.squeeze(1)
        x = self.transform(wave.float())
        if x.dim() == 3:                 # (B, M, T) -> (B, 1, M, T)
            x = x.unsqueeze(1)
        return x


# --------------------------------------------------------------------------- #
# SpecAugment (own implementation)
# --------------------------------------------------------------------------- #
class SpecAugment(nn.Module):
    """Random frequency and time masking applied per sample, only in train mode.

    Args:
        freq_mask: max width (in mel bins) of each frequency mask (F in the paper)
        time_mask: max width (in frames) of each time mask (T in the paper)
        n_freq_masks / n_time_masks: how many masks of each kind
    Masked regions are filled with the per-sample mean, so the statistics of
    the input don't shift (important because we normalise with BatchNorm).
    """

    def __init__(self, freq_mask: int = 24, time_mask: int = 40,
                 n_freq_masks: int = 2, n_time_masks: int = 2):
        super().__init__()
        self.freq_mask, self.time_mask = freq_mask, time_mask
        self.n_freq_masks, self.n_time_masks = n_freq_masks, n_time_masks

    @staticmethod
    def _mask_along(x: torch.Tensor, max_width: int, n_masks: int, dim: int) -> torch.Tensor:
        B, size = x.size(0), x.size(dim)
        if max_width <= 0 or n_masks <= 0 or size <= 1:
            return x
        max_width = min(max_width, size - 1)
        idx = torch.arange(size, device=x.device).view(1, size)          # (1, size)
        mask = torch.zeros(B, size, dtype=torch.bool, device=x.device)
        for _ in range(n_masks):
            width = torch.randint(0, max_width + 1, (B, 1), device=x.device)
            start = (torch.rand(B, 1, device=x.device) * (size - width)).long()
            mask |= (idx >= start) & (idx < start + width)
        shape = [B, 1, 1, 1]
        shape[dim] = size
        mask = mask.view(*shape)
        fill = x.mean(dim=(1, 2, 3), keepdim=True)
        return torch.where(mask, fill, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, M, T)
        if not self.training:
            return x
        x = self._mask_along(x, self.freq_mask, self.n_freq_masks, dim=2)
        x = self._mask_along(x, self.time_mask, self.n_time_masks, dim=3)
        return x

    def extra_repr(self) -> str:
        return (f"F={self.freq_mask}, T={self.time_mask}, "
                f"mF={self.n_freq_masks}, mT={self.n_time_masks}")


# --------------------------------------------------------------------------- #
# Conv block
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    """[Conv3x3 -> BN -> ReLU] x 2 -> AvgPool -> Dropout2d."""

    def __init__(self, c_in: int, c_out: int, pool=(2, 2), dropout: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(pool) if pool and tuple(pool) != (1, 1) else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.pool(self.body(x)))


# --------------------------------------------------------------------------- #
# Attention pooling (own implementation)
# --------------------------------------------------------------------------- #
class AttentionPool(nn.Module):
    """Learnable soft-attention pooling over the time axis.
    x: (B, T, D) -> (B, D);  alpha_t = softmax_t(w^T tanh(W x_t))."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.score(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        a = F.softmax(a, dim=1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1)       # (B, D)


class ASTFrontend(nn.Module):
    """waveform (B, T) at `sample_rate` -> AST input (B, max_length, 128).

    Re-implements the HuggingFace `ASTFeatureExtractor` on tensors so it runs on
    GPU in the training loop: resample to 16 kHz, Kaldi-compatible 128-bin fbank
    (25 ms window, 10 ms shift), pad/crop to `max_length` frames and normalise with
    the AudioSet statistics used during AST pre-training (Gong et al., 2021).
    5-s ESC-50 clips give 498 frames, so max_length=512 (instead of 1024) halves
    the compute; positional embeddings are resized in models/ast.py.
    """

    def __init__(self, sample_rate: int = 16000, target_sr: int = 16000, num_mel_bins: int = 128,
                 max_length: int = 512, mean: float = -4.2677393, std: float = 4.5689974):
        super().__init__()
        self.sample_rate, self.target_sr = sample_rate, target_sr
        self.num_mel_bins, self.max_length = num_mel_bins, max_length
        self.mean, self.std = mean, std
        self.source = "kaldi fbank (AST)"

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        import torchaudio

        if wave.dim() == 3:
            wave = wave.mean(1)
        wave = wave.float()
        if self.sample_rate != self.target_sr:
            wave = torchaudio.functional.resample(wave, self.sample_rate, self.target_sr)
        feats = []
        for w in wave:
            fb = torchaudio.compliance.kaldi.fbank(
                w.unsqueeze(0), htk_compat=True, sample_frequency=self.target_sr, use_energy=False,
                window_type="hanning", num_mel_bins=self.num_mel_bins, dither=0.0, frame_shift=10)
            n = fb.size(0)
            if n < self.max_length:
                fb = F.pad(fb, (0, 0, 0, self.max_length - n))
            feats.append(fb[: self.max_length])
        x = torch.stack(feats)                       # (B, max_length, mel)
        return (x - self.mean) / (self.std * 2)


def build_frontend(fcfg: dict | None, sample_rate: int = 16000) -> nn.Module:
    """Front-end factory driven by the `frontend:` block of a config.
    type: logmel (CNN / CRNN / Conformer)  |  ast_fbank (AST)."""
    fcfg = dict(fcfg or {})
    kind = fcfg.get("type", "logmel")
    sr = fcfg.get("sample_rate", sample_rate)
    if kind == "logmel":
        keys = ("n_mels", "n_fft", "hop_length", "use_dataset_logmel")
        return LogMelFrontend(sample_rate=sr, **{k: fcfg[k] for k in keys if k in fcfg})
    if kind == "ast_fbank":
        keys = ("target_sr", "num_mel_bins", "max_length", "mean", "std")
        return ASTFrontend(sample_rate=sr, **{k: fcfg[k] for k in keys if k in fcfg})
    raise ValueError(f"Unknown frontend type '{kind}'")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
