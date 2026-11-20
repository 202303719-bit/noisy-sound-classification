"""Noise-mixing augmentation with DEMAND (own implementation).

Used for
  * training-time noise augmentation (only the 10 TRAIN environments), and
  * the Gradio demo (mix a chosen environment at a chosen SNR).
The official noisy val/test sets are produced by datasets.esc50.ESC50 (deterministic),
NOT by this class, so evaluation is identical for every model.

Noise files are never loaded completely: for every clip a random segment is read
from disk with soundfile (DEMAND = 16 channels x 5 min per environment).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEMAND_SPLITS = {
    "train": ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "OHALLWAY", "OOFFICE",
              "PSTATION", "SPSQUARE", "TCAR"],
    "val": ["NRIVER", "OMEETING", "PRESTO"],
    "test": ["PCAFETER", "STRAFFIC", "TBUS", "TMETRO"],
}


def mix_at_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db) -> torch.Tensor:
    """clean, noise: (B, T); snr_db: float or (B,) tensor. Returns clean + g*noise with
    10*log10(P_clean / P_scaled_noise) = snr_db."""
    snr = torch.as_tensor(snr_db, dtype=clean.dtype, device=clean.device).reshape(-1, 1)
    p_c = clean.pow(2).mean(-1, keepdim=True).clamp_min(1e-10)
    p_n = noise.pow(2).mean(-1, keepdim=True).clamp_min(1e-10)
    gain = torch.sqrt(p_c / (p_n * 10 ** (snr / 10)))
    return clean + gain * noise


class NoiseMixer:
    def __init__(self, demand_dir: str, envs: list[str], sample_rate: int = 16000,
                 channels: list[int] | None = (1,), snr_range=(-5.0, 20.0), p: float = 1.0,
                 seed: int = 0):
        self.sample_rate, self.snr_range, self.p = sample_rate, tuple(snr_range), p
        self.rng = np.random.default_rng(seed)
        envs = {e.upper() for e in envs}
        self.files: dict[str, list[tuple[str, int, int]]] = {}
        for path in sorted(Path(demand_dir).rglob("*.wav")):
            env = path.parent.name.upper().replace("_48K", "").replace("_16K", "")
            if env not in envs:
                continue
            m = re.search(r"ch(\d+)", path.stem.lower())
            if channels and (m is None or int(m.group(1)) not in channels):
                continue
            info = sf.info(str(path))
            self.files.setdefault(env, []).append((str(path), info.frames, info.samplerate))
        missing = envs - set(self.files)
        if missing:
            raise FileNotFoundError(f"DEMAND environments not found under {demand_dir}: {sorted(missing)}")
        self.envs = sorted(self.files)

    # ------------------------------------------------------------------ #
    def _segment(self, env: str, n_out: int, rng) -> torch.Tensor:
        path, frames, sr = self.files[env][rng.integers(len(self.files[env]))]
        n_in = int(np.ceil(n_out * sr / self.sample_rate)) + 16
        start = int(rng.integers(0, max(1, frames - n_in)))
        data, _ = sf.read(path, start=start, frames=n_in, dtype="float32", always_2d=True)
        x = torch.from_numpy(data[:, 0].copy())
        if sr != self.sample_rate:
            import torchaudio

            x = torchaudio.functional.resample(x, sr, self.sample_rate)
        if x.numel() < n_out:                                  # very short file: loop it
            x = x.repeat(int(np.ceil(n_out / max(1, x.numel()))))
        return x[:n_out]

    def sample_noise(self, batch: int, n_samples: int, env: str | None = None, rng=None) -> torch.Tensor:
        rng = rng or self.rng
        return torch.stack([self._segment(env or self.envs[rng.integers(len(self.envs))], n_samples, rng)
                            for _ in range(batch)])

    # ------------------------------------------------------------------ #
    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """Training augmentation: each clip gets random-environment noise at a random
        SNR ~ U(snr_range) with probability p. wave: (B, T) or (B, 1, T)."""
        squeeze = wave.dim() == 3
        x = wave.squeeze(1) if squeeze else wave
        B, T = x.shape
        apply = torch.from_numpy(self.rng.random(B) < self.p).to(x.device)
        if apply.any():
            noise = self.sample_noise(B, T).to(x.device, x.dtype)
            snr = torch.from_numpy(self.rng.uniform(*self.snr_range, size=B)).to(x.device, x.dtype)
            x = torch.where(apply[:, None], mix_at_snr(x, noise, snr), x)
        return x.unsqueeze(1) if squeeze else x

    def mix_fixed(self, wave: torch.Tensor, env: str, snr_db: float, seed: int = 0) -> torch.Tensor:
        """Deterministic mixing for the demo. wave: (T,) or (B, T)."""
        single = wave.dim() == 1
        x = wave[None] if single else wave
        noise = self.sample_noise(x.size(0), x.size(-1), env=env.upper(),
                                  rng=np.random.default_rng(seed)).to(x.device, x.dtype)
        out = mix_at_snr(x, noise, snr_db)
        return out[0] if single else out


def build_noise_mixer(cfg: dict, root, seed: int = 0) -> NoiseMixer | None:
    """From the `noise_aug:` block of a config (train environments only)."""
    ncfg = cfg.get("noise_aug") or {}
    if not ncfg.get("enabled", False):
        return None
    envs = ncfg.get("envs") or DEMAND_SPLITS["train"]
    leak = set(e.upper() for e in envs) & set(DEMAND_SPLITS["val"] + DEMAND_SPLITS["test"])
    if leak:
        raise ValueError(f"noise_aug uses val/test environments {leak} -> data leakage")
    demand = Path(cfg["data"]["demand_dir"])
    demand = demand if demand.is_absolute() else Path(root) / demand
    return NoiseMixer(str(demand), envs, cfg["data"].get("sample_rate", 16000),
                      channels=ncfg.get("channels", [1]), snr_range=ncfg.get("snr_range", [-5, 20]),
                      p=ncfg.get("p", 1.0), seed=seed)
