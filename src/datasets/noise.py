"""Noise utilities written from scratch (Core Requirement 5)."""
import os
import numpy as np

# DEMAND environments split into train / val / test (test = UNSEEN noise types)
NOISE_SPLITS = {
    "train": ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK",
              "OHALLWAY", "OOFFICE", "PSTATION", "SPSQUARE", "TCAR"],
    "val":   ["NRIVER", "OMEETING", "PRESTO"],
    "test":  ["PCAFETER", "STRAFFIC", "TBUS", "TMETRO"],
}


def _power(x: np.ndarray) -> float:
    return float(np.mean(x.astype(np.float64) ** 2))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Return clean + g * noise so that the SNR equals snr_db."""
    assert clean.shape == noise.shape, "crop/loop noise to the clean length first"
    p_clean, p_noise = _power(clean), _power(noise)
    if p_clean < 1e-12 or p_noise < 1e-12:
        return clean.copy()
    g = np.sqrt(p_clean / (p_noise * 10.0 ** (snr_db / 10.0)))
    mixed = clean + g * noise
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def random_crop(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    start = rng.integers(0, len(noise) - length + 1)
    return noise[start:start + length]


class NoiseBank:
    """Loads one channel (default ch01) of every DEMAND environment of a split."""

    def __init__(self, demand_dir: str, split: str, sr: int = 16000, channel: str = "ch01"):
        import soundfile as sf
        self.envs, self.signals = [], []
        for env in NOISE_SPLITS[split]:
            path = os.path.join(demand_dir, env, f"{channel}.wav")
            if not os.path.exists(path):
                print(f"[NoiseBank] WARNING: missing {path}, skipping")
                continue
            x, file_sr = sf.read(path, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            if file_sr != sr:
                raise ValueError(f"{path} is {file_sr} Hz, expected {sr}. Use the *_16k zips.")
            self.envs.append(env)
            self.signals.append(x)
        if not self.signals:
            raise RuntimeError(f"No DEMAND environments found for split '{split}' in {demand_dir}")

    def sample(self, length: int, rng: np.random.Generator):
        i = rng.integers(0, len(self.signals))
        return self.envs[i], random_crop(self.signals[i], length, rng)