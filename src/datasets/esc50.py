"""ESC-50 dataset with leakage-free official-fold split + optional noise mixing.
train = folds 1-3, val = fold 4, test = fold 5.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .noise import NoiseBank, mix_at_snr

SR = 16000
CLIP_LEN = 5 * SR
FOLDS = {"train": [1, 2, 3], "val": [4], "test": [5]}


def load_wav_16k(path: str) -> np.ndarray:
    import soundfile as sf
    import torchaudio.functional as AF
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
    if len(x) < CLIP_LEN:
        x = np.pad(x, (0, CLIP_LEN - len(x)))
    return x[:CLIP_LEN]


class ESC50(Dataset):
    """
    split   : 'train' | 'val' | 'test'
    snr_db  : None -> clean | float -> fixed SNR | list -> random SNR from list
    demand_dir : path to data/DEMAND (needed if snr_db is not None)
    For val/test the noise and SNR are fixed per clip (deterministic).
    """

    def __init__(self, root, split, snr_db=None, demand_dir=None, seed=0,
                 cache=True, noise_split=None):
        meta = pd.read_csv(os.path.join(root, "meta", "esc50.csv"))
        self.df = meta[meta["fold"].isin(FOLDS[split])].reset_index(drop=True)
        self.audio_dir = os.path.join(root, "audio")
        self.split, self.snr_db, self.seed = split, snr_db, seed
        self.deterministic = split != "train"
        self.noise_bank = None
        if snr_db is not None:
            self.noise_bank = NoiseBank(demand_dir, noise_split or split, SR)
        self.cache = {} if cache else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def _clean(self, i):
        if self.cache is not None and i in self.cache:
            return self.cache[i]
        x = load_wav_16k(os.path.join(self.audio_dir, self.df.loc[i, "filename"]))
        if self.cache is not None:
            self.cache[i] = x
        return x

    def __getitem__(self, i):
        x = self._clean(i)
        if self.noise_bank is not None:
            rng = np.random.default_rng(self.seed * 100003 + i) if self.deterministic else self.rng
            snr = self.snr_db if np.isscalar(self.snr_db) else float(rng.choice(self.snr_db))
            _, n = self.noise_bank.sample(CLIP_LEN, rng)
            x = mix_at_snr(x, n, snr)
        return torch.from_numpy(x), int(self.df.loc[i, "target"])


class LogMel(torch.nn.Module):
    """waveform (B, T) -> log-mel (B, 1, n_mels, frames)"""

    def __init__(self, sr=SR, n_fft=1024, hop=320, n_mels=64, fmin=50, fmax=8000):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, f_min=fmin, f_max=fmax)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, wav):
        return self.db(self.mel(wav)).unsqueeze(1)