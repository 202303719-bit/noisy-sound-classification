"""Shared inference for every model family (used by evaluate.py and demo/app.py).

    pred = Predictor("checkpoints/cnn_scratch_seed0.pt")
    probs = pred.predict_proba(wave_batch)          # (B, 50) numpy, rows sum to 1
"""
from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
import torch

import models as M


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_class_names(esc50_root: str | Path, n_classes: int = 50) -> list[str]:
    """Class names from ESC-50/meta/esc50.csv (target -> category)."""
    names = [str(i) for i in range(n_classes)]
    meta = Path(esc50_root) / "meta" / "esc50.csv"
    if meta.exists():
        with open(meta, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names[int(row["target"])] = row["category"]
    return names


class Predictor:
    def __init__(self, ckpt_path: str | Path, device: str | torch.device = "cpu", n_classes: int = 50):
        self.path = Path(ckpt_path)
        self.device = torch.device(device)
        self.n_classes = n_classes
        if self.path.suffix == ".joblib":
            b = joblib.load(self.path)
            self.kind, self.name = "sklearn", b["model_name"]
            self.clf = b["model"]
            self.extractor = M.MFCCStats(**b["feature_config"]).to(self.device).eval()
            self.sample_rate = b["feature_config"].get("sample_rate", 16000)
        else:
            ck = torch.load(self.path, map_location=self.device, weights_only=False)
            self.kind, self.name = "torch", ck["model_name"]
            params = dict(ck.get("model_params") or {})
            if self.name == "ast":
                params["pretrained"] = False          # weights come from the checkpoint
                params["arch_config"] = ck.get("model_arch")
            self.model = M.build_torch_model(self.name, ck["n_classes"], params).to(self.device)
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.sample_rate = ck.get("sample_rate", 16000)
            self.frontend = M.build_frontend(ck.get("frontend"), self.sample_rate).to(self.device).eval()
            self.n_classes = ck["n_classes"]

    @torch.no_grad()
    def predict_proba(self, wave: torch.Tensor) -> np.ndarray:
        wave = torch.as_tensor(wave).float().to(self.device)
        if wave.dim() == 1:
            wave = wave[None]
        if wave.dim() == 3:
            wave = wave.mean(1)
        if self.kind == "torch":
            return torch.softmax(self.model(self.frontend(wave)).float(), dim=1).cpu().numpy()
        feats = self.extractor(wave).cpu().numpy()
        clf = self.clf.named_steps["clf"]
        if hasattr(self.clf, "predict_proba") and hasattr(clf, "predict_proba") and \
                getattr(clf, "probability", True) is not False:
            p = self.clf.predict_proba(feats)
        else:
            p = softmax_np(self.clf.decision_function(feats))
        out = np.zeros((len(feats), self.n_classes), dtype=np.float32)
        out[:, clf.classes_.astype(int)] = p          # classes missing from training -> 0
        return out

    def predict_long(self, wave: torch.Tensor, clip_samples: int) -> np.ndarray:
        """Any-length mono waveform (T,): split into 5-s windows (50% overlap),
        average the probabilities. Short audio is zero-padded."""
        wave = torch.as_tensor(wave).float()
        if wave.numel() <= clip_samples:
            return self.predict_proba(torch.nn.functional.pad(wave, (0, clip_samples - wave.numel())))[0]
        hop = clip_samples // 2
        starts = list(range(0, wave.numel() - clip_samples + 1, hop))
        windows = torch.stack([wave[s:s + clip_samples] for s in starts])
        return self.predict_proba(windows).mean(0)


def find_checkpoints(ckpt_dir: str | Path, run: str, seeds=None) -> dict[int, Path]:
    """checkpoints/<run>_seed<k>.(pt|joblib) -> {seed: path}"""
    out = {}
    for p in sorted(Path(ckpt_dir).glob(f"{run}_seed*")):
        if p.suffix not in (".pt", ".joblib"):
            continue
        try:
            seed = int(p.stem.rsplit("_seed", 1)[1])
        except ValueError:
            continue
        if seeds is None or seed in seeds:
            out[seed] = p
    return out
