"""Baseline: hand-crafted MFCC statistics + classical classifier (SVM / Random Forest).

Feature vector per clip (default n_mfcc=40 -> 240 dims):
    [mean(MFCC), std(MFCC), mean(dMFCC), std(dMFCC), mean(ddMFCC), std(ddMFCC)]
computed over all frames of the 5 s clip. This is the classic "bag of frames"
representation used in the original ESC-50 paper (Piczak, 2015).
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torchaudio
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# Feature extractor
# --------------------------------------------------------------------------- #
class MFCCStats(torch.nn.Module):
    """waveform (B, T) or (B, 1, T) -> fixed-length MFCC statistics (B, 6*n_mfcc)."""

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40, n_mels: int = 128,
                 win_ms: float = 46.0, hop_ms: float = 23.0, f_min: float = 0.0,
                 f_max: float | None = None):
        super().__init__()
        win = int(round(sample_rate * win_ms / 1000))
        hop = int(round(sample_rate * hop_ms / 1000))
        n_fft = 1 << (win - 1).bit_length()  # next power of two
        self.n_mfcc = n_mfcc
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            log_mels=True,
            melkwargs=dict(n_fft=n_fft, win_length=win, hop_length=hop, n_mels=n_mels,
                           f_min=f_min, f_max=f_max, center=True),
        )
        self.deltas = torchaudio.transforms.ComputeDeltas(win_length=9)

    @property
    def dim(self) -> int:
        return 6 * self.n_mfcc

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 3:          # (B, 1, T) -> (B, T)
            wave = wave.mean(1)
        m = self.mfcc(wave)          # (B, n_mfcc, frames)
        d1 = self.deltas(m)
        d2 = self.deltas(d1)
        feats = [f(x, dim=-1) for x in (m, d1, d2) for f in (torch.mean, torch.std)]
        return torch.cat(feats, dim=1)


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #
def build_svm(C: float = 10.0, gamma="scale", kernel: str = "rbf", class_weight=None,
              seed: int = 0, **_) -> Pipeline:
    # No `probability=True` (deprecated in scikit-learn 1.9). For scores in the demo /
    # error analysis use `decision_function` (one-vs-rest margins) + softmax.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, gamma=gamma, kernel=kernel, class_weight=class_weight,
                    decision_function_shape="ovr", random_state=seed)),
    ])


def build_rf(n_estimators: int = 500, max_depth=None, max_features="sqrt",
             min_samples_leaf: int = 1, class_weight=None, seed: int = 0,
             n_jobs: int = -1, **_) -> Pipeline:
    # Trees don't need scaling, but keeping the same pipeline shape makes
    # inference code identical for both baselines.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       class_weight=class_weight, n_jobs=n_jobs,
                                       random_state=seed)),
    ])


SKLEARN_BUILDERS = {"svm": build_svm, "rf": build_rf}


def build_classifier(name: str, params: dict, seed: int = 0) -> Pipeline:
    if name not in SKLEARN_BUILDERS:
        raise KeyError(f"Unknown baseline '{name}'. Options: {list(SKLEARN_BUILDERS)}")
    return SKLEARN_BUILDERS[name](seed=seed, **params)


def param_grid(fixed: dict, search: dict | None) -> list[dict]:
    """Cartesian product of `search` values on top of `fixed` params.
    Model selection is done on the *validation* split (never on test)."""
    if not search:
        return [dict(fixed)]
    keys = list(search)
    return [{**fixed, **dict(zip(keys, vals))}
            for vals in itertools.product(*(search[k] for k in keys))]


def count_sklearn_params(pipe: Pipeline) -> int:
    """Rough 'parameter count' for the computational-cost table."""
    clf = pipe.named_steps["clf"]
    if isinstance(clf, SVC):
        # support vectors * feature dim + dual coefs + intercepts
        sv = clf.support_vectors_
        return int(sv.size + clf.dual_coef_.size + clf.intercept_.size)
    if isinstance(clf, RandomForestClassifier):
        return int(sum(est.tree_.node_count for est in clf.estimators_))  # total tree nodes
    return 0


def to_numpy_features(extractor: MFCCStats, loader, device="cpu") -> tuple[np.ndarray, np.ndarray]:
    extractor = extractor.to(device).eval()
    xs, ys = [], []
    for wave, y in loader:
        xs.append(extractor(wave.to(device).float()).cpu().numpy())
        ys.append(np.asarray(y))
    return np.concatenate(xs), np.concatenate(ys)
