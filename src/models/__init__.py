"""Model registry. One file per model family:
    baseline.py  -> MFCC + SVM / Random Forest        (baseline, sklearn)
    cnn.py       -> CNN / CRNN on log-mel, scratch    (classic)
    ast.py       -> Audio Spectrogram Transformer     (modern, pretrained vs scratch)
    conformer.py -> Conformer encoder, scratch        (advanced)
"""
from .baseline import MFCCStats, build_classifier  # noqa: F401
from .cnn import CNN, CRNN
from .conformer import Conformer
from .layers import (ASTFrontend, LogMelFrontend, SpecAugment, build_frontend,  # noqa: F401
                     count_parameters)


def _ast(**kw):
    from .ast import AST  # lazy: needs `transformers`
    return AST(**kw)


TORCH_MODELS = {
    "cnn": CNN,
    "crnn": CRNN,
    "conformer": Conformer,
    "ast": _ast,
}

SKLEARN_MODELS = ("svm", "rf")


def model_type(name: str) -> str:
    if name in SKLEARN_MODELS:
        return "sklearn"
    if name in TORCH_MODELS:
        return "torch"
    raise KeyError(f"Unknown model '{name}'. Torch: {list(TORCH_MODELS)}, sklearn: {SKLEARN_MODELS}")


def build_torch_model(name: str, n_classes: int, params: dict | None = None):
    return TORCH_MODELS[name](n_classes=n_classes, **(params or {}))
