"""Model registry. One file per model family:
    baseline.py -> MFCC + SVM / Random Forest   (type: sklearn)
    cnn.py      -> CNN / CRNN on log-mel        (type: torch)
Other families (modern / advanced) can register themselves in TORCH_MODELS.
"""
from .baseline import MFCCStats, build_classifier  # noqa: F401
from .cnn import CNN, CRNN
from .layers import LogMelFrontend, SpecAugment, count_parameters  # noqa: F401

TORCH_MODELS = {
    "cnn": CNN,
    "crnn": CRNN,
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
