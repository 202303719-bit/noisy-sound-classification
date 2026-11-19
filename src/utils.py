"""Shared helpers: config loading (with `base:` inheritance + CLI overrides),
seeding, and small bookkeeping utilities used by train.py."""
from __future__ import annotations

import copy
import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import yaml


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_path(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. A config may contain `base: other.yaml`
    (relative to its own folder); the base is loaded first and then overridden.
    `overrides` are strings like "train.lr=0.001" (values parsed as YAML)."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_config(path.parent / cfg.pop("base"))
        cfg = deep_merge(base_cfg, cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key=value, got: {ov}")
        key, val = ov.split("=", 1)
        set_by_path(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def dataset_kwargs(cond: dict) -> dict:
    """Drop None values so we only pass arguments the dataset really needs
    (e.g. ESC50(root, 'train') for clean audio)."""
    return {k: v for k, v in cond.items() if v is not None}


def condition_name(cond: dict) -> str:
    snr = cond.get("snr_db")
    return f"{cond['split']}_clean" if snr is None else f"{cond['split']}_snr{snr}"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:  # DataLoader worker_init_fn
    import torch

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def hardware_info() -> dict:
    info = {"platform": platform.platform(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info


def save_json(obj, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def file_size_mb(path: str | os.PathLike) -> float:
    return os.path.getsize(path) / 1024**2
