#!/usr/bin/env python
"""Train any model of the project from a YAML config and save a checkpoint.

Examples (run from the repository root):
    python src/train.py --config configs/baseline_svm.yaml
    python src/train.py --config configs/baseline_rf.yaml
    python src/train.py --config configs/cnn.yaml --seeds 0 1 2
    python src/train.py --config configs/crnn.yaml --set train.lr=0.0005 train.batch_size=32
    python src/train.py --config configs/cnn.yaml --fast-dev-run      # quick smoke test

Outputs
    checkpoints/<name>_seed<k>.pt | .joblib     best checkpoint (selected on validation)
    experiments/<name>/seed<k>/log.csv          per-epoch (torch) or per-grid-point (sklearn) log
    experiments/<name>/seed<k>/summary.json     best val metrics + computational cost
    experiments/<name>/summary_seeds.json       mean +- SD over seeds
The test split is NEVER touched here; use src/evaluate.py for final evaluation.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

import models as M  # noqa: E402
from utils import (condition_name, dataset_kwargs, file_size_mb, hardware_info,  # noqa: E402
                   load_config, save_json, seed_worker, set_seed)

FAST_DEV_CLIPS = 64  # clips per split in --fast-dev-run


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _resolve(p):
    if p is None:
        return None
    p = Path(p)
    return str(p if p.is_absolute() else ROOT / p)


def build_datasets(cfg: dict, fast: bool):
    """Returns (train_ds, {condition_name: val_ds}). Conditions come from cfg['data']."""
    from datasets.esc50 import ESC50  # team's shared loader (src/datasets/esc50.py)

    d = cfg["data"]
    root, demand = _resolve(d["root"]), _resolve(d.get("demand_dir"))

    def make(cond: dict):
        kw = dataset_kwargs(cond)
        split = kw.pop("split")
        if "snr_db" in kw:                      # noisy condition -> needs DEMAND
            kw["demand_dir"] = demand
        ds = ESC50(root, split, **kw)
        if fast:
            ds = Subset(ds, range(min(len(ds), FAST_DEV_CLIPS)))
        return ds

    train = make(d["train"])
    vals = {condition_name(c): make(c) for c in d["val"]}
    return train, vals


def make_loader(ds, cfg, shuffle: bool, seed: int, batch_size: int | None = None):
    nw = cfg["data"].get("num_workers", 0)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size or cfg["train"]["batch_size"], shuffle=shuffle,
                      drop_last=shuffle and len(ds) > cfg["train"]["batch_size"],
                      num_workers=nw, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=g, persistent_workers=nw > 0)


def compute_metrics(y_true, y_pred) -> dict:
    return {"acc": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def selection_score(val_metrics: dict, metric: str) -> float:
    """Model selection = mean of the chosen metric over all validation conditions."""
    return float(np.mean([m[metric] for m in val_metrics.values()]))


def get_device(cfg) -> torch.device:
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


# --------------------------------------------------------------------------- #
# Baseline (scikit-learn)
# --------------------------------------------------------------------------- #
def train_sklearn(cfg: dict, seed: int, run_dir: Path, ckpt_path: Path, fast: bool) -> dict:
    device = get_device(cfg)
    train_ds, val_dss = build_datasets(cfg, fast)
    feat_cfg = dict(cfg.get("features", {}))
    extractor = M.MFCCStats(**feat_cfg)
    bs = cfg["train"].get("batch_size", 64)

    t0 = time.perf_counter()
    X_tr, y_tr = M.baseline.to_numpy_features(extractor, make_loader(train_ds, cfg, False, seed, bs), device)
    X_val = {k: M.baseline.to_numpy_features(extractor, make_loader(v, cfg, False, seed, bs), device)
             for k, v in val_dss.items()}
    feat_time = time.perf_counter() - t0
    print(f"[features] train {X_tr.shape}, extracted in {feat_time:.1f}s")

    mcfg = cfg["model"]
    grid = M.baseline.param_grid(mcfg.get("params", {}), mcfg.get("search"))
    metric = cfg["train"].get("select_metric", "macro_f1")
    best, rows = None, []
    for i, params in enumerate(grid):
        clf = M.build_classifier(mcfg["name"], params, seed)
        t = time.perf_counter()
        clf.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t
        val = {k: compute_metrics(y, clf.predict(X)) for k, (X, y) in X_val.items()}
        score = selection_score(val, metric)
        row = {"grid_id": i, **{f"p_{k}": v for k, v in params.items()}, "fit_time_s": round(fit_time, 2),
               **{f"{c}_{m}": v for c, mm in val.items() for m, v in mm.items()}, f"val_mean_{metric}": score}
        rows.append(row)
        print(f"[grid {i + 1}/{len(grid)}] {params} -> val mean {metric}={score:.4f} ({fit_time:.1f}s)")
        if best is None or score > best["score"]:
            best = {"score": score, "params": params, "clf": clf, "val": val, "fit_time": fit_time}

    _write_csv(rows, run_dir / "log.csv")

    joblib.dump({"model": best["clf"], "model_name": mcfg["name"], "params": best["params"],
                 "feature_config": feat_cfg, "config": cfg, "seed": seed}, ckpt_path)

    # inference time per clip (feature extraction + prediction), CPU/GPU as configured
    first_val = next(iter(val_dss.values()))
    wave, _ = first_val[0]
    wave = torch.as_tensor(wave)[None].float().to(device)
    n_rep = 20
    t = time.perf_counter()
    for _ in range(n_rep):
        best["clf"].predict(extractor.to(device)(wave).cpu().numpy())
    infer_ms = (time.perf_counter() - t) / n_rep * 1000

    return {
        "model": mcfg["name"], "seed": seed, "best_params": best["params"],
        "val": best["val"], f"val_mean_{metric}": best["score"],
        "feature_dim": int(X_tr.shape[1]),
        "n_params": M.baseline.count_sklearn_params(best["clf"]),
        "train_time_s": round(feat_time + sum(r["fit_time_s"] for r in rows), 2),
        "best_fit_time_s": round(best["fit_time"], 2),
        "inference_ms_per_clip": round(infer_ms, 2),
        "model_size_mb": round(file_size_mb(ckpt_path), 3),
        "checkpoint": str(ckpt_path), "hardware": hardware_info(),
    }


# --------------------------------------------------------------------------- #
# Deep models (PyTorch)
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, tcfg: dict):
    name = tcfg.get("optimizer", "adamw").lower()
    lr, wd = tcfg["lr"], tcfg.get("weight_decay", 0.0)
    # no weight decay on BatchNorm / bias parameters
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() <= 1]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=tcfg.get("momentum", 0.9), nesterov=True)
    raise ValueError(f"Unknown optimizer {name}")


def build_scheduler(opt, tcfg: dict, steps_per_epoch: int):
    total = max(1, tcfg["epochs"] * steps_per_epoch)
    warmup = int(tcfg.get("warmup_epochs", 0) * steps_per_epoch)
    min_ratio = tcfg.get("min_lr_ratio", 0.01)
    kind = tcfg.get("scheduler", "cosine")

    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        prog = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


@torch.no_grad()
def evaluate_torch(model, frontend, loaders: dict, device, criterion, fast: bool = False) -> dict:
    model.eval()
    out = {}
    for name, loader in loaders.items():
        preds, ys, losses = [], [], []
        for i, (wave, y) in enumerate(loader):
            if fast and i >= 2:
                break
            y = torch.as_tensor(y).long().to(device)
            logits = model(frontend(wave.to(device)))
            losses.append(criterion(logits, y).item() * len(y))
            preds.append(logits.argmax(1).cpu())
            ys.append(y.cpu())
        p, t = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        out[name] = {**compute_metrics(t, p), "loss": float(sum(losses) / len(t))}
    return out


def train_torch(cfg: dict, seed: int, run_dir: Path, ckpt_path: Path, fast: bool) -> dict:
    device = get_device(cfg)
    tcfg, mcfg = cfg["train"], cfg["model"]
    if fast:
        tcfg = {**tcfg, "epochs": 1, "warmup_epochs": 0}
    n_classes = cfg["data"].get("n_classes", 50)

    train_ds, val_dss = build_datasets(cfg, fast)
    train_loader = make_loader(train_ds, cfg, True, seed)
    val_loaders = {k: make_loader(v, cfg, False, seed, tcfg["batch_size"] * 2) for k, v in val_dss.items()}

    frontend = M.LogMelFrontend(**cfg.get("frontend", {})).to(device).eval()
    model = M.build_torch_model(mcfg["name"], n_classes, mcfg.get("params")).to(device)
    n_params = M.count_parameters(model)
    print(f"[model] {mcfg['name']}: {n_params:,} trainable params | front-end: {frontend.source} | device: {device}")

    opt = build_optimizer(model, tcfg)
    sched = build_scheduler(opt, tcfg, len(train_loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mixup_alpha = tcfg.get("mixup_alpha", 0.0)
    metric = tcfg.get("select_metric", "macro_f1")
    patience = tcfg.get("early_stopping_patience", 0)
    rng = np.random.default_rng(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_score, best_epoch, best_val, bad_epochs, rows = -1.0, 0, None, 0, []
    t_start = time.perf_counter()

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t_ep, run_loss, n_seen, correct = time.perf_counter(), 0.0, 0, 0
        for i, (wave, y) in enumerate(train_loader):
            if fast and i >= 2:
                break
            y = torch.as_tensor(y).long().to(device, non_blocking=True)
            with torch.no_grad():
                x = frontend(wave.to(device, non_blocking=True))      # fp32 log-mel

            if mixup_alpha > 0:
                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1 - lam) * x[perm]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                if mixup_alpha > 0:
                    loss = lam * loss + (1 - lam) * criterion(logits, y[perm])

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg.get("grad_clip"):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * len(y)
            n_seen += len(y)
            correct += (logits.argmax(1) == y).sum().item()

        val = evaluate_torch(model, frontend, val_loaders, device, criterion, fast)
        score = selection_score(val, metric)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               "train_loss": run_loss / max(1, n_seen), "train_acc": correct / max(1, n_seen),
               **{f"{c}_{m}": v for c, mm in val.items() for m, v in mm.items()},
               f"val_mean_{metric}": score, "epoch_time_s": round(time.perf_counter() - t_ep, 1)}
        rows.append(row)
        _write_csv(rows, run_dir / "log.csv")

        improved = score > best_score
        if improved:
            best_score, best_epoch, best_val, bad_epochs = score, epoch, val, 0
            torch.save({"model_name": mcfg["name"], "model_params": mcfg.get("params", {}),
                        "n_classes": n_classes, "state_dict": model.state_dict(),
                        "frontend": cfg.get("frontend", {}), "frontend_source": frontend.source,
                        "epoch": epoch, "val_metrics": val, "seed": seed, "config": cfg}, ckpt_path)
        else:
            bad_epochs += 1
        print(f"[ep {epoch:3d}] loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val mean {metric} {score:.4f}{' *' if improved else ''} | {row['epoch_time_s']}s")
        if patience and bad_epochs >= patience:
            print(f"[early stop] no improvement for {patience} epochs")
            break

    train_time = time.perf_counter() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None

    # inference latency for a single 5 s clip (front-end + model) with the best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    wave, _ = next(iter(val_dss.values()))[0]
    wave = torch.as_tensor(wave)[None].float().to(device)
    with torch.no_grad():
        for _ in range(3):
            model(frontend(wave))
        if device.type == "cuda":
            torch.cuda.synchronize()
        n_rep, t = 30, time.perf_counter()
        for _ in range(n_rep):
            model(frontend(wave))
        if device.type == "cuda":
            torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t) / n_rep * 1000

    return {
        "model": mcfg["name"], "seed": seed, "best_epoch": best_epoch, "val": best_val,
        f"val_mean_{metric}": best_score, "n_params": n_params,
        "model_size_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2, 3),
        "checkpoint_size_mb": round(file_size_mb(ckpt_path), 3),
        "train_time_s": round(train_time, 1), "epochs_run": len(rows),
        "inference_ms_per_clip": round(infer_ms, 2), "peak_gpu_mem_mb": peak_mem,
        "checkpoint": str(ckpt_path), "hardware": hardware_info(),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def aggregate_seeds(summaries: list[dict], metric: str) -> dict:
    key = f"val_mean_{metric}"
    out = {"n_seeds": len(summaries), "seeds": [s["seed"] for s in summaries]}
    vals = np.array([s[key] for s in summaries])
    out[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    for cond in summaries[0]["val"]:
        for m in summaries[0]["val"][cond]:
            v = np.array([s["val"][cond][m] for s in summaries])
            out[f"{cond}_{m}"] = {"mean": float(v.mean()),
                                  "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a YAML config in configs/")
    ap.add_argument("--seeds", type=int, nargs="+", help="override cfg 'seeds' (e.g. --seeds 0 1 2)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. train.lr=0.0005 model.params.dropout=0.5")
    ap.add_argument("--fast-dev-run", action="store_true",
                    help=f"1 epoch, 2 batches, {FAST_DEV_CLIPS} clips per split (smoke test)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    seeds = args.seeds or cfg.get("seeds", [0])
    name = cfg["name"] + ("_debug" if args.fast_dev_run else "")
    exp_dir = ROOT / cfg.get("output", {}).get("experiment_dir", "experiments") / name
    ckpt_dir = ROOT / cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    kind = M.model_type(cfg["model"]["name"])
    metric = cfg["train"].get("select_metric", "macro_f1")

    summaries = []
    for seed in seeds:
        print(f"\n===== {name} | seed {seed} =====")
        set_seed(seed, cfg.get("deterministic", False))
        run_dir = exp_dir / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        ext = ".joblib" if kind == "sklearn" else ".pt"
        ckpt_path = ckpt_dir / f"{name}_seed{seed}{ext}"
        fn = train_sklearn if kind == "sklearn" else train_torch
        summary = fn(cfg, seed, run_dir, ckpt_path, args.fast_dev_run)
        save_json(summary, run_dir / "summary.json")
        summaries.append(summary)
        print(f"[done] seed {seed}: val mean {metric} = {summary[f'val_mean_{metric}']:.4f} -> {ckpt_path}")

    agg = aggregate_seeds(summaries, metric)
    save_json(agg, exp_dir / "summary_seeds.json")
    m = agg[f"val_mean_{metric}"]
    print(f"\n[{name}] val mean {metric} over {len(seeds)} seed(s): {m['mean']:.4f} +- {m['std']:.4f}")


if __name__ == "__main__":
    main()
