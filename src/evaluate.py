#!/usr/bin/env python
"""Shared final evaluation for ALL models on the TEST split (unseen DEMAND noise:
PCAFETER, STRAFFIC, TBUS, TMETRO) at several SNRs.

Examples (from the repository root):
  # main comparison: Table 4, accuracy-vs-SNR curve, Table 6 (errors), Table 7 (cost)
  python src/evaluate.py --runs baseline_svm baseline_rf cnn_scratch crnn_scratch \
                                ast_pretrained ast_scratch conformer_full
  # ablation (Table 5) / data-strategy study: rows + deltas defined in the study file
  python src/evaluate.py --study configs/studies/ablation.yaml --out experiments/eval_ablation
  python src/evaluate.py --study configs/studies/data_strategy.yaml --out experiments/eval_data_strategy

Outputs in --out (default experiments/eval):
  results_long.csv        run, seed, snr, acc, macro_f1        (every single evaluation)
  table4_main.md          mean +- SD (and 95% CI) over seeds per model and SNR
  acc_vs_snr.png / f1_vs_snr.png   the key figure: metric vs SNR for all models
  per_class_f1.csv        per-class F1 (clean and error SNR), mean over seeds
  confusion_<run>.png     confusion matrix at eval.error_snr (summed over seeds)
  errors_<run>.csv        every misclassified test clip at error_snr (true, pred, conf, type)
  table6_errors.md        error types, counts, examples, likely cause
  error_examples/         noisy wavs of example failures (for the report / slides)
  table5_ablation.md      only with --study / --reference: change vs reference
  table7_cost.md          params, size, train time, inference time, memory, hardware
  predictions/*.npz       raw probabilities (so analysis can be redone without models)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from inference import Predictor, find_checkpoints, load_class_names  # noqa: E402
from utils import load_config, save_json  # noqa: E402

ESC_GROUPS = ["Animals", "Natural soundscapes & water", "Human non-speech", "Interior/domestic",
              "Exterior/urban"]   # ESC-50 major category = target // 10


def snr_label(s) -> str:
    return "clean" if s is None else f"{s:g}dB"


def mean_sd_ci(v) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return float(v.mean()), 0.0, 0.0
    from scipy import stats

    sd = v.std(ddof=1)
    return float(v.mean()), float(sd), float(stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)))


def fmt(v, pct=True) -> str:
    m, sd, _ = mean_sd_ci(v)
    k = 100 if pct else 1
    return f"{m * k:.1f} ± {sd * k:.1f}"


def clip_names(ds) -> list[str]:
    """Try to recover file names from the dataset object (falls back to indices)."""
    for attr in ("files", "filenames", "paths", "file_list"):
        v = getattr(ds, attr, None)
        if v is not None and len(v) == len(ds):
            return [Path(str(x)).name for x in v]
    for attr in ("df", "meta", "metadata"):
        v = getattr(ds, attr, None)
        if v is not None and hasattr(v, "columns") and "filename" in v.columns and len(v) == len(ds):
            return [str(x) for x in v["filename"]]
    return [f"test#{i}" for i in range(len(ds))]


# --------------------------------------------------------------------------- #
def run_predictions(cfg, runs: dict[str, dict[int, Path]], snrs, device, batch_size, num_workers):
    """For every SNR the test set is loaded ONCE and fed to every model/seed."""
    from datasets.esc50 import ESC50

    d = cfg["data"]
    root = str(ROOT / d["root"]) if not Path(d["root"]).is_absolute() else d["root"]
    demand = str(ROOT / d["demand_dir"]) if not Path(d["demand_dir"]).is_absolute() else d["demand_dir"]

    preds = defaultdict(dict)            # preds[(run, seed)][snr_label] = probs (N, C)
    labels, names, waves_cache = {}, None, {}
    predictors = {(run, seed): Predictor(path, device)          # load every checkpoint once
                  for run, seeds in runs.items() for seed, path in seeds.items()}
    for snr in snrs:
        kw = {} if snr is None else {"snr_db": snr, "demand_dir": demand}
        ds = ESC50(root, "test", **kw)
        names = names or clip_names(ds)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        out = defaultdict(list)
        ys = []
        print(f"[eval] test {snr_label(snr)}: {len(ds)} clips x {len(predictors)} models")
        for wave, y in loader:
            ys.append(np.asarray(y))
            for key, pred in predictors.items():
                out[key].append(pred.predict_proba(wave))
        labels[snr_label(snr)] = np.concatenate(ys)
        for key in predictors:
            preds[key][snr_label(snr)] = np.concatenate(out[key])
        waves_cache[snr_label(snr)] = ds
    del predictors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds, labels, names, waves_cache


# --------------------------------------------------------------------------- #
def plot_vs_snr(rows, runs, snrs, metric, path, display):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs_lab = [snr_label(s) for s in snrs]
    order = sorted(range(len(snrs)), key=lambda i: (snrs[i] is None, snrs[i] if snrs[i] is not None else 0))
    xs_lab = [xs_lab[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        m, s = [], []
        for lab in xs_lab:
            v = [r[metric] for r in rows if r["run"] == run and r["snr"] == lab]
            mm, sd, _ = mean_sd_ci(v)
            m.append(mm * 100)
            s.append(sd * 100)
        x = np.arange(len(xs_lab))
        ax.plot(x, m, marker="o", label=display.get(run, run))
        ax.fill_between(x, np.array(m) - s, np.array(m) + s, alpha=0.15)
    ax.set_xticks(np.arange(len(xs_lab)), xs_lab)
    ax.set_xlabel("SNR (unseen test noise: PCAFETER, STRAFFIC, TBUS, TMETRO)")
    ax.set_ylabel({"acc": "Accuracy (%)", "macro_f1": "Macro-F1 (%)"}[metric])
    ax.set_title(f"{'Accuracy' if metric == 'acc' else 'Macro-F1'} vs SNR (mean ± SD over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, class_names, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def error_analysis(run, preds, labels, names, class_names, err_lab, out_dir, datasets, max_wavs=8):
    """Confusion matrix + misclassification list + Table 6 rows for one run."""
    seeds = sorted(s for (r, s) in preds if r == run)
    y = labels[err_lab]
    C = len(class_names)
    cm = np.zeros((C, C), dtype=int)
    rows = []
    for s in seeds:
        p = preds[(run, s)][err_lab]
        yhat = p.argmax(1)
        cm += confusion_matrix(y, yhat, labels=range(C))
        clean = preds[(run, s)].get("clean")
        clean_ok = (clean.argmax(1) == labels["clean"]) if clean is not None else None
        for i in np.where(yhat != y)[0]:
            etype = ("noise-induced (correct on clean)" if clean_ok is not None and clean_ok[i]
                     else "inherent (wrong also on clean)" if clean_ok is not None else "error")
            rows.append({"seed": s, "idx": int(i), "file": names[i], "true": class_names[y[i]],
                         "pred": class_names[yhat[i]], "confidence": round(float(p[i].max()), 3),
                         "p_true": round(float(p[i, y[i]]), 3), "type": etype,
                         "same_major_category": (y[i] // 10) == (yhat[i] // 10)})
    plot_confusion(cm, class_names, out_dir / f"confusion_{run}.png",
                   f"{run} - test @ {err_lab} (summed over {len(seeds)} seeds)")
    if rows:
        with open(out_dir / f"errors_{run}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -r["confidence"]))

    # ---- Table 6 rows (error type, count, example, likely cause) ----
    n_seeds = max(1, len(seeds))
    table = []
    by_type = Counter(r["type"] for r in rows)
    for t, n in by_type.items():
        ex = next(r for r in rows if r["type"] == t)
        cause = ("additive noise masks the event (low SNR, overlapping spectrum)" if t.startswith("noise")
                 else "class confusion independent of noise (similar acoustic texture / short event)")
        table.append((t, n / n_seeds, f"{ex['file']}: {ex['true']} → {ex['pred']}", cause))
    pair_counts = Counter((r["true"], r["pred"]) for r in rows)
    for (t_cls, p_cls), n in pair_counts.most_common(5):
        ex = next(r for r in rows if r["true"] == t_cls and r["pred"] == p_cls)
        same = ex["same_major_category"]
        cause = (f"same ESC-50 major category ({ESC_GROUPS[class_names.index(t_cls) // 10]}) → similar spectra"
                 if same else "different categories → noise texture resembles predicted class")
        table.append((f"confusion {t_cls} → {p_cls}", n / n_seeds, ex["file"], cause))
    hi = [r for r in rows if r["confidence"] >= 0.8]
    if hi:
        table.append(("high-confidence errors (conf ≥ 0.8)", len(hi) / n_seeds,
                      f"{hi[0]['file']}: {hi[0]['true']} → {hi[0]['pred']} ({hi[0]['confidence']})",
                      "over-confident model / possible label ambiguity"))

    # ---- export a few failure examples as wav (seed with most errors first) ----
    try:
        import soundfile as sf

        ex_dir = out_dir / "error_examples"
        ex_dir.mkdir(exist_ok=True)
        seen = set()
        for r in sorted(rows, key=lambda r: -r["confidence"]):
            if len(seen) >= max_wavs or r["idx"] in seen:
                continue
            seen.add(r["idx"])
            wave, _ = datasets[err_lab][r["idx"]]
            wave = np.asarray(torch.as_tensor(wave).float().reshape(-1))
            fname = f"{run}_{err_lab}_{r['idx']}_{r['true']}_as_{r['pred']}.wav".replace(" ", "")
            sf.write(ex_dir / fname, wave, getattr(datasets[err_lab], "sample_rate", 16000))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not export error examples: {e}")
    return table, cm


def cost_table(runs, display) -> list[str]:
    lines = ["| Model | Params | Size (MB) | Train time / seed | Inference (ms/clip) | Peak GPU mem (MB) | Hardware |",
             "|---|---|---|---|---|---|---|"]
    for run in runs:
        files = sorted((ROOT / "experiments" / run).glob("seed*/summary.json"))
        if not files:
            lines.append(f"| {display.get(run, run)} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        S = [json.load(open(f, encoding="utf-8")) for f in files]
        tt = np.array([s.get("train_time_s", np.nan) for s in S], dtype=float)
        inf = np.array([s.get("inference_ms_per_clip", np.nan) for s in S], dtype=float)
        mem = [s.get("peak_gpu_mem_mb") for s in S if s.get("peak_gpu_mem_mb")]
        size = S[0].get("model_size_mb") or S[0].get("checkpoint_size_mb")
        hw = S[0].get("hardware", {})
        hw_s = hw.get("gpu") or hw.get("cpu") or "?"
        n = S[0].get("n_params_total", S[0].get("n_params", 0))
        n_s = f"{n / 1e6:.2f} M" if n >= 1e5 else f"{n:,}"
        lines.append(f"| {display.get(run, run)} | {n_s} | {size} | {np.nanmean(tt) / 60:.1f} min | "
                     f"{np.nanmean(inf):.1f} | {f'{np.mean(mem):.0f}' if mem else '-'} | {hw_s} |")
    return lines


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run names (= config `name`)")
    ap.add_argument("--study", help="study YAML (configs/studies/*.yaml): runs + deltas for Table 5")
    ap.add_argument("--reference", help="compare every run against this run (Table 5)")
    ap.add_argument("--config", default="configs/base.yaml", help="config with data + eval settings")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--snrs", nargs="*", help="override eval.snrs, e.g. clean 20 10 5 0 -5")
    ap.add_argument("--analyze", nargs="*", help="runs for error analysis (default: best run)")
    ap.add_argument("--out", default="experiments/eval")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = load_config(ROOT / args.config)
    out_dir = ROOT / args.out
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))

    runs, display, compare = list(args.runs), {}, {}
    if args.study:
        st = yaml.safe_load(open(ROOT / args.study, encoding="utf-8"))
        for r in st["runs"]:
            runs.append(r["name"])
            display[r["name"]] = r["name"]
            compare[r["name"]] = r.get("compare_to", st.get("reference"))
        removed = {r["name"]: r.get("removed", "") for r in st["runs"]}
    else:
        removed = {}
    if args.reference:
        compare.update({r: args.reference for r in runs})
    runs = list(dict.fromkeys(runs))
    if not runs:
        ap.error("give --runs and/or --study")

    ckpt_dir = ROOT / cfg.get("output", {}).get("checkpoint_dir", "checkpoints")
    found = {r: find_checkpoints(ckpt_dir, r, args.seeds) for r in runs}
    for r, s in found.items():
        print(f"[ckpt] {r}: seeds {sorted(s) or 'NONE (skipped)'}")
    found = {r: s for r, s in found.items() if s}
    runs = [r for r in runs if r in found]

    snrs = cfg.get("eval", {}).get("snrs", [None, 20, 10, 5, 0, -5])
    if args.snrs:
        snrs = [None if s.lower() in ("clean", "none", "null") else float(s) for s in args.snrs]
    err_snr = cfg.get("eval", {}).get("error_snr", 0)
    err_lab = snr_label(err_snr if err_snr in snrs else snrs[-1])
    class_names = load_class_names(ROOT / cfg["data"]["root"], cfg["data"].get("n_classes", 50))

    preds, labels, names, datasets = run_predictions(cfg, found, snrs, device, args.batch_size,
                                                     args.num_workers)

    # ---------------- metrics ----------------
    rows, per_class = [], defaultdict(list)
    for (run, seed), by_snr in preds.items():
        np.savez_compressed(out_dir / "predictions" / f"{run}_seed{seed}.npz",
                            labels=np.stack([labels[k] for k in by_snr]), snrs=np.array(list(by_snr)),
                            probs=np.stack(list(by_snr.values())))
        for lab, p in by_snr.items():
            yhat = p.argmax(1)
            y = labels[lab]
            rows.append({"run": run, "seed": seed, "snr": lab, "acc": accuracy_score(y, yhat),
                         "macro_f1": f1_score(y, yhat, average="macro", zero_division=0)})
            if lab in ("clean", err_lab):
                per_class[(run, lab)].append(f1_score(y, yhat, average=None, labels=range(len(class_names)),
                                                      zero_division=0))
    with open(out_dir / "results_long.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    labs = [snr_label(s) for s in snrs]
    noisy = [lab for lab in labs if lab != "clean"]

    def vals(run, lab, m):
        return [r[m] for r in rows if r["run"] == run and r["snr"] == lab]

    def avg_noisy(run, m):  # per-seed average over noisy SNRs
        seeds = sorted({r["seed"] for r in rows if r["run"] == run})
        return [np.mean([r[m] for r in rows if r["run"] == run and r["seed"] == s and r["snr"] in noisy])
                for s in seeds]

    # ---------------- Table 4 ----------------
    t4 = ["**Table 4 - Main results on the test split (mean ± SD over seeds, %).** Primary metric: "
          "macro-F1; secondary: accuracy. Noise = unseen DEMAND test environments.", "",
          "| Model | Seeds | " + " | ".join(f"F1 {lab}" for lab in labs) + " | F1 avg noisy | Acc clean | "
          f"Acc {err_lab} |",
          "|---|---|" + "---|" * (len(labs) + 3)]
    for run in runs:
        n = len(found[run])
        t4.append(f"| {display.get(run, run)} | {n} | " + " | ".join(fmt(vals(run, lab, 'macro_f1')) for lab in labs)
                  + f" | **{fmt(avg_noisy(run, 'macro_f1'))}** | {fmt(vals(run, 'clean', 'acc'))} | "
                  f"{fmt(vals(run, err_lab, 'acc'))} |")
    t4 += ["", "95% CI half-width (t-distribution) of F1 avg noisy: " +
           ", ".join(f"{r}: ±{mean_sd_ci(avg_noisy(r, 'macro_f1'))[2] * 100:.1f}" for r in runs)]
    (out_dir / "table4_main.md").write_text("\n".join(t4), encoding="utf-8")

    for m, fn in (("acc", "acc_vs_snr.png"), ("macro_f1", "f1_vs_snr.png")):
        plot_vs_snr(rows, runs, snrs, m, out_dir / fn, display)

    # ---------------- per-class ----------------
    with open(out_dir / "per_class_f1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "class", "f1_clean_mean", f"f1_{err_lab}_mean", "drop"])
        for run in runs:
            c = np.mean(per_class.get((run, "clean"), [np.zeros(len(class_names))]), axis=0)
            e = np.mean(per_class.get((run, err_lab), [np.zeros(len(class_names))]), axis=0)
            for k, name in enumerate(class_names):
                w.writerow([run, name, round(c[k], 4), round(e[k], 4), round(c[k] - e[k], 4)])

    # ---------------- Table 5 (deltas) ----------------
    if compare:
        t5 = ["**Table 5 - Ablation / study results (test, macro-F1 %, mean ± SD over seeds).**", "",
              f"| Run | Component removed / strategy | F1 clean | F1 {err_lab} | F1 avg noisy | Δ avg noisy vs reference |",
              "|---|---|---|---|---|---|"]
        for run in runs:
            ref = compare.get(run)
            delta = "-"
            if ref and ref != run and ref in found:
                d = np.mean(avg_noisy(run, "macro_f1")) - np.mean(avg_noisy(ref, "macro_f1"))
                delta = f"{d * 100:+.1f} (vs {ref})"
            t5.append(f"| {run} | {removed.get(run, '')} | {fmt(vals(run, 'clean', 'macro_f1'))} | "
                      f"{fmt(vals(run, err_lab, 'macro_f1'))} | {fmt(avg_noisy(run, 'macro_f1'))} | {delta} |")
        (out_dir / "table5_ablation.md").write_text("\n".join(t5), encoding="utf-8")

    # ---------------- Table 6 (errors) ----------------
    analyze = args.analyze or [max(runs, key=lambda r: np.mean(avg_noisy(r, "macro_f1")))]
    t6 = [f"**Table 6 - Error analysis (test @ {err_lab}; counts are per seed, averaged).**", "",
          "| Model | Error type | Count | Example | Likely cause |", "|---|---|---|---|---|"]
    for run in analyze:
        if run not in runs:
            continue
        table, _ = error_analysis(run, preds, labels, names, class_names, err_lab, out_dir, datasets)
        for etype, cnt, ex, cause in table:
            t6.append(f"| {run} | {etype} | {cnt:.1f} | {ex} | {cause} |")
    t6 += ["", "> Listen to `error_examples/*.wav` and refine the 'likely cause' column by hand."]
    (out_dir / "table6_errors.md").write_text("\n".join(t6), encoding="utf-8")

    # ---------------- Table 7 (cost) ----------------
    t7 = ["**Table 7 - Computational cost** (from experiments/<run>/seed*/summary.json).", ""]
    t7 += cost_table(runs, display)
    (out_dir / "table7_cost.md").write_text("\n".join(t7), encoding="utf-8")

    save_json({"runs": runs, "snrs": labs, "error_snr": err_lab,
               "avg_noisy_f1": {r: mean_sd_ci(avg_noisy(r, "macro_f1"))[:2] for r in runs}},
              out_dir / "summary.json")
    print("\n".join(t4))
    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
