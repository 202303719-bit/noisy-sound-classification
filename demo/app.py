#!/usr/bin/env python
"""Gradio live demo: environmental sound classification under noise.

    python demo/app.py                                  # all checkpoints in checkpoints/
    python demo/app.py --share                          # public link (Colab / Kaggle)
    python demo/app.py --ckpt checkpoints/ast_pretrained_seed0.pt

Upload a file or record from the microphone, optionally mix in an UNSEEN DEMAND
noise environment at a chosen SNR, and get the top-5 classes + the log-mel
spectrogram of what the model actually heard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from augment import DEMAND_SPLITS, NoiseMixer  # noqa: E402
from inference import Predictor, load_class_names  # noqa: E402
from utils import load_config  # noqa: E402

CLIP_SECONDS = 5.0


def list_checkpoints(ckpt_dir: Path) -> list[Path]:
    files = [p for p in sorted(ckpt_dir.glob("*")) if p.suffix in (".pt", ".joblib") and "_debug" not in p.stem]
    # prefer seed0 of each run to keep the dropdown short
    seed0 = [p for p in files if p.stem.endswith("_seed0")]
    return seed0 or files


def to_mono_float(audio) -> tuple[int, np.ndarray]:
    sr, x = audio
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1) if x.shape[1] <= 8 else x.mean(axis=0)
    return int(sr), x


def logmel_figure(wave: torch.Tensor, sr: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=512, n_mels=128)(wave)
    db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel).numpy()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma",
              extent=[0, wave.numel() / sr, 0, 128])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel bin")
    ax.set_title("log-mel spectrogram of the (noisy) input")
    fig.tight_layout()
    return fig


def build_app(cfg, ckpts: list[Path], device):
    import gradio as gr

    sr = cfg["data"].get("sample_rate", 16000)
    clip = int(CLIP_SECONDS * sr)
    class_names = load_class_names(ROOT / cfg["data"]["root"], cfg["data"].get("n_classes", 50))
    demand = ROOT / cfg["data"]["demand_dir"]
    envs = DEMAND_SPLITS["test"] + DEMAND_SPLITS["val"] + DEMAND_SPLITS["train"]
    try:
        mixer = NoiseMixer(str(demand), envs, sample_rate=sr)
        env_choices = ["none"] + mixer.envs
    except FileNotFoundError:
        try:
            mixer = NoiseMixer(str(demand), DEMAND_SPLITS["test"], sample_rate=sr)
            env_choices = ["none"] + mixer.envs
        except FileNotFoundError:
            mixer, env_choices = None, ["none"]
            print("[demo] DEMAND not found -> noise mixing disabled")

    cache: dict[str, Predictor] = {}

    def get_pred(name: str) -> Predictor:
        if name not in cache:
            cache[name] = Predictor(next(p for p in ckpts if p.name == name), device)
        return cache[name]

    def classify(audio, model_name, env, snr):
        if audio is None:
            raise gr.Error("Upload or record a sound first.")
        in_sr, x = to_mono_float(audio)
        wave = torch.from_numpy(x)
        if in_sr != sr:
            wave = torchaudio.functional.resample(wave, in_sr, sr)
        peak = wave.abs().max()
        if peak > 1.0:
            wave = wave / peak
        if mixer is not None and env != "none":
            wave = mixer.mix_fixed(wave, env, float(snr), seed=0)
        probs = get_pred(model_name).predict_long(wave, clip)
        top = np.argsort(probs)[::-1][:5]
        label = {class_names[i]: float(probs[i]) for i in top}
        out_audio = (sr, np.clip(wave.numpy(), -1, 1).astype(np.float32))
        return label, out_audio, logmel_figure(wave, sr)

    with gr.Blocks(title="ESC-50 under noise") as app:
        gr.Markdown("## Environmental sound classification under noise\n"
                    "Upload / record a sound, choose a model, add unseen DEMAND noise at an SNR and "
                    "see the top-5 predictions. (Test environments: PCAFETER, STRAFFIC, TBUS, TMETRO.)")
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Input sound")
                model = gr.Dropdown([p.name for p in ckpts], value=ckpts[0].name, label="Model checkpoint")
                env = gr.Dropdown(env_choices, value=env_choices[1] if len(env_choices) > 1 else "none",
                                  label="Noise environment (DEMAND)")
                snr = gr.Slider(-10, 30, value=10, step=1, label="SNR (dB) - lower = noisier")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                label = gr.Label(num_top_classes=5, label="Top-5 predictions")
                noisy = gr.Audio(label="What the model heard (input + noise)", type="numpy")
                plot = gr.Plot(label="log-mel")
        btn.click(classify, [audio, model, env, snr], [label, noisy, plot])
        snr.release(classify, [audio, model, env, snr], [label, noisy, plot])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", nargs="*", help="specific checkpoint files (default: checkpoints/*_seed0.*)")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    ckpts = [Path(p) for p in args.ckpt] if args.ckpt else list_checkpoints(ROOT / "checkpoints")
    if not ckpts:
        sys.exit("No checkpoints found in checkpoints/. Train a model first (src/train.py).")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_app(cfg, ckpts, device).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
