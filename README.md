# Environmental Sound Classification under Noise

Deep Learning (AI403) final project, Pharos University in Alexandria, Fall 2026.
Idea 17: Environmental sound classification under noise.

Team: <
youssef walid El_bastwesy             202303719
farid elsayed Elgedawy                202302103
taha abdelhaleem                      202301786
Mohamed Elsayed Badawy Mohamed        202201851
Youssef Shabaan Saleh Shabaan         202303487
>

## Datasets

- **ESC-50** (sounds): https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0
- **DEMAND** (noise): https://zenodo.org/records/1227121, CC BY-SA 3.0

Full details (versions, licences, preprocessing, split): see `docs/DATASET.md`.

## Environment setup

pip install torch torchaudio soundfile pandas numpy

## Dataset download and preprocessing

1. Clone ESC-50 into `data/ESC-50`:

   git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50

2. Download the 17 `*_16k.zip` files from the DEMAND Zenodo page and unzip each into
   `data/DEMAND/<ENV>/` (for example `data/DEMAND/DKITCHEN/ch01.wav`).

3. Verify the data and the noise mixer:

   python data/check_data.py

Preprocessing (16 kHz mono, 5 s clips, log-mel, noise mixing at a given SNR) happens on the fly
inside `src/datasets/esc50.py`, so there is no separate preprocessing command.

## Using the dataset in code

    from datasets.esc50 import ESC50, LogMel
    train = ESC50("data/ESC-50", "train", snr_db=[20, 10, 5, 0], demand_dir="data/DEMAND")
    test  = ESC50("data/ESC-50", "test", snr_db=5, demand_dir="data/DEMAND")

## Credits

- ESC-50: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015.
- DEMAND: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)", 2013.
- The SNR noise mixer (`src/datasets/noise.py`) was written by the team.


All models are trained with one script, which reads a YAML config, trains, selects the best
epoch / grid point on the **validation** split and saves a checkpoint. It never touches the test split.

```bash
# 0) smoke test (1 epoch, 2 batches, 64 clips): checks the data + model pipeline in about 1 min
python src/train.py --config configs/cnn.yaml --fast-dev-run

# 1) Baselines: MFCC statistics + SVM / Random Forest (CPU, a few minutes)
python src/train.py --config configs/baseline_svm.yaml
python src/train.py --config configs/baseline_rf.yaml

# 2) Classic models trained from scratch on log-mel (GPU recommended)
python src/train.py --config configs/cnn.yaml
python src/train.py --config configs/crnn.yaml

# options
python src/train.py --config configs/cnn.yaml --seeds 0            # single seed
python src/train.py --config configs/cnn.yaml --set train.lr=0.0003 model.params.dropout=0.5
python src/train.py --config configs/cnn.yaml --set model.params.spec_augment.enabled=false   # ablation
python src/train.py --config configs/cnn.yaml --set data.train.snr_db=10                      # noisy training
```

| Output | Content |
|---|---|
| `checkpoints/<name>_seed<k>.pt` | best CNN/CRNN weights + config + front-end info |
| `checkpoints/<name>_seed<k>.joblib` | best scikit-learn pipeline (scaler + classifier) + feature config |
| `experiments/<name>/seed<k>/log.csv` | per-epoch (deep) or per-grid-point (baseline) metrics |
| `experiments/<name>/seed<k>/summary.json` | best val metrics, params, train time, inference ms/clip, size, hardware |
| `experiments/<name>/summary_seeds.json` | mean ± SD over the seeds |

### Loading a checkpoint (for evaluate.py / demo)

```python
import sys, torch, joblib; sys.path.insert(0, "src")
import models as M

ck = torch.load("checkpoints/cnn_scratch_seed0.pt", map_location="cpu", weights_only=False)
model = M.build_torch_model(ck["model_name"], ck["n_classes"], ck["model_params"])
model.load_state_dict(ck["state_dict"]); model.eval()
frontend = M.LogMelFrontend(**ck["frontend"])
logits = model(frontend(wave[None]))                  # wave: (T,) tensor from ESC50

bl = joblib.load("checkpoints/baseline_svm_seed0.joblib")
feats = M.MFCCStats(**bl["feature_config"])(wave[None]).numpy()
pred = bl["model"].predict(feats)
```
## Training, Evaluation, Studies & Live Demo

### 1. Training all model families (`src/train.py`)

`src/train.py` reads a YAML config (inheriting shared split/preprocessing settings from `configs/base.yaml`), trains across 3 seeds (`0, 1, 2`), selects the best checkpoint on the **validation** conditions (`val_clean` + `val_snr5`), and never touches the test split.

```bash
# 0) Smoke test (1 epoch, 2 batches, 64 clips)
python src/train.py --config configs/cnn.yaml --fast-dev-run

# 1) Baselines: MFCC statistics + SVM / Random Forest (CPU, ~1-2 min)
python src/train.py --config configs/baseline_svm.yaml --seeds 0 1 2
python src/train.py --config configs/baseline_rf.yaml  --seeds 0 1 2

# 2) Classic models from scratch on log-mel (GPU recommended)
python src/train.py --config configs/cnn.yaml  --seeds 0 1 2
python src/train.py --config configs/crnn.yaml --seeds 0 1 2

# 3) Modern model: Audio Spectrogram Transformer (Pretrained vs. Scratch)
#    Slowest model (~10 min/seed on Colab/Kaggle T4 GPU) -> start early on GPU!
python src/train.py --config configs/ast.yaml         --seeds 0 1 2
python src/train.py --config configs/ast_scratch.yaml --seeds 0 1 2

# 4) Advanced model: Conformer from scratch + on-the-fly DEMAND train noise mixing
python src/train.py --config configs/conformer.yaml --seeds 0 1 2
```

### 2. Running the Ablation & Data-Strategy Studies (`scripts/run_study.py`)

```bash
# Train all ablation variants (Table 5: remove MHSA, ConvModule, AttentionPool, SpecAugment, NoiseAug)
python scripts/run_study.py configs/studies/ablation.yaml --seeds 0 1 2 --skip-existing

# Train all data-strategy variants on CNN (None vs. SpecAugment vs. DEMAND Noise vs. Mixup vs. All)
python scripts/run_study.py configs/studies/data_strategy.yaml --seeds 0 1 2 --skip-existing
```

### 3. Shared Final Evaluation on the Test Split (`src/evaluate.py`)

Evaluates all trained checkpoints on the **unseen test split** mixed with the 4 unseen DEMAND test environments (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$, and automatically outputs:
- `table4_main.md` (mean ± SD and 95% CI over 3 seeds)
- `acc_vs_snr.png` & `f1_vs_snr.png` (Accuracy & Macro-F1 vs. SNR robustness curves)
- `per_class_f1.csv` (per-class F1 on clean vs. noisy test audio)
- `confusion_<run>.png`, `errors_<run>.csv`, `table6_errors.md`, and `error_examples/*.wav` (Table 6 error analysis)
- `table7_cost.md` (Table 7 computational cost: parameters, model size, train time, inference ms/clip, peak GPU RAM, hardware)

```bash
# Main model comparison (Tables 4, 6, 7 + Accuracy/F1 vs. SNR curves)
python src/evaluate.py --runs baseline_svm baseline_rf cnn_scratch crnn_scratch \
                              ast_pretrained ast_scratch conformer_full \
                       --out experiments/eval

# Ablation study table (Table 5 with deltas vs. reference model)
python src/evaluate.py --study configs/studies/ablation.yaml --out experiments/eval_ablation

# Data-strategy study table
python src/evaluate.py --study configs/studies/data_strategy.yaml --out experiments/eval_data_strategy
```

### 4. Interactive Gradio Live Demo (`demo/app.py`)

Allows uploading an audio file or recording from the microphone, picking any trained checkpoint, selecting a DEMAND noise environment, adjusting the SNR slider ($-10\text{ dB}$ to $+30\text{ dB}$), listening to the mixed audio, inspecting its log-mel spectrogram, and viewing top-5 class probabilities:

```bash
python demo/app.py                       # local http://127.0.0.1:7860
python demo/app.py --share               # public Gradio link (for Colab / Kaggle / live presentation)
```
