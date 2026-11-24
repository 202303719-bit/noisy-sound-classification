# Environmental Sound Classification under Noise

Deep Learning (AI403) final project, Pharos University in Alexandria, Fall 2026.
Idea 17: Environmental sound classification under noise.

## Team

| Name | Student ID |
|---|---|
| Youssef Walid El_bastwesy | 202303719 |
| Farid Elsayed Elgedawy | 202302103 |
| Taha Abdelhaleem | 202301786 |
| Mohamed Elsayed Badawy Mohamed | 202201851 |
| Youssef Shabaan Saleh Shabaan | 202303487 |

## Project status

| Item | Status |
|---|---|
| Data pipeline, leakage-free split, SNR noise mixer | Done |
| Baselines (MFCC + SVM, MFCC + RF), 3 seeds, test evaluation | Done |
| Classic models (2-D CNN, CRNN), 3 seeds, test evaluation | Done |
| Modern model (AST pretrained), 3 seeds, test evaluation | Done |
| Advanced model (Conformer + noise mixing), 3 seeds, test evaluation | Done (seed 0 used a mis-sized config, see report Section 10.3) |
| Error analysis | Written up for the Random Forest and the pretrained AST; files for all models are in `experiments/eval_baselines/` and `experiments/eval_deep/` |
| Computational cost table (Table 7) | Done |
| AST from scratch (`ast_scratch`, pretrained vs scratch) | **Not run** |
| Ablation study and data-strategy study | **Not run** |
| Learning-rate / batch-size study | **Not run** |
| Gradio demo | Done (tested locally with the CNN and Random Forest checkpoints; not tested with the AST) |

Everything in the report comes from real runs. Items marked "Not run" are stated as such in the report.

## Datasets

- **ESC-50** (sounds): https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0
- **DEMAND** (noise): https://zenodo.org/records/1227121, CC BY-SA 3.0

Full details (versions, licences, preprocessing, split): see `docs/DATASET.md`.

## Environment setup

```bash
pip install -r requirements.txt
```

Hardware used: baselines, evaluation of the baselines and the demo on a 11th Gen Intel Core i5-11400H (CPU, no GPU); CNN, CRNN, Conformer and AST training and their evaluation on a Google Colab Tesla T4 GPU.

## Dataset download and preprocessing

1. Clone ESC-50 into `data/ESC-50`:

   ```bash
   git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50
   ```

2. Download the 17 `*_16k.zip` files from the DEMAND Zenodo page and unzip each into
   `data/DEMAND/<ENV>/` (for example `data/DEMAND/DKITCHEN/ch01.wav`).

3. Verify the data and the noise mixer:

   ```bash
   python data/check_data.py
   ```

Preprocessing (16 kHz mono, 5 s clips, log-mel, noise mixing at a given SNR) happens on the fly
inside `src/datasets/esc50.py`, so there is no separate preprocessing command.

```python
from datasets.esc50 import ESC50, LogMel
train = ESC50("data/ESC-50", "train", snr_db=[20, 10, 5, 0], demand_dir="data/DEMAND")
test  = ESC50("data/ESC-50", "test", snr_db=5, demand_dir="data/DEMAND")
```

## Training (`src/train.py`)

One script trains all models. It reads a YAML config (inheriting `configs/base.yaml`), trains for the given seeds, selects the best epoch / grid point on the **validation** split and saves a checkpoint. It never touches the test split.

```bash
# 0) Smoke test (1 epoch, 64 clips per seed)
python src/train.py --config configs/cnn.yaml --fast-dev-run

# 1) Baselines (CPU, about 1 min per seed)
python src/train.py --config configs/baseline_svm.yaml --seeds 0 1 2
python src/train.py --config configs/baseline_rf.yaml  --seeds 0 1 2

# 2) Classic models on log-mel (GPU recommended)
python src/train.py --config configs/cnn.yaml  --seeds 0 1 2
python src/train.py --config configs/crnn.yaml --seeds 0 1 2

# 3) Modern model: AST pretrained (GPU, about 22 min per seed on a T4)
python src/train.py --config configs/ast.yaml --seeds 0 1 2

# 4) Advanced model: Conformer + DEMAND train noise mixing (GPU, about 8 min per seed)
python src/train.py --config configs/conformer.yaml --seeds 0 1 2
```

Config values can be overridden, e.g. `--set train.lr=0.0003 model.params.dropout=0.5`.

Not run (configs and runner exist, but were never executed):

```bash
python src/train.py --config configs/ast_scratch.yaml --seeds 0 1 2
python scripts/run_study.py configs/studies/ablation.yaml --seeds 0 1 2
python scripts/run_study.py configs/studies/data_strategy.yaml --seeds 0 1 2
```

| Output | Content |
|---|---|
| `checkpoints/<name>_seed<k>.pt` | best deep-model weights + config + front-end info (**not in the repo**) |
| `checkpoints/<name>_seed<k>.joblib` | best scikit-learn pipeline (**not in the repo**) |
| `experiments/<name>/seed<k>/log.csv` | per-epoch (deep) or per-grid-point (baseline) metrics |
| `experiments/<name>/seed<k>/summary.json` | best val metrics, params, train time, inference ms/clip, size, hardware |
| `experiments/<name>/summary_seeds.json` | mean ± SD over the seeds |

Checkpoints are large (up to 330 MB each) and excluded from Git (`.gitignore`). To reproduce the results, re-run the training commands above.

## Evaluation on the test split (`src/evaluate.py`)

Evaluates trained checkpoints on the test fold mixed with the 4 unseen DEMAND test environments (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) at clean, 20, 10, 5, 0 and -5 dB. The two baselines were evaluated on CPU and the deep models on a Colab T4:

```bash
python src/evaluate.py --runs baseline_svm baseline_rf --out experiments/eval_baselines
python src/evaluate.py --runs cnn_scratch crnn_scratch conformer_full ast_pretrained --out experiments/eval_deep
```

Each output folder contains `table4_main.md` (mean ± SD, 95% CI), `acc_vs_snr.png`, `f1_vs_snr.png`, `per_class_f1.csv`, `confusion_<run>.png`, `errors_<run>.csv`, `table6_errors.md`, `error_examples/*.wav` and `table7_cost.md`. Results are already in `experiments/eval_baselines/` and `experiments/eval_deep/`.

## Demo (`demo/app.py`)

Gradio app: upload or record audio, choose a checkpoint, add DEMAND noise at a chosen SNR, and see the top-5 classes. The demo was run locally with the CNN and Random Forest checkpoints; it was not tested with the AST. It needs trained checkpoints in `checkpoints/`, which are not stored in the repo (they are created by `src/train.py`).

```bash
python demo/app.py
```

## Credits

- ESC-50: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015.
- DEMAND: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)", 2013.
- AST: Y. Gong, Y.-A. Chung, J. Glass, "AST: Audio Spectrogram Transformer", Interspeech 2021 (pretrained weights `MIT/ast-finetuned-audioset-10-10-0.4593` from Hugging Face).
- The SNR noise mixer (`src/datasets/noise.py`), the SpecAugment module, the attention pooling layer, the Conformer and the AST positional-embedding resizing were written by the team.