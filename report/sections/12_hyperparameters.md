# 12. Hyperparameters

All runs are launched with `python src/train.py --config <file>`. Each YAML file in `configs/`
inherits `configs/base.yaml`, which fixes the parts that must be identical for every model
(split, noise conditions, log-mel front-end, seeds, selection rule). So a model config only
contains what really differs between models. Any value can be overridden from the command line for
the hyperparameter study, e.g. `--set train.lr=0.0003 model.params.dropout=0.5`.

## 12.1 Shared settings (all models)

| Setting | Value |
|---|---|
| Input | 5-s ESC-50 clip, mono, 16 kHz (shared `datasets.esc50.ESC50` loader) |
| Deep-model input | 64-band log-mel (1024-pt FFT, hop 320 → 64 × 251), shared `LogMel` transform |
| Train audio | Clean ESC-50 train folds (noise-augmented training is part of the data-strategy study) |
| Validation conditions | Val folds clean + val folds mixed with **val** DEMAND noise (NRIVER, OMEETING, PRESTO) at 5 dB |
| Model selection | Highest mean macro-F1 over the validation conditions (best epoch / best grid point) |
| Seeds | 0, 1, 2 for every final configuration, reported as mean ± SD |
| Test data | Never used in `train.py`; evaluated once with `src/evaluate.py` (unseen noise: PCAFETER, STRAFFIC, TBUS, TMETRO) |

## 12.2 Table 2: Model architecture comparison (baseline + classic models)

| Model | Family | Input | Main layers | Parameters | Pretrained |
|---|---|---|---|---|---|
| MFCC + SVM | Classical ML (baseline) | 240-d MFCC statistics (40 MFCC + Δ + ΔΔ, mean & std) | StandardScaler → RBF-kernel SVM (one-vs-one, 50 classes) | Non-parametric: #support vectors × 240 (reported after fitting) | No |
| MFCC + RF | Classical ML (baseline) | 240-d MFCC statistics | StandardScaler → Random Forest (300–1000 trees, Gini) | Non-parametric: total tree nodes (reported after fitting) | No |
| CNN (scratch) | Classic DL: 2-D CNN | Log-mel 1 × 64 × 251 | BN(input) → SpecAugment → 4 × ConvBlock [2 × (Conv3×3-BN-ReLU), AvgPool 2×2, Dropout2d 0.1] with 32-64-128-256 channels → freq-mean → time mean ‖ max pooling (512-d) → FC 256-ReLU → FC 50 | **1.32 M** (5.0 MB fp32) | No (trained from scratch) |
| CRNN (scratch) | Classic DL: CNN + RNN | Log-mel 1 × 64 × 251 | BN(input) → SpecAugment → 3 × ConvBlock (64-128-256; pool 2×2, 2×2, 2×1) → freq-mean → BiGRU (1 layer, 128 per direction) over 62 frames → attention pooling (own implementation) → FC 50 | **1.49 M** (5.7 MB fp32) | No (trained from scratch) |

Design notes:
* **Time mean + max pooling (CNN).** Max pooling captures short impulsive events (door knock, dog bark)
  and mean pooling captures stationary textures (rain, engine). Using both gave a richer 512-d embedding
  at no parameter cost.
* **CRNN keeps temporal resolution.** The last block pools only along frequency, so the GRU sees 107
  time steps (~0.05 s each). Attention pooling learns which frames contain the event and which contain
  only background noise, which is the motivation for hypothesis H2 (temporal modelling helps under noise).
* **SpecAugment and attention pooling** are written from scratch in `src/models/layers.py`
  (requirement 5). SpecAugment fills masked regions with the per-sample mean so input statistics do not
  shift.

## 12.3 Table 3: Hyperparameter configuration

| | MFCC + SVM | MFCC + RF | CNN (scratch) | CRNN (scratch) |
|---|---|---|---|---|
| Optimizer | libsvm (SMO, convex QP) | Greedy CART splits | AdamW (no decay on BN/bias) | AdamW (no decay on BN/bias) |
| Learning rate | n/a | n/a | 1e-3, 5-epoch linear warm-up → cosine decay to 1e-5 | 1e-3, 5-epoch linear warm-up → cosine decay to 1e-5 |
| Batch size | n/a (full batch) | n/a (bootstrap samples) | 32 | 32 |
| Epochs | n/a | n/a | 100 max, early stopping (patience 20 on val macro-F1) | 100 max, early stopping (patience 20) |
| Regularization | C ∈ {1, 10, **100**} (grid), γ ∈ {scale, 1e-3, 1e-2} | max_features ∈ {sqrt, log2}, min_samples_leaf ∈ {1, 2}, n_trees ∈ {300, 500, 1000} | Weight decay 0.01, dropout 0.3 (head), Dropout2d 0.1 (blocks), label smoothing 0.1, grad-clip 5.0 | Weight decay 0.01, dropout 0.3, Dropout2d 0.1, label smoothing 0.1, grad-clip 5.0 |
| Loss | Hinge (SVM dual) | Gini impurity | Cross-entropy (label smoothing 0.1) | Cross-entropy (label smoothing 0.1) |
| Augmentation | None | None | SpecAugment: 2 freq masks (F ≤ 24 bins) + 2 time masks (T ≤ 40 frames); mixup off (α = 0, data-strategy study) | Same as CNN |
| Selection | Best grid point on val (mean macro-F1) | Best grid point on val | Best epoch on val | Best epoch on val |
| Precision / HW | CPU float64 | CPU, all cores | Mixed precision (AMP) on GPU | Mixed precision (AMP) on GPU |
| Config file | `configs/baseline_svm.yaml` | `configs/baseline_rf.yaml` | `configs/cnn.yaml` | `configs/crnn.yaml` |

> Values in bold or in grids are replaced with the selected values after the runs (see
> `experiments/<name>/seed*/summary.json → best_params`).

## 12.4 Justification of the choices

* **AdamW + warm-up + cosine.** AdamW decouples weight decay from the adaptive update [3]. Warm-up avoids
  unstable early updates with BatchNorm on a small dataset (~1.2 k training clips), and cosine decay is a
  common default for audio CNNs trained from scratch [4].
* **Label smoothing (0.1)** reduces over-confidence on a 50-class problem with only 40 clips per class,
  which also helps calibration under noise [5].
* **SpecAugment** [6] is the standard spectrogram augmentation. It forces the model not to depend on a
  single frequency band or time segment, which we expect to matter when noise masks parts of the
  spectrum (tested in the ablation: `--set model.params.spec_augment.enabled=false`).
* **Early stopping on noisy + clean validation** selects weights that generalise to *unseen* noise
  environments instead of the clean training distribution.
* **Equal tuning budget.** Each baseline gets 9–12 grid points. Each deep model gets the same
  hyperparameter-study budget (learning rate ∈ {3e-4, 1e-3, 3e-3} × dropout ∈ {0.3, 0.5}) through
  `--set` overrides, logged under `experiments/`.

## References

[3] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," *ICLR*, 2019.
[4] Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition,"
*IEEE/ACM TASLP*, 2020.
[5] R. Müller, S. Kornblith and G. Hinton, "When Does Label Smoothing Help?," *NeurIPS*, 2019.
[6] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition,"
*Interspeech*, 2019.
