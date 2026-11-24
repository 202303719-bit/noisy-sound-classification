# 12. Hyperparameters

All runs are launched with `python src/train.py --config <file>`. Each YAML file in `configs/`
inherits `configs/base.yaml`, which fixes the parts that must be identical for every model
(split, noise conditions, log-mel front-end, seeds, selection rule). So a model config only
contains what really differs between models. Any value can be overridden from the command line,
e.g. `--set train.lr=0.0003 model.params.dropout=0.5`.

**Status of the runs.** Final results exist for `baseline_svm`, `baseline_rf`, `cnn_scratch`, `crnn_scratch`, `conformer_full` and `ast_pretrained` (3 seeds each). `ast_scratch` is configured below but **was never trained**. The learning-rate / batch-size study (Core Requirement 8) was not run, so every value below is a single chosen configuration, not the result of a search (except the grid search of the two baselines on the validation split).

## 12.1 Shared settings (all models)

| Setting | Value |
|---|---|
| Input | 5-s ESC-50 clip, mono, resampled to 16 kHz by the shared `datasets.esc50.ESC50` loader |
| Deep-model input | 64-band log-mel ($1024$-pt FFT, hop $320 \to 64 \times 251$, shared `LogMel` transform) for CNN/CRNN/Conformer; 128-bin Kaldi fbank (16 kHz, AudioSet normalization) for AST |
| Train audio | Clean ESC-50 train folds by default; on-the-fly DEMAND noise mixing (10 train envs, $\text{SNR} \sim \mathcal{U}(-5, 20)\text{ dB}$, $p = 0.8$) enabled only in `conformer.yaml` |
| Validation conditions | Val fold clean + val fold mixed with **val** DEMAND noise (`NRIVER`, `OMEETING`, `PRESTO`) at 5 dB |
| Model selection | Highest mean macro-F1 over the validation conditions (best epoch / best grid point) |
| Seeds | `0, 1, 2` for every final configuration, reported as mean ± SD |
| Test data | Never used in `train.py`; evaluated once with `src/evaluate.py` across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$ (unseen noise: `PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) |

## 12.2 Table 2: Model architecture comparison (all model families)

| Model | Family | Input | Main layers | Parameters | Pretrained |
|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | Classical ML (baseline) | 240-d MFCC statistics (40 MFCC + Δ + ΔΔ, mean & std) | StandardScaler → RBF-kernel SVM (one-vs-one, 50 classes) | 0.35 M (348,025 as counted by `train.py`) | No |
| MFCC + RF (`baseline_rf`) | Classical ML (baseline) | 240-d MFCC statistics | StandardScaler → Random Forest (1000 trees selected, Gini) | 0.55 M (549,154 as counted by `train.py`) | No |
| 2-D CNN (`cnn_scratch`) | Classic DL: 2-D CNN | Log-mel $1 \times 64 \times 251$ | BN(input) → SpecAugment → 4 × ConvBlock [2 × (Conv3×3-BN-ReLU), AvgPool 2×2, Dropout2d 0.1] with (32, 64, 128, 256) channels → freq-mean → time mean ‖ max pooling (512-d) → FC 256-ReLU → FC 50 | **1.32 M** (5.0 MB fp32) | No (scratch) |
| CRNN (`crnn_scratch`) | Classic DL: CNN + RNN | Log-mel $1 \times 64 \times 251$ | BN(input) → SpecAugment → 3 × ConvBlock (64, 128, 256; pool 2×2, 2×2, 2×1) → freq-mean → BiGRU (1 layer, 128/dir) → AttentionPool (custom) → FC 50 | **1.49 M** (5.7 MB fp32) | No (scratch) |
| AST Pretrained (`ast_pretrained`) | Modern DL: Audio Spectrogram Transformer | Kaldi fbank (16 kHz, 128 bins) | SpecAugment → $16\times 16$ patch embed (stride $10\times 10$) + bilinear-resized 2-D pos. embed → 12 × ViT-Base Transformer encoder blocks ($D=768$, 12 heads) → LayerNorm → Linear 50 | **85.76 M** (327.1 MB fp32) | **Yes** (ImageNet + AudioSet `MIT/ast-finetuned-audioset-10-10-0.4593`) |
| AST Scratch (`ast_scratch`) | Modern DL: Transformer (scratch control) | Kaldi fbank (16 kHz, 128 bins) | Identical to `ast_pretrained` (random initialization, dropout 0.1). **Not trained.** | 85.76 M (configured only) | No (scratch) |
| Conformer (`conformer_full`) | Advanced DL: Convolution-augmented Transformer | Log-mel $1 \times 64 \times 251$ | BN(input) → SpecAugment → 2× stride-2 Conv2d subsampling ($\div 4$ in time and frequency) + sinusoidal PE → 4 × Macaron ConformerBlock ($\tfrac{1}{2}$FFN + 4-head MHSA + depthwise ConvModule $k=15$ + $\tfrac{1}{2}$FFN, $d=144$) → AttentionPool (custom) → FC 50 | **2.15 M** (8.19 MB fp32; seed 0 had 2.29 M, see Section 10.3) | No (scratch + DEMAND noise mixing) |

## 12.3 Table 3: Hyperparameter configuration

"Max epochs" is the upper limit; all deep models use early stopping on validation macro-F1 (patience in brackets), so the number of epochs actually run is lower or equal. `ast_pretrained` ran 16, 25 and 13 epochs for seeds 0, 1 and 2.

| | MFCC + SVM | MFCC + RF | CNN (scratch) | CRNN (scratch) | AST (pretrained) | AST (scratch, not trained) | Conformer (full) |
|---|---|---|---|---|---|---|---|
| Optimizer | libsvm (SMO, convex QP) | Greedy CART splits | AdamW (no decay on BN/bias) | AdamW (no decay on BN/bias) | AdamW (no decay on LN/bias/pos/tokens) | AdamW (no decay on LN/bias/pos/tokens) | AdamW (no decay on BN/LN/bias) |
| Learning rate | n/a | n/a | $10^{-3}$, 5-ep warm-up → cosine to $10^{-5}$ | $10^{-3}$, 5-ep warm-up → cosine to $10^{-5}$ | $2\times 10^{-5}$, 1-ep warm-up → cosine to $2\times 10^{-7}$ | $10^{-4}$, 5-ep warm-up → cosine to $10^{-6}$ | $10^{-3}$, 10-ep warm-up → cosine to $10^{-5}$ |
| Batch size | full batch | bootstrap | 32 | 32 | 16 | 16 | 32 |
| Max epochs | n/a | n/a | 100 (patience 20) | 100 (patience 20) | 25 (patience 8) | 60 (patience 15) | 120 (patience 30) |
| Regularization / search | Grid on validation: $C \in \{1, 10, 100\}$, $\gamma \in \{\text{scale}, 10^{-3}, 10^{-2}\}$. Selected (seed 0): $C=10$, $\gamma=$ scale | Grid on validation: `max_features` $\in \{\text{sqrt}, \text{log2}\}$, `min_leaf` $\in \{1, 2\}$, $N \in \{300, 500, 1000\}$. Selected (seed 0): $N=1000$, sqrt, `min_leaf` 2 | WD $0.01$, drop $0.3$, drop2d $0.1$, clip $5.0$ | WD $0.01$, drop $0.3$, drop2d $0.1$, clip $5.0$ | WD $0.01$, drop $0.0$, clip $5.0$ | WD $0.05$, drop $0.1$, attn-drop $0.1$, clip $5.0$ | WD $0.05$, drop $0.1$, head-drop $0.3$, clip $5.0$ |
| Loss | Hinge (SVM dual) | Gini impurity | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) |
| Augmentation | None | None | SpecAugment ($2\times F\le 24$, $2\times T\le 40$) | SpecAugment ($2\times F\le 24$, $2\times T\le 40$) | SpecAugment ($2\times F\le 24$, $2\times T\le 48$) | SpecAugment ($2\times F\le 24$, $2\times T\le 48$) | SpecAugment ($2\times F\le 24$, $2\times T\le 40$) + DEMAND train noise ($\mathcal{U}(-5, 20)\text{ dB}$, $p=0.8$) |
| Config file | `configs/baseline_svm.yaml` | `configs/baseline_rf.yaml` | `configs/cnn.yaml` | `configs/crnn.yaml` | `configs/ast.yaml` | `configs/ast_scratch.yaml` | `configs/conformer.yaml` |

## 12.4 Justification of the choices

* **AdamW + warm-up + cosine.** AdamW decouples weight decay from the adaptive update [3]. Warm-up avoids unstable early updates (especially for Transformer/Conformer attention layers), and cosine decay is common in AST, Conformer and PANNs recipes [1, 2, 4]. These values were chosen from the literature and not tuned by us.
* **Small fine-tuning LR ($2\times 10^{-5}$) for `ast_pretrained`.** Pretrained weights are fine-tuned with a small learning rate to avoid destroying the pretrained representations, in line with the fine-tuning practice of Gong et al. [1] (who report a similar order of magnitude, $10^{-5}$, for ESC-50). We did not tune this value. The larger rate in the `ast_scratch` configuration ($10^{-4}$) was intended for random initialisation, but that run was not performed.
* **Label smoothing ($0.1$)** reduces over-confidence on a 50-class problem with only 24 training clips per class [5]. We did not measure its effect.
* **SpecAugment** [6] discourages the model from relying on narrow frequency bands or short time slices. We did not run a with/without comparison (Section 15), so its benefit here is not tested.
* **On-the-fly DEMAND noise mixing** (Conformer only) exposes the model to non-stationary background noise using only the 10 training environments, leaving validation and test environments unseen. The training SNR range ($-5$ to $20$ dB) covers the evaluation SNR levels, but the noise environments differ, so this is not leakage.
* **Early stopping on validation macro-F1** selects the epoch without touching the test split.

## References

[1] Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," *Interspeech*, 2021.
[2] A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," *Interspeech*, 2020.
[3] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," *ICLR*, 2019.
[4] Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," *IEEE/ACM TASLP*, 2020.
[5] R. Müller, S. Kornblith, and G. Hinton, "When Does Label Smoothing Help?," *NeurIPS*, 2019.
[6] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," *Interspeech*, 2019.