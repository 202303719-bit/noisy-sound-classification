# 12. Hyperparameters

All runs are launched with `python src/train.py --config <file>`. Each YAML file in `configs/`
inherits `configs/base.yaml`, which fixes the parts that must be identical for every model
(split, noise conditions, log-mel front-end, seeds, selection rule). So a model config only
contains what really differs between models. Any value can be overridden from the command line for
the hyperparameter or ablation studies, e.g. `--set train.lr=0.0003 model.params.dropout=0.5`.

## 12.1 Shared settings (all models)

| Setting | Value |
|---|---|
| Input | 5-s ESC-50 clip, mono, 44.1 kHz (shared `datasets.esc50.ESC50` loader) |
| Deep-model input | 128-band log-mel ($2048$-pt FFT, hop $512 \to 128 \times 431$, shared `LogMel` transform) for CNN/CRNN/Conformer; 16 kHz 128-bin Kaldi fbank ($512 \times 128$, AudioSet normalization) for AST |
| Train audio | Clean ESC-50 train folds by default; on-the-fly DEMAND noise mixing (10 train envs, $\text{SNR} \sim \mathcal{U}(-5, 20)\text{ dB}$, $p = 0.8$) enabled in `conformer.yaml` and the data-strategy study |
| Validation conditions | Val folds clean + val folds mixed with **val** DEMAND noise (`NRIVER`, `OMEETING`, `PRESTO`) at 5 dB |
| Model selection | Highest mean macro-F1 over the validation conditions (best epoch / best grid point) |
| Seeds | `0, 1, 2` for every final configuration, reported as mean ± SD |
| Test data | Never used in `train.py`; evaluated once with `src/evaluate.py` across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$ (unseen noise: `PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) |

## 12.2 Table 2: Model architecture comparison (all model families)

| Model | Family | Input | Main layers | Parameters | Pretrained |
|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | Classical ML (baseline) | 240-d MFCC statistics (40 MFCC + Δ + ΔΔ, mean & std) | StandardScaler → RBF-kernel SVM (one-vs-one, 50 classes) | Non-parametric: #SV × 240 (reported after fitting) | No |
| MFCC + RF (`baseline_rf`) | Classical ML (baseline) | 240-d MFCC statistics | StandardScaler → Random Forest (300–1000 trees, Gini) | Non-parametric: total tree nodes (reported after fitting) | No |
| 2-D CNN (`cnn_scratch`) | Classic DL: 2-D CNN | Log-mel $1 \times 128 \times 431$ | BN(input) → SpecAugment → 4 × ConvBlock [2 × (Conv3×3-BN-ReLU), AvgPool 2×2, Dropout2d 0.1] with (32, 64, 128, 256) channels → freq-mean → time mean ‖ max pooling (512-d) → FC 256-ReLU → FC 50 | **1.32 M** (5.0 MB fp32) | No (scratch) |
| CRNN (`crnn_scratch`) | Classic DL: CNN + RNN | Log-mel $1 \times 128 \times 431$ | BN(input) → SpecAugment → 3 × ConvBlock (64, 128, 256; pool 2×2, 2×2, 2×1) → freq-mean → BiGRU (1 layer, 128/dir) over 107 frames → AttentionPool (custom) → FC 50 | **1.49 M** (5.7 MB fp32) | No (scratch) |
| AST Pretrained (`ast_pretrained`) | Modern DL: Audio Spectrogram Transformer | Kaldi fbank $512 \times 128$ (16 kHz) | SpecAugment → $16\times 16$ patch embed (stride $10\times 10 \to 600$ patches) + bilinear-resized 2-D pos. embed → 12 × ViT-Base Transformer encoder blocks ($D=768$, 12 heads) → LayerNorm → Linear 50 | **85.76 M** (327.1 MB fp32) | **Yes** (ImageNet + AudioSet `MIT/ast-finetuned-audioset-10-10-0.4593`) |
| AST Scratch (`ast_scratch`) | Modern DL: Transformer (scratch control) | Kaldi fbank $512 \times 128$ (16 kHz) | Identical to `ast_pretrained` (random initialization, dropout 0.1) | **85.76 M** (327.1 MB fp32) | No (scratch) |
| Conformer (`conformer_full`) | Advanced DL: Convolution-augmented Transformer | Log-mel $1 \times 128 \times 431$ | BN(input) → SpecAugment → 2× stride-2 Conv2d subsampling ($\div 4 \to 108$ frames) + sinusoidal PE → 4 × Macaron ConformerBlock ($\tfrac{1}{2}$FFN + 4-head MHSA + depthwise ConvModule $k=15$ + $\tfrac{1}{2}$FFN, $d=144$) → AttentionPool (custom) → FC 50 | **2.29 M** (8.75 MB fp32) | No (scratch + DEMAND noise mixing) |

## 12.3 Table 3: Hyperparameter configuration

| | MFCC + SVM | MFCC + RF | CNN (scratch) | CRNN (scratch) | AST (pretrained) | AST (scratch) | Conformer (full) |
|---|---|---|---|---|---|---|---|
| Optimizer | libsvm (SMO, convex QP) | Greedy CART splits | AdamW (no decay on BN/bias) | AdamW (no decay on BN/bias) | AdamW (no decay on LN/bias/pos/tokens) | AdamW (no decay on LN/bias/pos/tokens) | AdamW (no decay on BN/LN/bias) |
| Learning rate | n/a | n/a | $10^{-3}$, 5-ep warm-up → cosine to $10^{-5}$ | $10^{-3}$, 5-ep warm-up → cosine to $10^{-5}$ | $2\times 10^{-5}$, 1-ep warm-up → cosine to $2\times 10^{-7}$ | $10^{-4}$, 5-ep warm-up → cosine to $10^{-6}$ | $10^{-3}$, 10-ep warm-up → cosine to $10^{-5}$ |
| Batch size | full batch | bootstrap | 32 | 32 | 16 | 16 | 32 |
| Epochs | n/a | n/a | 100 (patience 20) | 100 (patience 20) | 25 (patience 8) | 60 (patience 15) | 120 (patience 30) |
| Regularization | $C \in \{1, 10, 100\}$, $\gamma \in \{\text{scale}, 10^{-3}, 10^{-2}\}$ | `max_features` $\in \{\text{sqrt}, \text{log2}\}$, `min_leaf` $\in \{1, 2\}$, $N \in \{300, 500, 1000\}$ | WD $0.01$, drop $0.3$, drop2d $0.1$, clip $5.0$ | WD $0.01$, drop $0.3$, drop2d $0.1$, clip $5.0$ | WD $0.01$, drop $0.0$, clip $5.0$ | WD $0.05$, drop $0.1$, attn-drop $0.1$, clip $5.0$ | WD $0.05$, drop $0.1$, head-drop $0.3$, clip $5.0$ |
| Loss | Hinge (SVM dual) | Gini impurity | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) | CE (label smoothing $0.1$) |
| Augmentation | None | None | SpecAugment ($2\times F\le 24$, $2\times T\le 40$) | SpecAugment ($2\times F\le 24$, $2\times T\le 40$) | SpecAugment ($2\times F\le 24$, $2\times T\le 48$) | SpecAugment ($2\times F\le 24$, $2\times T\le 48$) | SpecAugment + DEMAND train noise ($\mathcal{U}(-5, 20)\text{ dB}$, $p=0.8$) |
| Config file | `configs/baseline_svm.yaml` | `configs/baseline_rf.yaml` | `configs/cnn.yaml` | `configs/crnn.yaml` | `configs/ast.yaml` | `configs/ast_scratch.yaml` | `configs/conformer.yaml` |

## 12.4 Justification of the choices

* **AdamW + warm-up + cosine.** AdamW decouples weight decay from the adaptive update [3]. Warm-up avoids unstable early updates (especially for Transformer/Conformer attention layers), and cosine decay is standard across PANNs, AST, and Conformer training recipes [1, 2, 4].
* **Small fine-tuning LR ($2\times 10^{-5}$) for `ast_pretrained` vs. $10^{-4}$ for `ast_scratch`.** Following Gong et al. [1], pretrained AudioSet weights require an order-of-magnitude smaller learning rate to prevent catastrophic forgetting of pretrained patch representations.
* **Label smoothing ($0.1$)** reduces over-confidence on a 50-class problem with only 24 training clips per class and improves probability calibration under distribution shift [5].
* **SpecAugment** [6] forces models not to rely on narrow frequency bands or short time slices that may be masked by environmental noise.
* **On-the-fly DEMAND noise mixing** exposes the model to realistic non-stationary background interference across $-5$ to $20\text{ dB}$ SNR using only the 10 training environments, leaving validation and test environments strictly unseen.

## References

[1] Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," *Interspeech*, 2021.
[2] A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," *Interspeech*, 2020.
[3] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," *ICLR*, 2019.
[4] Q. Kong et al., "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition," *IEEE/ACM TASLP*, 2020.
[5] R. Müller, S. Kornblith, and G. Hinton, "When Does Label Smoothing Help?," *NeurIPS*, 2019.
[6] D. S. Park et al., "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition," *Interspeech*, 2019.
