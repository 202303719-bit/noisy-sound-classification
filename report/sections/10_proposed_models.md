# 10. Proposed and Advanced Models

We cover the three required deep-learning categories (Core Requirement 4): **Classic** (2-D CNN and CRNN with attention pooling), **Modern** (Audio Spectrogram Transformer, AST, pretrained) and **Advanced** (Conformer trained with on-the-fly DEMAND noise mixing). All models are trained for 3 seeds (0, 1, 2) and selected on the validation split only. Trained results are reported in Section 13 and computational cost in Section 16.2.

---

## 10.1 Classic Models: 2-D CNN and CRNN (`src/models/cnn.py`)

Both models are trained **from scratch** on the shared log-mel spectrogram (`datasets.esc50.LogMel`: 64 mel bands, 1,024-point FFT, hop 320, input $1 \times 64 \times 251$) and start with a single-channel `BatchNorm2d` followed by our `SpecAugment` module (`src/models/layers.py`).

### 2-D CNN (1.32 M parameters)
* **Architecture:** four VGG-style `ConvBlock` stages with channel widths $(32, 64, 128, 256)$. Each stage applies two $3 \times 3$ convolutions with BatchNorm and ReLU, followed by $2 \times 2$ average pooling and spatial dropout.
* **Pooling and head:** the feature map is averaged over frequency and pooled over time with **concatenated mean and max pooling** ($256 + 256 = 512$ dimensions), followed by a two-layer MLP classifier ($512 \to 256 \to 50$).
* **Rationale:** max pooling keeps short impulsive events even when noise dominates the rest of the clip, while mean pooling captures stationary textures.

### CRNN with attention pooling (1.49 M parameters)
* **Architecture:** three `ConvBlock` stages $(64, 128, 256)$ where the last stage pools only along frequency, so most of the time resolution is kept. The frequency-averaged sequence goes through a 1-layer bidirectional GRU (128 hidden units per direction) and is aggregated by our **learnable soft-attention pooling** (`AttentionPool` in `src/models/layers.py`):
  $$\alpha_t = \frac{\exp(w^\top \tanh(W h_t + b))}{\sum_{\tau=1}^{T'} \exp(w^\top \tanh(W h_\tau + b))}, \qquad z = \sum_{t=1}^{T'} \alpha_t h_t.$$
* **Rationale:** under non-stationary noise, uniform averaging mixes noise-only frames with event frames, and attention pooling can in principle down-weight noise-dominated frames. We did not run an ablation (Section 15), so this effect is a motivation, not a tested result.

---

## 10.2 Modern Model: Audio Spectrogram Transformer (`src/models/ast.py`)

### Architecture and front-end
The **Audio Spectrogram Transformer (AST)** [1] is an attention-based architecture adapted from the Vision Transformer (DeiT-Base: 12 layers, 12 heads, embedding dimension 768). In our implementation it has **85,757,234 trainable parameters**.
* **Front-end (`ASTFrontend`, `src/models/layers.py`):** resamples to 16 kHz and extracts 128-bin Kaldi-compatible log-mel filterbanks (the format AST was pretrained with), normalised with the AudioSet global statistics. Unlike the CNN and CRNN, AST therefore does not use our 64-band `LogMel`.
* **Positional-embedding interpolation (custom component):** the official AudioSet checkpoint (`MIT/ast-finetuned-audioset-10-10-0.4593`) was trained on longer inputs than our 5-s clips, so we implement 2-D positional-embedding resizing (`resize_pos_embed` in `src/models/ast.py`) to fit the pretrained positional grid to our shorter sequence.

### Pretrained vs. scratch (Core Requirement 7): configured but not run
`configs/ast.yaml` (`ast_pretrained`) fine-tunes the pretrained checkpoint [1] end-to-end with AdamW. `configs/ast_scratch.yaml` (`ast_scratch`) defines the same architecture with random initialisation and a different learning-rate and regularisation setup, but **it was never trained**. Only `ast_pretrained` has results. Therefore **the required pretrained-vs-scratch comparison was not performed**, and we cannot attribute the performance of `ast_pretrained` to pretraining alone (the model is also much larger than the other models).

### Training in practice
`ast_pretrained` was trained for 3 seeds on a Colab Tesla T4 with early stopping on validation macro-F1 (patience 8). The runs stopped after 16, 25 and 13 epochs for seeds 0, 1 and 2, at about 72 s per epoch, with a measured mean training time of **22.3 min per seed** (Table 7). Validation macro-F1 was 0.9236, 0.9202 and 0.9125.

---

## 10.3 Advanced Model: Conformer with On-the-Fly Noise Mixing (`src/models/conformer.py`)

### Motivation
AST is large (85.76 M parameters, 327 MB checkpoint). We wanted a compact model that combines local spectral-temporal patterns (convolution) with global sequence modelling (self-attention), so we implemented a **Conformer** [2] from scratch (**2.15 M parameters**, about 40 times fewer than AST) and trained it with leakage-free DEMAND noise mixing (`src/augment.py`).

### Architecture (`Conformer`)
1. **Convolutional subsampling (`ConvSubsampling`):** two stride-2 $3 \times 3$ convolutions with SiLU reduce the normalised log-mel input by $4\times$ in time and frequency, followed by a projection to $d_{\text{model}} = 144$ and sinusoidal positional encodings.
2. **Macaron Conformer blocks ($N = 4$, $d_{\text{model}} = 144$, $H = 4$ heads):** each block places multi-head self-attention and a depthwise-separable convolution module between two half-step feed-forward networks:
   $$\tilde{x} = x + \tfrac{1}{2}\text{FFN}_1(x), \quad x' = \tilde{x} + \text{MHSA}(\tilde{x}), \quad x'' = x' + \text{ConvModule}(x'), \quad y = \text{LN}\!\left(x'' + \tfrac{1}{2}\text{FFN}_2(x'')\right).$$
   The `ConvModule` applies LayerNorm, pointwise convolution, GLU, 1-D depthwise convolution (kernel 15), BatchNorm, SiLU and a pointwise convolution.
3. **Pooling and head:** the output sequence is aggregated with our `AttentionPool` layer and classified by a linear head.
4. **Ablation switches:** `use_mhsa`, `use_conv_module` and `pooling` can be toggled from the configuration. **No ablation was run** (Section 15), so the contribution of each sub-module is unknown.

### Noise-mixing augmentation (`src/augment.py → NoiseMixer`)
With `noise_aug.enabled: true` (`configs/conformer.yaml`), each training clip is mixed with probability 0.8 with a random segment of the **10 training DEMAND environments**, at an SNR drawn uniformly from -5 to 20 dB, using the same mixing equation as Section 7.2. `build_noise_mixer` raises an error if a validation (`NRIVER`, `OMEETING`, `PRESTO`) or test (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) environment is requested, which prevents noise-environment leakage.

The Conformer is the only from-scratch model trained with noise mixing, so its better robustness in Section 13 cannot be attributed to the architecture alone. Its training SNR range (-5 to 20 dB) covers the evaluation levels, but the test noise environments are never seen in training.

**Configuration note:** seed 0 of the Conformer was trained with `n_mels: 128` (2,294,532 parameters) before the configuration was corrected to 64 mel bands. Seeds 1 and 2 used the corrected configuration (2,147,076 parameters).

## References

[1] Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," *Proc. Interspeech*, 2021.
[2] A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," *Proc. Interspeech*, 2020.