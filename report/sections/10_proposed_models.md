# 10. Proposed and Advanced Models

To test our hypotheses across the three required deep-learning categories (Core Requirement 4), we move from local time–frequency pattern detectors (**Classic**: 2-D CNN and CRNN) to large-scale transfer learning with global self-attention (**Modern**: Audio Spectrogram Transformer, AST) and finally to a hybrid local-and-global sequence architecture trained with domain-randomized noise mixing (**Advanced**: Conformer).

---

## 10.1 Classic Models: 2-D CNN and CRNN (`src/models/cnn.py`)

Both classic architectures are trained **from scratch** on the shared 128-band log-mel spectrogram ($1 \times 128 \times 431$) and begin with a single-channel `BatchNorm2d` layer followed by our custom `SpecAugment` module (`src/models/layers.py`).

### 2-D CNN (1.32 M parameters)
* **Architecture:** Four VGG-style `ConvBlock` stages with channel widths $(32, 64, 128, 256)$. Each stage applies two $3 \times 3$ convolutions with BatchNorm and ReLU, followed by $2 \times 2$ average pooling and spatial `Dropout2d(0.1)`.
* **Dual Global Pooling:** After the fourth block, the feature map $(B, 256, 8, 26)$ is averaged across the 8 frequency bins and pooled across time using **concatenated mean and max pooling** ($256 + 256 = 512$-d), feeding a two-layer MLP classifier ($512 \to 256 \to 50$).
* **Rationale:** Max pooling isolates short impulsive events (e.g., *door wood knock*, *dog*, *glass breaking*) even when background noise dominates the rest of the 5-s clip, whereas mean pooling captures stationary acoustic textures (e.g., *rain*, *engine*, *vacuum cleaner*).

### CRNN with Attention Pooling (1.49 M parameters)
* **Architecture:** Three `ConvBlock` stages $(64, 128, 256)$ where the third stage pools only along frequency (`pool=(2, 1)`), preserving $T' = 107$ time steps ($\approx 47\text{ ms}$ per frame). The frequency-averaged sequence $(B, 107, 256)$ is processed by a 1-layer bidirectional GRU ($128$ hidden units per direction $\to 256$-d) and aggregated by our custom **learnable soft-attention pooling** (`AttentionPool` in `src/models/layers.py`):
  $$\alpha_t = \frac{\exp(w^\top \tanh(W h_t + b))}{\sum_{\tau=1}^{T'} \exp(w^\top \tanh(W h_\tau + b))}, \qquad z = \sum_{t=1}^{T'} \alpha_t h_t.$$
* **Rationale:** Under non-stationary noise (e.g., *PCAFETER* babble or *TBUS* acceleration), uniform temporal averaging mixes noise-only frames with target-event frames. Attention pooling learns to down-weight noise-dominated frames and focus on high-SNR segments.

---

## 10.2 Modern Model: Audio Spectrogram Transformer (`src/models/ast.py`)

### Architecture and Input Front-End (`ASTFrontend`)
The **Audio Spectrogram Transformer (AST)** [1] is a purely attention-based architecture adapted from Vision Transformer (ViT / DeiT-Base, 12 layers, 12 heads, embedding dimension $D = 768$, $\approx 85.76\text{ M}$ parameters).
* **Front-End (`src/models/layers.py → ASTFrontend`):** Resamples waveforms to $16\text{ kHz}$ and extracts 128-dimensional Kaldi-compatible log-mel filterbanks (25 ms Povey/Hanning window, 10 ms hop $\to 498$ frames for a 5-s clip, padded to $T = 512$), normalized with the AudioSet global statistics ($\mu = -4.268$, $\sigma = 4.569 \times 2$).
* **Patchification and 2-D Positional Embedding Interpolation (Custom Component):** AST splits the spectrogram into $16 \times 16$ patches with stride $10 \times 10$, yielding $F_p = 12$ frequency patches and $T_p = 50$ time patches ($12 \times 50 = 600$ patch tokens plus `[CLS]` and distillation tokens $= 602$ tokens). Because the official AudioSet checkpoint (`MIT/ast-finetuned-audioset-10-10-0.4593`) was trained with $T = 1024$ frames ($12 \times 101 = 1212$ patches), we implement 2-D bilinear positional-embedding interpolation (`resize_pos_embed` in `src/models/ast.py`) to reshape the pretrained positional grid from $(12, 101)$ to $(12, 50)$. This cuts sequence length in half and reduces self-attention compute and memory by $\sim 4\times$ without padding waste.

### Pretrained vs. Scratch Comparison (Core Requirement 7)
Because ESC-50 provides only 1,200 training clips (24 clips per class in our split), Transformers lack the inductive biases of translation equivariance and locality and therefore overfit severely when trained from scratch on small datasets. We test this directly with two paired configurations using the **exact same architecture and parameter count**:
1. **`configs/ast.yaml` (`ast_pretrained`):** Initialized from the ImageNet + AudioSet pretrained weights [1] and fine-tuned end-to-end with AdamW ($\text{lr} = 2 \times 10^{-5}$, 25 epochs, 1-epoch linear warm-up, cosine decay).
2. **`configs/ast_scratch.yaml` (`ast_scratch`):** Identical 86.2 M-parameter AST architecture initialized randomly (`pretrained: false`), trained with higher learning rate and regularization ($\text{lr} = 10^{-4}$, weight decay $0.05$, dropout $0.1$, 60 epochs, 5-epoch warm-up).

> **Practical Training Note (Kaggle / Colab GPU):** Fine-tuning `ast_pretrained` for 25 epochs takes $\sim 8\text{--}12$ minutes per seed on a single T4 / P100 GPU with mixed precision (`amp: true`, batch size 16). All 3 seeds can be run in one session via `python src/train.py --config configs/ast.yaml --seeds 0 1 2`.

---

## 10.3 Advanced Model: Conformer with On-the-Fly Noise Mixing (`src/models/conformer.py`)

### Motivation
While AST excels via large-scale pretraining, its 85.8 M parameters (~327 MB) are expensive for real-time edge deployment, and pure self-attention trained from scratch struggles on small audio datasets. Conversely, pure CNNs lack global temporal context across a 5-s clip. To combine **local spectral-temporal invariance** with **global sequence modeling** in a compact architecture ($\sim 2.29\text{ M}$ parameters, $37\times$ smaller than AST), we implement a **Conformer** (Convolution-augmented Transformer) [2] from scratch and pair it with **leakage-free on-the-fly DEMAND noise mixing** (`src/augment.py`).

### Architecture (`Conformer`)
1. **2-D Convolutional Subsampling (`ConvSubsampling`):** Two stride-2 $3 \times 3$ convolutions with SiLU activations downsample the normalized log-mel spectrogram $(1, 128, 431)$ by $4\times$ in both frequency and time to $(64, 32, 108)$, which is projected to $d_{\text{model}} = 144$ and summed with sinusoidal positional encodings.
2. **Macaron Conformer Blocks ($N = 4$ layers, $d_{\text{model}} = 144$, $H = 4$ heads):** Each block sandwiches Multi-Head Self-Attention (MHSA) and a Depthwise Separable Convolution Module between two half-step Macaron Feed-Forward Networks (FFN, expansion factor 4, SiLU):
   $$\tilde{x} = x + \tfrac{1}{2}\text{FFN}_1(x), \quad x' = \tilde{x} + \text{MHSA}(\tilde{x}), \quad x'' = x' + \text{ConvModule}(x'), \quad y = \text{LN}\!\left(x'' + \tfrac{1}{2}\text{FFN}_2(x'')\right).$$
   The `ConvModule` applies LayerNorm $\to$ pointwise conv ($D \to 2D$) $\to$ Gated Linear Unit (GLU) $\to$ 1-D depthwise conv (kernel size 15, covering $\sim 0.7\text{ s}$) $\to$ BatchNorm $\to$ SiLU $\to$ pointwise conv ($D \to D$).
3. **Attention Pooling & Head:** The output sequence $(B, 108, 144)$ is aggregated with our `AttentionPool` layer and classified by a linear head.
4. **Built-in Ablation Switches:** `use_mhsa`, `use_conv_module`, and `pooling` (`"attention"` vs. `"mean"`) can be toggled directly via YAML/CLI overrides to isolate the contribution of each sub-module in Table 5.

### Leakage-Free Noise-Mixing Augmentation (`src/augment.py → NoiseMixer`)
During training (`noise_aug.enabled: true` in `configs/conformer.yaml`), each clean waveform is mixed with probability $p = 0.8$ with a random segment drawn exclusively from the **10 training DEMAND environments** (`DKITCHEN`, `DLIVING`, `DWASHING`, `NFIELD`, `NPARK`, `OHALLWAY`, `OOFFICE`, `PSTATION`, `SPSQUARE`, `TCAR`) at a uniformly sampled SNR $\sim \mathcal{U}(-5, 20)\text{ dB}$:
$$x_{\text{noisy}}(t) = x_{\text{clean}}(t) + \sqrt{\frac{P_{\text{clean}}}{P_{\text{noise}} \cdot 10^{\text{SNR}_{\text{dB}}/10}}} \, n(t).$$
`build_noise_mixer` explicitly checks and raises an error if any validation (`NRIVER`, `OMEETING`, `PRESTO`) or test (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) environment is passed, guaranteeing zero acoustic-environment leakage.

## References

[1] Y. Gong, Y.-A. Chung, and J. Glass, "AST: Audio Spectrogram Transformer," *Proc. Interspeech*, pp. 571–575, 2021.
[2] A. Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition," *Proc. Interspeech*, pp. 5036–5040, 2020.
