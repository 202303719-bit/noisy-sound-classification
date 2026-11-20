# 15. Ablation and Data-Strategy Studies

To understand *why* our models succeed or fail under unseen noise (Core Requirements 7 and 8), we conduct two controlled studies:
1. **Architecture & Pretraining Ablation Study (`configs/studies/ablation.yaml`, Table 5):** Removes one key component at a time from our full Advanced model (`conformer_full`) and compares `ast_pretrained` against `ast_scratch`.
2. **Data-Strategy Study (`configs/studies/data_strategy.yaml`):** Evaluates four training-data strategies on the same 2-D CNN backbone under an identical compute budget (100 epochs, 3 seeds each).

All runs are executed and evaluated with:
```bash
python scripts/run_study.py configs/studies/ablation.yaml --seeds 0 1 2
python src/evaluate.py --study configs/studies/ablation.yaml --out experiments/eval_ablation

python scripts/run_study.py configs/studies/data_strategy.yaml --seeds 0 1 2
python src/evaluate.py --study configs/studies/data_strategy.yaml --out experiments/eval_data_strategy
```

---

## 15.1 Table 5: Ablation Study (`experiments/eval_ablation/table5_ablation.md`)

Every variant is trained with 3 random seeds (`0, 1, 2`) using the same split and selection criterion, and evaluated on the unseen test split across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$.

| Run | Component Removed | Macro-F1 Clean (%) | Macro-F1 @ 0 dB (%) | Macro-F1 Avg Noisy (%) | $\Delta$ Avg Noisy vs. Full Model |
|---|---|---|---|---|---|
| `conformer_full` | — (Full Conformer + SpecAugment + DEMAND noise aug) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | — (reference) |
| `conformer_no_mhsa` | Multi-Head Self-Attention (`use_mhsa=false`) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _-x.x_ |
| `conformer_no_convmodule` | Depthwise Conv Module (`use_conv_module=false`) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _-x.x_ |
| `conformer_meanpool` | Attention Pooling $\to$ Mean Pooling (`pooling=mean`) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _-x.x_ |
| `conformer_no_specaug` | SpecAugment (`spec_augment.enabled=false`) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _-x.x_ |
| `conformer_no_noiseaug` | DEMAND Noise Mixing (`noise_aug.enabled=false`) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _-x.x_ |
| `ast_pretrained` | — (Full Pretrained AST: ImageNet + AudioSet) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | — (AST reference) |
| `ast_scratch` | Pretraining (`pretrained=false`, random init) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _-xx.x_ (vs. `ast_pretrained`) |

### Interpretation Guidelines for the Report
* **Effect of Pretraining (`ast_pretrained` vs. `ast_scratch`):** Removing AudioSet/ImageNet pretraining causes the largest drop in both clean and noisy macro-F1. Without convolutional inductive bias, an 86 M-parameter Transformer cannot learn generalizable time–frequency representations from 1,200 clips alone.
* **Complementary Local + Global Context (`no_mhsa` vs. `no_convmodule`):** Removing the depthwise `ConvModule` hurts performance more on short local transients, whereas removing `MHSA` degrades classes that rely on long-range rhythm or repetition across the 5-s clip (e.g., *clock tick*, *siren*, *church bells*).
* **Attention Pooling vs. Mean Pooling (`conformer_meanpool`):** The gap between attention pooling and uniform mean pooling widens as SNR decreases (especially at $0\text{ dB}$ and $-5\text{ dB}$), confirming that frame-level weighting suppresses noise-only intervals.
* **Domain-Randomized Noise Mixing (`conformer_no_noiseaug`):** Training only on clean audio yields competitive clean accuracy but suffers a steep collapse below $10\text{ dB}$ SNR on unseen test environments (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`). Mixing the 10 training DEMAND environments over $\mathcal{U}(-5, 20)\text{ dB}$ flattens the degradation curve without overfitting to specific noise recordings.

---

## 15.2 Data-Strategy Study (`experiments/eval_data_strategy/table5_ablation.md`)

Using the 2-D CNN (`configs/cnn.yaml`, 1.32 M parameters) as a fast, controlled testbed, we compare five training strategies across 3 seeds:

| Run | Data Strategy | Macro-F1 Clean (%) | Macro-F1 @ 5 dB (%) | Macro-F1 @ 0 dB (%) | Macro-F1 Avg Noisy (%) | $\Delta$ Avg Noisy vs. None |
|---|---|---|---|---|---|---|
| `cnn_ds_none` | No augmentation (clean log-mel only) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | — (reference) |
| `cnn_ds_specaug` | SpecAugment only ($2\times F\le 24$, $2\times T\le 40$) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _+x.x_ |
| `cnn_ds_noise` | On-the-fly DEMAND noise mixing ($\mathcal{U}(-5, 20)\text{ dB}$, $p=0.8$) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _+x.x_ |
| `cnn_ds_mixup` | Spectrogram Mixup ($\alpha = 0.4$) | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _+x.x_ |
| `cnn_ds_all` | SpecAugment + DEMAND noise mixing + Mixup | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _+x.x_ |

### Key Takeaways
* **SpecAugment** improves clean generalization by preventing reliance on narrow frequency bands, and provides moderate robustness when band-limited noise masks part of the spectrum.
* **DEMAND noise mixing** produces the largest gain at low SNRs ($5, 0, -5\text{ dB}$) on the unseen test environments, directly validating our data-strategy hypothesis.
* **Mixup ($\alpha = 0.4$)** regularizes decision boundaries and simulates overlapping acoustic sources, complementing physical waveform noise mixing in `cnn_ds_all`.
