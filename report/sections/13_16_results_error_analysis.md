# 13–14 & 16. Experimental Results, Model Comparison, Error Analysis, and Computational Cost

All final tables and figures in these sections are generated automatically from the saved checkpoints by a single command that touches the **test split** (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$:

```bash
python src/evaluate.py --runs baseline_svm baseline_rf cnn_scratch crnn_scratch \
                              ast_pretrained ast_scratch conformer_full \
                       --out experiments/eval
```

---

## 13. Main Experimental Results (`experiments/eval/table4_main.md`)

Following Core Requirement 9, we report **macro-averaged F1** as our primary metric (alongside accuracy and per-class F1 in `experiments/eval/per_class_f1.csv`), with **mean ± SD** and 95% $t$-confidence intervals over 3 random seeds (`0, 1, 2`).

### Table 4: Main Experimental Results on the Test Split (Macro-F1 & Accuracy, %)

| Model | Seeds | F1 Clean | F1 20 dB | F1 10 dB | F1 5 dB | F1 0 dB | F1 -5 dB | **F1 Avg Noisy** | Acc Clean | Acc 0 dB |
|---|---|---|---|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |
| MFCC + RF (`baseline_rf`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |
| 2-D CNN (`cnn_scratch`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |
| CRNN + Attn (`crnn_scratch`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |
| AST Scratch (`ast_scratch`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |
| AST Pretrained (`ast_pretrained`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |
| Conformer + Noise Aug (`conformer_full`) | 3 | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | **_xx.x ± x.x_** | _xx.x ± x.x_ | _xx.x ± x.x_ |

### Robustness Curve: Accuracy & Macro-F1 vs. SNR
`src/evaluate.py` saves `experiments/eval/acc_vs_snr.png` and `experiments/eval/f1_vs_snr.png` (shaded bands show $\pm 1\text{ SD}$ across the 3 seeds). Include both figures side-by-side in Section 14 to illustrate:
1. The rapid collapse of hand-crafted MFCC baselines below $10\text{ dB}$ SNR (verdict on **Hypothesis H1**).
2. The advantage of sequence/attention pooling (`CRNN`, `Conformer`) over bag-of-frames pooling as stationary/non-stationary noise increases (verdict on **Hypothesis H2**).
3. The cross-over or trade-off between massive clean-pretrained representations (`ast_pretrained`) and compact noise-trained models (`conformer_full`) at extreme low SNR ($0\text{ dB}, -5\text{ dB}$) (verdict on **Hypothesis H3**).

---

## 16. Error Analysis (`experiments/eval/table6_errors.md`)

For the selected model(s) at `eval.error_snr = 0 dB`, `src/evaluate.py` produces:
* **Normalized 50×50 Confusion Matrix:** `experiments/eval/confusion_<run>.png`.
* **Full Misclassification Log:** `experiments/eval/errors_<run>.csv`, tagging every error as either **noise-induced** (correctly classified on clean audio, broken by $0\text{ dB}$ noise) or **inherent** (already misclassified on clean audio), and recording whether the confusion occurs within the same ESC-50 major category.
* **Exported Audio Failure Cases:** `experiments/eval/error_examples/*.wav` (top high-confidence failures saved as `.wav` files for listening and live presentation).
* **Per-Class Degradation:** `experiments/eval/per_class_f1.csv` (`f1_clean_mean`, `f1_0dB_mean`, and `drop`), identifying the most noise-vulnerable classes (typically low-energy broadband sounds such as *wind*, *breathing*, *footsteps*, or *mouse click* masked by *TMETRO* / *STRAFFIC* rumble).

### Table 6: Error Analysis on Test Split @ 0 dB SNR (Template auto-filled by `evaluate.py`)

| Model | Error Type | Count / Seed | Concrete Example (Clip: True $\to$ Pred) | Likely Cause |
|---|---|---|---|---|
| _best_model_ | Noise-induced errors (correct on clean, wrong @ 0 dB) | _xx.x_ | `test#...: ... → ...` | Additive low-frequency traffic/metro noise masks low-energy target harmonics |
| _best_model_ | Inherent errors (wrong on both clean and 0 dB) | _xx.x_ | `test#...: ... → ...` | Intrinsic acoustic ambiguity or short transient duration independent of noise |
| _best_model_ | Intra-category confusion (top pair #1) | _x.x_ | `test#...: helicopter → airplane` | Overlapping broadband engine/rotor spectral envelope within Exterior/Urban category |
| _best_model_ | Noise-texture hijack (top pair #2) | _x.x_ | `test#...: mouse_click → rain / engine` | High-energy background DEMAND noise dominates quiet intervals and triggers texture class |
| _best_model_ | High-confidence failures ($\text{conf} \ge 0.80$) | _x.x_ | `test#...: ... → ... (0.9x)` | Calibration shift under severe unseen SNR / background vocal babble in `PCAFETER` |

---

## 16.2 Table 7: Computational Cost (`experiments/eval/table7_cost.md`)

`src/train.py` records exact parameter counts, model file sizes, wall-clock training time, single-clip inference latency (averaged over 30 warm runs including feature extraction), peak CUDA memory (`torch.cuda.max_memory_allocated`), and hardware specs into `experiments/<run>/seed*/summary.json`. `src/evaluate.py` compiles them into `table7_cost.md`:

| Model | Trainable Params | Model Size (MB) | Train Time / Seed | Inference (ms / 5-s clip) | Peak GPU Mem (MB) | Hardware |
|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | #SV $\times 240$ | _~1–3_ | _< 1 min_ | _~x.x_ | — (CPU) | CPU |
| MFCC + RF (`baseline_rf`) | #tree nodes | _~15–40_ | _< 1 min_ | _~xx.x_ | — (CPU) | CPU |
| 2-D CNN (`cnn_scratch`) | **1.32 M** | **5.02** | _~x.x min_ | _~x.x_ | _~xxx_ | GPU (T4 / P100) |
| CRNN + Attn (`crnn_scratch`) | **1.49 M** | **5.67** | _~x.x min_ | _~x.x_ | _~xxx_ | GPU (T4 / P100) |
| Conformer (`conformer_full`) | **2.29 M** | **8.75** | _~x.x min_ | _~x.x_ | _~xxx_ | GPU (T4 / P100) |
| AST (`ast_pretrained` / `ast_scratch`) | **85.76 M** | **327.1** | _~8–15 min_ | _~xx.x_ | _~xxxx_ | GPU (T4 / P100) |
