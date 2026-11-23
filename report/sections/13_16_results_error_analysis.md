# 13–14 & 16. Experimental Results, Model Comparison, Error Analysis, and Computational Cost

All final tables and figures in these sections are generated automatically from the saved checkpoints by a single command that touches the **test split** (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$:

```bash
python src/evaluate.py --runs baseline_svm baseline_rf --out experiments/eval_baselines
```

---

## 13. Main Experimental Results (`experiments/eval/table4_main.md`)

Following Core Requirement 9, we report **macro-averaged F1** as our primary metric (alongside accuracy and per-class F1 in `experiments/eval/per_class_f1.csv`), with **mean ± SD** and 95% $t$-confidence intervals over 3 random seeds (`0, 1, 2`).

### Table 4: Main Experimental Results on the Test Split (Macro-F1 & Accuracy, %)

| Model | Seeds | F1 Clean | F1 20 dB | F1 10 dB | F1 5 dB | F1 0 dB | F1 -5 dB | **F1 Avg Noisy** | Acc Clean | Acc 0 dB |
|---|---|---|---|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 3 | 37.4 ± 0.0 | 28.6 ± 0.0 | 24.9 ± 0.0 | 18.4 ± 0.0 | 11.7 ± 0.0 | 6.8 ± 0.0 | **18.1 ± 0.0** | 39.0 ± 0.0 | 14.2 ± 0.0 |
| MFCC + RF (`baseline_rf`) | 3 | 43.5 ± 0.6 | 35.9 ± 0.9 | 31.0 ± 0.3 | 24.6 ± 0.3 | 18.6 ± 0.3 | 13.6 ± 0.2 | **24.7 ± 0.3** | 46.0 ± 0.8 | 23.4 ± 0.1 |
| 2-D CNN (`cnn_scratch`) | 3 | trained on Kaggle (val macro-F1 55.8 ± 0.2), checkpoints lost, not evaluated on test | - | - | - | - | - | - | - | - |
| CRNN + Attn (`crnn_scratch`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |
| AST Scratch (`ast_scratch`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |
| AST Pretrained (`ast_pretrained`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |
| Conformer + Noise Aug (`conformer_full`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |

Only the two baselines were evaluated on the test split. Both degrade monotonically as the SNR decreases: the Random Forest falls from 43.5% macro-F1 on clean audio to 13.6% at -5 dB, and the SVM from 37.4% to 6.8% (see `experiments/eval_baselines/f1_vs_snr.png`). The deep models (CNN, CRNN, AST, Conformer) were not trained within the project time, so the hypotheses that compare them could not be tested.
---

## 16. Error Analysis (`experiments/eval/table6_errors.md`)

For the selected model(s) at `eval.error_snr = 0 dB`, `src/evaluate.py` produces:
* **Normalized 50×50 Confusion Matrix:** `experiments/eval/confusion_<run>.png`.
* **Full Misclassification Log:** `experiments/eval/errors_<run>.csv`, tagging every error as either **noise-induced** (correctly classified on clean audio, broken by $0\text{ dB}$ noise) or **inherent** (already misclassified on clean audio), and recording whether the confusion occurs within the same ESC-50 major category.
* **Exported Audio Failure Cases:** `experiments/eval/error_examples/*.wav` (top high-confidence failures saved as `.wav` files for listening and live presentation).
* **Per-Class Degradation:** `experiments/eval/per_class_f1.csv` (`f1_clean_mean`, `f1_0dB_mean`, and `drop`), identifying the most noise-vulnerable classes (typically low-energy broadband sounds such as *wind*, *breathing*, *footsteps*, or *mouse click* masked by *TMETRO* / *STRAFFIC* rumble).

### Table 6: Error Analysis on Test Split @ 0 dB SNR (best baseline, MFCC + Random Forest, mean over 3 seeds)

| Model | Error Type | Count / Seed | Concrete Example (Clip: True → Pred) | Likely Cause |
|---|---|---|---|---|
| baseline_rf | Noise-induced errors (correct on clean, wrong @ 0 dB) | 113.7 | `5-103415-A-2.wav: pig → toilet_flush` | Additive noise masks the event (low SNR, overlapping spectrum) |
| baseline_rf | Inherent errors (wrong on both clean and 0 dB) | 192.7 | `5-103416-A-2.wav: pig → chainsaw` | Class confusion independent of noise (similar acoustic texture, short event) |
| baseline_rf | Intra-category confusion: crickets → chirping_birds | 5.7 | `5-204352-A-13.wav` | Same ESC-50 major category (natural soundscapes), similar spectra |
| baseline_rf | Cross-category confusion: door_wood_creaks → snoring | 3.0 | `5-161270-A-33.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| baseline_rf | Cross-category confusion: sneezing → frog | 3.0 | `5-202220-A-21.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
---

## 16.2 Table 7: Computational Cost (`experiments/eval/table7_cost.md`)

`src/train.py` records exact parameter counts, model file sizes, wall-clock training time, single-clip inference latency (averaged over 30 warm runs including feature extraction), peak CUDA memory (`torch.cuda.max_memory_allocated`), and hardware specs into `experiments/<run>/seed*/summary.json`. `src/evaluate.py` compiles them into `table7_cost.md`:

| Model | Trainable Params | Model Size (MB) | Train Time / Seed | Inference (ms / 5-s clip) | Peak GPU Mem (MB) | Hardware |
|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 0.35 M | 3.14 | 0.8 min | 2.7 | — (CPU) | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| MFCC + RF (`baseline_rf`) | 0.55 M | 243.7 | 1.0 min | 96.7 | — (CPU) | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| 2-D CNN (`cnn_scratch`) | **1.32 M** | **5.02** | _~x.x min_ | _~x.x_ | _~xxx_ | GPU (T4 / P100) |
| CRNN + Attn (`crnn_scratch`) | **1.49 M** | **5.67** | _~x.x min_ | _~x.x_ | _~xxx_ | GPU (T4 / P100) |
| Conformer (`conformer_full`) | **2.29 M** | **8.75** | _~x.x min_ | _~x.x_ | _~xxx_ | GPU (T4 / P100) |
| AST (`ast_pretrained` / `ast_scratch`) | **85.76 M** | **327.1** | _~8–15 min_ | _~xx.x_ | _~xxxx_ | GPU (T4 / P100) |
