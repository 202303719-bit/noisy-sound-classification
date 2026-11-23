# 13–14 & 16. Experimental Results, Model Comparison, Error Analysis, and Computational Cost

The final tables and figures in these sections were generated from the saved checkpoints by a single command that touches the **test split** (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) across $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$. Only the two baselines were evaluated:

```bash
python src/evaluate.py --runs baseline_svm baseline_rf --out experiments/eval_baselines
```

---

## 13. Main Experimental Results (`experiments/eval_baselines/table4_main.md`)

Following Core Requirement 9, we report **macro-averaged F1** as our primary metric (alongside accuracy and per-class F1 in `experiments/eval_baselines/per_class_f1.csv`), with **mean ± SD** and 95% $t$-confidence intervals over 3 random seeds (`0, 1, 2`).

### Table 4: Main Experimental Results on the Test Split (Macro-F1 & Accuracy, %)

| Model | Seeds | F1 Clean | F1 20 dB | F1 10 dB | F1 5 dB | F1 0 dB | F1 -5 dB | **F1 Avg Noisy** | Acc Clean | Acc 0 dB |
|---|---|---|---|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 3 | 37.4 ± 0.0 | 28.6 ± 0.0 | 24.9 ± 0.0 | 18.4 ± 0.0 | 11.7 ± 0.0 | 6.8 ± 0.0 | **18.1 ± 0.0** | 39.0 ± 0.0 | 14.2 ± 0.0 |
| MFCC + RF (`baseline_rf`) | 3 | 43.5 ± 0.6 | 35.9 ± 0.9 | 31.0 ± 0.3 | 24.6 ± 0.3 | 18.6 ± 0.3 | 13.6 ± 0.2 | **24.7 ± 0.3** | 46.0 ± 0.8 | 23.4 ± 0.1 |
| 2-D CNN (`cnn_scratch`) | 3 | trained on Kaggle (val macro-F1 55.8 ± 0.2); checkpoints lost, not evaluated on test | - | - | - | - | - | - | - | - |
| CRNN + Attn (`crnn_scratch`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |
| AST Scratch (`ast_scratch`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |
| AST Pretrained (`ast_pretrained`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |
| Conformer + Noise Aug (`conformer_full`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |

Only the two baselines were evaluated on the test split. Both degrade monotonically as the SNR decreases: the Random Forest falls from 43.5% macro-F1 on clean audio to 13.6% at -5 dB, and the SVM from 37.4% to 6.8% (see `experiments/eval_baselines/f1_vs_snr.png` and `acc_vs_snr.png`). The SVM has no standard deviation across seeds because it is deterministic (same features and same hyperparameter grid in every seed). The CNN was trained on Kaggle, reaching a validation macro-F1 of 55.8 ± 0.2 over 3 seeds, which is well above both baselines on validation, but its checkpoints were lost with the Kaggle session, so it has no test results. The remaining deep models (CRNN, AST, Conformer) were not trained within the project time, so the hypotheses that compare them could not be tested.

---

## 16. Error Analysis (`experiments/eval_baselines/table6_errors.md`)

For the best baseline (Random Forest) at `eval.error_snr = 0 dB`, `src/evaluate.py` produces:

* **Normalized 50×50 Confusion Matrix:** `experiments/eval_baselines/confusion_baseline_rf.png`.
* **Full Misclassification Log:** `experiments/eval_baselines/errors_baseline_rf.csv`, tagging every error as either **noise-induced** (correctly classified on clean audio, broken by $0\text{ dB}$ noise) or **inherent** (already misclassified on clean audio), and recording whether the confusion occurs within the same ESC-50 major category.
* **Exported Audio Failure Cases:** `experiments/eval_baselines/error_examples/*.wav` (8 failure clips saved for listening).
* **Per-Class Degradation:** `experiments/eval_baselines/per_class_f1.csv` (`f1_clean_mean`, `f1_0dB_mean`, and `drop`).

### Table 6: Error Analysis on Test Split @ 0 dB SNR (best baseline, MFCC + Random Forest, mean over 3 seeds)

| Model | Error Type | Count / Seed | Concrete Example (Clip: True → Pred) | Likely Cause |
|---|---|---|---|---|
| baseline_rf | Noise-induced errors (correct on clean, wrong @ 0 dB) | 113.7 | `5-103415-A-2.wav: pig → toilet_flush` | Additive noise masks the event (low SNR, overlapping spectrum) |
| baseline_rf | Inherent errors (wrong on both clean and 0 dB) | 192.7 | `5-103416-A-2.wav: pig → chainsaw` | Class confusion independent of noise (similar acoustic texture, short event) |
| baseline_rf | Intra-category confusion: crickets → chirping_birds | 5.7 | `5-204352-A-13.wav` | Same ESC-50 major category (natural soundscapes), similar spectra |
| baseline_rf | Cross-category confusion: door_wood_creaks → snoring | 3.0 | `5-161270-A-33.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| baseline_rf | Cross-category confusion: sneezing → frog | 3.0 | `5-202220-A-21.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |

We analysed the errors of the best baseline, Random Forest, on the test split at 0 dB (averaged over three seeds). Of its misclassified clips, about 114 per seed were noise-induced (correct on the clean clip, wrong once noise was added) and about 193 were inherent (wrong even on clean audio). Even for this weak model, most errors at 0 dB therefore exist without noise: the hand-crafted MFCC features separate many ESC-50 classes poorly to begin with. The most frequent single confusion was crickets predicted as chirping birds (about 5.7 clips per seed), two classes of the same ESC-50 major category (natural soundscapes) with similar spectra. The remaining frequent confusions (door creak as snoring, sneezing as frog, pouring water as brushing teeth, can opening as hand saw; about 3 clips each) involve different categories, which suggests that the noise texture resembles the predicted class. This is a hypothesis, since we did not verify it by listening to every example.

---

## 16.2 Table 7: Computational Cost (`experiments/eval_baselines/table7_cost.md`)

`src/train.py` records parameter counts, model file sizes, wall-clock training time, single-clip inference latency, peak CUDA memory, and hardware specs into `experiments/<run>/seed*/summary.json`. `src/evaluate.py` compiles them into `table7_cost.md`:

| Model | Trainable Params | Model Size (MB) | Train Time / Seed | Inference (ms / 5-s clip) | Peak GPU Mem (MB) | Hardware |
|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 0.35 M | 3.14 | 0.8 min | 2.7 | — (CPU) | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| MFCC + RF (`baseline_rf`) | 0.55 M | 243.7 | 1.0 min | 96.7 | — (CPU) | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| 2-D CNN (`cnn_scratch`) | 1.32 M | 5.02 | not measured | not measured | not measured | Kaggle GPU (Tesla T4), checkpoints lost |
| CRNN + Attn (`crnn_scratch`) | 1.49 M | 5.67 | not trained | not trained | not trained | not trained |
| Conformer (`conformer_full`) | 2.29 M | 8.75 | not trained | not trained | not trained | not trained |
| AST (`ast_pretrained` / `ast_scratch`) | 85.76 M | 327.1 | not trained | not trained | not trained | not trained |

Parameter counts and sizes for the models that were not evaluated are architecture estimates from the code, not measured values. The SVM has 0.35 M parameters and a 3.1 MB checkpoint; training took 0.8 min per seed and inference 2.7 ms per clip. The Random Forest has 0.55 M parameters and a 244 MB checkpoint and trains in 1.0 min per seed, but inference takes 97 ms per clip. Both ran on CPU (11th Gen Intel Core i5-11400H, 2.70 GHz, no GPU).