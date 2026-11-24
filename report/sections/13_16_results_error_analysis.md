# 13–14 & 16. Experimental Results, Model Comparison, Error Analysis, and Computational Cost

The test-split results were generated from the saved checkpoints by `src/evaluate.py`. The test split uses unseen DEMAND noise environments (`PCAFETER`, `STRAFFIC`, `TBUS`, `TMETRO`) at $\text{SNR} \in \{\text{clean}, 20, 10, 5, 0, -5\}\text{ dB}$. The two baselines were evaluated locally (CPU) and the four deep models on a Colab T4 GPU:

```bash
python src/evaluate.py --runs baseline_svm baseline_rf --out experiments/eval_baselines
python src/evaluate.py --runs cnn_scratch crnn_scratch conformer_full ast_pretrained --out experiments/eval_deep
```

Both runs use the same fixed noisy test set (400 clips), so the numbers are directly comparable. The baselines were evaluated separately because the saved SVM/RF checkpoints were produced with a newer scikit-learn version than the one available on Colab.

---

## 13. Main Experimental Results (`experiments/eval_baselines/table4_main.md`, `experiments/eval_deep/table4_main.md`)

Following Core Requirement 9, we report **macro-averaged F1** as our primary metric (alongside accuracy and per-class F1 in `per_class_f1.csv`), with **mean ± SD** over 3 random seeds (`0, 1, 2`).

### Table 4: Main Experimental Results on the Test Split (Macro-F1 & Accuracy, %)

| Model | Seeds | F1 Clean | F1 20 dB | F1 10 dB | F1 5 dB | F1 0 dB | F1 -5 dB | **F1 Avg Noisy** | Acc Clean | Acc 0 dB |
|---|---|---|---|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 3 | 37.4 ± 0.0 | 28.6 ± 0.0 | 24.9 ± 0.0 | 18.4 ± 0.0 | 11.7 ± 0.0 | 6.8 ± 0.0 | **18.1 ± 0.0** | 39.0 ± 0.0 | 14.2 ± 0.0 |
| MFCC + RF (`baseline_rf`) | 3 | 43.5 ± 0.6 | 35.9 ± 0.9 | 31.0 ± 0.3 | 24.6 ± 0.3 | 18.6 ± 0.3 | 13.6 ± 0.2 | **24.7 ± 0.3** | 46.0 ± 0.8 | 23.4 ± 0.1 |
| 2-D CNN (`cnn_scratch`) | 3 | 70.3 ± 0.7 | 67.3 ± 1.1 | 58.8 ± 1.2 | 52.7 ± 1.3 | 43.5 ± 2.7 | 33.1 ± 1.5 | **51.1 ± 1.4** | 71.8 ± 0.6 | 45.2 ± 2.5 |
| CRNN + Attn (`crnn_scratch`) | 3 | 70.2 ± 1.5 | 68.3 ± 1.4 | 61.4 ± 2.0 | 55.6 ± 2.2 | 46.3 ± 1.8 | 35.3 ± 2.0 | **53.4 ± 1.4** | 71.5 ± 1.4 | 47.9 ± 1.8 |
| Conformer + Noise Aug (`conformer_full`) | 3 | 65.3 ± 0.4 | 64.6 ± 2.1 | 62.4 ± 1.3 | 60.2 ± 0.8 | 55.3 ± 1.9 | 46.6 ± 1.2 | **57.8 ± 1.3** | 66.9 ± 0.6 | 56.9 ± 1.8 |
| AST Pretrained (`ast_pretrained`) | 3 | 92.0 ± 2.4 | 91.9 ± 2.7 | 90.7 ± 2.3 | 87.4 ± 3.4 | 81.8 ± 2.4 | 76.3 ± 2.8 | **85.6 ± 2.7** | 92.4 ± 1.9 | 82.8 ± 2.1 |
| AST Scratch (`ast_scratch`) | 0 | not trained | not trained | not trained | not trained | not trained | not trained | **not trained** | not trained | not trained |

95% t-confidence interval half-widths of F1 avg noisy: CNN ±3.5, CRNN ±3.5, Conformer ±3.2, AST pretrained ±6.6 (the baselines' are in their own `table4_main.md`).

### Discussion

All models degrade as the SNR decreases. The pretrained AST is the best model at every noise level, with 92.0% macro-F1 on clean audio and 76.3% at -5 dB. The three models trained from scratch are clearly better than the MFCC baselines (the best baseline, the Random Forest, reaches 24.7% average noisy F1 against 51.1–57.8% for the from-scratch deep models).

The Conformer is weaker than the CNN and CRNN on clean audio (65.3% vs about 70%) but degrades much more slowly: at 0 dB it reaches 55.3% against 43.5% (CNN) and 46.3% (CRNN). We cannot attribute this to the architecture alone: the Conformer was the only from-scratch model trained with noise-mixing augmentation (`noise_aug` on, SpecAugment on), whereas the CNN and CRNN were trained without it. No ablation was run to separate the two factors, so the difference is better explained by the augmentation than by the architecture, but this is not proven.

The CNN and CRNN differ by less than their confidence intervals (51.1 vs 53.4 average noisy F1, CI ±3.5), so we do not claim that either is better. The pretrained AST was trained without noise augmentation and is still the most robust model, which is consistent with the hypothesis that pretraining helps under noise. However, the scratch counterpart (`ast_scratch`) was not trained because of time, so the pretrained-vs-scratch comparison required by the project was not run, and we cannot say that pretraining alone is the cause. The model sizes also differ widely (85.76 M parameters for AST against 1.3–2.3 M for the others), which is a confounder in all comparisons with the AST.

The data-strategy study and the ablation study were not run either, so the hypotheses that depend on them (effect of noise augmentation, of pretraining, of the attention component) are not tested.

Validation macro-F1 (fold 4, clean audio) is reported by `train.py`: CNN 0.5655 ± 0.0062, CRNN 0.5751 ± 0.0115, Conformer 0.6634 ± 0.0061, AST pretrained 0.9163 ± 0.0054 over the first two seeds in the final log, and 0.9236, 0.9202, 0.9125 for seeds 0, 1, 2. These were used only for model selection; the test split was used only for the final evaluation. The SVM has no standard deviation across seeds because it is deterministic (same features and same hyperparameter grid in every seed).

---

## 16. Error Analysis (`experiments/eval_baselines/table6_errors.md`, `experiments/eval_deep/table6_errors.md`)

The error analysis below covers the best baseline (Random Forest, Table 6) and the best model (pretrained AST, Table 6b). It was not written up for the CNN, CRNN and Conformer, but the same files exist for them in `experiments/eval_deep/`. At `eval.error_snr = 0 dB`, `src/evaluate.py` produces:

* **Normalized 50×50 Confusion Matrix:** `experiments/eval_baselines/confusion_baseline_rf.png`.
* **Full Misclassification Log:** `experiments/eval_baselines/errors_baseline_rf.csv`, tagging every error as either **noise-induced** (correctly classified on clean audio, broken by 0 dB noise) or **inherent** (already misclassified on clean audio).
* **Exported Audio Failure Cases:** `experiments/eval_baselines/error_examples/*.wav` (8 failure clips saved for listening).
* **Per-Class Degradation:** `experiments/eval_baselines/per_class_f1.csv`.

### Table 6: Error Analysis on Test Split @ 0 dB SNR (best baseline, MFCC + Random Forest, mean over 3 seeds)

| Model | Error Type | Count / Seed | Concrete Example (Clip: True → Pred) | Likely Cause |
|---|---|---|---|---|
| baseline_rf | Noise-induced errors (correct on clean, wrong @ 0 dB) | 113.7 | `5-103415-A-2.wav: pig → toilet_flush` | Additive noise masks the event (low SNR, overlapping spectrum) |
| baseline_rf | Inherent errors (wrong on both clean and 0 dB) | 192.7 | `5-103416-A-2.wav: pig → chainsaw` | Class confusion independent of noise (similar acoustic texture, short event) |
| baseline_rf | Intra-category confusion: crickets → chirping_birds | 5.7 | `5-204352-A-13.wav` | Same ESC-50 major category (natural soundscapes), similar spectra |
| baseline_rf | Cross-category confusion: door_wood_creaks → snoring | 3.0 | `5-161270-A-33.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| baseline_rf | Cross-category confusion: sneezing → frog | 3.0 | `5-202220-A-21.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |

We analysed the errors of the best baseline, Random Forest, on the test split at 0 dB (averaged over three seeds). Of its misclassified clips, about 114 per seed were noise-induced (correct on the clean clip, wrong once noise was added) and about 193 were inherent (wrong even on clean audio). Even for this weak model, most errors at 0 dB therefore exist without noise: the hand-crafted MFCC features separate many ESC-50 classes poorly to begin with. The most frequent single confusion was crickets predicted as chirping birds (about 5.7 clips per seed), two classes of the same ESC-50 major category (natural soundscapes) with similar spectra. The remaining frequent confusions (door creak as snoring, sneezing as frog, pouring water as brushing teeth, can opening as hand saw; about 3 clips each) involve different categories, which suggests that the noise texture resembles the predicted class. This is a hypothesis, since we did not verify it by listening to every example.

### Table 6b: Error Analysis on Test Split @ 0 dB SNR (best model, AST pretrained, mean over 3 seeds)

| Model | Error Type | Count / Seed | Concrete Example (Clip: True → Pred) | Likely Cause |
|---|---|---|---|---|
| ast_pretrained | Noise-induced errors (correct on clean, wrong @ 0 dB) | 43.0 | `5-103416-A-2.wav: pig → hen` | Additive noise masks the event (low SNR, overlapping spectrum) |
| ast_pretrained | Inherent errors (wrong on both clean and 0 dB) | 26.0 | `5-141683-A-35.wav: washing_machine → glass_breaking` | Class confusion independent of noise (similar acoustic texture, short event) |
| ast_pretrained | Cross-category confusion: thunderstorm → train | 4.3 | `5-156999-C-19.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| ast_pretrained | Cross-category confusion: fireworks → clapping | 2.0 | `5-160614-D-48.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| ast_pretrained | Cross-category confusion: airplane → wind | 2.0 | `5-215445-A-47.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| ast_pretrained | Intra-category confusion: breathing → snoring | 2.0 | `5-232816-A-23.wav` | Same ESC-50 major category (human non-speech), similar spectra |
| ast_pretrained | Cross-category confusion: helicopter → wind | 1.7 | `5-177957-A-40.wav` | Hypothesis: noise texture resembles the predicted class (not verified by listening) |
| ast_pretrained | High-confidence errors (confidence ≥ 0.8) | 1.3 | `5-260875-A-35.wav: washing_machine → train (0.849)` | Over-confident prediction or possible label ambiguity (not verified) |

We also analysed the errors of the pretrained AST at 0 dB (mean over three seeds). It misclassifies about 69 of the 400 test clips per seed: about 43 are noise-induced (correct on the clean clip, wrong once noise was added) and about 26 are inherent (wrong even on clean audio). Unlike the Random Forest, where most errors already exist on clean audio, most of the AST's errors at 0 dB are therefore caused by the noise. This is consistent with its high clean accuracy (92.4%), which leaves little room for inherent errors. The most frequent confusion is thunderstorm predicted as train (about 4.3 clips per seed), followed by several confusions between sounds with continuous broadband texture (fireworks as clapping, airplane and helicopter as wind). A plausible explanation is that these sounds and the DEMAND noise share a similar spectral texture, but this is a hypothesis: we did not verify it by listening to the examples. High-confidence errors (confidence of at least 0.8) are rare, about 1.3 clips per seed. This analysis covers 0 dB only; errors at -5 dB were not analysed.

---

## 16.2 Table 7: Computational Cost (`experiments/eval_deep/table7_cost.md`)

Parameter counts, checkpoint sizes, training time per seed, single-clip inference latency (5-s clip, including feature extraction) and peak GPU memory are recorded by `src/train.py` in `experiments/<run>/seed*/summary.json` and compiled by `src/evaluate.py`.

| Model | Params | Size (MB) | Train time / seed | Inference (ms/clip) | Peak GPU mem (MB) | Hardware |
|---|---|---|---|---|---|---|
| MFCC + SVM (`baseline_svm`) | 0.35 M | 3.14 | 0.8 min | 2.7 | - (CPU) | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| MFCC + RF (`baseline_rf`) | 0.55 M | 243.7 | 1.0 min | 96.7 | - (CPU) | 11th Gen Intel Core i5-11400H @ 2.70 GHz |
| 2-D CNN (`cnn_scratch`) | 1.32 M | 5.02 | 4.1 min | 2.1 | 633 | Tesla T4 (Colab) |
| CRNN + Attn (`crnn_scratch`) | 1.49 M | 5.67 | 6.6 min | 1.9 | 1588 | Tesla T4 (Colab) |
| Conformer (`conformer_full`) | 2.15 M | 8.19 | 8.4 min | 6.4 | 245 | Tesla T4 (Colab) |
| AST pretrained (`ast_pretrained`) | 85.76 M | 327.1 | 22.3 min | 43.3 | 4987 | Tesla T4 (Colab) |
| AST scratch (`ast_scratch`) | not trained | not trained | not trained | not trained | not trained | not trained |

The AST is about 40 to 65 times larger than the other deep models (85.76 M against 1.3 to 2.3 M parameters) and takes about 5 times longer to train per seed than the CNN. Its inference is also slower (43.3 ms against about 2 to 6 ms per clip), but still far below real time for a 5-second clip, so it is usable in the demo. The Conformer has the lowest peak GPU memory (245 MB) in this measurement despite having more parameters than the CNN and CRNN; we did not investigate why. The SVM and Random Forest ran on CPU: the RF has a 244 MB checkpoint and 97 ms inference per clip, much slower than the SVM (2.7 ms). Training times were measured on a shared Colab T4, so they are indicative only.

The Conformer seed 0 was trained with a mis-sized configuration (`n_mels` set to 128, 2.29 M parameters) before the configuration was corrected to 64 mel bands (2.15 M, used for seeds 1 and 2); the table reports the corrected configuration.