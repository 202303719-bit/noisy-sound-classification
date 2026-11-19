# 9. Baseline

## 9.1 Purpose

The baseline is the reference point for every deep model. It answers one question: *how much of the
task can hand-crafted spectral features plus a classical classifier solve, without learning a
representation?* Following the protocol of the original ESC-50 paper [1], we use a **bag-of-frames
MFCC representation** with two classical classifiers: an **RBF Support Vector Machine (SVM)** and a
**Random Forest (RF)**. Both use the same data split, the same noisy validation protocol and the
same model-selection rule as the deep models (Section 11), so the comparison is fair.

## 9.2 Feature extraction (`src/models/baseline.py → MFCCStats`)

Each 5-s clip is turned into a fixed 240-dimensional vector:

1. **MFCC:** 40 coefficients from a 128-band log-mel spectrogram (46 ms Hann window, 23 ms hop,
   FFT size = next power of two). Computed with `torchaudio`.
2. **Dynamics:** first (Δ) and second (ΔΔ) order deltas (9-frame regression window), which capture
   how the spectrum changes over time (e.g. onset of a dog bark vs. steady rain).
3. **Temporal pooling:** mean and standard deviation over all frames of MFCC, Δ and ΔΔ
   → 3 × 2 × 40 = **240 features**.
4. **Standardisation:** z-score with statistics fitted on the training split only (`StandardScaler`
   inside the scikit-learn pipeline, so no validation/test statistics leak into training).

MFCCs decorrelate the log-mel energies with a DCT and keep the spectral envelope, the classic
compact descriptor for speech and environmental sound [1, 2]. Their weakness, which motivates
the deep models, is that the temporal pooling discards *when* things happen, and additive noise
shifts all low-order coefficients at once, so we expect the baseline to degrade quickly at low SNR
(hypothesis H1).

## 9.3 Classifiers

| | SVM | Random Forest |
|---|---|---|
| Model | `SVC(kernel="rbf")`, one-vs-one training, one-vs-rest decision scores | `RandomForestClassifier` |
| Why | Strong non-linear classifier on small, dense, standardised feature sets; the standard ESC baseline [1] | Robust to feature scale and irrelevant features, and gives feature importances for the error analysis |
| Searched on validation | C ∈ {1, 10, 100}, γ ∈ {scale, 10⁻³, 10⁻²} (9 configs) | n_estimators ∈ {300, 500, 1000}, max_features ∈ {sqrt, log2}, min_samples_leaf ∈ {1, 2} (12 configs) |

Model selection uses the **predefined validation split** (not cross-validation over train + val),
because the validation noise environments (NRIVER, OMEETING, PRESTO) must stay unseen during fitting.
The selected configuration is the one with the highest **mean macro-F1 over the validation conditions
(clean and 5 dB SNR)**. The test split (PCAFETER, STRAFFIC, TBUS, TMETRO noise) is used only once, in
`src/evaluate.py`.

## 9.4 Training protocol

```bash
python src/train.py --config configs/baseline_svm.yaml --seeds 0 1 2
python src/train.py --config configs/baseline_rf.yaml  --seeds 0 1 2
```

`train.py` extracts features once, runs the grid, logs every grid point to
`experiments/baseline_*/seed*/log.csv`, and saves the best pipeline (scaler + classifier + feature
config) to `checkpoints/baseline_*_seed*.joblib`. The SVM is almost deterministic, so its seed
variance is close to zero. For the RF the seed changes bootstrap sampling and feature
sub-sampling, so we report mean ± SD over 3 seeds, as for the deep models.

## 9.5 Results (validation; filled from `experiments/*/summary_seeds.json`)

| Model | Val clean macro-F1 | Val 5 dB macro-F1 | Val clean acc | Val 5 dB acc | Best params |
|---|---|---|---|---|---|
| MFCC + SVM | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _C=…, γ=…_ |
| MFCC + RF | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _xx.x ± x.x_ | _n=…, max_f=…_ |

> Test-set results across SNRs are reported together with all models in Table 4 (Section 13).

## References

[1] K. J. Piczak, "ESC: Dataset for Environmental Sound Classification," *Proc. ACM Multimedia*, 2015.
[2] S. Davis and P. Mermelstein, "Comparison of parametric representations for monosyllabic word
recognition in continuously spoken sentences," *IEEE TASSP*, 28(4), 1980.
