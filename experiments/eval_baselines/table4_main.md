**Table 4 - Main results on the test split (mean ± SD over seeds, %).** Primary metric: macro-F1; secondary: accuracy. Noise = unseen DEMAND test environments.

| Model | Seeds | F1 clean | F1 20dB | F1 10dB | F1 5dB | F1 0dB | F1 -5dB | F1 avg noisy | Acc clean | Acc 0dB |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_svm | 3 | 37.4 ± 0.0 | 28.6 ± 0.0 | 24.9 ± 0.0 | 18.4 ± 0.0 | 11.7 ± 0.0 | 6.8 ± 0.0 | **18.1 ± 0.0** | 39.0 ± 0.0 | 14.2 ± 0.0 |
| baseline_rf | 3 | 43.5 ± 0.6 | 35.9 ± 0.9 | 31.0 ± 0.3 | 24.6 ± 0.3 | 18.6 ± 0.3 | 13.6 ± 0.2 | **24.7 ± 0.3** | 46.0 ± 0.8 | 23.4 ± 0.1 |

95% CI half-width (t-distribution) of F1 avg noisy: baseline_svm: ±0.0, baseline_rf: ±0.7