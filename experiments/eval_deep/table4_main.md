**Table 4 - Main results on the test split (mean ± SD over seeds, %).** Primary metric: macro-F1; secondary: accuracy. Noise = unseen DEMAND test environments.

| Model | Seeds | F1 clean | F1 20dB | F1 10dB | F1 5dB | F1 0dB | F1 -5dB | F1 avg noisy | Acc clean | Acc 0dB |
|---|---|---|---|---|---|---|---|---|---|---|
| cnn_scratch | 3 | 70.3 ± 0.7 | 67.3 ± 1.1 | 58.8 ± 1.2 | 52.7 ± 1.3 | 43.5 ± 2.7 | 33.1 ± 1.5 | **51.1 ± 1.4** | 71.8 ± 0.6 | 45.2 ± 2.5 |
| crnn_scratch | 3 | 70.2 ± 1.5 | 68.3 ± 1.4 | 61.4 ± 2.0 | 55.6 ± 2.2 | 46.3 ± 1.8 | 35.3 ± 2.0 | **53.4 ± 1.4** | 71.5 ± 1.4 | 47.9 ± 1.8 |
| conformer_full | 3 | 65.3 ± 0.4 | 64.6 ± 2.1 | 62.4 ± 1.3 | 60.2 ± 0.8 | 55.3 ± 1.9 | 46.6 ± 1.2 | **57.8 ± 1.3** | 66.9 ± 0.6 | 56.9 ± 1.8 |
| ast_pretrained | 3 | 92.0 ± 2.4 | 91.9 ± 2.7 | 90.7 ± 2.3 | 87.4 ± 3.4 | 81.8 ± 2.4 | 76.3 ± 2.8 | **85.6 ± 2.7** | 92.4 ± 1.9 | 82.8 ± 2.1 |

95% CI half-width (t-distribution) of F1 avg noisy: cnn_scratch: ±3.5, crnn_scratch: ±3.5, conformer_full: ±3.2, ast_pretrained: ±6.6