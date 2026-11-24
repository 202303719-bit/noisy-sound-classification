**Table 7 - Computational cost** (from experiments/<run>/seed*/summary.json).

| Model | Params | Size (MB) | Train time / seed | Inference (ms/clip) | Peak GPU mem (MB) | Hardware |
|---|---|---|---|---|---|---|
| cnn_scratch | 1.32 M | 5.023 | 4.1 min | 2.1 | 633 | Tesla T4 |
| crnn_scratch | 1.49 M | 5.674 | 6.6 min | 1.9 | 1588 | Tesla T4 |
| conformer_full | 2.15 M | 8.19 | 8.4 min | 6.4 | 245 | Tesla T4 |
| ast_pretrained | 85.76 M | 327.138 | 22.3 min | 43.3 | 4987 | Tesla T4 |