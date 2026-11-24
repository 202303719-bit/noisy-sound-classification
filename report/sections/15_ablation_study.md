# 15. Ablation and Data-Strategy Studies

## Status: not run

The project plan included two controlled studies (Core Requirements 7 and 8): an ablation study and a data-strategy study. **Neither study was run**, because the project time ran out and GPU access was no longer available (the Colab usage limit was reached). No number in this report comes from these studies, and none of the hypotheses that depend on them was tested.

The configuration files and the study runner exist in the repository and were not executed:

```bash
python scripts/run_study.py configs/studies/ablation.yaml --seeds 0 1 2
python scripts/run_study.py configs/studies/data_strategy.yaml --seeds 0 1 2
```

## 15.1 Ablation study (Table 5): not run

Planned variants: removing multi-head self-attention, the depthwise convolution module, attention pooling (replaced by mean pooling), SpecAugment and noise mixing from the Conformer, and comparing `ast_pretrained` against `ast_scratch` (random initialisation).

| Run | Component removed | Result |
|---|---|---|
| `conformer_no_mhsa` | Multi-head self-attention | not run |
| `conformer_no_convmodule` | Depthwise conv module | not run |
| `conformer_meanpool` | Attention pooling → mean pooling | not run |
| `conformer_no_specaug` | SpecAugment | not run |
| `conformer_no_noiseaug` | DEMAND noise mixing | not run |
| `ast_scratch` | Pretraining | not run |

## 15.2 Data-strategy study: not run

Planned strategies on the 2-D CNN: no augmentation, SpecAugment, DEMAND noise mixing, mixup, and all combined. Not run.

## 15.3 What the available results can and cannot say

Without these studies, we cannot separate the effect of individual components. In particular, the Conformer's better robustness at low SNR (Section 13) coincides with it being the only from-scratch model trained with noise mixing and SpecAugment, so the cause could be the augmentation, the architecture, or both. Likewise, the pretrained AST is the most robust model, but without `ast_scratch` we cannot say how much of this comes from pretraining and how much from the model's size. These questions remain open.