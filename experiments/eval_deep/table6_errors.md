**Table 6 - Error analysis (test @ 0dB; counts are per seed, averaged).**

| Model | Error type | Count | Example | Likely cause |
|---|---|---|---|---|
| ast_pretrained | noise-induced (correct on clean) | 43.0 | 5-103416-A-2.wav: pig → hen | additive noise masks the event (low SNR, overlapping spectrum) |
| ast_pretrained | inherent (wrong also on clean) | 26.0 | 5-141683-A-35.wav: washing_machine → glass_breaking | class confusion independent of noise (similar acoustic texture / short event) |
| ast_pretrained | confusion thunderstorm → train | 4.3 | 5-156999-C-19.wav | different categories → noise texture resembles predicted class |
| ast_pretrained | confusion fireworks → clapping | 2.0 | 5-160614-D-48.wav | different categories → noise texture resembles predicted class |
| ast_pretrained | confusion airplane → wind | 2.0 | 5-215445-A-47.wav | different categories → noise texture resembles predicted class |
| ast_pretrained | confusion breathing → snoring | 2.0 | 5-232816-A-23.wav | same ESC-50 major category (Human non-speech) → similar spectra |
| ast_pretrained | confusion helicopter → wind | 1.7 | 5-177957-A-40.wav | different categories → noise texture resembles predicted class |
| ast_pretrained | high-confidence errors (conf ≥ 0.8) | 1.3 | 5-260875-A-35.wav: washing_machine → train (0.849) | over-confident model / possible label ambiguity |

> Listen to `error_examples/*.wav` and refine the 'likely cause' column by hand.