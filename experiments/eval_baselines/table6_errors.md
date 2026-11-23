**Table 6 - Error analysis (test @ 0dB; counts are per seed, averaged).**

| Model | Error type | Count | Example | Likely cause |
|---|---|---|---|---|
| baseline_rf | noise-induced (correct on clean) | 113.7 | 5-103415-A-2.wav: pig → toilet_flush | additive noise masks the event (low SNR, overlapping spectrum) |
| baseline_rf | inherent (wrong also on clean) | 192.7 | 5-103416-A-2.wav: pig → chainsaw | class confusion independent of noise (similar acoustic texture / short event) |
| baseline_rf | confusion crickets → chirping_birds | 5.7 | 5-204352-A-13.wav | same ESC-50 major category (Natural soundscapes & water) → similar spectra |
| baseline_rf | confusion door_wood_creaks → snoring | 3.0 | 5-161270-A-33.wav | different categories → noise texture resembles predicted class |
| baseline_rf | confusion sneezing → frog | 3.0 | 5-202220-A-21.wav | different categories → noise texture resembles predicted class |
| baseline_rf | confusion pouring_water → brushing_teeth | 3.0 | 5-212730-A-17.wav | different categories → noise texture resembles predicted class |
| baseline_rf | confusion can_opening → hand_saw | 3.0 | 5-237795-A-34.wav | different categories → noise texture resembles predicted class |

> Listen to `error_examples/*.wav` and refine the 'likely cause' column by hand.