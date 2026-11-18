# Dataset Documentation

Project: Environmental Sound Classification under Noise (Deep Learning AI403, Idea 17)

The system uses two public datasets: **ESC-50** (the sounds to classify) and **DEMAND** (the background noise mixed into them).

## 1\. Datasets

||ESC-50|DEMAND|
|-|-|-|
|Role|Clean audio to classify (train / val / test)|Noise source, mixed into ESC-50 clips|
|Source|https://github.com/karolpiczak/ESC-50|https://zenodo.org/records/1227121|
|Version used|GitHub `master`, downloaded 18/11/2026 |Version 1.0, 16 kHz zips (`\*\_16k.zip`)|
|Size|2,000 clips, 5 s each|17 environments used, 300 s each, 16 channels each|
|Format|WAV, mono, 44.1 kHz|WAV, mono per channel, 16 kHz|
|Classes|50 (5 groups of 10: animals, natural soundscapes, human non-speech, domestic, urban noises)|17 environments (no class labels)|
|Licence|CC BY-NC 3.0|CC BY-SA 3.0|
|Citation|Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015|Thiemann, Ito, Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)", Proc. Meetings on Acoustics 19, 2013|

ESC-50 clips were extracted from Freesound recordings. The dataset is balanced: 40 clips per class, 400 clips per fold.

Note: some third-party mirrors of DEMAND (Kaggle, Hugging Face) list the licence as CC-BY-4.0. We use the original Zenodo release, which is CC BY-SA 3.0.

## 2\. Preprocessing

1. **Resampling:** all audio is converted to mono, 16 kHz.
2. **Fixed length:** every clip is exactly 5 s (80,000 samples). Shorter clips are zero-padded, longer ones truncated.
3. **DEMAND channel:** only channel `ch01` of each environment is used. The other 15 channels are not used.
4. **Features:** log-mel spectrogram (64 mel bands, n\_fft 1024, hop 320 samples = 20 ms, 50 to 8000 Hz), giving a 64 x 251 input per clip. Models that take raw waveforms or their own front end (e.g. AST) use the waveform directly.

## 3\. Train / validation / test split

### 3.1 Audio (ESC-50)

We use the official ESC-50 folds:

|Split|Folds|Clips|
|-|-|-|
|Train|1, 2, 3|1,200|
|Validation|4|400|
|Test|5|400|

**Why there is no leakage:** many ESC-50 clips are cut from the same original Freesound recording. The dataset authors assigned all clips from one recording to the same fold, so no recording appears in two splits. We never re-split randomly. The test fold is used only for final evaluation, never for tuning.

### 3.2 Noise (DEMAND), split by environment

The noise environments are also split, so the test set is mixed with noise types the models never saw during training:

|Split|Environments|
|-|-|
|Train (10)|DKITCHEN, DLIVING, DWASHING, NFIELD, NPARK, OHALLWAY, OOFFICE, PSTATION, SPSQUARE, TCAR|
|Validation (3)|NRIVER, OMEETING, PRESTO|
|Test (4)|PCAFETER, STRAFFIC, TBUS, TMETRO|

This tests generalisation to unseen noise, not only to unseen sounds.

## 4\. Noise generation

Implemented from scratch in `src/datasets/noise.py` (`mix\_at\_snr`).

For a clean clip `x` and a noise segment `n` of the same length, the mixture is `x + g \* n` with

```
g = sqrt( P\_x / (P\_n \* 10^(SNR\_dB / 10)) ),   P = mean(signal^2)
```

so the mixture has exactly the requested SNR. If the mixture peak exceeds 1.0, the whole mixture is scaled down (this keeps the SNR unchanged and avoids clipping). The noise segment is a random crop of the 300 s recording.

* **Evaluation SNR levels:** 20, 10, 5, 0, -5 dB, plus the clean condition.
* **Training:** either clean, or noise augmentation with SNR drawn at random from {20, 10, 5, 0} dB using training environments only.
* **Validation / test are deterministic:** the noise segment and environment for each clip are fixed by a seed and the clip index. Every model is therefore evaluated on exactly the same noisy audio, which makes the comparison fair.

## 5\. Limitations

* SNR is computed over the whole 5 s clip. ESC-50 clips that are zero-padded contain silent parts, so the SNR over the part where the sound is active is lower than the nominal value.
* ESC-50 clips are not perfectly clean: some already contain background noise.
* Noise is mixed digitally and has no room acoustics or microphone effects.
* Only 17 noise environments, one channel each.
* ESC-50 is licensed for non-commercial use only (CC BY-NC), so models trained on it are for academic use.

## 6\. Reproducing the data

1. Clone ESC-50 into `data/ESC-50/` (`git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50`).
2. Download the 17 `\*\_16k.zip` files from https://zenodo.org/records/1227121, unzip each into `data/DEMAND/<ENV>/`.
3. Run `python data/check\_data.py` to verify the files and the noise mixer.

Raw data is not stored in the repository (`.gitignore`).

