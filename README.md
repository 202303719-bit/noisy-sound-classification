# Environmental Sound Classification under Noise

Deep Learning (AI403) final project, Pharos University in Alexandria, Fall 2026.
Idea 17: Environmental sound classification under noise.

Team: <
youssef walid El_bastwesy             202303719
farid elsayed Elgedawy                202302103
taha abdelhaleem                      202301786
Mohamed Elsayed Badawy Mohamed        202201851
Youssef Shabaan Saleh Shabaan         202303487
>

## Datasets

- **ESC-50** (sounds): https://github.com/karolpiczak/ESC-50, CC BY-NC 3.0
- **DEMAND** (noise): https://zenodo.org/records/1227121, CC BY-SA 3.0

Full details (versions, licences, preprocessing, split): see `docs/DATASET.md`.

## Environment setup

pip install torch torchaudio soundfile pandas numpy

## Dataset download and preprocessing

1. Clone ESC-50 into `data/ESC-50`:

   git clone https://github.com/karolpiczak/ESC-50.git data/ESC-50

2. Download the 17 `*_16k.zip` files from the DEMAND Zenodo page and unzip each into
   `data/DEMAND/<ENV>/` (for example `data/DEMAND/DKITCHEN/ch01.wav`).

3. Verify the data and the noise mixer:

   python data/check_data.py

Preprocessing (16 kHz mono, 5 s clips, log-mel, noise mixing at a given SNR) happens on the fly
inside `src/datasets/esc50.py`, so there is no separate preprocessing command.

## Using the dataset in code

    from datasets.esc50 import ESC50, LogMel
    train = ESC50("data/ESC-50", "train", snr_db=[20, 10, 5, 0], demand_dir="data/DEMAND")
    test  = ESC50("data/ESC-50", "test", snr_db=5, demand_dir="data/DEMAND")

## Credits

- ESC-50: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", ACM Multimedia 2015.
- DEMAND: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)", 2013.
- The SNR noise mixer (`src/datasets/noise.py`) was written by the team.