import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "datasets"))
from noise import mix_at_snr, NOISE_SPLITS

ESC = "data/ESC-50"
DEMAND = "data/DEMAND"

n_wav = len([f for f in os.listdir(f"{ESC}/audio") if f.endswith(".wav")])
meta = pd.read_csv(f"{ESC}/meta/esc50.csv")
print(f"ESC-50 wav files: {n_wav} (expected 2000)")
print("clips per fold:", meta.groupby("fold").size().to_dict())
print("classes:", meta["target"].nunique(), "(expected 50)")

found = [d for d in sorted(os.listdir(DEMAND)) if os.path.isdir(f"{DEMAND}/{d}")]
print("DEMAND environments found:", found)
for split, envs in NOISE_SPLITS.items():
    missing = [e for e in envs if e not in found]
    print(f"  {split}: {len(envs) - len(missing)}/{len(envs)} present",
          "missing: " + str(missing) if missing else "")

rng = np.random.default_rng(0)
clean, noise = rng.standard_normal(80000) * 0.03, rng.standard_normal(80000)
for snr in [20, 10, 5, 0, -5]:
    mixed = mix_at_snr(clean, noise, snr)
    g_noise = mixed - clean
    measured = 10 * np.log10(np.mean(clean ** 2) / np.mean(g_noise ** 2))
    print(f"target SNR {snr:>3} dB -> measured {measured:6.2f} dB")