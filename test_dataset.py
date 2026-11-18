import sys
sys.path.append("src")
from datasets.esc50 import ESC50, LogMel

# test set مخلوط بـ noise غير مشوف عند SNR = 5 dB
ds = ESC50("data/ESC-50", "test", snr_db=5, demand_dir="data/DEMAND")
x, y = ds[0]
print("clips in test:", len(ds))
print("waveform:", x.shape, "label:", y)
print("log-mel:", LogMel()(x[None]).shape)

# نفس الـ clip لازم يطلع نفس الـ noise كل مرة (عشان كل الموديلات تتقيّم على نفس الداتا)
x2, _ = ds[0]
print("deterministic:", bool((x == x2).all()))

# clean
ds_clean = ESC50("data/ESC-50", "train")
print("train clips:", len(ds_clean))