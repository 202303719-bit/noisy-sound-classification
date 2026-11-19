## Training (baseline + classic models)

> Paste this section into the main README (items 5 and 8 of the README structure).

All models are trained with one script, which reads a YAML config, trains, selects the best
epoch / grid point on the **validation** split and saves a checkpoint. It never touches the test split.

```bash
# 0) smoke test (1 epoch, 2 batches, 64 clips): checks the data + model pipeline in about 1 min
python src/train.py --config configs/cnn.yaml --fast-dev-run

# 1) Baselines: MFCC statistics + SVM / Random Forest (CPU, a few minutes)
python src/train.py --config configs/baseline_svm.yaml
python src/train.py --config configs/baseline_rf.yaml

# 2) Classic models trained from scratch on log-mel (GPU recommended)
python src/train.py --config configs/cnn.yaml
python src/train.py --config configs/crnn.yaml

# options
python src/train.py --config configs/cnn.yaml --seeds 0            # single seed
python src/train.py --config configs/cnn.yaml --set train.lr=0.0003 model.params.dropout=0.5
python src/train.py --config configs/cnn.yaml --set model.params.spec_augment.enabled=false   # ablation
python src/train.py --config configs/cnn.yaml --set data.train.snr_db=10                      # noisy training
```

| Output | Content |
|---|---|
| `checkpoints/<name>_seed<k>.pt` | best CNN/CRNN weights + config + front-end info |
| `checkpoints/<name>_seed<k>.joblib` | best scikit-learn pipeline (scaler + classifier) + feature config |
| `experiments/<name>/seed<k>/log.csv` | per-epoch (deep) or per-grid-point (baseline) metrics |
| `experiments/<name>/seed<k>/summary.json` | best val metrics, params, train time, inference ms/clip, size, hardware |
| `experiments/<name>/summary_seeds.json` | mean ± SD over the seeds |

### Loading a checkpoint (for evaluate.py / demo)

```python
import sys, torch, joblib; sys.path.insert(0, "src")
import models as M

ck = torch.load("checkpoints/cnn_scratch_seed0.pt", map_location="cpu", weights_only=False)
model = M.build_torch_model(ck["model_name"], ck["n_classes"], ck["model_params"])
model.load_state_dict(ck["state_dict"]); model.eval()
frontend = M.LogMelFrontend(**ck["frontend"])
logits = model(frontend(wave[None]))                  # wave: (T,) tensor from ESC50

bl = joblib.load("checkpoints/baseline_svm_seed0.joblib")
feats = M.MFCCStats(**bl["feature_config"])(wave[None]).numpy()
pred = bl["model"].predict(feats)
```
