#!/usr/bin/env python
"""Run every training run of a study (ablation / data strategy) with train.py.

    python scripts/run_study.py configs/studies/ablation.yaml
    python scripts/run_study.py configs/studies/data_strategy.yaml --seeds 0 1 2
    python scripts/run_study.py configs/studies/ablation.yaml --only conformer_no_mhsa --skip-existing
    python scripts/run_study.py configs/studies/ablation.yaml --dry-run       # print commands only

After training, evaluate on the test split with
    python src/evaluate.py --study <same yaml> --out experiments/eval_<study>
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--only", nargs="*", help="run only these run names")
    ap.add_argument("--skip-existing", action="store_true", help="skip runs with summary_seeds.json")
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], help="extra --set overrides for every run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    st = yaml.safe_load(open(ROOT / args.study, encoding="utf-8"))
    failed = []
    for r in st["runs"]:
        name = r["name"]
        if args.only and name not in args.only:
            continue
        if args.skip_existing and (ROOT / "experiments" / name / "summary_seeds.json").exists():
            print(f"[skip] {name} (already trained)")
            continue
        cmd = [sys.executable, str(ROOT / "src" / "train.py"), "--config", str(ROOT / r["config"]),
               "--set", f"name={name}", *r.get("set", []), *args.extra]
        if args.seeds:
            cmd += ["--seeds", *map(str, args.seeds)]
        if args.fast_dev_run:
            cmd.append("--fast-dev-run")
        print("\n>>> " + " ".join(cmd))
        if args.dry_run:
            continue
        if subprocess.call(cmd, cwd=ROOT) != 0:
            failed.append(name)
    print(f"\n[{st['study']}] finished. Failed: {failed or 'none'}")
    print(f"Evaluate: python src/evaluate.py --study {args.study} --out experiments/eval_{st['study']}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
