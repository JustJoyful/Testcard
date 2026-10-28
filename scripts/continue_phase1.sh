#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
phase1_stage='resuming checksum-verified feature extraction'
trap 'phase1_exit=$?; printf "Phase 1 queue exit %s at stage: %s\n" "$phase1_exit" "$phase1_stage"; if [ "$phase1_exit" -ne 0 ]; then .venv/bin/python scripts/06_eval.py --config configs/protocol.yaml --results results/phase1 --features configs/features_v2.yaml || true; fi' EXIT
# One writer; completed video entries are checksum-verified and skipped.
for dataset in konvid livevqc; do
    for backbone in dino resnet50; do
        .venv/bin/python scripts/02_extract_spatial.py --config configs/features_v2.yaml --dataset "$dataset" --backbone "$backbone"
    done
done
phase1_stage='verifying complete extraction and parity'
.venv/bin/python - <<'PY'
import importlib
from pathlib import Path
import yaml
training=importlib.import_module('scripts.05_train')
parity=importlib.import_module('scripts.04_parity_check')
protocol=yaml.safe_load(Path('configs/protocol.yaml').read_text())
provenance=training.prerequisites(protocol,Path('configs/features_v2.yaml'))
print(parity.validate_parity(Path('results/phase1/parity.json'),provenance,Path('cache/v2')),flush=True)
PY
phase1_stage='initial five-seed leakage control'
.venv/bin/python scripts/05_train.py --mode control-initial --config configs/protocol.yaml --features configs/features_v2.yaml
phase1_stage='training pilot and measured runtime gate'
.venv/bin/python scripts/05_train.py --mode pilot --config configs/protocol.yaml --features configs/features_v2.yaml
phase1_stage='shared rank-weight selection'
.venv/bin/python scripts/05_train.py --mode rank --config configs/protocol.yaml --features configs/features_v2.yaml
phase1_stage='equal-budget head tuning'
.venv/bin/python scripts/05_train.py --mode tune --config configs/protocol.yaml --features configs/features_v2.yaml
phase1_stage='final five-seed leakage control'
.venv/bin/python scripts/05_train.py --mode control --config configs/protocol.yaml --features configs/features_v2.yaml
phase1_stage='definitive DINO CV and ridge'
.venv/bin/python scripts/05_train.py --mode cv --config configs/protocol.yaml --features configs/features_v2.yaml
phase1_stage='offline tests and final exit report'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/06_eval.py --config configs/protocol.yaml --results results/phase1 --features configs/features_v2.yaml
phase1_stage='complete; inspect results/phase1/phase_gate.json'
