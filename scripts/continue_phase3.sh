#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
if [[ -z "${TMUX:-}" ]]; then
  echo 'Phase 3 must run inside an existing tmux session.' >&2
  exit 1
fi

mkdir -p results/phase3
log=results/phase3/session.log
gate=results/phase3/phase_gate.json
failed_gate() {
  status=$?
  if (( status != 0 )); then
    printf '{"status":"failed","reason":"Phase 3 pipeline stopped before verified completion"}\n' > "${gate}.partial"
    mv "${gate}.partial" "$gate"
  fi
}
trap failed_gate EXIT

stage() {
  printf '\n=== Phase 3: %s ===\n' "$1" | tee -a "$log"
  shift
  .venv/bin/python -m "$@" 2>&1 | tee -a "$log"
}

if ! stage check scripts.08_phase3 --config configs/phase3.yaml --mode check; then
  for branch in dino resnet50; do
    list="results/phase3/repair_${branch}_ids.txt"
    if [[ -s "$list" ]]; then
      printf 'Repair only %s with:\n.venv/bin/python scripts/02_extract_spatial.py --config configs/features_v2.yaml --dataset livevqc --backbone %s --sample-ids %s\n' "$branch" "$branch" "$list" | tee -a "$log"
    fi
  done
  exit 1
fi
stage cpu scripts.08_cpu_eligibility --config configs/phase3.yaml
if [[ -f results/phase3/control_fixed_epoch/qualification.json ]]; then
  stage qualified_control scripts.09_phase3_refit --config configs/phase3.yaml --mode qualification
else
  stage control scripts.09_phase3_refit --config configs/phase3.yaml --mode control
fi
stage fit scripts.09_phase3_refit --config configs/phase3.yaml --mode fit
stage predict scripts.10_phase3_predict --config configs/phase3.yaml
stage report scripts.11_phase3_report --config configs/phase3.yaml
