#!/bin/bash
set -euo pipefail
if [[ -z "${TMUX:-}" ]]; then
    printf 'Run inside tmux: tmux new -s train\n' >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONUNBUFFERED=1
mkdir -p results/phase2
run_pipeline() (
    phase2_stage='starting'
    trap 'phase2_exit=$?; if [[ "$phase2_exit" -ne 0 ]]; then printf "{\"status\":\"failed\",\"stage\":\"%s\",\"reason\":\"worker exit %s; see session.log and progress.json\"}\n" "$phase2_stage" "$phase2_exit" > results/phase2/phase_gate.json.partial; mv -- results/phase2/phase_gate.json.partial results/phase2/phase_gate.json; printf "Phase 2 exit %s at stage: %s; see session.log and progress.json\n" "$phase2_exit" "$phase2_stage"; fi' EXIT
    printf '{"status":"failed","reason":"Phase 2 queue in progress"}\n' > results/phase2/phase_gate.json.partial
    mv -- results/phase2/phase_gate.json.partial results/phase2/phase_gate.json
    for phase2_stage in check control-initial control cv select; do
        printf '\nPhase 2 stage: %s\n' "$phase2_stage"
        .venv/bin/python -u scripts/05_phase2.py --mode "$phase2_stage" --config configs/phase2.yaml --features configs/features_v2.yaml
    done
    phase2_stage='bootstrap'
    .venv/bin/python -u scripts/07_bootstrap.py --config configs/phase2.yaml --features configs/features_v2.yaml
    phase2_stage='tables'
    .venv/bin/python -u scripts/13_make_tables.py --config configs/phase2.yaml --features configs/features_v2.yaml
)
run_pipeline 2>&1 | tee -a results/phase2/session.log
