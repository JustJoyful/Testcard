# Historical checkpoint before the decoder amendment

This saved checkpoint precedes the user-approved packaging amendment. See [current generated report](report.md) and [gate](phase_gate.json) for closure status.

# Phase 4a implementation status

Incomplete: Tasks 1–3 implemented locally; Task 4 stopped at the required exact static decoder archive. Tasks 5–6 remain pending. This is not a passed phase gate.

Implemented: selected portable bundle, CPU-only dependency locks/environment, shared raw-video CLI, FastAPI upload/form, pre-parser admission and limits, cancellation lifetime and zero cooperative retention.

Validation (generated from saved command outputs):

- serving: `48 passed, 1 warning in 6.16s`; command: `.venv-serving/bin/python -m pytest tests/test_serving_bundle.py tests/test_predict_cli.py tests/test_serving_api.py -q`; [log](final_serving_tests.log).

- cpu_shared: `82 passed, 1 deselected, 1 warning in 9.83s`; command: `.venv-serving/bin/python -m pytest tests/test_serving_api.py tests/test_predict_cli.py tests/test_serving_bundle.py tests/test_phase0_preprocess.py tests/test_phase0_features.py tests/test_phase1_ridge.py tests/test_phase1_parity.py tests/test_phase3_gate.py --deselect tests/test_phase0_preprocess.py::test_slowfast_matches_installed_reference_on_synthetic_pixels -q`; [log](final_cpu_shared_tests.log).

- research: `256 passed, 1 deselected in 27.16s`; command: `.venv/bin/python -m pytest -q --ignore tests/test_predict_cli.py --ignore tests/test_serving_api.py --ignore tests/test_serving_bundle.py --deselect tests/test_phase1_pilot.py::test_cli_repairs_interrupted_protocol_freeze_without_refitting`; [log](final_research_tests.log).

The existing `tests/test_phase1_pilot.py::test_cli_repairs_interrupted_protocol_freeze_without_refitting` failed at its historical calendar deadline; it remains unchanged and is excluded from the final research regression run. The legacy SlowFast reference test requires the research environment; it is intentionally excluded from the CPU-only shared regression command and passes in research regression. Starlette emits a TestClient/HTTPX deprecation warning.

Local [CLI smoke](cli_smoke.json) and [HTTP smoke](api_smoke.json) verify selected-model prediction, local score agreement, busy rejection, responsive health, invalid-file rejection and cleanup. They do not replace Docker fixed-sample parity or request-latency gates.

Packaging: No exact n9.0.2 static FFmpeg/ffprobe archive found in checked BtbN releases; verified candidate is a later git revision. John Van Sickle release advertises 7.0.2. No substitution authorized by the spec.

Checked candidate: `ffmpeg version n9.0.2-37-gae19868753-20261029 Copyright (c) 2000-2026 the FFmpeg developers`. Full [host](decoder_host.txt), [candidate](decoder_candidate.txt) and [archive/metadata evidence](decoder_packaging.json) are retained. No Docker image was built.

[Historical verification](historical_verification.json) checks the original source roster, all historical bindings and Phase 3 outputs. Phase 3 remains **passed with qualification** and its control remains **failed**.

One fresh-context read-only review found an input/output alias data-loss bug. Same-path, symlink and hardlink regression cases failed before the guard and pass after it. The CLI now rejects those exports before inference.

[Operations](../../docs/phase4a-operations.md) and [NOTES](../../NOTES.md) document commands, limits and unresolved items. No commit, push, publish, styling or final README work was performed. An exact static decoder archive with immutable URL/full SHA256, or an explicit packaging-spec amendment, is required to resume.
