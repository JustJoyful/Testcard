# Repository Guidelines

## Project Structure & Module Organization

This project studies cached-feature video quality assessment: frozen backbones feed mean-pool, Bi-GRU, Transformer, and Mamba heads. The current checkout contains planning documents and dataset ZIP archives; application code is not yet present.

Read `RULES.md` for binding requirements, `PHASES.md` for ordered milestones, `ARCHITECTURE.md` for interfaces and preprocessing, `PRD.md` for research goals, and `DECISIONS.md` for dated changes.

Follow the planned layout in `ARCHITECTURE.md`:
- `src/`: shared preprocessing, features, heads, and metrics.
- `scripts/`: numbered pipeline stages, such as `00_timing.py` and `05_train.py`.
- `configs/` and `env/requirements.lock`: experiment settings and locked dependencies.
- `data/splits/` and `data/manifests/`: committed indices and metadata; `data/raw/` and `cache/`: uncommitted videos and features.
- `results/`: predictions, metrics, and generated tables.
- `serving/`: planned FastAPI API, Celery worker, and React frontend.

## Build, Test, and Development Commands

No build, application, or test commands are implemented in this checkout. Phase 0 must establish the environment before experiments begin.

Once `env/requirements.lock` exists, install it with `python -m pip install -r env/requirements.lock`. Run numbered scripts in stage order using their declared config inputs; document exact CLI arguments when implemented. Do not assume undocumented flags. Docker Compose belongs to Phase 4.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` modules and functions, and descriptive class names. Preserve numbered script names and use YAML for experiment configs. No formatter or linter is configured yet.

Offline, serving, and notebook code must import the same `src/preprocess.py` and `src/features.py`; never duplicate them.

## Testing Guidelines

No test framework or coverage threshold is established. When adding tests, use descriptive names such as `test_padding_invariance`.

Required checks include padding invariance for every head, Mamba versus a naive recurrence, causal-variant causality, shuffled-label leakage control, and serving/cloud parity. Follow thresholds in the planning documents. Never tune on held-out folds or predict LIVE-VQC before freezing the KoNViD configuration.

## Commit & Pull Request Guidelines

Git history is unavailable here, so no existing commit convention can be verified. Use concise, imperative subjects. PRs should explain the change, affected phase, validation performed, and linked issue when applicable. Include screenshots for UI changes and dated `DECISIONS.md` entries for plan changes.

Never commit videos or feature caches. Generate reported metrics from saved outputs; do not enter results manually.
