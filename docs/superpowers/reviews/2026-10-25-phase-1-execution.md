# Phase 1 execution record

The complete implementation is on branch `phase1-core-slice`; the reviewed fix commit is `978aa18`. Offline verification: `.venv/bin/python -m pytest -q` → **143 passed**. Independent review and the test-first fix evidence are in [the review record](2026-10-25-phase-1-review.md).

At launch, real fixed-sample GPU extraction passed for DINO and ResNet. Full extraction is running in order: KoNViD DINO, KoNViD ResNet, LIVE-VQC DINO, LIVE-VQC ResNet. CPU fp32 parity is running independently against the original fixed sample. No head training or LIVE-VQC prediction has run yet. This is an execution-start record, not a Phase 1 completion claim.

A live sequential continuation waits for extraction and parity, verifies both, then runs initial control → pilot/runtime gate → rank selection → head tuning → final control → definitive DINO CV/ridge → normal pytest → final report. It commits each successful stage's eligible artifacts locally. It stops on any failed prerequisite, failed control, runtime no-go, missed calendar checkpoint or failed command. The common two-seed tuning fallback may activate only from the measured no-go; reported runs and controls keep all five seeds. Further cuts remain proposals.

All commands run from `/home/joydeep/Projects/VQA` using `.venv/bin/python`. No merge or push is queued. The training continuation is a shell sequence of the declared CLIs, not a scheduler or another extraction writer.

Monitor:

```bash
cd /home/joydeep/Projects/VQA
tail -n 10 .superpowers/sdd/2026-10-25-phase-1-core-slice/extract-full.log
tail -n 10 .superpowers/sdd/2026-10-25-phase-1-core-slice/parity.log
tail -n 30 .superpowers/sdd/2026-10-25-phase-1-core-slice/remaining.log
ps -eo pid,etime,args | rg '02_extract_spatial.py|04_parity_check.py|05_train.py|run-remaining.sh'
```

At launch the extraction shell PID was `877532`, parity PID `969024`, and continuation shell PID `1134033`. These are launch identities, not durable status. Tool execution sessions are `49026` (extraction), `8140` (parity), `60374` (continuation). The ignored execution ledger is `.superpowers/sdd/2026-10-25-phase-1-core-slice/progress.md`; the exact continuation is `run-remaining.sh` in the same directory. Do not launch another continuation or extraction writer while these are active.

The authoritative data/result acceptance command is:

```bash
.venv/bin/python scripts/06_eval.py \
  --config configs/protocol.yaml \
  --results results/phase1 \
  --features configs/features_v1.yaml
```

It exits nonzero and publishes missing/failed gates until actual evidence is complete. A passing `results/phase1/phase_gate.json` certifies the exit; final summaries, time accounting and phase documents are generated then. Keep the machine awake and preserve the running execution sessions. Avoid editing files consumed by extraction/training while their jobs run. If a gate fails, inspect its saved evidence before restarting; never reroll the control permutation or regenerate splits.

Implementation rulings: work in the named branch in place so ignored inputs remain available; code commit is audit-only for otherwise compatible saved artifacts. Hardware remains gate-dependent; single-writer and later-phase exclusions follow the approved scope; artifact checks validate provenance relationships without an added attestation system. The costs are no directory isolation, unsupported simultaneous external writers, and a possible explicit stop for a scope decision. No review findings remain deferred.

Parity subsequently completed successfully: 160 real CPU/GPU comparisons; minimum timestep cosine 0.9998409678411763. Evidence: `results/phase1/parity.json`, independently revalidated against current sample arrays and source hashes. Full extraction and all head-training stages remain pending.
