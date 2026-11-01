# Ten-day scope feasibility

Status: **go for Phase 1**. Spatial preprocessing/cache policy is frozen in `configs/features_v1.yaml`.

| Planning input | Allowance |
|---|---:|
| Deadline | 2026-11-04 |
| Hands-on hours | 80 |
| Unattended GPU hours | 100 |

Active extraction projection (DINO + ResNet, both datasets, 2× allowance): **10.37 GPU hours**.
Predicted fp16 cache (all planned DINO statistics plus ResNet): **0.87 GiB**.
Additional disk including atomic-write headroom, transient raw video and reserve: **6.78 GiB**.

Completed historical preflight coverage: all measurements and pretrained CPU/GPU checks passed. The original conservative readiness report includes SlowFast; the active projection above excludes it.

Cuts: SlowFast; all YouTube-UGC/cloud work; separate layer comparison. Tuning is 10 trials per head. Serving is FastAPI + HTML upload page + one Dockerfile.

The primary DINO representation stays CLS plus block-12 mean; cache all planned layers anyway. Ridge, four heads, five folds/seeds, shuffled-label control, correctness tests, confidence intervals, LIVE-VQC and serving parity remain mandatory.

Training, tuning and end-to-end API runtime remain unmeasured. Passing Phase 0 confirms extraction feasibility, not a guaranteed finish date for the full project.
