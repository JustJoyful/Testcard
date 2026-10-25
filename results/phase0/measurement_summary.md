# Historical Phase 0 measured extraction costs

All fixed-sample measurements succeeded. Values are end-to-end extraction time normalized to a 10-second video; these are engineering estimates, not head/API latency.

| Backbone | GPU fp16 mean (s) | CPU fp32 mean (s) | CPU fp32 worst sample (s) |
|---|---:|---:|---:|
| dino | 8.24 | 27.72 | 29.93 |
| resnet50 | 3.83 | 6.61 | 8.12 |
| slowfast_r50 | 16.45 | 33.86 | 45.23 |

KoNViD plus LIVE-VQC extraction, summing all three backbones with a 2× planning allowance: **24.50 GPU hours**. LIVE-VQC is extrapolated from KoNViD. Training remains unmeasured.

Predicted fp16 feature arrays: **1.15 GiB**. Additional disk including atomic-write headroom, one transient raw clip and reserve: **7.32 GiB**.

Phase 0 readiness now passes. SlowFast above is historical only; the active spatial scope and freeze are recorded in `scope_report.md` and `preprocessing_freeze.json`.
