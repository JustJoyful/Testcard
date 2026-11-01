# Phase 0 feasibility

Status: **go**

Missing or failed gates:


Measured projections:

```json
{
  "dino": {
    "seconds_per_decoded_second": 0.8244966632509566,
    "repeat_video_ratio_std": 0.019027891939534428,
    "konvid_seconds": 7916.924089860584,
    "livevqc_seconds": 4833.135920754197,
    "combined_planning_seconds": 25500.120021229563,
    "note": "LIVE-VQC extrapolated from KoNViD; includes 2x planning allowance; training unmeasured."
  },
  "resnet50": {
    "seconds_per_decoded_second": 0.38258486484854337,
    "repeat_video_ratio_std": 0.01986346259230352,
    "konvid_seconds": 3673.6295826749356,
    "livevqc_seconds": 2242.6830034041855,
    "combined_planning_seconds": 11832.625172158241,
    "note": "LIVE-VQC extrapolated from KoNViD; includes 2x planning allowance; training unmeasured."
  },
  "slowfast_r50": {
    "seconds_per_decoded_second": 1.6451685357215016,
    "repeat_video_ratio_std": 0.02437993409878888,
    "konvid_seconds": 15797.122041681114,
    "livevqc_seconds": 9643.851212615502,
    "combined_planning_seconds": 50881.94650859323,
    "note": "LIVE-VQC extrapolated from KoNViD; includes 2x planning allowance; training unmeasured."
  }
}
```

DINO-only extraction estimate; head inference and API overhead unmeasured; 25% extraction headroom required.

Current free bytes already exclude retained archives, installed environments, weights and retained sample clips; additional capacity includes two full feature copies for atomic-write headroom, a transient raw video and reserve.

Hardware checks are separate from offline unit tests. Preprocessing is not frozen until all gates pass. No training or LIVE-VQC predictions have been run.
