# Phase 4b local demo

Status: **passed**. Phase 3: **passed with qualification**; control: **failed**.

Fixed videos: 20, each through the CLI `--json`, `/predict` and `/upload`→`/status`. Minimum per-timestep cosine: 0.999840967841. Maximum cache/raw score difference: 0.00484944152853 MOS. Maximum front-door score difference: 0.

Warm upload-to-result wall seconds per ten seconds of video (browser polling contract): maximum 12.7292; median 10.0395; budget 120. Polling overhead beyond upload and internal inference: median 0.595 s, maximum 0.974 s.

Production JavaScript: 9087 bytes gzip (budget 30000, fonts excluded). Lighthouse accessibility on the built container: 1.00.

Cold startup/first task and the excluded warmup are in [cold](cold.json) and [warmup](warmup.json); every warm row with upload, internal extraction/head/total and polling overhead is in [raw timings](latency.raw.json) and [CSV](latency.csv). Per-clip raw CLI, API and status responses are under `cli/`, `api/` and `status/`.

CPU image: `sha256:85a3e58c672186cb2f896e49ffecc395715dff26d88e4e55b11e4fa38e6aeb8a`. Phase 4a anchor image: `sha256:7414a60bd7fe92ffe8fa5f8c5fde06bae0f33ba2b0f8d7fc2b57ab783e9e1950` ([baseline](baseline/manifest.json)). [Build](build.json), [image sources](image_sources.json), [decoder versions](decoder_versions.json), [parity](parity.json), [tests](tests.json), [Lighthouse](lighthouse.json), [operations](../../docs/phase4b-operations.md), [NOTES](../../NOTES.md).

Measurements are local loopback on one CPU host under the recorded caps, not hosted latency. Scores are raw, unclipped and uncalibrated KoNViD MOS-scale ridge outputs. Job metadata is in memory only and video bytes are deleted after each job. Hosting, final README and publication remain outside this phase.
