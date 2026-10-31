# Phase 4a serving core

Status: **passed**. Phase 3: **passed with qualification**; control: **failed**.

Fixed videos: 20. Minimum per-timestep cosine: 0.999840967841. Maximum cache/raw score difference: 0.00484944152853 MOS. Maximum CLI/API score difference: 0.

Warm loopback request seconds per ten seconds: maximum 10.647; median 8.82536; budget 120.

Cold startup/first request and excluded warmup are separate in [cold](cold.json) and [warmup](warmup.json). Extraction/head/internal total and actual client round-trip are saved in [raw timings](latency.raw.json) and [CSV](latency.csv).

The [approved decoder amendment](packaging_amendment.json) uses the same release branch plus 37 maintenance commits; [host/image full versions](decoder_versions.json) are recorded side by side. All predeclared thresholds are unchanged.

CPU image: `sha256:7e98d3fa13066c7b401f2c4587d7cfd162cfcc7fa152c7143153b353d55445d5`. [Build](build.json), [parity](parity.json), [tests](tests.json), [operations](../../docs/phase4a-operations.md), [NOTES](../../NOTES.md).

Measurements are local loopback under the recorded caps, not hosted/network-independent latency. Pathological decode may occupy the single slot until decoder exit; forced process death may leave local temporary files. Public-internet hardening, presentation, hosting, final README and publication remain outside scope.
