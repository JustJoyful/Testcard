# Decoder repair focused review

**No Important or Critical findings.** Reviewed `repair.diff`, active v2 config, freeze regression, real VFR regression, dated decision, extraction/training/report/parity consumers and saved diagnosis evidence. No source or Git mutations; this requested report is the only write.

- `-enc_time_base demux` is confined to raw RGB output timing. Strict FFmpeg diagnostics, `-xerror`, `-err_detect explode`, truncated-read checks, extra-frame detection and final exit/error checks remain intact. Sampling, rotation, spatial transforms and pretrained features are unchanged.
- Saved evidence now contains successful shared-decoder smoke for all six failed clips and A001. All seven have identical legacy/repaired full RGB hashes, matching probe/decode frame counts and empty repaired stderr. The real generated VFR test exercises the original collision and compares sampled pixels with the legacy decoded stream.
- Config, scope and protocol select v2 consistently. Extraction resolves its root from cache_version; training resolves protocol cache_root; parity takes explicit feature/cache inputs. The common freeze resolver selects the declared v2 record and binds frozen status/version/config/source hashes. A v1 cache/parity record cannot satisfy the v2 identities.
- Compared with HEAD: v1 feature config, original freeze, readiness snapshot and frozen features module are byte-identical. Archived v1 parity matches the current historical artifact, and cumulative extraction invocations preserve the archived v1 prefix. Budget consumers sum all invocations without filtering out the old version, retaining its wall time.

**Pending, not a defect:** `preprocessing_freeze_v2.json` does not yet exist, as intentionally sequenced after smoke evidence. Generate and validate it before launching v2 extraction. Complete the fresh v2 caches and parity before training; current guards fail closed until those artifacts exist.

Declined to judge: full v2 extraction/parity outcomes and unchanged Phase 1 behavior outside this repair. Did not repeat the 145-test suite, expensive decoding or hardware jobs; assessed their saved evidence and the concrete consumer paths.

## Verified repair and live continuation

The full offline suite passes (145 tests). All six failed clips subsequently completed real GPU extraction with both backbones; `results/phase1/decoder_repair/verification.json` validates all 48 video/branch arrays, source hashes, shapes, dtypes, timesteps and checksums. The active v2 freeze was validated against actual config/source/lock/weights; original v1 config/freeze/splits are unchanged.

Fresh v2 fixed-sample extraction, full dataset extraction and parity are queued, followed by the original gated training sequence. GPU priming shell PID: 613777 (session 46339); full extraction shell PID: 629143 (session 39694); CPU parity watcher PID: 629308 (session 16281); training continuation PID: 642290 (session 96054). These launch identities replace the stopped v1 queue.

Monitor `.superpowers/decoder-repair/{priming,extraction,parity,remaining}.log`. Exact shell sequences are `prime.sh`, `extract-v2.sh`, and `run-remaining-v2.sh` in that directory. Keep one extraction writer. All active commands use `configs/features_v2.yaml` and `cache/v2`; the final report still must pass before Phase 1 is complete.

## User-requested pause


User-requested pause at 2026-10-26T19:54:53.316048+05:30: all extraction and continuation processes stopped. Published cache groups: {"konvid": {"dino": 1200, "resnet50": 1200}, "livevqc": {"dino": 463, "resnet50": 6}}. Resume with `bash /home/joydeep/Projects/VQA/scripts/continue_phase1.sh`; this sequence has no stale PID waits and revalidates completed cache entries. Resume shell verified by bash -n plus two isolated execution-order/failure-stop tests; no real workload was restarted.

The old process IDs above are historical; no pipeline jobs remain active. The resume command performs extraction first, then verifies cache/parity before the remaining gated stages. The unfinished video may be recomputed.
