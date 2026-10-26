# Decoder repair focused review

**No Important or Critical findings.** Reviewed `repair.diff`, active v2 config, freeze regression, real VFR regression, dated decision, extraction/training/report/parity consumers and saved diagnosis evidence. No source or Git mutations; this requested report is the only write.

- `-enc_time_base demux` is confined to raw RGB output timing. Strict FFmpeg diagnostics, `-xerror`, `-err_detect explode`, truncated-read checks, extra-frame detection and final exit/error checks remain intact. Sampling, rotation, spatial transforms and pretrained features are unchanged.
- Saved evidence now contains successful shared-decoder smoke for all six failed clips and A001. All seven have identical legacy/repaired full RGB hashes, matching probe/decode frame counts and empty repaired stderr. The real generated VFR test exercises the original collision and compares sampled pixels with the legacy decoded stream.
- Config, scope and protocol select v2 consistently. Extraction resolves its root from cache_version; training resolves protocol cache_root; parity takes explicit feature/cache inputs. The common freeze resolver selects the declared v2 record and binds frozen status/version/config/source hashes. A v1 cache/parity record cannot satisfy the v2 identities.
- Compared with HEAD: v1 feature config, original freeze, readiness snapshot and frozen features module are byte-identical. Archived v1 parity matches the current historical artifact, and cumulative extraction invocations preserve the archived v1 prefix. Budget consumers sum all invocations without filtering out the old version, retaining its wall time.

**Pending, not a defect:** `preprocessing_freeze_v2.json` does not yet exist, as intentionally sequenced after smoke evidence. Generate and validate it before launching v2 extraction. Complete the fresh v2 caches and parity before training; current guards fail closed until those artifacts exist.

Declined to judge: full v2 extraction/parity outcomes and unchanged Phase 1 behavior outside this repair. Did not repeat the 145-test suite, expensive decoding or hardware jobs; assessed their saved evidence and the concrete consumer paths.
