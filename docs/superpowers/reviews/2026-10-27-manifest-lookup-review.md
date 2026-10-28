# Manifest lookup optimization review

**No Important or Critical findings.** Reviewed the two source diffs and appended regression test against HEAD.

- Index keys use the same `(dataset, video_id, feature_set)` tuple as the previous filter. Each key retains a list of all matching records, so duplicate entries still fail the existing exactly-one-record check rather than being overwritten.
- Missing records, failed status, source SHA/size, array SHA, shape, dtype, finite values and nonempty sequence checks remain on the same `read_branch` path. Existing list callers retain their previous behavior.
- Coverage and dataset construction each index their already-read manifest snapshot once. Dataset iteration still follows the inventory and then explicitly reorders items using committed split IDs; indexing does not alter feature order, label association, timestep counts or batches.
- The added regression checks correct retrieval, source mismatch, duplicate rejection and corrupt-array checksum rejection. Saved `lookup_validation.json` reports bitwise-equal results across the real branches checked and appropriately limits the timing claim to this diagnostic.

No fitting, preprocessing, frozen config or artifact-identity behavior changes were found. No tests/fits were rerun; source and Git state were untouched. This requested review file is the only write.
