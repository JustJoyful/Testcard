# Phase 4a implementation review

One fresh-context whole-implementation static review by gpt-6-astra. No test or inference load was introduced by the reviewer during timing collection. Earlier partial CLI review had already fixed source-overwrite aliases.

Important findings addressed in one fix pass:

- Report entry-point settings errors could preserve an old passed gate. Unsupported-winner and malformed-YAML regressions failed first; both pass after invalidation moved before settings loading.
- New tested runtime sources could be paired with an old measured image. A post-build source-edit regression failed first; it passes with a mandatory complete runtime-source proof captured from the actual immutable image and checked against the current checkout. The proof is separate from existing measurement records; no measured scores or timings were rewritten.

The exact original executed measurement-tool source is retained by its SHA256 under provenance/. Current validation reproduces every saved per-video comparison and warm summary before final publication. Thresholds, runtime source, image, research outputs and original source roster are unchanged by these evidence-only fixes.

Deferred minor: src/predict.py does not normalize subprocess.SubprocessError from an unusual decoder process-wait timeout; it can yield a generic API 500 or CLI traceback. This is rare failure presentation, not a false successful score, data-loss or admission-lifetime issue. Cleanup still executes. Shared preprocessing is frozen; no timeout or numerical policy was changed.

Review scope excluded public-internet hardening, forcible-process-death recovery, exhaustive pathological codec/aspect-ratio stress, hostile local rewriting of code/proofs, alternative architectures, presentation/hosting/README/publication and integration. Linux x86_64 and the recorded local resource caps are the packaging scope. Final actual latency, tests and publication are verified by the executor after the static review.
