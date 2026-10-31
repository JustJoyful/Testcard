# Phase 4a operations

The generated [report](../results/phase4a/report.md) and [gate](../results/phase4a/phase_gate.json) give the verified phase status and measured results. The upstream Phase 3 gate passed with qualification; its fixed-epoch shuffled-label control remains failed.

## Setup

Run from the repository root using CPython 3.11.15 on Linux x86_64. The runtime uses CPU-only torch 2.14.1+cpu and torchvision 0.29.1+cpu. Research dependencies remain separate.

```bash
git lfs pull --include="results/phase3/refit/resnet50/ridge/seedNone/model.npz"
uv venv --python .venv/bin/python .venv-serving
uv pip sync --python .venv-serving/bin/python --require-hashes --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match env/serving-test.lock
uv pip check --python .venv-serving/bin/python
.venv-serving/bin/python -m scripts.12_prepare_serving --config configs/serving.yaml --output artifacts/serving
```

Production needs `env/serving.lock`; the test lock additionally installs pytest and HTTPX. Both locks require the official PyTorch CPU index. Preparation checks the historical source roster and all Phase 3 outputs; added serving files do not replace the historical digest. The bundle includes only the selected model, public ResNet V2 checkpoint and numerical metadata. If absent, preparation downloads the public checkpoint from torchvision's explicit `ResNet50_Weights.IMAGENET1K_V2.url` and verifies its complete SHA256 before using it. A corrupt existing checkpoint fails rather than being replaced. No ordinary inference download, cache or labels are needed.

## CLI

```bash
.venv-serving/bin/python -m src.predict sample.mp4
.venv-serving/bin/python -m src.predict sample.mp4 --json
.venv-serving/bin/python -m src.predict sample.mp4 --bundle artifacts/serving/manifest.json --json --features-out /tmp/features.npy
```

The default is a human-readable score, clip duration, extraction/backbone and ridge timings, CPU thread count, short model/weight hashes and the full preprocessing SHA256. Use `--json` for exactly the original JSON response schema, including full score precision and hashes; evidence tooling always passes this flag explicitly. Only the known read-only-frame warning is hidden by the CLI; other warnings remain visible. Errors use stderr and a nonzero exit. `score` is raw, unclipped KoNViD MOS-scale ridge output; predictions may lie outside 1–5. No LIVE logistic mapping is applied. `verified` identifies the actual checked model, weights, preprocessing freeze and four CPU threads. CPU fp32 applies to backbone features; pooling and ridge arithmetic remain float64. The optional feature export contains the exact raw fp32 rows used for the score.

`startup_seconds` covers validation and model loading. Extraction includes admission probing, decoding, transforms and the backbone; head timing includes float64 pooling/ridge. Internal total excludes startup; it includes the optional local feature write when requested. Saved local smoke responses are setup checks, not the fixed-video Docker parity gate.

## Media limits

MP4/MOV with an accepted ISO base-media signature and WebM with an EBML/WebM signature are accepted. URLs, playlists, empty files and unsupported containers fail. Maximum file size is 100 MiB; duration 30 seconds; source area 2073600 pixels; longest side 1920; average frame rate 60 fps. Unknown/nonfinite metadata and invalid orientation fail. The shared decoder applies the frozen rotation and VFR sampling behavior. The CLI retains the user's video.

## API and plain upload page

```bash
.venv-serving/bin/uvicorn serving.app:create_app --factory --workers 1 --host 127.0.0.1 --port 8000
curl --fail http://127.0.0.1:8000/health
curl --fail -F file=@sample.mp4 http://127.0.0.1:8000/predict
```

Open `http://127.0.0.1:8000/` for the plain form. There is one prediction endpoint and one loaded runtime. Startup fails on corrupt/missing/pointer artifacts or frozen source mismatches. Never use multiple workers or reload: admission is per process.

Exactly one request is admitted before multipart parsing/body reads. Concurrent uploads receive 503. Actual received body bytes are bounded to 101 MiB independently of Content-Length; receipt is limited to 60 seconds. One file named `file`, no extra fields, at most 100 MiB of file bytes. Metadata/container limits match the CLI. Rejections use 413 (bytes), 408 (receipt), 415 (container/request type), 422 (metadata/decode/multipart) and 503 (busy); unexpected failures return a generic 500. Responses do not include filesystem paths or tracebacks.

All parser spools and generated temporary files are closed/deleted after success, failure or disconnect: zero retention. Client filenames never determine disk paths. A cancelled request retains admission and its file until its running inference thread actually completes. Health remains responsive because inference runs in a thread in the same process. No queue, separate worker or durable jobs exist.

The receipt deadline is not an inference deadline. Pathological decode may occupy the only slot until the shared decoder exits. Existing decoder subprocess timeouts and finalization remain unchanged; CPU/memory/temp-disk caps are required for the Docker deployment. General public-internet hardening is outside this phase's claim.

## Approved packaging amendment and clean build

The user-approved decoder is BtbN `n9.0.2-37-gae19868753-20261029`, pinned to the dated archive and full SHA256 in the Dockerfile and `results/phase4a/packaging_amendment.json`. The exact historical decision-log prefix remains verified; the approved appended entry is checked separately. Parity tolerances and the latency budget are unchanged. If parity fails, stop/report; fallback is building tag `n9.0.2` from source in a Docker builder stage, not changing tolerances.

Prepare the portable bundle on the host with the setup command, which verifies all historical artifacts. The Docker builder independently checks that bundle's selected model, backbone, preprocessing and shared source identities. It does not need to receive neural research checkpoints merely to reconstruct a bundle already verified at setup.

```bash
git lfs pull --include="results/phase3/refit/resnet50/ridge/seedNone/model.npz"
.venv-serving/bin/python -m scripts.12_prepare_serving --config configs/serving.yaml --output artifacts/serving
docker build --no-cache -t vqa:phase4a .
docker run --rm --cpus=4 --memory=4g --pids-limit=128 --read-only --tmpfs /tmp:rw,nosuid,size=256m -p 127.0.0.1:8000:8000 vqa:phase4a
docker run --rm --network none --cpus=4 --memory=4g --pids-limit=128 --read-only --tmpfs /tmp:rw,nosuid,size=256m --entrypoint python --mount type=bind,src="$PWD/sample.mp4",dst=/input/video.mp4,readonly vqa:phase4a -m src.predict /input/video.mp4
```

The Dockerfile pins CPython 3.11.15 slim by digest, verifies the static decoder archive before extraction, installs only the hash-pinned CPU runtime lock, and runs as UID/GID 10001 with one Uvicorn worker. The context allowlist includes only runtime source, lock and portable artifact files; no research caches, videos, archives, neural/DINO checkpoints, labels or environments. No runtime downloads are needed. The host's installed Docker client uses its legacy builder; Buildx is not required by these commands.

For a private host video, add `--user "$(id -u):$(id -g)"` to the Docker CLI command. Evidence CLI runs use that non-root UID/GID so mode-0600 source files remain private and exported arrays are host-readable. The image's public application/model bundle is readable by unprivileged users; the API still runs as UID 10001. No source file is chmodded or overwritten by evidence tooling.

Evidence commands, in order after a verified image is running:

```bash
.venv-serving/bin/python -m scripts.14_serving_check --mode reference --image vqa:phase4a --url http://127.0.0.1:8000
.venv-serving/bin/python -m scripts.14_serving_check --mode parity --image vqa:phase4a --url http://127.0.0.1:8000
.venv-serving/bin/python -m scripts.14_serving_check --mode latency --image vqa:phase4a --url http://127.0.0.1:8000
.venv-serving/bin/python -m scripts.14_serving_check --mode report --image vqa:phase4a --url http://127.0.0.1:8000
```

The tool uses only the original ordered fixed sample and pinned full-refit model; no sample or threshold override exists. Reference scores use cached fp16 rows pooled in float64, without MOS. Raw CLI fp32 feature exports stay ignored under `results/phase4a/features/`; response JSON, source/feature hashes, every-clip results, raw latency rows and generated summaries are saved beside them. `reference` must precede `parity`; `latency` revalidates passed parity; `report` reproduces saved comparisons/aggregates and verifies notes/tests/build bindings before writing the passed gate last. A failing report command leaves an explicit failed gate.

## Measurement contract

Require every aligned timestep cosine >= 0.999, each cache-reference/raw score difference <= 0.01 MOS, and CLI/API score difference <= 1e-6. Measure three warm requests per original fixed clip after one excluded warmup; keep cold startup/first request separate. The maximum normalized warm loopback request must be <= 120 seconds per ten seconds of video. Preserve individual failures and stop; no threshold override or clipping is allowed. CPU hardware, image/package/decoder identities and container caps accompany the raw measurements.

The checked build and interface evidence are in [build.json](../results/phase4a/build.json); complete paired decoder outputs are in [decoder_versions.json](../results/phase4a/decoder_versions.json) and the [comparison](../results/phase4a/decoder_comparison.md). Final metrics and gate are generated from saved outputs by report mode; these setup checks alone do not close Phase 4a.

The final gate also binds a complete source roster read from the actual image to the current checkout. An invalid report configuration invalidates any previous passed gate before parsing. Measurement-tool source snapshots are retained under `results/phase4a/provenance/` when later changes affect evidence validation only.

Known minor: an unusual shared-decoder process-wait timeout can currently surface as a generic API 500 or a CLI traceback. Cleanup still runs and no successful score is returned. Shared preprocessing remains byte-for-byte frozen; this serving error-normalization improvement is deferred.

CLI format refresh: the human summary performs one additional shared metadata-only probe for clip duration after inference; this display probe is excluded from inference timings and never runs in JSON/API mode. The frozen preprocessing files and model arithmetic are unchanged.
