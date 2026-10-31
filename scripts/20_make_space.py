"""Stage the Hugging Face Docker Space folder hf_release/testcard-space/ (Dockerfile + README.md); uploads nothing.

    .venv-serving/bin/python -m scripts.20_make_space            # stage; Dockerfile keeps a placeholder until a digest is recorded
    .venv-serving/bin/python -m scripts.20_make_space --record   # after `docker push`: record image ID and manifest digest, restage

The Space pins the registry manifest digest (not the local image ID) recorded in results/phase4/hf_space.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.cache import atomic_json

OUT = Path('hf_release/testcard-space')
RECORD = Path('results/phase4/hf_space.json')
REGISTRY = 'ghcr.io/justjoyful/testcard'
TAG = '0.1.0'


def inspect(ref, field):
    return subprocess.run(['docker', 'image', 'inspect', ref, '--format', field], capture_output=True, text=True, check=True).stdout.strip()


def record():
    gated = json.loads(Path('results/phase4/testcard_tag.json').read_text())
    image_id = inspect(f'{REGISTRY}:{TAG}', '{{.Id}}')
    if image_id != gated['image_id']:
        raise SystemExit('the pushed image is not the gated testcard:0.1.0 image')
    digests = [d for d in json.loads(inspect(f'{REGISTRY}:{TAG}', '{{json .RepoDigests}}')) if d.startswith(REGISTRY + '@')]
    if len(digests) != 1:
        raise SystemExit(f'expected one {REGISTRY} manifest digest after push, found {digests}')
    atomic_json(RECORD, dict(image=f'{REGISTRY}:{TAG}', local_image_id=image_id, manifest_digest=digests[0].split('@', 1)[1],
                             pinned_reference=digests[0], gated_tag=gated['tag'], gate_commit=gated['gate_commit'],
                             note='The Space pins the registry manifest digest; the local image ID is the config digest of the same gated image.'))


def main():
    if sys.argv[1:] == ['--record']:
        record()
    pinned = json.loads(RECORD.read_text())['pinned_reference'] if RECORD.is_file() else f'{REGISTRY}@sha256:<manifest digest>'
    limit_mib = yaml.safe_load(Path('configs/serving.yaml').read_text())['limits']['file_bytes'] // 2**20
    readme = f"""---
title: Testcard
sdk: docker
app_port: 8000
license: cc-by-nc-4.0
short_description: No-reference video quality score for a short clip
models:
- JustJoyful/testcard-r50-ridge
---

# Testcard

Upload a short video and Testcard predicts its perceived quality as a mean opinion score on the KoNViD-1k 1–5 scale. The page shows the score on a TV screen, with the analysis time, the verified model, weights and preprocessing hashes, and the full JSON result.

- **Your video:** one upload is processed at a time, up to {limit_mib} MB. The video is deleted after scoring.
- **The score is an estimate.** It is a model prediction from a ridge regression on ResNet-50 features, not a ground-truth rating, and it is unclamped, so it can fall slightly outside 1–5.
- **Speed:** this Space runs on 2 shared vCPUs, so it is slower than the CPU latency table in the GitHub README, which was measured with 4 CPUs.
- **First visit:** the Space may be asleep. The first request then waits for it to start.

Links:
- Code, method and results: [GitHub](https://github.com/JustJoyful/Testcard)
- Model checkpoint and card: [JustJoyful/testcard-r50-ridge](https://huggingface.co/JustJoyful/testcard-r50-ridge)

This Space runs the gated image unchanged, pinned by registry digest: `{pinned}`.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    extra = {p.name for p in OUT.iterdir()} - {'Dockerfile', 'README.md'}
    if extra:
        raise SystemExit(f'unexpected files in {OUT}: {sorted(extra)}')
    (OUT / 'Dockerfile').write_text(f'FROM {pinned}\n')
    (OUT / 'README.md').write_text(readme)
    print(f'staged {OUT} with FROM {pinned}')


if __name__ == '__main__':
    main()
