"""Alias-tag the gated Phase 4b image as testcard:<version> (no rebuild) and record the digest.

    .venv-serving/bin/python -m results.phase4.tag_release 0.1.0

Refuses unless the Phase 4b gate and the latest gate rerun passed for exactly the image vqa:phase4b points at.
Writes results/phase4/testcard_tag.json.
"""
import json
import subprocess
import sys
from pathlib import Path

from src.cache import atomic_json


def image_id(tag):
    return subprocess.run(['docker', 'image', 'inspect', tag, '--format', '{{.Id}}'], capture_output=True, text=True, check=True).stdout.strip()


def main(version):
    gate = json.loads(Path('results/phase4b/phase_gate.json').read_text())
    rerun = json.loads(Path('results/phase4/gate_rerun/summary.json').read_text())
    image = image_id('vqa:phase4b')
    if gate['status'] != 'passed' or rerun['status'] != 'passed' or gate['image_id'] != image or rerun['phase4b_image_id'] != image:
        raise SystemExit('refusing to tag: the gate did not pass for the current vqa:phase4b image')
    tag = f'testcard:{version}'
    subprocess.run(['docker', 'tag', image, tag], check=True)
    if image_id(tag) != image:
        raise SystemExit('alias tag does not point at the gated image')
    record = dict(tag=tag, image_id=image, source_tag='vqa:phase4b', gate_commit=rerun['commit'],
                  model_name='testcard-r50-ridge', deployed_candidate_id='resnet50/ridge', note='alias of the gated image; no rebuild')
    atomic_json(Path('results/phase4/testcard_tag.json'), record)
    print(json.dumps(record))


if __name__ == '__main__':
    main(sys.argv[1])
