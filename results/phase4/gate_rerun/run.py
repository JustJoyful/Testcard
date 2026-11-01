"""Rerun the serving gate checks (scripts 12, 14, 15) and save every output with the commit hash.

    .venv-serving/bin/python -m results.phase4.gate_rerun.run

Scripts 12 and 14 run through their verification functions only: their normal modes rewrite the shipped
bundle (artifacts/serving) and Phase 4a's evidence/gate, which the plan forbids for changed source.
Script 15 runs all four evidence modes against the Phase 4b container on 127.0.0.1:8001.
"""
import filecmp
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.cache import atomic_json, atomic_write
from src.serving_bundle import read_settings

OUT = Path('results/phase4/gate_rerun')
URL, IMAGE = 'http://127.0.0.1:8001', 'vqa:phase4b'


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, check=True).stdout.strip()


def script_12():
    prepare = importlib.import_module('scripts.12_prepare_serving')
    with tempfile.TemporaryDirectory() as tmp:
        prepare.prepare_bundle(Path('configs/serving.yaml'), Path(tmp))
        files = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob('*') if p.is_file())
        identical = all(filecmp.cmp(Path(tmp) / f, Path('artifacts/serving') / f, shallow=False) for f in files)
    if not identical:
        raise ValueError('freshly prepared bundle differs from artifacts/serving')
    return dict(status='passed', files=files, identical_to_artifacts_serving=identical)


def script_14():
    module = importlib.import_module('scripts.14_serving_check')
    context = module.research_inputs(read_settings(Path('configs/serving.yaml')))
    references = module.reference_for(context)
    return dict(status='passed', reference_rows=len(references), bindings=context['bindings'],
                **module.qualified_provenance())


def main():
    record = dict(commit=git('rev-parse', 'HEAD'),
                  dirty_tracked_files=git('status', '--short', '--untracked-files=no').splitlines(), steps={})
    ok = True
    for name, step in (('script_12_prepare_serving', script_12), ('script_14_serving_check', script_14)):
        try:
            record['steps'][name] = step()
        except Exception as exc:  # recorded, then stop: nothing later runs on a failed prerequisite
            record['steps'][name] = dict(status='failed', error=str(exc))
            ok = False
            break
    if ok:
        for mode in ('parity', 'latency', 'tests', 'report'):
            result = subprocess.run([sys.executable, '-m', 'scripts.15_demo_check', '--mode', mode, '--image', IMAGE,
                                     '--url', URL], capture_output=True, text=True)
            log = OUT / f'script_15_{mode}.log'
            atomic_write(log, (result.stdout + result.stderr).encode())
            record['steps'][f'script_15_{mode}'] = dict(status='passed' if result.returncode == 0 else 'failed',
                                                       exit_code=result.returncode, log=str(log),
                                                       last_line=(result.stdout.strip().splitlines() or [''])[-1])
            if result.returncode:
                ok = False
                break
    gate = json.loads(Path('results/phase4b/phase_gate.json').read_text())
    record.update(status='passed' if ok else 'failed', phase4b_gate_status=gate['status'],
                  phase4b_image_id=gate.get('image_id'))
    atomic_json(OUT / 'summary.json', record)
    print(json.dumps({k: v['status'] for k, v in record['steps'].items()}
                     | {'overall': record['status'], 'commit': record['commit']}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
