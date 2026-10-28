"""Exercise the real shell sequence with computation/Git replaced at their boundaries."""
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_resume(tmp_path, fail=False):
    log = tmp_path / 'calls.log'
    env = {**os.environ, 'PHASE1_TEST_LOG': str(log), 'PHASE1_TEST_FAIL': '1' if fail else '0'}
    shell = r'''
function .venv/bin/python() {
    printf '%s\n' "$*" >> "$PHASE1_TEST_LOG"
    if [[ "$PHASE1_TEST_FAIL" == 1 && "$*" == *'--dataset livevqc --backbone dino'* ]]; then
        return 23
    fi
}
function git() { return 0; }
source scripts/continue_phase1.sh
'''
    result = subprocess.run(['bash', '-c', shell], cwd=ROOT, env=env, capture_output=True, text=True)
    return result, log.read_text().splitlines() if log.exists() else []


def test_resume_extracts_before_control_and_preserves_stage_order(tmp_path):
    result, calls = run_resume(tmp_path)
    assert result.returncode == 0, result.stderr
    extraction = [line for line in calls if '02_extract_spatial.py' in line]
    assert len(extraction) == 4
    assert [' '.join(line.split('--dataset ')[1].split()[:3]) for line in extraction] == [
        'konvid --backbone dino', 'konvid --backbone resnet50',
        'livevqc --backbone dino', 'livevqc --backbone resnet50',
    ]
    assert [' '.join(line.split('--mode ')[1].split()[:1]) for line in calls if '--mode ' in line] == [
        'control-initial', 'pilot', 'rank', 'tune', 'control', 'cv',
    ]
    assert calls.index(extraction[-1]) < next(i for i, line in enumerate(calls) if '--mode ' in line)
    assert any('-m pytest -q' == line for line in calls)
    assert calls[-1].startswith('scripts/06_eval.py ')


def test_extraction_failure_stops_resume_before_training(tmp_path):
    result, calls = run_resume(tmp_path, fail=True)
    assert result.returncode == 23
    assert sum('02_extract_spatial.py' in line for line in calls) == 3
    assert not any('--mode ' in line for line in calls)


def test_resume_never_mutates_git(tmp_path):
    log=tmp_path/'git.log'
    env={**os.environ,'PHASE1_GIT_LOG':str(log)}
    shell=r'''
function .venv/bin/python() { return 0; }
function git() { printf '%s\n' "$*" >> "$PHASE1_GIT_LOG"; return 0; }
source scripts/continue_phase1.sh
'''
    result=subprocess.run(['bash','-c',shell],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    calls=log.read_text().splitlines() if log.exists() else []
    assert not any(c.split()[0] in ('add','commit','push') for c in calls)
