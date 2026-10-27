import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]


def run_queue(tmp_path,stage=None,tmux=True):
    root=tmp_path/'checkout with spaces'
    (root/'scripts').mkdir(parents=True);(root/'.venv/bin').mkdir(parents=True)
    shutil.copy(ROOT/'scripts/continue_phase2.sh',root/'scripts/continue_phase2.sh')
    python=root/'.venv/bin/python'
    python.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> calls.log
if [[ -n "${FAIL_STAGE:-}" && "$*" == *"$FAIL_STAGE"* ]]; then exit 23; fi
''')
    python.chmod(0o755)
    env={**os.environ,'FAIL_STAGE':stage or ''}
    env.pop('TMUX',None)
    if tmux:env['TMUX']='fixture'
    result=subprocess.run(['bash',str(root/'scripts/continue_phase2.sh')],cwd=tmp_path,env=env,capture_output=True,text=True)
    calls=(root/'calls.log').read_text().splitlines() if (root/'calls.log').exists() else []
    return result,root,calls


def test_runner_requires_tmux_before_any_python(tmp_path):
    result,root,calls=run_queue(tmp_path,tmux=False)
    assert result.returncode!=0 and not calls
    assert 'tmux' in result.stderr


def test_runner_stage_order_and_declared_flags_with_spaces(tmp_path):
    result,root,calls=run_queue(tmp_path)
    assert result.returncode==0,result.stderr
    assert len(calls)==7
    assert [c.split('--mode ')[1].split()[0] for c in calls[:5]]==['check','control-initial','control','cv','select']
    assert '07_bootstrap.py' in calls[5] and '13_make_tables.py' in calls[6]
    assert all(c.startswith('-u scripts/') and '--config configs/phase2.yaml --features configs/features_v2.yaml' in c for c in calls)
    assert (root/'results/phase2/session.log').exists()


@pytest.mark.parametrize('stage,count',[('control-initial',2),('--mode control ',3),('--mode cv',4),('07_bootstrap.py',6)])
def test_runner_failure_stops_before_next_stage_and_invalidates_gate(tmp_path,stage,count):
    result,root,calls=run_queue(tmp_path,stage=stage)
    assert result.returncode==23 and len(calls)==count
    gate=json.loads((root/'results/phase2/phase_gate.json').read_text())
    assert gate['status']=='failed'
    assert 'stage' in gate and '23' in result.stdout
