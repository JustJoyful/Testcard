import importlib
from pathlib import Path
import pytest
import yaml


def test_absent_hardware_parity_fails(tmp_path):
    check=importlib.import_module('scripts.04_parity_check').validate_parity
    with pytest.raises(ValueError,match='parity'):
        check(tmp_path/'parity.json',{},Path('cache/v1'))


def test_missing_phase_gates_do_not_pass(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.06_eval')
    training=importlib.import_module('scripts.05_train')
    def absent(*a,**kw):
        raise ValueError('incomplete cache')
    monkeypatch.setattr(training,'prerequisites',absent)
    p=yaml.safe_load(Path('configs/protocol.yaml').read_text())
    p['results_root']=str(tmp_path)
    result=script.phase_report(p,tmp_path,Path('configs/features_v1.yaml'))
    assert result['status']=='failed'
    names={r['gate'] for r in result['missing_or_failed_gates']}
    assert {'cache_and_committed_splits','initial_control','final_control','feature_parity','runtime_pilot'}<=names
