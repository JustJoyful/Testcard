import importlib
import json
from pathlib import Path

import pytest

from phase2_fixtures import phase2_tree,make_cv


def test_resnet_collection_and_completed_grid_use_no_optimizer_or_role_loading(tmp_path,monkeypatch):
    from src.phase2 import load_phase2_context,validate_rows
    phase2_tree(tmp_path,monkeypatch)
    c=load_phase2_context(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    p=c['resnet_protocol'];configs=c['resnet_configs'];provenance=c['base_provenance']
    make_cv(p,configs,provenance)
    report=importlib.import_module('scripts.06_eval');training=importlib.import_module('scripts.05_train')
    rows,runs=report.collect_cv(p,provenance,configs,Path(p['results_root']))
    validate_rows(p,rows)
    assert len(runs)==100 and len(rows)==420
    events=[]
    monkeypatch.setattr(training,'role_data',lambda *a:pytest.fail('complete grid loaded role data'))
    monkeypatch.setattr(training,'fit_run',lambda *a,**kw:pytest.fail('complete grid called fit'))
    training.fit_cv_grid(p,provenance,configs,progress=events.append)
    assert events[-1]['cv_complete']==100 and events[-1]['ridge_complete']==5
    assert max(e.get('cv_complete',0) for e in events)==100
    bad=Path(p['results_root'])/'cv/mean_pool/fold0/seed0/best.pt';bad.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        training.fit_cv_grid(p,provenance,configs,progress=events.append)


def test_copied_or_missing_control_cannot_unlock_cv(tmp_path,monkeypatch):
    import src.phase2 as phase2
    phase2_tree(tmp_path,monkeypatch)
    c=phase2.load_phase2_context(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    p=c['resnet_protocol']
    for damage in ('missing','copied'):
        if damage=='copied':
            monkeypatch.setattr(phase2,'require_control',lambda *a:{'status':'passed','config_sha256':'DINO'})
        with pytest.raises((ValueError,OSError)):
            phase2.require_resnet_controls(c)


def test_phase2_cli_requires_both_controls_before_cv(tmp_path,monkeypatch):
    import src.phase2 as phase2
    cli=importlib.import_module('scripts.05_phase2')
    monkeypatch.setattr(cli,'load_phase2_context',lambda *a:{'policy':{'results_root':str(tmp_path)}})
    monkeypatch.setattr(cli,'require_resnet_controls',lambda *a:(_ for _ in ()).throw(ValueError('control failed')))
    with pytest.raises(ValueError,match='control failed'):
        cli.run_mode('cv',Path('unused'),Path('unused'))


def test_external_control_config_cannot_change_feature_or_output_root(tmp_path):
    training=importlib.import_module('scripts.05_train')
    p={'feature_set':'resnet50','results_root':str(tmp_path),'seeds':list(range(5))}
    with pytest.raises(ValueError,match='config'):
        training.control_runs(p,{},config={**p,'feature_set':'dino'})
