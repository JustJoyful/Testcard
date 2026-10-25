import importlib
import pytest


def test_cv_complete_coverage_and_no_duplicates():
    validate=importlib.import_module('scripts.06_eval').validate_oof
    folds={'a':0,'b':1}
    rows=[{'video_id':key,'fold':fold,'head':'mean_pool','seed':seed,'label_mode':'real','run_kind':'cv','prediction':2.,'mos':3.}
          for key,fold in folds.items() for seed in [0,1]]
    assert validate(rows,folds,['mean_pool'],[0,1])['status']=='complete'
    for bad in [[],rows[:-1],rows+[rows[0]], [{**r,'fold':1-r['fold']} for r in rows]]:
        assert validate(bad,folds,['mean_pool'],[0,1])['status']=='failed'


def test_control_or_pilot_rows_cannot_enter_headline():
    validate=importlib.import_module('scripts.06_eval').validate_oof
    for kind,mode in [('pilot','real'),('cv','shuffled'),('control_final','shuffled')]:
        row={'video_id':'a','fold':0,'head':'mean_pool','seed':0,'label_mode':mode,'run_kind':kind,'prediction':2.,'mos':3.}
        assert validate([row],{'a':0},['mean_pool'],[0])['status']=='failed'


def test_failed_control_blocks_real_cv(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.05_train')
    monkeypatch.setattr(script,'selected_configs',lambda *a:{})
    with pytest.raises(ValueError,match='control'):
        script.cv_runs({'results_root':str(tmp_path)}, {})


def test_reusing_complete_dino_cv_never_calls_fit(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.05_train')
    import torch
    from src.training import fit_one
    from src.heads import build_head
    import yaml
    from pathlib import Path
    config=yaml.safe_load(Path('configs/protocol.yaml').read_text())
    config.update(device='cpu',max_epochs=2,patience=2,cpu_threads=2)
    config=script.resolved_config(config)
    data=[{'video_id':str(i),'features':torch.randn(2,1536),'mos':float(i)} for i in range(8)]
    provenance={'head':'mean_pool','fold':0,'run_kind':'cv','label_mode':'real'}
    fit_one(build_head('mean_pool',1536,config),data,data,config,tmp_path,0,provenance)
    def forbidden(*a,**kw):
        raise AssertionError('complete run called fit')
    monkeypatch.setattr(script,'fit_one',forbidden)
    _,result=script.fit_run(config,config,{},'mean_pool',0,0,tmp_path,data,data,run_kind='cv')
    assert result['status']=='complete'
