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


def report_fixture(tmp_path,monkeypatch):
    import json
    from src.cache import atomic_write,csv_bytes
    from src.training import config_hash
    script=importlib.import_module('scripts.06_eval')
    training=importlib.import_module('scripts.05_train')
    parity=importlib.import_module('scripts.04_parity_check')
    source=tmp_path/'inventory.csv';splits=tmp_path/'splits.csv'
    atomic_write(source,csv_bytes([{'video_id':'a','mos':2.},{'video_id':'b','mos':3.}],['video_id','mos']))
    atomic_write(splits,csv_bytes([{'video_id':'a','outer_fold':0,'role':'test'},{'video_id':'b','outer_fold':1,'role':'test'}],['video_id','outer_fold','role']))
    p=yaml.safe_load(Path('configs/protocol.yaml').read_text())
    p.update(inventory=str(source),splits=str(splits),results_root=str(tmp_path))
    configs={'mean_pool':{'rank_weight':.1}}
    monkeypatch.setattr(training,'prerequisites',lambda *a:{'split_sha256':'fixture'})
    monkeypatch.setattr(training,'selected_configs',lambda *a:configs)
    final={'status':'passed','config_sha256':config_hash(configs['mean_pool'])}
    monkeypatch.setattr(script,'require_control',lambda *a:final)
    monkeypatch.setattr(parity,'validate_parity',lambda *a:{'status':'passed'})
    monkeypatch.setattr(script,'validate_runtime',lambda *a:{'status':'go'})
    monkeypatch.setattr(script,'validate_parameter_budget',lambda *a:{'status':'passed'})
    rows=[{'video_id':key,'fold':fold,'head':head,'seed':seed,'label_mode':'real','run_kind':'cv','mos':mos,'prediction':mos}
          for key,fold,mos in [('a',0,2.),('b',1,3.)] for head in training.HEADS+['ridge'] for seed in ([None] if head=='ridge' else range(5))]
    monkeypatch.setattr(script,'collect_cv',lambda *a:(rows,[]))
    monkeypatch.setattr(script,'control_history',lambda *a:{'disclosure':'approved amendment','post_hoc_amendment':True})
    monkeypatch.setattr(script,'summarize',lambda *a:({},[]))
    monkeypatch.setattr(script,'time_accounting',lambda *a:{})
    monkeypatch.setattr(script,'update_phase_docs',lambda *a:None)
    return script,training,p,final


def test_final_control_must_match_selected_mean_pool(tmp_path,monkeypatch):
    script,training,p,final=report_fixture(tmp_path,monkeypatch)
    assert script.phase_report(p,tmp_path,Path('configs/features_v1.yaml'))['status']=='passed'
    final['config_sha256']='different selected configuration'
    result=script.phase_report(p,tmp_path,Path('configs/features_v1.yaml'))
    assert result['status']=='failed'
    assert any(r['gate']=='final_control' for r in result['missing_or_failed_gates'])


def test_dirty_split_replaces_stale_passing_gate(tmp_path,monkeypatch):
    import json
    import subprocess
    script,training,p,final=report_fixture(tmp_path,monkeypatch)
    assert script.phase_report(p,tmp_path,Path('configs/features_v1.yaml'))['status']=='passed'
    def dirty(*a):
        raise subprocess.CalledProcessError(1,['git','diff','--exit-code','HEAD','--',p['splits']])
    monkeypatch.setattr(training,'prerequisites',dirty)
    result=script.phase_report(p,tmp_path,Path('configs/features_v1.yaml'))
    assert result['status']=='failed'
    assert json.loads((tmp_path/'phase_gate.json').read_text())['status']=='failed'


def test_time_accounting_uses_saved_job_wall_times(tmp_path):
    from src.cache import atomic_json
    script=importlib.import_module('scripts.06_eval')
    atomic_json(tmp_path/'extraction.json',{'invocations':[{'wall_seconds':3600.}]})
    atomic_json(tmp_path/'cv/x/run.json',{'status':'complete','wall_seconds':1800.,'config':{'device':'cuda'}})
    atomic_json(tmp_path/'ridge/fold0/run.json',{'status':'complete','wall_seconds':60.})
    atomic_json(tmp_path/'parity.json',{'wall_seconds':120.})
    actual=script.time_accounting(tmp_path)
    assert actual['gpu_job_wall_hours']==1.5 and actual['remaining_gpu_allowance_hours']==98.5
    assert actual['cpu_ridge_wall_hours']==1/60 and actual['cpu_parity_wall_hours']==1/30


def test_control_history_preserves_failed_original_and_declares_amendment(tmp_path):
    from src.cache import atomic_json
    script=importlib.import_module('scripts.06_eval')
    original={'status':'failed','bootstrap':{'status':'ok','srcc':.129,'ci95':[.004,.255]}}
    atomic_json(tmp_path/'control_final/diagnostic.json',original)
    result=script.control_history(tmp_path)
    assert result['original_final_control']['status']=='failed'
    assert result['original_final_control']['ci95']==[.004,.255]
    assert result['active_policy']['schema_version']==2
    assert result['post_hoc_amendment'] is True


def test_phase_doc_update_preserves_later_amendments(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.06_eval')
    monkeypatch.chdir(tmp_path)
    (tmp_path/'DECISIONS.md').write_text('# Decisions\n')
    (tmp_path/'PHASES.md').write_text('# Phases\n\n## Phase 1 execution status (2026-10-25)\n\nPreviously blocked.\n\n### Approved amendment\n\nKeep original failed control visible.\n')
    script.update_phase_docs({'gpu_job_wall_hours':1.,'remaining_gpu_allowance_hours':99.})
    text=(tmp_path/'PHASES.md').read_text()
    assert 'passed the generated exit gate' in text
    assert 'Keep original failed control visible.' in text


def test_headline_table_uses_saved_summary_metrics_and_run_count(tmp_path):
    import csv
    script=importlib.import_module('scripts.06_eval')
    summary={'mamba':{'status':'complete','run_count':25,'srcc_mean':.71347,'srcc_std':.04234,
        'krcc_mean':.52345,'krcc_std':.0321,'plcc_raw_mean':.74567,'plcc_raw_std':.031,
        'plcc_mapped_mean':.76543,'plcc_mapped_std':.02567}}
    table=script.headline_table(tmp_path,summary)
    assert '| mamba | 25 | 0.7135 ± 0.0423 | 0.5234 ± 0.0321 | 0.7457 ± 0.0310 | 0.7654 ± 0.0257 |' in table
    row=list(csv.DictReader((tmp_path/'headline.csv').open()))[0]
    assert row['head']=='mamba' and int(row['run_count'])==25
    assert float(row['srcc_mean'])==.71347 and float(row['plcc_mapped_mean'])==.76543


def test_output_failure_cannot_leave_a_passing_phase_gate(tmp_path,monkeypatch):
    import json
    script,training,p,final=report_fixture(tmp_path,monkeypatch)
    assert script.phase_report(p,tmp_path,Path('configs/features_v2.yaml'))['status']=='passed'
    monkeypatch.setattr(script,'summarize',lambda *a:({'mamba':{'status':'complete'}},[]))
    def failed_output(*a):
        raise OSError('output unavailable')
    monkeypatch.setattr(script,'headline_table',failed_output)
    with pytest.raises(OSError):
        script.phase_report(p,tmp_path,Path('configs/features_v2.yaml'))
    assert json.loads((tmp_path/'phase_gate.json').read_text())['status']!='passed'


def test_empty_provenance_cannot_skip_missing_cv_check(tmp_path,monkeypatch):
    script,training,p,final=report_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(training,'prerequisites',lambda *a:{})
    def missing(*a):
        raise ValueError('missing CV artifacts')
    monkeypatch.setattr(script,'collect_cv',missing)
    result=script.phase_report(p,tmp_path,Path('configs/features_v2.yaml'))
    assert result['status']=='failed'
    assert any(r['gate']=='cv_artifacts' for r in result['missing_or_failed_gates'])
