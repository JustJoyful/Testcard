import copy
import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from phase2_fixtures import HEADS,make_cv,phase2_tree
from src.cache import atomic_json,atomic_write
from src.features import sha256
from src.training import config_hash


def report_tree(tmp_path,monkeypatch):
    import src.phase2 as phase2
    phase2_tree(tmp_path,monkeypatch)
    c=phase2.load_phase2_context(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    make_cv(c['resnet_protocol'],c['resnet_configs'],c['base_provenance'])
    monkeypatch.setattr(phase2,'require_resnet_controls',lambda *a:{'initial':{'status':'passed'},'final':{'status':'passed'}})
    report1=importlib.import_module('scripts.06_eval')
    monkeypatch.setattr(report1,'control_history',lambda *a:{'disclosure':'Original training-only control failed; post-hoc amendment.', 'post_hoc_amendment':True})
    atomic_json('results/phase1/time_accounting.json',{'gpu_job_wall_hours':1.,'remaining_gpu_allowance_hours':99.,'cpu_ridge_wall_hours':.1,'cpu_parity_wall_hours':.1})
    Path('configs/scope.yaml').write_text('serving:\n  cpu_budget_seconds_per_10s: 120\n')
    Path('DECISIONS.md').write_text('# Decisions\n')
    selection=phase2.save_selection(c)
    outputs,sources=phase2.collect_feature_outputs(c)
    ids,y,values=phase2.aligned_ensembles(c,outputs)
    bootstrap=importlib.import_module('scripts.07_bootstrap')
    from src.metrics import make_bootstrap_indices,bootstrap_indices_hash
    bank=make_bootstrap_indices(len(ids))
    bank_hash=bootstrap_indices_hash(bank)
    # All fixture predictions have the same ranking: every paired difference is exactly zero.
    records=[]
    for pair in bootstrap.comparison_roster(selection['selected_feature_set']):
        metrics={m:{'status':'ok','reason':None,'metric':m,'delta':0.,'ci95':[0.,0.],'p_raw':1.,
                    'valid_draws':10000,'invalid_draws':0,'draws':10000,'seed':0,
                    'pvalue_method':'centered_bootstrap_add_one','resample_indices_sha256':bank_hash} for m in ('srcc','krcc')}
        metrics['srcc']['classification']='equivalent within margin'
        if pair['family']=='primary_srcc':metrics['srcc'].update(p_holm=1.,holm_reject=False)
        records.append({**pair,'metrics':metrics})
    atomic_json('results/phase2/comparisons.json',records)
    atomic_write('results/phase2/comparisons.csv',bootstrap.comparison_csv(records))
    atomic_json('results/phase2/comparisons.meta.json',{'source_identity':c['source_identity'],'cv_sources_sha256':sources,
        'selection_sha256':sha256('results/phase2/selection.json'),'video_ids_sha256':config_hash(ids),'draws':10000,'seed':0,
        'pvalue_method':'centered_bootstrap_add_one','resample_indices_sha256':bank_hash,'policy':c['policy']['statistics'],
        'json_sha256':sha256('results/phase2/comparisons.json'),'csv_sha256':sha256('results/phase2/comparisons.csv')})
    return c,outputs


def test_complete_tables_and_freeze_use_all_sources_without_choosing_unmeasured_deployment(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.13_make_tables')
    c,outputs=report_tree(tmp_path,monkeypatch)
    before={p:p.read_bytes() for p in Path('results/phase1').rglob('*') if p.is_file()}
    gate=script.phase2_report(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    assert gate['status']=='passed'
    assert all(p.read_bytes()==v for p,v in before.items())
    rows=list(csv.DictReader(Path('results/phase2/headline.csv').open()))
    assert len(rows)==10 and {r['feature_set'] for r in rows}=={'dino','resnet50'}
    assert all(int(r['run_count'])==(5 if r['head']=='ridge' else 25) for r in rows)
    heads=list(csv.DictReader(Path('results/phase2/head_comparison.csv').open()))
    assert len(heads)==10 and sum(bool(r['srcc_p_holm']) for r in heads)==6
    freeze=json.loads(Path('results/phase2/konvid_freeze.json').read_text())
    assert freeze['deployment']['winner'] is None
    assert freeze['deployment']['cpu_budget_seconds_per_10s']==120
    assert len(freeze['deployment']['candidates'])==10
    for head in HEADS:
        policy=freeze['refit'][head]
        assert policy['median_best_epoch']==13 and len(policy['cv_epochs'])==25
    assert freeze['refit']['ridge']['phase3_alpha']==1.
    assert len(freeze['refit']['ridge']['cv_selected_alphas'])==5
    assert sha256('results/phase2/konvid_freeze.json')==gate['outputs_sha256']['konvid_freeze.json']
    assert 'Phase 2 exit verified' in Path('DECISIONS.md').read_text()


@pytest.mark.parametrize('damage',['missing_pair','duplicate_pair','krcc','ridge_pair','draws','bank','p_holm','ci_class','delta','csv','metadata','epoch','ridge_alpha','selected','controls','write'])
def test_failed_regeneration_replaces_stale_pass(tmp_path,monkeypatch,damage):
    import src.phase2 as phase2
    script=importlib.import_module('scripts.13_make_tables')
    c,outputs=report_tree(tmp_path,monkeypatch)
    atomic_json('results/phase2/phase_gate.json',{'status':'passed'})
    path=Path('results/phase2/comparisons.json');records=json.loads(path.read_text())
    if damage=='missing_pair':records.pop(0)
    elif damage=='duplicate_pair':records[-1]=copy.deepcopy(records[0])
    elif damage=='krcc':records[0]['metrics'].pop('krcc')
    elif damage=='ridge_pair':records=[r for r in records if r['pair_id']!='ridge/mamba']
    elif damage=='draws':records[0]['metrics']['srcc']['valid_draws']=1
    elif damage=='bank':records[0]['metrics']['krcc']['resample_indices_sha256']='other'
    elif damage=='p_holm':records[0]['metrics']['srcc']['p_holm']=.001
    elif damage=='ci_class':records[0]['metrics']['srcc']['classification']='A better than B (practically meaningful)'
    elif damage=='delta':records[0]['metrics']['srcc']['delta']=.9
    elif damage=='csv':Path('results/phase2/comparisons.csv').write_text('edited')
    elif damage=='metadata':
        m=Path('results/phase2/comparisons.meta.json');meta=json.loads(m.read_text());meta['video_ids_sha256']='changed';atomic_json(m,meta)
    elif damage=='epoch':
        p=Path('results/phase2/resnet50/cv/mean_pool/fold0/seed0/run.json');r=json.loads(p.read_text());r['best_epoch']=0;atomic_json(p,r)
    elif damage=='ridge_alpha':
        p=Path('results/phase2/resnet50/ridge/fold0/run.json');r=json.loads(p.read_text());r['alpha']=7;atomic_json(p,r)
    elif damage=='selected':
        p=Path('results/phase2/selection.json');r=json.loads(p.read_text());r['selected_feature_set']='resnet50';atomic_json(p,r)
    elif damage=='controls':
        monkeypatch.setattr(phase2,'require_resnet_controls',lambda *a:(_ for _ in ()).throw(ValueError('control failed')))
    elif damage=='write':
        original=script.atomic_write
        def write(p,data):
            if Path(p).name=='headline.csv':raise OSError('unavailable')
            return original(p,data)
        monkeypatch.setattr(script,'atomic_write',write)
    if damage in ('missing_pair','duplicate_pair','krcc','ridge_pair','draws','bank','p_holm','ci_class','delta'):
        atomic_json(path,records)
        m=Path('results/phase2/comparisons.meta.json');meta=json.loads(m.read_text());meta['json_sha256']=sha256(path);atomic_json(m,meta)
    result=script.phase2_report(Path('configs/phase2.yaml'),Path('configs/features_v2.yaml'))
    assert result['status']=='failed'
    assert json.loads(Path('results/phase2/phase_gate.json').read_text())['status']=='failed'


def test_freeze_rejects_edited_epoch_sources_and_ridge_policy(tmp_path,monkeypatch):
    from src.phase2 import build_konvid_freeze,save_selection
    c,outputs=report_tree(tmp_path,monkeypatch)
    for output in outputs.values():
        output['headlines']={h:{'srcc_mean':.5} for h in (*HEADS,'ridge')}
    selected=save_selection(c)
    for mutation in ('epoch','alpha','missing','score'):
        damaged=copy.deepcopy(outputs)
        if mutation=='epoch':damaged['dino']['neural_runs'][0]['best_epoch']=199
        elif mutation=='alpha':damaged['dino']['ridge_runs'][0]['alpha']=10
        elif mutation=='missing':damaged['dino']['neural_runs'].pop()
        else:damaged['dino']['headlines']['ridge']['srcc_mean']=float('nan')
        with pytest.raises(ValueError):build_konvid_freeze(c,selected,damaged)
