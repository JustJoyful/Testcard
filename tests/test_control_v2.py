import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def data(prefix,n):
    return [{'video_id':f'{prefix}{i}','features':torch.tensor([[float(i)]]),'mos':float(i)} for i in range(n)]


def test_role_permutations_are_independent_and_keep_features_attached():
    from src.tuning import permute_control_role
    train,val=data('train',8),data('val',6)
    shuffled_train,train_map=permute_control_role(train,'train')
    shuffled_val,val_map=permute_control_role(val,'val')
    assert [r['mos'] for r in shuffled_train]==[2.,4.,3.,6.,5.,0.,1.,7.]
    assert [r['mos'] for r in shuffled_val]==[3.,2.,5.,4.,0.,1.]
    assert [r['video_id'] for r in shuffled_train]==[r['video_id'] for r in train]
    for original,changed in zip(val,shuffled_val):
        torch.testing.assert_close(original['features'],changed['features'])
    assert {r['label_source_id'] for r in train_map}=={r['video_id'] for r in train}
    assert {r['label_source_id'] for r in val_map}=={r['video_id'] for r in val}
    assert [r['mos'] for r in val]==list(map(float,range(6)))
    with pytest.raises(ValueError,match='role'):
        permute_control_role(data('test',5),'test')


def test_control_fits_shuffled_validation_and_reports_true_test_mos(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.05_train')
    train,val,test=data('train',8),data('val',6),data('test',5)
    monkeypatch.setattr(script,'role_data',lambda p,f,r:{'train':train,'val':val,'test':test}[r])
    monkeypatch.setattr(script,'resolved_config',lambda p:p)
    observed=[]
    def fit(p,c,provenance,head,fold,seed,output,fit_rows,val_rows,*a,**kw):
        observed.append((copy.deepcopy(fit_rows),copy.deepcopy(val_rows),provenance))
        return None,{'target_stats':{'mean':0.,'std':1.},'checkpoint_sha256':'checked'}
    monkeypatch.setattr(script,'fit_run',fit)
    monkeypatch.setattr(script,'predict',lambda m,d,s,device:[{'video_id':r['video_id'],'prediction':float(i)} for i,r in enumerate(d)])
    import src.metrics
    monkeypatch.setattr(src.metrics,'bootstrap_srcc',lambda *a,**kw:{'status':'ok','srcc':0.,'ci95':[-.1,.1]})
    result=script.control_runs({'results_root':str(tmp_path),'seeds':list(range(5)),'device':'cpu'},{})
    assert len(observed)==5
    assert all([r['mos'] for r in x[1]]==[3.,2.,5.,4.,0.,1.] for x in observed)
    assert all(x[2]['validation_permutation_sha256'] for x in observed)
    assert all(x[2]['control_policy_sha256'] for x in observed)
    assert (tmp_path/'control_initial_v2/diagnostic.json').exists()
    rows=json.loads((tmp_path/'control_initial_v2/seed0/predictions.json').read_text())
    assert [r['mos'] for r in rows]==[0.,1.,2.,3.,4.]
    assert result['control_policy']['schema_version']==2


def test_frozen_protocol_does_not_reproject_completed_work(tmp_path,monkeypatch):
    script=importlib.import_module('scripts.05_train')
    import src.tuning
    path=tmp_path/'runtime_budget.json';path.write_text('{"status":"go"}\n')
    before=path.read_bytes()
    monkeypatch.setattr(script,'require_selection_gates',lambda *a,**kw:None)
    monkeypatch.setattr(src.tuning,'runtime_budget',lambda *a,**kw:pytest.fail('reprojected already completed selection work'))
    protocol={'status':'frozen','results_root':str(tmp_path)}
    assert script.freeze_protocol(protocol,tmp_path/'protocol.yaml',{}, {})==protocol
    assert path.read_bytes()==before
