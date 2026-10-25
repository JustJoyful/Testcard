import csv
import json
from pathlib import Path

import numpy as np
import pytest


def control_fixture(tmp_path,mutation=None):
    from src.cache import atomic_json,atomic_write,csv_bytes
    from src.features import sha256
    from src.metrics import bootstrap_srcc
    from src.splits import freeze_splits
    from src.training import config_hash
    source=tmp_path/'inventory.csv';splits=tmp_path/'splits.csv'
    rows=[{'video_id':f'v{i:03}','mos':1+i/100,'status':'ok'} for i in range(100)]
    atomic_write(source,csv_bytes(rows,['video_id','mos','status']))
    freeze_splits(source,splits)
    roles=list(csv.DictReader(splits.open()))
    train=[r['video_id'] for r in roles if r['outer_fold']=='0' and r['role']=='train']
    test=[r['video_id'] for r in roles if r['outer_fold']=='0' and r['role']=='test']
    lookup={r['video_id']:r['mos'] for r in rows}
    order=np.random.default_rng(0).permutation(len(train))
    permutation=[{'video_id':key,'label_source_id':train[int(j)]} for key,j in zip(train,order)]
    if mutation=='permutation':
        permutation=[]
    atomic_json(tmp_path/'control_permutation.json',permutation)
    permutation_sha=sha256(tmp_path/'control_permutation.json')
    config={'inventory':str(source),'splits':str(splits)}
    provenance={'split_sha256':sha256(splits)}
    labels=np.array([lookup[key] for key in test])
    predictions=np.roll(np.arange(len(test),dtype=float),4)
    if mutation=='role':
        test[0]=train[0]
    if mutation=='mos':
        labels[0]+=.0001
    if mutation=='missing':
        test,labels=test[1:],labels[1:]
        predictions=np.roll(np.arange(len(test),dtype=float),4)
    root=tmp_path/'control_initial'
    for seed in range(5):
        directory=root/f'seed{seed}'
        atomic_write(directory/'best.pt',b'test checkpoint bytes')
        checksum=sha256(directory/'best.pt')
        run={'status':'complete','seed':seed,'config':config,
             'provenance':{**provenance,'head':'mean_pool','fold':0,'label_mode':'shuffled','run_kind':'control_initial','permutation_sha256':permutation_sha},
             'identity':{'config_sha256':config_hash(config)},'checkpoint_sha256':checksum}
        atomic_json(directory/'run.json',run)
        atomic_json(directory/'predictions.json',[{'video_id':key,'mos':float(mos),'prediction':float(pred),
            'seed':seed,'label_mode':'shuffled','run_kind':'control_initial','checkpoint_sha256':checksum}
            for key,mos,pred in zip(test,labels,predictions)])
    bootstrap=bootstrap_srcc(labels,predictions)
    assert abs(bootstrap['srcc'])<.15 and bootstrap['ci95'][0]<0<bootstrap['ci95'][1]
    atomic_json(root/'diagnostic.json',{'status':'passed','seeds':list(range(5)),'provenance':provenance,
        'bootstrap':bootstrap,'config_sha256':config_hash(config),'permutation_sha256':permutation_sha,
        'video_ids':test,'seed_averaged_predictions':predictions.tolist()})
    return root,provenance


def test_valid_control_uses_true_roles_mos_and_fixed_permutation(tmp_path):
    from src.tuning import require_control
    root,provenance=control_fixture(tmp_path)
    assert require_control(root,provenance)['status']=='passed'


@pytest.mark.parametrize('mutation',['role','mos','missing','permutation'])
def test_self_consistent_wrong_control_inputs_fail(tmp_path,mutation):
    from src.tuning import require_control
    root,provenance=control_fixture(tmp_path,mutation)
    with pytest.raises(ValueError,match='control'):
        require_control(root,provenance)
