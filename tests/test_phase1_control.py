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
    from src.tuning import control_policy,CONTROL_POLICY_PATH
    source=tmp_path/'inventory.csv';splits=tmp_path/'splits.csv'
    rows=[{'video_id':f'v{i:03}','mos':1+i/100,'status':'ok'} for i in range(100)]
    atomic_write(source,csv_bytes(rows,['video_id','mos','status']))
    freeze_splits(source,splits)
    roles=list(csv.DictReader(splits.open()))
    train=[r['video_id'] for r in roles if r['outer_fold']=='0' and r['role']=='train']
    val=[r['video_id'] for r in roles if r['outer_fold']=='0' and r['role']=='val']
    test=[r['video_id'] for r in roles if r['outer_fold']=='0' and r['role']=='test']
    lookup={r['video_id']:r['mos'] for r in rows}
    order=np.random.default_rng(0).permutation(len(train))
    permutation=[{'video_id':key,'label_source_id':train[int(j)]} for key,j in zip(train,order)]
    if mutation=='permutation':
        permutation=[]
    atomic_json(tmp_path/'control_permutation.json',permutation)
    permutation_sha=sha256(tmp_path/'control_permutation.json')
    val_order=np.random.default_rng(0).permutation(len(val))
    val_map=[{'video_id':key,'label_source_id':val[int(j)]} for key,j in zip(val,val_order)]
    if mutation=='validation_permutation':
        val_map[0]['label_source_id']=train[0]
    atomic_json(tmp_path/'control_validation_permutation_v2.json',val_map)
    val_sha=sha256(tmp_path/'control_validation_permutation_v2.json')
    policy=control_policy();policy_sha=sha256(CONTROL_POLICY_PATH)
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
    root=tmp_path/'control_initial_v2'
    for seed in range(5):
        directory=root/f'seed{seed}'
        atomic_write(directory/'best.pt',b'test checkpoint bytes')
        checksum=sha256(directory/'best.pt')
        run={'status':'complete','seed':seed,'config':config,
             'provenance':{**provenance,'head':'mean_pool','fold':0,'label_mode':'shuffled','run_kind':'control_initial','permutation_sha256':permutation_sha,'validation_permutation_sha256':val_sha,'control_policy_sha256':policy_sha},
             'identity':{'config_sha256':config_hash(config)},'checkpoint_sha256':checksum}
        atomic_json(directory/'run.json',run)
        atomic_json(directory/'predictions.json',[{'video_id':key,'mos':float(mos),'prediction':float(pred),
            'seed':seed,'label_mode':'shuffled','run_kind':'control_initial','checkpoint_sha256':checksum}
            for key,mos,pred in zip(test,labels,predictions)])
    bootstrap=bootstrap_srcc(labels,predictions)
    assert abs(bootstrap['srcc'])<.15 and bootstrap['ci95'][0]<0<bootstrap['ci95'][1]
    atomic_json(root/'diagnostic.json',{'status':'passed','seeds':list(range(5)),'provenance':provenance,
        'control_policy':policy,'control_policy_sha256':policy_sha,'validation_permutation_sha256':val_sha,
        'bootstrap':bootstrap,'config_sha256':config_hash(config),'permutation_sha256':permutation_sha,
        'video_ids':test,'seed_averaged_predictions':predictions.tolist()})
    return root,provenance


def test_valid_control_uses_true_roles_mos_and_fixed_permutation(tmp_path):
    from src.tuning import require_control
    root,provenance=control_fixture(tmp_path)
    assert require_control(root,provenance)['status']=='passed'


@pytest.mark.parametrize('mutation',['role','mos','missing','permutation','validation_permutation'])
def test_self_consistent_wrong_control_inputs_fail(tmp_path,mutation):
    from src.tuning import require_control
    root,provenance=control_fixture(tmp_path,mutation)
    with pytest.raises(ValueError,match='control'):
        require_control(root,provenance)


def test_legacy_control_policy_cannot_unlock_cv(tmp_path):
    from src.cache import atomic_json
    from src.tuning import require_control
    root,provenance=control_fixture(tmp_path)
    path=root/'diagnostic.json'
    record=json.loads(path.read_text())
    record.pop('control_policy')
    atomic_json(path,record)
    with pytest.raises(ValueError,match='legacy|policy'):
        require_control(root,provenance)


def test_validation_permutation_identity_binds_each_run(tmp_path):
    from src.cache import atomic_json
    from src.tuning import require_control
    root,provenance=control_fixture(tmp_path)
    path=root/'seed2/run.json'
    record=json.loads(path.read_text())
    record['provenance']['validation_permutation_sha256']='wrong'
    atomic_json(path,record)
    with pytest.raises(ValueError,match='control'):
        require_control(root,provenance)
