"""One validation-selected fp32 fit, with immutable completed artifacts."""
import hashlib
import io
import json
import math
import os
import random
import time
import warnings
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.cache import atomic_json, atomic_write
from src.data import collate_features, collate_videos
from src.features import sha256
from src.losses import quality_loss
from src.metrics import correlations, fit_target_stats


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest()


def seed_everything(seed,threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


@torch.inference_mode()
def predict(model,dataset,target_stats,device):
    model.eval()
    rows=[]
    for batch in DataLoader(dataset,batch_size=64,shuffle=False,collate_fn=collate_features):
        values=model(batch['features'].to(device),batch['lengths']).cpu().numpy()
        values=values*target_stats['std']+target_stats['mean']
        if not np.isfinite(values).all():
            raise ValueError(f'nonfinite predictions: {batch["video_id"]}')
        rows.extend({'video_id':key,'prediction':float(value)} for key,value in zip(batch['video_id'],values,strict=True))
    return rows


def load_completed(model,output,identity,device):
    output=Path(output)
    path=output/'run.json'
    if not path.exists():
        return None
    record=json.loads(path.read_text())
    if record['status']!='complete':
        return None
    if record['identity']!=identity:
        raise ValueError(f'{output}: incompatible completed run')
    checkpoint=output/'best.pt'
    if not checkpoint.exists() or sha256(checkpoint)!=record['checkpoint_sha256']:
        raise ValueError(f'{output}: checkpoint checksum mismatch')
    saved=torch.load(checkpoint,map_location=device,weights_only=True)
    model.load_state_dict(saved['state_dict'])
    return record


def _train_epoch(model, loader, optimizer, config, stats, device, epoch, emit):
    warm=config['warmup_epochs']
    multiplier=epoch/max(1,warm) if epoch<=warm else .5*(1+math.cos(math.pi*(epoch-warm)/max(1,config['max_epochs']-warm)))
    learning_rate=config['learning_rate']*multiplier
    for group in optimizer.param_groups:
        group['lr']=learning_rate
    model.train()
    losses=[]
    for batch_index,batch in enumerate(loader,1):
        optimizer.zero_grad(set_to_none=True)
        target=(batch['mos'].to(device)-stats['mean'])/stats['std']
        value=model(batch['features'].to(device),batch['lengths'])
        loss=quality_loss(value,target,config['rank_weight'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        emit('batch_end',epoch=epoch,batch=batch_index,batches=len(loader),train_loss=float(np.mean(losses)))
    return float(np.mean(losses)),learning_rate


def fit_one(model,train_data,val_data,config,output,seed,provenance,*,progress=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    device=config['device']
    seed_everything(seed,config['cpu_threads'])
    model.to(device=device,dtype=torch.float32)
    identity={'config_sha256':config_hash(config),'seed':seed,'provenance':{k:v for k,v in provenance.items() if k!='code_commit'}}
    def emit(event,**values):
        if progress is not None:
            progress({'event':event,'run_path':str(output),'seed':seed,'head':provenance.get('head'),
                      'fold':provenance.get('fold'),'feature_set':config.get('feature_set'),
                      'device':device,'max_epochs':config['max_epochs'],**values})
    existing=load_completed(model,output,identity,device)
    if existing:
        emit('fit_reused',epoch=existing['best_epoch'])
        return existing
    emit('fit_start')
    resolved=output/'config.json'
    if resolved.exists() and json.loads(resolved.read_text())!=config:
        raise ValueError(f'{output}: incompatible saved config')
    atomic_json(resolved,config)
    record={'status':'running','identity':identity,'seed':seed,'config':config,'provenance':provenance,
            'checkpoint_path':str(output/'best.pt'),'run_json_path':str(output/'run.json')}
    atomic_json(output/'run.json',record)
    start=time.perf_counter()
    observed=[]
    try:
        stats=fit_target_stats(np.array([train_data[i]['mos'] for i in range(len(train_data))]))
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        loader=DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,collate_fn=collate_videos,
                          generator=torch.Generator().manual_seed(seed))
        best=-math.inf
        best_epoch=0
        history=[]
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            for epoch in range(1,config['max_epochs']+1):
                synchronize(device)
                epoch_start=time.perf_counter()
                mean_loss,_=_train_epoch(model,loader,optimizer,config,stats,device,epoch,emit)
                emit('validation_start',epoch=epoch)
                predictions=predict(model,val_data,stats,device)
                score=correlations(np.array([val_data[i]['mos'] for i in range(len(val_data))]),
                                   np.array([r['prediction'] for r in predictions]))
                synchronize(device)
                history.append({'epoch':epoch,'mean_loss':mean_loss,
                                'validation_srcc':score['srcc'],'wall_seconds':time.perf_counter()-epoch_start})
                if progress is not None:
                    validation_l1=float(np.mean([abs(val_data[i]['mos']-r['prediction']) for i,r in enumerate(predictions)])/stats['std'])
                    emit('epoch_end',epoch=epoch,train_loss=mean_loss,validation_l1_z=validation_l1,
                         validation_srcc=score['srcc'],validation_epoch=epoch)
                if score['status']=='ok' and score['srcc']>best:
                    best,best_epoch=score['srcc'],epoch
                    buffer=io.BytesIO()
                    torch.save({'state_dict':model.state_dict(),'target_stats':stats,'config':config,'epoch':epoch},buffer)
                    atomic_write(output/'best.pt',buffer.getvalue())
                if epoch-best_epoch>=config['patience']:
                    break
            observed=sorted({str(w.message) for w in captured})
        if best_epoch==0:
            raise ValueError('no defined validation SRCC; no checkpoint selected')
        synchronize(device)
        record.update(status='complete',target_stats=stats,best_epoch=best_epoch,best_validation_srcc=best,
                      completed_epochs=len(history),history=history,warnings=observed,
                      wall_seconds=time.perf_counter()-start,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
                      checkpoint_sha256=sha256(output/'best.pt'))
        model.load_state_dict(torch.load(output/'best.pt',map_location=device,weights_only=True)['state_dict'])
        atomic_json(output/'run.json',record)
        atomic_write(output/'warnings.log',('\n'.join(observed)+'\n').encode())
    except Exception as exc:
        record.update(status='failed',reason=str(exc),warnings=observed,wall_seconds=time.perf_counter()-start)
        atomic_json(output/'run.json',record)
        try:
            emit('fit_failed',reason=str(exc))
        except Exception as observer_error:
            exc.add_note(f'Progress reporting also failed: {observer_error}')
        raise
    emit('fit_complete',epoch=record['completed_epochs'],best_epoch=record['best_epoch'],wall_seconds=record['wall_seconds'])
    return record


def fit_fixed_epochs(model,train_data,config,output,seed,provenance,epochs,*,progress=None):
    if type(epochs) is not int or not 1<=epochs<=config['max_epochs']:
        raise ValueError('invalid frozen epoch')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    device=config['device']
    seed_everything(seed,config['cpu_threads'])
    model.to(device=device,dtype=torch.float32)
    identity={'config_sha256':config_hash(config),'seed':seed,'epochs':epochs,
              'provenance':{k:v for k,v in provenance.items() if k!='code_commit'}}
    existing=load_completed(model,output,identity,device)
    if existing:
        return existing
    config_path=output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError(f'{output}: incompatible saved config')
    atomic_json(config_path,config)
    record={'status':'running','identity':identity,'seed':seed,'config':config,'provenance':provenance,
            'selection':'fixed_final_epoch','fold':None,'checkpoint_path':str(output/'best.pt')}
    atomic_json(output/'run.json',record)
    started=time.perf_counter()
    def emit(event,**fields):
        if progress is not None:
            progress({'event':event,'run_path':str(output),'seed':seed,'head':provenance.get('head'),
                      'fold':None,'feature_set':config.get('feature_set'),**fields})
    try:
        stats=fit_target_stats(np.array([train_data[i]['mos'] for i in range(len(train_data))]))
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
        loader=DataLoader(train_data,batch_size=config['batch_size'],shuffle=True,collate_fn=collate_videos,
                          generator=torch.Generator().manual_seed(seed))
        history=[]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            for epoch in range(1,epochs+1):
                tick=time.perf_counter()
                mean_loss,learning_rate=_train_epoch(model,loader,optimizer,config,stats,device,epoch,emit)
                history.append({'epoch':epoch,'mean_loss':mean_loss,'learning_rate':learning_rate,
                                'wall_seconds':time.perf_counter()-tick})
                emit('epoch_end',epoch=epoch,train_loss=mean_loss)
        buffer=io.BytesIO()
        torch.save({'state_dict':model.state_dict(),'target_stats':stats,'config':config,
                    'epoch':epochs,'selection':'fixed_final_epoch'},buffer)
        atomic_write(output/'best.pt',buffer.getvalue())
        record.update(status='complete',target_stats=stats,best_epoch=epochs,completed_epochs=epochs,
                      history=history,warnings=sorted({str(w.message) for w in captured}),
                      wall_seconds=time.perf_counter()-started,checkpoint_sha256=sha256(output/'best.pt'))
        atomic_json(output/'run.json',record)
        return record
    except Exception as exc:
        record.update(status='failed',reason=str(exc),wall_seconds=time.perf_counter()-started)
        atomic_json(output/'run.json',record)
        raise
