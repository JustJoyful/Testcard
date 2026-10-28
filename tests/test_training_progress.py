import copy
import io
import json
import random

import numpy as np
import pytest
import torch

from test_phase1_training import Tiny, config, dataset


def reporter(tmp_path, **kwargs):
    from src.progress import ProgressReporter
    return ProgressReporter(tmp_path/'progress.json', **kwargs)


def test_progress_pending_throttling_cpu_and_failure(tmp_path, monkeypatch):
    import time
    ticks=iter([0., .1, .2, 3., 4.])
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *a: pytest.fail('CPU queried CUDA'))
    stream=io.StringIO()
    show=reporter(tmp_path, stream=stream)
    show({'event':'fit_start','device':'cpu','run_path':'r','seed':0,'head':'mean_pool',
          'feature_set':'resnet50','fold':0,'max_epochs':200,'cv_complete':0,'ridge_complete':0})
    assert 'pending' in stream.getvalue() and 'GPU: N/A' in stream.getvalue()
    show({'event':'batch_end','epoch':1,'batch':1,'batches':3,'train_loss':.3})
    size=len(stream.getvalue())
    show({'event':'batch_end','epoch':1,'batch':2,'batches':3,'train_loss':.2})
    assert len(stream.getvalue())==size
    show({'event':'epoch_end','epoch':1,'batch':3,'batches':3,'train_loss':.2,
          'validation_l1_z':.4,'validation_srcc':.6,'validation_epoch':1})
    show({'event':'fit_failed','reason':'broken'})
    saved=json.loads((tmp_path/'progress.json').read_text())
    assert saved['event']=='fit_failed' and saved['reason']=='broken'
    text=stream.getvalue()
    for value in ('Epoch: 1/200 maximum','Batch: 3/3','Train Loss:','Val L1(z):','Val SRCC:', 'CV fits complete: 0/100', 'broken'):
        assert value in text
    assert '\x1b' not in text


def test_reporter_does_not_change_training_or_rng(tmp_path, monkeypatch):
    from src.training import fit_one
    torch.manual_seed(5)
    first=Tiny(); second=copy.deepcopy(first)
    calls=[]
    original=Tiny.forward
    def forward(self, *args):
        calls.append(1)
        return original(self,*args)
    monkeypatch.setattr(Tiny,'forward',forward)
    cfg={**config(),'batch_size':4}
    a=fit_one(first,dataset(),dataset(),cfg,tmp_path/'a',0,{})
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    count=len(calls); calls.clear()
    stream=io.StringIO()
    b=fit_one(second,dataset(),dataset(),cfg,tmp_path/'b',0,{},progress=reporter(tmp_path,stream=stream,min_interval=0))
    assert count==len(calls)
    assert random.getstate()==rng[0]
    assert np.array_equal(np.random.get_state()[1],rng[1][1])
    assert torch.equal(torch.get_rng_state(),rng[2])
    for key in ('identity','best_epoch','completed_epochs','target_stats','best_validation_srcc'):
        assert a[key]==b[key]
    for x,y in zip(a['history'],b['history']):
        for key in ('epoch','mean_loss','validation_srcc'):
            assert x[key]==y[key]
    for key,value in first.state_dict().items():
        assert torch.equal(value,second.state_dict()[key])
    assert 'Batch: 2/2' in stream.getvalue()
    assert json.loads((tmp_path/'progress.json').read_text())['event']=='fit_complete'
    before=(tmp_path/'b/best.pt').stat().st_mtime_ns
    fit_one(Tiny(),dataset(),dataset(),cfg,tmp_path/'b',0,{},progress=reporter(tmp_path,stream=stream))
    assert (tmp_path/'b/best.pt').stat().st_mtime_ns==before
    assert 'fit_reused' in stream.getvalue()


def test_observer_failure_keeps_success_and_original_model_failure(tmp_path):
    from src.training import fit_one
    def observer(event):
        if event['event'] in ('fit_complete','fit_failed'):
            raise OSError('observer broke')
    with pytest.raises(OSError,match='observer broke'):
        fit_one(Tiny(),dataset(),dataset(),config(),tmp_path/'ok',0,{},progress=observer)
    assert json.loads((tmp_path/'ok/run.json').read_text())['status']=='complete'
    class Broken(Tiny):
        def forward(self,*args):
            raise ValueError('model broke')
    with pytest.raises(ValueError,match='model broke'):
        fit_one(Broken(),dataset(),dataset(),config(),tmp_path/'bad',0,{},progress=observer)
    assert json.loads((tmp_path/'bad/run.json').read_text())['reason']=='model broke'


def test_status_failure_happens_before_training(tmp_path, monkeypatch):
    from src.training import fit_one
    show=reporter(tmp_path,stream=io.StringIO())
    (tmp_path/'progress.json').mkdir()
    monkeypatch.setattr(torch.optim.AdamW,'step',lambda *a,**kw: pytest.fail('optimizer ran'))
    with pytest.raises(OSError):
        fit_one(Tiny(),dataset(),dataset(),config(),tmp_path/'fit',0,{},progress=show)
    assert not (tmp_path/'fit/best.pt').exists()


def test_reused_run_clears_other_runs_validation_and_shows_selected_epoch(tmp_path):
    stream=io.StringIO();show=reporter(tmp_path,stream=stream,min_interval=0)
    show({'event':'fit_start','seed':0,'device':'cpu'})
    show({'event':'epoch_end','epoch':3,'validation_l1_z':.2,'validation_srcc':.8,'validation_epoch':3})
    show({'event':'fit_complete','epoch':3,'best_epoch':1,'wall_seconds':2.5})
    assert 'Best epoch: 1' in stream.getvalue() and 'Elapsed: 2.5s' in stream.getvalue()
    show({'event':'fit_reused','seed':1,'epoch':2})
    state=json.loads((tmp_path/'progress.json').read_text())
    assert 'validation_srcc' not in state
