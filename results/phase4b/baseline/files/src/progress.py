"""Plain terminal progress and an atomic diagnostic snapshot; never a research gate."""
import sys
import time
from pathlib import Path

import torch

from src.cache import atomic_json


class ProgressReporter:
    def __init__(self, output: Path, stream=None, min_interval=2.0):
        self.output=Path(output)
        self.stream=sys.stdout if stream is None else stream
        self.min_interval=min_interval
        self.state={}
        self.last_display=-float('inf')
        self.started=None

    def __call__(self,event):
        if event['event'] in ('fit_start','fit_reused'):
            for key in ('epoch','batch','batches','train_loss','validation_l1_z','validation_srcc','validation_epoch','reason','best_epoch','wall_seconds'):
                self.state.pop(key,None)
        self.state.update(event)
        now=time.monotonic()
        if event['event'] in ('fit_start','fit_reused'):
            self.started=now
        self.state['elapsed_seconds']=self.state.get('wall_seconds',now-self.started if self.started is not None else 0.)
        if (event['event']=='batch_end' and event['batch'] not in (1,event['batches'])
                and now-self.last_display<self.min_interval):
            return
        self.last_display=now
        s=self.state
        gpu='N/A'
        if str(s.get('device','cpu')).startswith('cuda'):
            device=s['device']
            gpu=f"{torch.cuda.memory_allocated(device)/2**30:.2f} / {torch.cuda.get_device_properties(device).total_memory/2**30:.2f} GiB allocated/total"
        s['gpu']=gpu
        atomic_json(self.output,s)
        val='pending' if 'validation_epoch' not in s else f"{s['validation_l1_z']:.4f} | Val SRCC: {s['validation_srcc']} (epoch {s['validation_epoch']})"
        loss='pending' if 'train_loss' not in s else f"{s['train_loss']:.4f}"
        text=(f"Stage: {s.get('stage','training')} | {s['event']} | Head: {s.get('head','?')} | "
              f"Fold: {s.get('fold','?')} | Seed: {s.get('seed','?')} | {s.get('run_path','')}\n"
              f"Epoch: {s.get('epoch',0)}/{s.get('max_epochs',200)} maximum | Batch: {s.get('batch',0)}/{s.get('batches','?')}\n"
              f"Train Loss: {loss}\nVal L1(z): {val}\nGPU: {gpu}\n"
              f"CV fits complete: {s.get('cv_complete',0)}/100 ({s.get('cv_complete',0)}%) | "
              f"Ridge folds: {s.get('ridge_complete',0)}/5 | Control seeds: {s.get('control_complete',0)}/5\n")
        text+=f"Best epoch: {s.get('best_epoch','pending')} | Elapsed: {s['elapsed_seconds']:.1f}s\n"
        if s.get('reason'):
            text+=f"Reason: {s['reason']}\n"
        print(text,file=self.stream,flush=True)
