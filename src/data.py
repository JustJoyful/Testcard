"""Checked immutable cached features; raw MOS without dataset-level normalization."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.cache import DINO_BRANCHES, index_manifest, read_branch, read_manifest
from src.splits import freeze_splits


class CachedFeatureDataset(Dataset):
    def __init__(self,root,manifest,splits,fold,role,feature_set):
        if role not in ('train','val','test') or fold not in range(5) or feature_set not in ('dino','resnet50'):
            raise ValueError('invalid fold/role/features')
        freeze_splits(manifest,splits)
        inventory=list(csv.DictReader(Path(manifest).open()))
        sources={r['video_id']:r for r in inventory}
        roles=list(csv.DictReader(Path(splits).open()))
        self.ids=[r['video_id'] for r in roles if int(r['outer_fold'])==fold and r['role']==role]
        self.items=[]
        records=index_manifest(read_manifest(root))
        branches=DINO_BRANCHES if feature_set=='dino' else ['resnet50']
        # Check the entire authoritative inventory, so no invalid video disappears from fitting.
        for key,row in sources.items():
            arrays={b:read_branch(root,row['dataset'],key,b,records,row) for b in branches}
            if len({len(x) for x in arrays.values()})!=1:
                raise ValueError(f'{key}: inconsistent timesteps')
            if key in self.ids:
                x=np.concatenate([arrays['dino_cls'],arrays['dino_patchmean_L12']],axis=1) if feature_set=='dino' else arrays['resnet50']
                self.items.append({'video_id':key,'features':torch.from_numpy(x.astype(np.float32,copy=True)), 'mos':float(row['mos'])})
        by_id={r['video_id']:r for r in self.items}
        self.items=[by_id[key] for key in self.ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self,index):
        item=self.items[index]
        return {**item,'features':item['features'].clone()}


def collate_videos(items):
    if not items:
        raise ValueError('empty video batch')
    dimension=items[0]['features'].shape[-1]
    for item in items:
        x=item['features']
        if x.ndim!=2 or not len(x) or x.shape[1]!=dimension or x.dtype!=torch.float32 or not torch.isfinite(x).all() or not np.isfinite(item['mos']):
            raise ValueError(f'{item["video_id"]}: invalid features/MOS')
    return {'video_id':[r['video_id'] for r in items],
            'features':torch.nn.utils.rnn.pad_sequence([r['features'] for r in items],batch_first=True),
            'lengths':torch.tensor([len(r['features']) for r in items],dtype=torch.int64),
            'mos':torch.tensor([r['mos'] for r in items],dtype=torch.float32)}
