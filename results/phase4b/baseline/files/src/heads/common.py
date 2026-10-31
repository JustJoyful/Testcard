"""Shared projection and length-aware sequence operations."""
import torch
from torch import nn


def valid_mask(lengths,T):
    if lengths.ndim!=1 or lengths.dtype not in (torch.int32,torch.int64) or torch.any(lengths<1) or torch.any(lengths>T):
        raise ValueError('lengths must be integer in [1,T]')
    return torch.arange(T,device=lengths.device)[None,:]<lengths[:,None]


def sanitize(x,lengths):
    mask=valid_mask(lengths.to(x.device),x.shape[1])
    if len(lengths)!=len(x):
        raise ValueError('batch/length mismatch')
    return torch.where(mask[:,:,None],x,torch.zeros((),dtype=x.dtype,device=x.device))


def reverse_valid(x,lengths):
    x=sanitize(x,lengths)
    positions=torch.arange(x.shape[1],device=x.device)[None,:]
    indices=(lengths.to(x.device)[:,None]-1-positions).clamp(min=0)
    return sanitize(x.gather(1,indices[:,:,None].expand_as(x)),lengths)


def masked_mean(x,lengths):
    return sanitize(x,lengths).sum(1)/lengths.to(x.device)[:,None]


class FeatureFrontEnd(nn.Module):
    def __init__(self,input_dim,dropout):
        super().__init__()
        self.layers=nn.Sequential(nn.LayerNorm(input_dim),nn.Dropout(dropout),nn.Linear(input_dim,256))

    def forward(self,x):
        return self.layers(x)


def readout(width,hidden,dropout):
    return nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))


class VideoHead(nn.Module):
    def __init__(self,input_dim,config,head):
        super().__init__()
        self.front_end=FeatureFrontEnd(input_dim,config['input_dropout'])
        self.head=head

    def forward(self,features,lengths):
        x=self.front_end(sanitize(features,lengths))
        return self.head(sanitize(x,lengths),lengths).squeeze(-1)


def parameter_counts(model):
    front=sum(p.numel() for p in model.front_end.parameters() if p.requires_grad)
    head=sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    return {'front_end':front,'head':head,'total':front+head}
