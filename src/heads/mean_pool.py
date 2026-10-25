from torch import nn
from .common import masked_mean, readout


class MeanPoolHead(nn.Module):
    def __init__(self,tier,dropout):
        super().__init__()
        self.readout=readout(256,tier['mean_pool_hidden'],dropout)

    def forward(self,x,lengths):
        return self.readout(masked_mean(x,lengths))
