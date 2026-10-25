from .common import VideoHead
from .mean_pool import MeanPoolHead
from .gru import BiGRUHead
from .transformer import TransformerHead


def build_head(name,input_dim,config):
    factories={'mean_pool':MeanPoolHead,'bigru':BiGRUHead,'transformer':TransformerHead}
    if name not in factories:
        raise ValueError(f'unknown head: {name}')
    tier=config['tier']
    if tier['front_width']!=256:
        raise ValueError('shared front width is frozen at 256')
    return VideoHead(input_dim,config,factories[name](tier,config['head_dropout']))
