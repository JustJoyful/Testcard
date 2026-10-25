"""Shared video-level L1 plus pairwise logistic rank loss."""
import math
import torch
from torch.nn import functional as F


def quality_loss(pred,target_z,rank_weight):
    if (pred.ndim!=1 or pred.shape!=target_z.shape or not len(pred) or not math.isfinite(rank_weight)
            or rank_weight<0 or not torch.isfinite(pred).all() or not torch.isfinite(target_z).all()):
        raise ValueError('invalid predictions, targets or rank weight')
    loss=F.l1_loss(pred,target_z)
    i,j=torch.triu_indices(len(pred),len(pred),offset=1,device=pred.device)
    signs=torch.sign(target_z[i]-target_z[j])
    unequal=signs!=0
    if unequal.any() and rank_weight:
        loss=loss+rank_weight*F.softplus(-signs[unequal]*(pred[i[unequal]]-pred[j[unequal]])).mean()
    return loss
