import pytest
import torch


def test_collate_unequal_lengths():
    from src.data import collate_videos
    actual=collate_videos([{'video_id':key,'features':torch.ones(n,3),'mos':3.} for key,n in [('a',2),('b',4)]])
    assert actual['lengths'].tolist()==[2,4]
    assert actual['video_id']==['a','b']
    assert actual['features'].shape==(2,4,3)
    assert (actual['features'][0,2:]==0).all()


def test_missing_or_nonfinite_cached_features_fail_by_id():
    from src.data import collate_videos
    with pytest.raises(ValueError,match='bad'):
        collate_videos([{'video_id':'bad','features':torch.tensor([[float('nan')]]),'mos':2.}])
