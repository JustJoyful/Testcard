import pytest
import yaml
from pathlib import Path


@pytest.mark.parametrize('dimension',[1536,2048])
def test_head_only_budget(dimension):
    from src.heads import build_head
    from src.heads.common import parameter_counts
    tier=yaml.safe_load(Path('configs/head_tiers.yaml').read_text())['matched']
    counts=[parameter_counts(build_head(h,dimension,{'tier':tier,'input_dropout':.1,'head_dropout':.1}))['head'] for h in ('mean_pool','bigru','transformer','mamba')]
    assert max(counts)/min(counts)<=1.05
