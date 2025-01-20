import pytest
import torch
from ..local_no_block import LocalNOBlocks

@pytest.mark.parametrize('n_dim', [1,2,3])
def test_LocalNOBlock_resolution_scaling_factor(n_dim):
    """Test LocalNOBlocks with upsampled or downsampled outputs
    """
    max_n_modes = [8, 8, 8]
    n_modes = [4, 4, 4]
    
    size = [10]*3
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    block = LocalNOBlocks(
        3, 4, default_in_shape=tuple(size[:n_dim]), n_modes=max_n_modes[:n_dim], 
        max_n_modes=max_n_modes[:n_dim], n_layers=1, diff_layers=[True], disco_layers=[(n_dim == 2)])
    
    assert block.convs[0].n_modes[:-1] == max_n_modes[:n_dim-1]
    assert block.convs[0].n_modes[-1] == max_n_modes[n_dim-1]//2 + 1

    block.n_modes = n_modes[:n_dim]
    assert block.convs[0].n_modes[:-1] == n_modes[:n_dim-1]
    assert block.convs[0].n_modes[-1] == n_modes[n_dim-1]//2 + 1

    block.n_modes = max_n_modes[:n_dim]
    assert block.convs[0].n_modes[:-1] == max_n_modes[:n_dim-1]
    assert block.convs[0].n_modes[-1] == max_n_modes[n_dim-1]//2 + 1

    # Downsample outputs
    block = LocalNOBlocks(
        3, 4, n_modes[:n_dim], default_in_shape=tuple(size[:n_dim]), n_layers=1, diff_layers=[True], 
        disco_layers=[(n_dim == 2)], resolution_scaling_factor=0.5, use_mlp=True, mlp_dropout=mlp_dropout, 
        mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert(list(res.shape[2:]) == [m//2 for m in size[:n_dim]])
    
    # Upsample outputs
    block = LocalNOBlocks(
        3, 4, n_modes[:n_dim], default_in_shape=tuple(size[:n_dim]), n_layers=1, diff_layers=[True], 
        disco_layers=[(n_dim == 2)], resolution_scaling_factor=2, use_mlp=True, mlp_dropout=mlp_dropout,
        mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert res.shape[1] == 4 # Check out channels
    assert(list(res.shape[2:]) == [m*2 for m in size[:n_dim]])

@pytest.mark.parametrize('norm', 
                         ['instance_norm', 'ada_in', 'group_norm'])
@pytest.mark.parametrize('n_dim', [1,2,3])
def test_LocalNOBlock_norm(norm, n_dim):
    """Test LocalNOBlock with normalization
    """
    modes = (8, 8, 8)
    size = [10]*3
    mlp_dropout=0
    mlp_expansion=0.5
    mlp_skip='linear'
    ada_in_features = 4
    block = LocalNOBlocks(
        3, 4, modes[:n_dim], default_in_shape=tuple(size[:n_dim]), n_layers=1, diff_layers=[True], 
        disco_layers=[(n_dim == 2)], use_mlp=True, norm=norm, ada_in_features=ada_in_features,
        mlp_dropout=mlp_dropout, mlp_expansion=mlp_expansion, mlp_skip=mlp_skip)

    if norm == 'ada_in':
        embedding = torch.randn(ada_in_features)
        block.set_ada_in_embeddings(embedding)

    x = torch.randn(2, 3, *size[:n_dim])
    res = block(x)
    assert(list(res.shape[2:]) == size[:n_dim])