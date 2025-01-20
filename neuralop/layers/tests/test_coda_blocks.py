import pytest
import torch
from ..coda_blocks import *
from ..spectral_convolution import *

device = 'cuda' if torch.backends.cuda.is_built() else 'cpu'
#device = 'cpu'

@pytest.mark.parametrize('token_codimension', [1, 2, 5])
@pytest.mark.parametrize('n_dim', [2, 3])
@pytest.mark.parametrize('norm', ['instance_norm', None])
@pytest.mark.parametrize('nonlinear_attention', ['True', 'False'])
@pytest.mark.parametrize('per_channel_attention', ['False', 'True'])
@pytest.mark.parametrize('output_scale', [1, 2])
def test_Codano(token_codimension,
                n_dim, norm,
                nonlinear_attention,
                per_channel_attention,
                output_scale):
    """
    Test CoDA-NO layers
    """
    n_modes = [64] * n_dim
    n_heads = 3
    head_codimension = 3
    codimension_size = None
    permutation_eq = True
    temperature = 1.0

    layer = CODABlocks(
        n_modes=n_modes,
        n_heads=n_heads,
        token_codimension=token_codimension,
        head_codimension=head_codimension,
        norm=norm,
        codimension_size=codimension_size,
        per_channel_attention=per_channel_attention,
        permutation_eq=permutation_eq,
        temperature=temperature,
        nonlinear_attention=nonlinear_attention,
    ).to(device)
    
    spatial_res = [64]*n_dim
    x = torch.randn(2, 10, *spatial_res).to(device)
    output_shape = [int(s * output_scale) for s in spatial_res]
    out = layer(x, output_shape)
    assert list(out.shape[-n_dim:]) == output_shape

    # test different spatial resolution 
    spatial_res = [48]*n_dim
    x = torch.randn(2, 10, *spatial_res).to(device)
    out = layer(x)
    assert out.shape == x.shape