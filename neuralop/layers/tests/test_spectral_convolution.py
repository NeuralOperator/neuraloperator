import pytest
import torch
from tltorch import FactorizedTensor
from ..spectral_convolution import SpectralConv



@pytest.mark.parametrize('factorization', ['Dense', 'CP', 'Tucker', 'TT'])
@pytest.mark.parametrize('implementation', ['factorized', 'reconstructed'])
@pytest.mark.parametrize('separable', [False, True])
@pytest.mark.parametrize('dim', [1,2,3,4])
@pytest.mark.parametrize('complex_data', [False, True])
def test_SpectralConv(factorization, implementation, separable, dim, complex_data):
    """Test for SpectralConv of any order
    
    Compares Factorized and Dense convolution output
    Verifies that a dense conv and factorized conv with the same weight produce the same output

    Checks the output size

    Verifies that dynamically changing the number of Fourier modes doesn't break the conv
    """
    modes = (10, 8, 6, 6)
    incremental_modes = (6, 6, 4, 4)
    dtype = torch.cfloat if complex_data else torch.float32

    # Test for Conv1D to Conv4D
    conv = SpectralConv(
        3, 3, modes[:dim],
        bias=False,
        implementation=implementation,
        factorization=factorization,
        complex_data=complex_data,
        separable=separable)

    conv_dense = SpectralConv(
        3, 3, modes[:dim],
        bias=False,
        implementation='reconstructed',
        factorization=None,
        complex_data=complex_data)

    x = torch.randn(2, 3, *(12, )*dim, dtype=dtype)

    assert torch.is_complex(conv.weight)
    assert torch.is_complex(conv_dense.weight)

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        conv_dense.weight = FactorizedTensor.from_tensor(conv.weight.to_tensor(),
                                                         rank=None,
                                                         factorization='ComplexDense')
    
    res_dense = conv_dense(x)
    res = conv(x)
    res_shape = res.shape

    # this closeness test only works if the weights in full form have the same shape
    if not separable:
        torch.testing.assert_close(res_dense, res)

    # Dynamically reduce the number of modes in Fourier space
    conv.n_modes = incremental_modes[:dim]
    res = conv(x)
    assert res_shape == res.shape

    # Downsample outputs
    block = SpectralConv(
        3, 4, modes[:dim], resolution_scaling_factor=0.5)

    x = torch.randn(2, 3, *(12, )*dim)
    res = block(x)
    assert(list(res.shape[2:]) == [12//2]*dim)
    
    # Upsample outputs
    block = SpectralConv(
        3, 4, modes[:dim], resolution_scaling_factor=2)

    x = torch.randn(2, 3, *(12, )*dim)
    res = block(x)
    assert res.shape[1] == 4 # Check out channels
    assert(list(res.shape[2:]) == [12*2]*dim)



def test_SpectralConv_resolution_scaling_factor():
    """Test SpectralConv with upsampled or downsampled outputs
    """
    modes = (4, 4, 4, 4)
    size = [6]*4
    for dim in [1, 2, 3, 4]:
        # Downsample outputs
        conv = SpectralConv(
            3, 3, modes[:dim], resolution_scaling_factor=0.5)
    
        x = torch.randn(2, 3, *size[:dim])
        res = conv(x)
        assert(list(res.shape[2:]) == [m//2 for m in size[:dim]])
        
        # Upsample outputs
        conv = SpectralConv(
            3, 3, modes[:dim], resolution_scaling_factor=2)
    
        x = torch.randn(2, 3, *size[:dim])
        res = conv(x)
        assert(list(res.shape[2:]) == [m*2 for m in size[:dim]])
