import torch
from torch.nn import Parameter
from torch.testing import assert_close
import pytest
import math

from ..adamw import AdamW
from ..galore_adamw import GaLoreAdamW

from ..tensor_galore_projector import TensorGaLoreProjector

from tensorly.tucker_tensor import validate_tucker_rank

@pytest.mark.parametrize('adam_optimizer_cls', [AdamW])
def test_correct_complex_adam_momentum(adam_optimizer_cls):
    # param = x * 2j
    x = torch.randn((3,3), dtype=torch.float64)
    param = Parameter(((0. + 1.0j) * x).to(torch.cfloat))

    optimizer = adam_optimizer_cls(params=[param],
                      betas=(0.5, 0.5))

    loss = torch.view_as_real((param * param.conj())).sum()
    # grad x^2 = 2x, grads are all 0 + 2j * x

    loss.backward()
    optimizer.step()

    # momentum value should be elemwise (2jx * -2jx * (1 - 0.5)) = 4x**2 * 0.5 = 2x**2
    # exp_avg_sq should be empty, meaning it is just momentum * (1-beta2)
    momentum = optimizer.state[param]["exp_avg_sq"]
    assert_close(momentum, (2 * x**2).to(torch.cfloat))


@pytest.mark.parametrize('galore_param_pct', [0.25, 0.5, 1.0])
@pytest.mark.parametrize('n_dim', [2,3,4])
def test_galore_projector(galore_param_pct, n_dim):
    full_rank_tensor = torch.randn([8]*n_dim)

    galore_rank = validate_tucker_rank(full_rank_tensor.shape, galore_param_pct)
    projector = TensorGaLoreProjector(rank=galore_rank, warm_restart=True)

    # test that the projection is the correct size
    # test 0th iter for full computation
    low_rank_tensor = projector.project(full_rank_tensor, 0)
    assert low_rank_tensor.numel() == math.prod(galore_rank)

@pytest.mark.parametrize('galore_param_pct', [0.25, 0.5, 1.0])
def test_galore_adamw_rank(galore_param_pct):
    x = torch.randn((8,8,8,8), dtype=torch.float64)
    param = Parameter(((0. + 1.0j) * x).to(torch.cfloat))
    galore_param = Parameter(((1. + 0j) * x).to(torch.cfloat))

    # pick mode-wise rank so that low_rank_grad.numel() / full_rank_grad.numel() == galore_param_pct
    galore_rank = validate_tucker_rank(galore_param.shape, galore_param_pct)

    optimizer = GaLoreAdamW(params=[param],
                            galore_params=[galore_param],
                            galore_rank=galore_rank,
                            betas=(0.5, 0.5))

    loss = torch.view_as_real((param * galore_param.conj())).sum()

    loss.backward()
    optimizer.step()

    momentum = optimizer.state[galore_param]["exp_avg_sq"]
    # make sure low-rank params are being stored
    assert momentum.numel() == math.prod(galore_rank)

