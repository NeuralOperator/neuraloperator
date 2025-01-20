import torch
from ..codano import CODANO
from ...layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
)
from functools import partial
import pytest


@pytest.mark.parametrize("hidden_variable_codimension", [1, 2])
@pytest.mark.parametrize("lifting_channel_ratio", [4, 2])
@pytest.mark.parametrize("use_positional_encoding", [True, False])
@pytest.mark.parametrize("n_variables", [3, 4])
@pytest.mark.parametrize("positional_encoding_dim", [4, 8])
@pytest.mark.parametrize("positional_encoding_modes", [[8, 8], [16, 16]])
@pytest.mark.parametrize("static_channel_dim", [0, 2])
@pytest.mark.parametrize("use_cls_token", [True, False])
def test_CODANO(
    hidden_variable_codimension,
    lifting_channel_ratio,
    use_positional_encoding,
    n_variables,
    positional_encoding_dim,
    positional_encoding_modes,
    static_channel_dim,
    use_cls_token,
):
    output_variable_codimension = 1
    n_layers = 5
    n_heads = [2] * n_layers
    n_modes = [[16, 16]] * n_layers
    attention_scalings = [0.5] * n_layers
    scalings = [[1, 1], [0.5, 0.5], [1, 1], [2, 2], [1, 1]]
    use_horizontal_skip_connection = True
    horizontal_skips_map = {3: 1, 4: 0}
    if use_positional_encoding:
        variable_ids = ["a" * (i + 1) for i in range(n_variables)]
    else:
        variable_ids = None

    model = CODANO(
        output_variable_codimension=output_variable_codimension,
        hidden_variable_codimension=hidden_variable_codimension,
        lifting_channel_ratio=lifting_channel_ratio,
        projection_channel_ratio=lifting_channel_ratio,
        use_positional_encoding=use_positional_encoding,
        positional_encoding_dim=positional_encoding_dim,
        positional_encoding_modes=positional_encoding_modes,
        use_horizontal_skip_connection=use_horizontal_skip_connection,
        static_channel_dim=static_channel_dim,
        horizontal_skips_map=horizontal_skips_map,
        variable_ids=variable_ids,
        n_layers=n_layers,
        n_heads=n_heads,
        n_modes=n_modes,
        attention_scaling_factors=attention_scalings,
        per_layer_scaling_factors=scalings,
        enable_cls_token=use_cls_token,
    )

    in_data = torch.randn(2, n_variables, 32, 32)

    if static_channel_dim > 0:
        static_channels = torch.randn(2, static_channel_dim, 32, 32)
    else:
        static_channels = None

    # Test forward pass
    out = model(in_data, static_channels, variable_ids)

    # Check output size
    assert list(out.shape) == [2, n_variables, 32, 32]

    # test different resolutions
    # also different sets of variables

    new_n_var = n_variables // 2
    if use_positional_encoding:
        input_var_ids = variable_ids[:new_n_var]
    else:
        input_var_ids = None

    in_data = torch.randn(2, new_n_var, 64, 64)
    if static_channel_dim > 0:
        static_channels = torch.randn(2, static_channel_dim, 64, 64)
    else:
        static_channels = None

    out = model(in_data, static_channels, input_var_ids)
    assert list(out.shape) == [2, new_n_var, 64, 64]

    # Test backward pass
    out.sum().backward()
