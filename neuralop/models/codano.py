import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.skip_connections import skip_connection
from ..layers.padding import DomainPadding
from ..layers.coda_blocks import CODABlocks
from ..layers.resample import resample
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND


class CODANO(nn.Module):
    """Codomain Attention Neural Operators (CoDA-NO) uses a specialized attention mechanism in the codomain space for data in
    infinite dimensional spaces as described in [1]_. The model treates each input channel as a variable of the physical system
    and uses attention mechanism to model the interactions between the variables. The model uses lifting and projection modules
    to map the input variables to a higher-dimensional space and then back to the output space. The model also supports positional
    encoding and static channel information for additional context of the physical system such as external force or inlet condition.


    Parameters
    ----------
    output_variable_codimension : int
        The number of output channels (or output codomain dimension) corresponding to each input variable (or input channel). Default is 1.
        Example: For a input with 3 variables (channels) and output_variable_codimension=2, the output will have 6 channels (3 variables × 2 codimension).

    lifting : bool
        Indicates whether to use a lifting module. The lifting module projects each input variable (i.e., each input channel) into a
        higher-dimensional space. Default is True.
        If `lifting=False`, the input channels are directly used as tokens for codoamin attention.

        Example: If lifting=True and hidden_variable_codimension=32, each input channel is lifted to 32 channels.

    hidden_variable_codimension : int
        The number of hidden channels corresponding to each input variable (or channel). Each input channel is independently lifted
        to `hidden_variable_codimension` channels by the lifting operation. This argument is ignored when `lifting=False`. Default is 32.

    lifting_channel_ratio : int
        ratio of lifting channels to hidden_channels, by default 2
        The number of intermidiate channels in the lifting block of the codano is
        lifting_channel_ratio * hidden_variable_codimension.
        This parameter is ignored when `lifting=False` as not lifting is performed.

    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of intermidiate channels in the projection block of the codano is
        projection_channel_ratio * hidden_variable_codimension.
        This parameter is ignored when `projection=False` and not ptrojection is performed.

    use_positional_encoding : bool
        Indicates whether to use variable-specific positional encoding. If True, a learnable positional encoding is concatenated
        to each variable (each input channel) before the lifting operation. The positinal encoding used here is a function space
        generalization of the learable positional encoding used in BERT [2]. In codano, the positional encoding is a function on
        domain which is learned directly in the Fourier Space. Default is False.

    positional_encoding_dim : int
        The dimension (number of channels) of the positional encoding learned of each input variable (i.e., input channel). Default is 8.

    positional_encoding_modes : list
        Number of Fourier modes used in positional encoding along each dimension. The positional embeddings are functions and are directly learned
        in Fourier space. This parameter must be specified when `use_positional_encoding=True`. Default is None.
        Example: For a 2D input, positional_encoding_modes could be [16, 16].

    static_channel_dim : int
        The number of channels for static information, such as boundary conditions in PDEs. These channels are concatenated with
        each variable before the lifting operation and use to provide additional information regarding the physical setup of the system.
        When `static_channel_dim > 0`, additional information must be provided during
        the forward pass. Default is 0.

        For example, static_channel_dim=1 can be used to provid mask of the domain pointing a hole or obstacle in the domain.

    variable_ids : list[str]
        The names of the variables in the dataset. Default is None.

        This parameter is **only** required when `use_positional_encoding=True` to initialize learnable positional embeddings for
        each unique physical varibles in the dataset.

        For example:
        If the dataset consists of only Navier Stokes equations, the variable_ids=['u_x', 'u_y', 'p'], representing the velocity
        components in x and y directions and pressure, respectively. Please note that we consider each input channel as a physical
        variable of the PDE.

        If the dataset consists of multiple PDEs, such as Navier Stokes and Heat equation, the variable_ids=['u_x', 'u_y', 'p', 'T'],
        where 'T' represents the temperature variable for thee Heat equation and 'u_x', 'u_y', 'p' are the velocity components and pressure
        for the Navier Stokes equations. This is required when we aim to learn a single solver for multiple different PDEs.

        This parameter is not required when `use_positional_encoding=False`.

    use_horizontal_skip_connection : bool
        Indicates whether to use horizontal skip connections, similar to U-shaped architectures. Default is False.

    horizontal_skips_map : dict
        A mapping that specifies horizontal skip connections between layers. Only required when `use_horizontal_skip_connection=True`. Default is None.
        Example: For a 5-layer architecture, horizontal_skips_map={4: 0, 3: 1} creates skip connections from layer 0 to layer 4 and layer 1 to layer 3.

    n_layers : int
        The number of codomain attention layers. Default is 4.

    n_modes : list
        The number of Fourier modes to use in integral operators in the CoDA-NO block along each dimension. Default is None.
        Example: For a 5-layer 2D CoDA-NO, n_modes=[[16, 16], [16, 16], [16, 16], [16, 16], [16, 16]].

    per_layer_scaling_factor : list
        The output scaling factor for each CoDANO_block along each dimension. The output of each of the CoDANO_block
        is resampled accroding to the scaling factor and then passed to the following CoDANO_blocks. Default is None ,i.e., no scaling.

        Example: For a 2D input and `n_layers=5`, per_layer_scaling_factor=[[1, 1], [0.5, 0.5], [1, 1], [2, 2], [1, 1]], which downsample the
        output of the second layer by a factor of 2 and upsample the output of the fourth layer by a factor of 2.

        The resolution of the output of the codano model is determined by the product of the scaling factors of all the layers.

    n_heads : list
        The number of attention heads for each layer. Default is None, i.e., single attention head for
        each codomain attention block.
        Example: For a 4-layer CoDA-NO, n_heads=[2, 2, 2, 2].

    attention_scaling_factors : list
        Scaling factors in the codomain attention mechanism to scale the key and query functions. These scaling factors are used to resample
        the key and query function before calculating the attention matrix. It does not have any effect on the value funnctions
        in the codoamin attention mechanism, i.e., it does not change the output shape of the block.  Default is None, which means no scaling.

        Example: For a 5-layer CoDA-NO, attention_scaling_factors=[0.5, 0.5, 0.5, 0.5, 0.5], which is downsample the key and query functions,
        reducing the resolution by a factor of 2.

    conv_module : nn.Module
        The convolution module to use in the CoDANO_block. Default is SpectralConv.

    nonlinear_attention : bool
        Indicates whether to use a non-linear attention mechanism, employing non-linear key, query, and value operators. Default is False.

    non_linearity : callable
        The non-linearity to use in the codomain attention block. Default is `F.gelu`.

    attention_token_dim : int
        The number of channels in each token function. `attention_token_dim` must divide `hidden_variable_codimension`. Default is 1.

    per_channel_attention : bool
        Indicates whether to use a per-channel attention mechanism in Codomain attention layer. Default is False.

    layer_kwargs : dict
        Additional arguments for the CoDA blocks. Default is an empty dictionary `{}`.

    projection : bool
        Indicates whether to use a projection module to map the hidden variables to the output variables. Default is True.
        if False, the output of the last CoDA block is returned directly.

    domain_padding : float
        The padding factor for each input channel. It zero pads each of the channel following the `domain_padding_mode`.  Default is 0.25.

    domain_padding_mode : str
        The domain padding mode, which can be 'one-sided' or 'two-sided'. Default is 'one-sided'.
        if one-sided, padding is only done along one side of each dimension.
        If two-sided, padding is done on both sides of each dimension.

    enable_cls_token : bool
        Indicates whether to use a learnable CLASS token during the attention mechanism. We use a function space greneralization of
        leanable [class] token used in vision transformers such as ViT, which is learned directly in Fourier space. Default is False.

    References
    -----------
    .. [1] :

    Rahman, Md Ashiqur, et al. "Pretraining codomain attention neural operators for solving multiphysics pdes." (2024).
    NeurIPS 2024. https://arxiv.org/pdf/2403.12553.

    .. [2] :
    Devlin, Jacob, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

    """

    def __init__(
        self,
        output_variable_codimension=1,
        lifting=True,
        hidden_variable_codimension=32,
        lifting_channel_ratio: int = 2,
        projection_channel_ratio: int = 2,
        use_positional_encoding=False,
        positional_encoding_dim=8,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        n_layers=4,
        n_modes=None,
        per_layer_scaling_factors=None,
        n_heads=None,
        attention_scaling_factors=None,
        conv_module=SpectralConv,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        layer_kwargs={},
        projection=True,
        domain_padding=0.25,
        domain_padding_mode="one-sided",
        enable_cls_token=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert len(n_modes) == n_layers, "number of modes for all layers are not given"
        assert (
            len(n_heads) == n_layers or n_heads is None
        ), "number of Attention head for all layers are not given"
        assert (
            len(per_layer_scaling_factors) == n_layers
            or per_layer_scaling_factors is None
        ), "scaling for all layers are not given"
        assert (
            len(attention_scaling_factors) == n_layers
            or attention_scaling_factors is None
        ), "attention scaling for all layers are not given"
        if use_positional_encoding:
            assert positional_encoding_dim > 0, "positional encoding dim is not given"
            assert (
                positional_encoding_modes is not None
            ), "positional encoding modes are not given"
        else:
            positional_encoding_dim = 0

        if attention_scaling_factors is None:
            attention_scaling_factors = [1] * n_layers

        input_variable_codimension = 1  # each channel is a variable
        if not lifting:
            hidden_variable_codimension = input_variable_codimension

        assert (
            hidden_variable_codimension % attention_token_dim == 0
        ), "attention token dim should divide hidden variable codimension"

        self.n_dim = len(n_modes[0])

        if n_heads is None:
            n_heads = [1] * n_layers
        if per_layer_scaling_factors is None:
            per_layer_scaling_factors = [[1] * self.n_dim] * n_layers
        if attention_scaling_factors is None:
            attention_scaling_factors = [1] * n_layers

        self.input_variable_codimension = input_variable_codimension
        self.hidden_variable_codimension = hidden_variable_codimension
        self.n_modes = n_modes
        self.per_layer_scale_factors = per_layer_scaling_factors
        self.non_linearity = non_linearity
        self.n_heads = n_heads
        self.lifting = lifting
        self.projection = projection
        self.enable_cls_token = enable_cls_token
        self.positional_encoding_dim = positional_encoding_dim
        self.variable_ids = variable_ids
        self.attention_scalings = attention_scaling_factors
        self.positional_encoding_modes = positional_encoding_modes
        self.static_channel_dim = static_channel_dim
        self.layer_kwargs = layer_kwargs
        self.use_positional_encoding = use_positional_encoding
        self.use_horizontal_skip_connection = use_horizontal_skip_connection
        self.horizontal_skips_map = horizontal_skips_map
        self.output_variable_codimension = output_variable_codimension
        lifting_variable_codimension = int(
            hidden_variable_codimension * lifting_channel_ratio
        )
        projection_variable_codimension = int(
            hidden_variable_codimension * projection_channel_ratio
        )
        if self.positional_encoding_modes is not None:
            self.positional_encoding_modes[-1] = self.positional_encoding_modes[-1] // 2

        # calculating scaling
        if self.per_layer_scale_factors is not None:
            self.end_to_end_scaling = [1] * len(self.per_layer_scale_factors[0])
            # multiplying scaling factors
            for k in self.per_layer_scale_factors:
                self.end_to_end_scaling = [
                    i * j for (i, j) in zip(self.end_to_end_scaling, k)
                ]
        else:
            self.end_to_end_scaling = [1] * self.n_dim

        if self.n_heads is None:
            self.n_heads = [1] * self.n_layers

        # Setting up domain padding for encoder and reconstructor
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=self.end_to_end_scaling,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        extended_variable_codimemsion = (
            input_variable_codimension + static_channel_dim + positional_encoding_dim
        )
        self.extended_variable_codimemsion = extended_variable_codimemsion
        if self.lifting:
            self.lifting = ChannelMLP(
                in_channels=extended_variable_codimemsion,
                out_channels=hidden_variable_codimension,
                hidden_channels=lifting_variable_codimension,
                n_layers=2,
                n_dim=self.n_dim,
            )
        else:
            self.lifting = None

        self.base = nn.ModuleList([])

        for i in range(self.n_layers):
            self.base.append(
                CODABlocks(
                    n_modes=self.n_modes[i],
                    n_heads=self.n_heads[i],
                    scale=self.attention_scalings[i],
                    token_codimension=attention_token_dim,
                    per_channel_attention=per_channel_attention,
                    nonlinear_attention=nonlinear_attention,
                    resolution_scaling_factor=self.per_layer_scale_factors[i],
                    conv_module=conv_module,
                    non_linearity=self.non_linearity,
                    **self.layer_kwargs,
                )
            )

        if self.use_horizontal_skip_connection:
            # horizontal skip connections
            # linear projection of the concated tokens from skip connections

            self.skip_map_module = nn.ModuleDict()
            for k in self.horizontal_skips_map.keys():
                self.skip_map_module[str(k)] = ChannelMLP(
                    in_channels=2 * hidden_variable_codimension,
                    out_channels=hidden_variable_codimension,
                    hidden_channels=None,
                    n_layers=1,
                    non_linearity=nn.Identity(),
                    n_dim=self.n_dim,
                )

        if self.projection:
            self.projection = ChannelMLP(
                in_channels=hidden_variable_codimension,
                out_channels=output_variable_codimension,
                hidden_channels=projection_variable_codimension,
                n_layers=2,
                n_dim=self.n_dim,
            )
        else:
            self.projection = None

        if enable_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(
                    1, hidden_variable_codimension, *self.n_modes[0], dtype=torch.cfloat
                )
            )

        if use_positional_encoding:
            self.positional_encoding = nn.ParameterDict()
            for i in self.variable_ids:
                self.positional_encoding[i] = nn.Parameter(
                    torch.randn(
                        1,
                        positional_encoding_dim,
                        *self.positional_encoding_modes,
                        dtype=torch.cfloat,
                    )
                )

    def _extend_positional_encoding(self, new_var_ids):
        """
        Add variable specific positional encoding for new variables. This function is required
        while adapting a pre-trained model to a new dataset/PDE with additional new variables.

        Parameters
        ----------
        new_var_ids : list[str]
            IDs of the new variables to add positional encoding.
        """
        for i in range(new_var_ids):
            self.positional_encoding[i] = nn.Parameter(
                torch.randn(
                    1,
                    self.positional_encoding_dim,
                    *self.positional_encoding_modes,
                    dtype=torch.cfloat,
                )
            )

        self.variable_ids += new_var_ids

    def _get_positional_encoding(self, x, input_variable_ids):
        encoding_list = []
        for i in input_variable_ids:
            encoding_list.append(
                torch.fft.irfftn(self.positional_encoding[i], s=x.shape[-self.n_dim :])
            )

        return torch.stack(encoding_list, dim=1)

    def _get_cls_token(self, x):
        cls_token = torch.fft.irfftn(self.cls_token, s=x.shape[-self.n_dim :])
        repeat_shape = [1 for _ in x.shape]
        repeat_shape[0] = x.shape[0]
        cls_token = cls_token.repeat(*repeat_shape)
        return cls_token

    def _extend_variables(self, x, static_channel, input_variable_ids):
        x = x.unsqueeze(2)
        if static_channel is not None:
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[1] = x.shape[1]
            static_channel = static_channel.unsqueeze(1).repeat(*repeat_shape)
            x = torch.cat([x, static_channel], dim=2)
        if self.use_positional_encoding:
            positional_encoding = self._get_positional_encoding(x, input_variable_ids)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat([x, positional_encoding.repeat(*repeat_shape)], dim=2)
        return x

    def forward(self, x: torch.Tensor, static_channel=None, input_variable_ids=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W, ...)
        static_channel : torch.Tensor
            static channel tensor of shape (batch_size, static_channel_dim, H, W, ...)
            These channels provide additional information regarding the physical setup of the system.
            Must be provided when `static_channel_dim > 0`.
        input_variable_ids : list[str]
            The names of the variables corresponding to the channels of input 'x'.
            This parameter is required when `use_positional_encoding=True`.

            For example, if input x represents and snapshot of the velocity field of a fluid flow, the variable_ids=['u_x', 'u_y'].
            The variable_ids must be in the same order as the channels in the input tensor 'x', i.e., variable_ids[0] corresponds to the
            first channel of 'x', i.e., x[:, 0, ...].

        Returns
        -------
        torch.Tensor
            output tensor of shape (batch_size, output_variable_codimension*num_inp_var, H, W, ...)
        """
        batch, num_inp_var, *spatial_shape = (
            x.shape
        )  # num_inp_var is the number of channels in the input

        # input validation
        if (
            self.static_channel_dim > 0
            and static_channel is None
            and static_channel.shape[1] != self.static_channel_dim
        ):
            raise ValueError(
                f"Epected static channel dimension is {self.static_channel_dim}, but got {static_channel.shape[1]}"
            )
        if self.use_positional_encoding:
            assert (
                input_variable_ids is not None
            ), "variable_ids are not provided for the input"
            assert x.shape[1] == len(
                input_variable_ids
            ), f"Expected number of variables in input is {len(input_variable_ids)}, but got {x.shape[1]}"

        # position encoding and static channels are concatenated with the input
        # variables
        x = self._extend_variables(x, static_channel, input_variable_ids)

        # input variables are lifted to a higher-dimensional space
        if self.lifting:
            x = x.reshape(
                batch * num_inp_var, self.extended_variable_codimemsion, *spatial_shape
            )
            x = self.lifting(x)
            x = x.reshape(
                batch, num_inp_var * self.hidden_variable_codimension, *spatial_shape
            )

        # getting the learnable CLASS token
        if self.enable_cls_token:
            cls_token = self._get_cls_token(x)
            x = torch.cat(
                [
                    cls_token,
                    x,
                ],
                dim=1,
            )
            num_inp_var += 1

        # zero padding the domain of the input
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        # calculating the output shape
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling)
        ]

        # forward pass through the Codomain Attention layers
        skip_outputs = {}
        for layer_idx in range(self.n_layers):

            if (
                self.horizontal_skips_map is not None
                and layer_idx in self.horizontal_skips_map.keys()
            ):
                # `horizontal skip connections`
                # tokens from skip connections are concatenated with the
                # current token and then linearly projected
                # to the `hidden_variable_codimension`
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                resolution_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                resolution_scaling_factors = resolution_scaling_factors[
                    -1 * self.n_dim :
                ]
                t = resample(
                    skip_val,
                    resolution_scaling_factors,
                    list(range(-self.n_dim, 0)),
                    output_shape=x.shape[-self.n_dim :],
                )
                x = x.reshape(
                    batch * num_inp_var,
                    self.hidden_variable_codimension,
                    *x.shape[-self.n_dim :],
                )
                t = t.reshape(
                    batch * num_inp_var,
                    self.hidden_variable_codimension,
                    *t.shape[-self.n_dim :],
                )
                x = torch.cat([x, t], dim=1)
                x = self.skip_map_module[str(layer_idx)](x)
                x = x.reshape(
                    batch,
                    num_inp_var * self.hidden_variable_codimension,
                    *x.shape[-self.n_dim :],
                )

            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape
            else:
                cur_output_shape = None

            x = self.base[layer_idx](x, output_shape=cur_output_shape)

            # storing the outputs for skip connections
            if (
                self.horizontal_skips_map is not None
                and layer_idx in self.horizontal_skips_map.values()
            ):
                skip_outputs[layer_idx] = x.clone()

        # removing the padding
        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # projecting the hidden variables to the output variables
        if self.projection:
            x = x.reshape(
                batch * num_inp_var,
                self.hidden_variable_codimension,
                *x.shape[-self.n_dim :],
            )
            x = self.projection(x)
            x = x.reshape(
                batch,
                num_inp_var * self.output_variable_codimension,
                *x.shape[-self.n_dim :],
            )
        else:
            return x

        # discarding the CLASS token
        if self.enable_cls_token:
            x = x[:, self.output_variable_codimension :, ...]
        return x
