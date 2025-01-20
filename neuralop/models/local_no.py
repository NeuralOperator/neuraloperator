from typing import Tuple, List, Union

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from neuralop.layers.local_no_block import LocalNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel

class LocalNO(BaseModel, name='LocalNO'):
    """N-Dimensional Local Fourier Neural Operator. The LocalNO shares
    its forward pass and architecture with the standard FNO, with the key difference
    that its Fourier convolution layers are replaced with LocalNOBlocks that place
    differential kernel layers and local integral layers in parallel to its 
    Fourier layers as detailed in [1]_.

    Parameters
    ----------
    n_modes : Tuple[int]
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the Local NO is inferred from ``len(n_modes)``
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        width of the Local NO (i.e. number of channels), by default 256
    default_in_shape : Tuple[int]
        Default input shape on spatiotemporal dimensions for structured DISCO convolutions
    n_layers : int, optional
        Number of Local NO block Layers, by default 4
    disco_layers : bool or bool list, optional
        Must be same length as n_layers, dictates whether to include a
        local integral kernel parallel connection at each layer. If a single
        bool, shared for all layers.
    disco_kernel_shape: Union[int, List[int]], optional
        kernel shape for local integral. Expects either a single integer for isotropic kernels or two integers for anisotropic kernels
    domain_length: torch.Tensor, optional
        extent/length of the physical domain. Assumes square domain [-1, 1]^2 by default
    disco_groups: int, optional
        number of groups in the local integral convolution, by default 1
    disco_bias: bool, optional
        whether to use a bias for the integral kernel, by default True
    radius_cutoff: float, optional
        cutoff radius (with respect to domain_length) for the local integral kernel, by default None
    diff_layers : bool or bool list, optional
        Must be same length as n_layers, dictates whether to include a
        differential kernel parallel connection at each layer. If a single
        bool, shared for all layers.
    conv_padding_mode : str in ['periodic', 'circular', 'replicate', 'reflect', 'zeros'], optional
        Padding mode for spatial convolution kernels.
    fin_diff_kernel_size : odd int, optional
        Conv kernel size for finite difference convolution.
    mix_derivatives : bool, optional
        Whether to mix derivatives across channels.
    
    Other parameters
    ------------------
    lifting_channel_ratio : int, optional
        ratio of lifting channels to hidden_channels, by default 2
        The number of liting channels in the lifting block of the Local FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 512)
    projection_channel_ratio : int, optional
        ratio of projection channels to hidden_channels, by default 2
        The number of projection channels in the projection block of the Local FNO is
        projection_channel_ratio * hidden_channels (e.g. default 512)
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the Local FNO. Defaults to "grid"

        * If "grid", appends a grid positional embedding with default settings to 
        the last channels of raw input. Assumes the inputs are discretized
        over a grid with entry [0,0,...] at the origin and side lengths of 1.

        * If an initialized GridEmbedding module, uses this module directly
        See :mod:`neuralop.embeddings.GridEmbeddingND` for details.

        * If None, does nothing

    non_linearity : nn.Module, optional
        Non-Linear activation function module to use, by default F.gelu
    norm : str {"ada_in", "group_norm", "instance_norm"}, optional
        Normalization layer to use, by default None
    complex_data : bool, optional
        Whether data is complex-valued (default False)
        if True, initializes complex-valued modules.
    channel_mlp_dropout : float, optional
        dropout parameter for ChannelMLP in FNO Block, by default 0
    channel_mlp_expansion : float, optional
        expansion parameter for ChannelMLP in FNO Block, by default 0.5
    channel_mlp_skip : str {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in channel-mixing mlp, by default 'soft-gating'
    local_no_skip : str {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use in FNO layers, by default 'linear'
    resolution_scaling_factor : Union[Number, List[Number]], optional
        layer-wise factor by which to scale the domain resolution of function, by default None
        
        * If a single number n, scales resolution by n at each layer

        * if a list of numbers [n_0, n_1,...] scales layer i's resolution by n_i.
    domain_padding : Union[Number, List[Number]], optional
        If not None, percentage of padding to use, by default None
        To vary the percentage of padding used along each input dimension,
        pass in a list of percentages e.g. [p1, p2, ..., pN] such that
        p1 corresponds to the percentage of padding along dim 1, etc.
    domain_padding_mode : str {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    local_no_block_precision : str {'full', 'half', 'mixed'}, optional
        precision mode in which to perform spectral convolution, by default "full"
    stabilizer : str {'tanh'} | None, optional
        whether to use a tanh stabilizer in FNO block, by default None

        Note: stabilizer greatly improves performance in the case
        `local_no_block_precision='mixed'`. 

    max_n_modes : Tuple[int] | None, optional

        * If not None, this allows to incrementally increase the number of
        modes in Fourier domain during training. Has to verify n <= N
        for (n, m) in zip(max_n_modes, n_modes).

        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the Local FNO layer weights to use, by default None.

        * If None, a dense tensor parametrizes the Spectral convolutions

        * Otherwise, the specified tensor factorization is used.
    rank : float, optional
        tensor rank to use in above factorization, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : str {'factorized', 'reconstructed'}, optional

        * If 'factorized', implements tensor contraction with the individual factors of the decomposition 
        
        * If 'reconstructed', implements with the reconstructed full tensorized weight.
    decomposition_kwargs : dict, optional
        extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`), by default dict()
    conv_module : nn.Module, optional
        module to use for FNOBlock's convolutions, by default SpectralConv
    
    Examples
    ---------
    
    >>> from neuralop.models import LocalNO
    >>> model = LocalNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (local_no_blocks): LocalNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    -----------
    .. [1] Liu-Schiaffini M., Berner J., Bonev B., Kurth T., Azizzadenesheli K., Anandkumar A.;
        "Neural Operators with Localized Integral and Differential Kernels" (2024).  
        ICML 2024, https://arxiv.org/pdf/2402.16845.

    """

    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        default_in_shape,
        n_layers: int=4,
        disco_layers: Union[bool, List[bool]]=True,
        disco_kernel_shape :List[int]=[2,4],
        radius_cutoff: bool=None,
        domain_length: List[int]=[2,2],
        disco_groups: int=1,
        disco_bias: bool=True,
        diff_layers: Union[bool, List[bool]]=True,
        conv_padding_mode: str='periodic',
        fin_diff_kernel_size: int=3,
        mix_derivatives: bool=True,
        lifting_channel_ratio: int=2,
        projection_channel_ratio: int=2,
        positional_embedding: Union[str, nn.Module]="grid",
        non_linearity: nn.Module=F.gelu,
        norm: str=None,
        complex_data: bool=False,
        channel_mlp_dropout: float=0,
        channel_mlp_expansion: float=0.5,
        channel_mlp_skip: str="soft-gating",
        local_no_skip: str="linear",
        resolution_scaling_factor: Union[Number, List[Number]]=None,
        domain_padding: Union[Number, List[Number]]=None,
        domain_padding_mode: str="one-sided",
        local_no_block_precision: str="full",
        stabilizer: str=None,
        max_n_modes: Tuple[int]=None,
        factorization: str=None,
        rank: float=1.0,
        fixed_rank_modes: bool=False,
        implementation: str="factorized",
        decomposition_kwargs: dict=dict(),
        separable: bool=False,
        preactivation: bool=False,
        conv_module: nn.Module=SpectralConv,
        **kwargs
    ):
        
        super().__init__()
        self.n_dim = len(n_modes)
        
        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in local_no blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = lifting_channel_ratio * self.hidden_channels

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = projection_channel_ratio * self.hidden_channels

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.local_no_skip = (local_no_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.local_no_block_precision = local_no_block_precision
        
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0., 1.]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(in_channels=self.in_channels,
                                                        dim=self.n_dim, 
                                                        grid_boundaries=spatial_grid_boundaries)
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(f'Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}')
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding == None:
            self.positional_embedding = None
        else:
            raise ValueError(f"Error: tried to instantiate positional embedding with {positional_embedding},\
                              expected one of \'grid\', GridEmbeddingND")
        
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode
        self.complex_data = self.complex_data

        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        self.local_no_blocks = LocalNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            default_in_shape=default_in_shape,
            resolution_scaling_factor=resolution_scaling_factor,
            disco_layers=disco_layers,
            disco_kernel_shape=disco_kernel_shape,
            radius_cutoff=radius_cutoff,
            domain_length=domain_length,
            disco_groups=disco_groups,
            disco_bias=disco_bias,
            diff_layers=diff_layers,
            conv_padding_mode=conv_padding_mode,
            fin_diff_kernel_size=fin_diff_kernel_size,
            mix_derivatives=mix_derivatives,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            local_no_skip=local_no_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            local_no_block_precision=local_no_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs
        )
        
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, **kwargs):
        """FNO's forward pass
        
        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent
            space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Local NO layers in sequence (Differential + optional DISCO + skip connections, nonlinearity) 

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor
        
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            
            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)
        
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.local_no_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.local_no_blocks.n_modes = n_modes
        self._n_modes = n_modes