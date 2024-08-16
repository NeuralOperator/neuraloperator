from typing import List


import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import ChannelMLP, LinearChannelMLP
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch


class GNOBlock(nn.Module):
    """GNOBlock implements a Graph Neural Operator layer as described in [1]_.

    A GNO layer is a resolution-invariant operator that maps a function defined
    over one coordinate mesh to another defined over another coordinate mesh using 
    a pointwise kernel integral that takes contributions from neighbors of distance 1
    within a graph constructed via neighbor search with a specified radius. 

    The kernel integral computed in IntegralTransform 
    computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy
    
    Parameters
    ----------
    ## TODO @REVIEWERS: how should we define input channels when the GNO can proceed
    with just the coordinate dimension in linear kernels? 
    in_channels : int
        number of channels in input function. Only used if transform_type
        is (c) "nonlinear" or (d) "nonlinear_kernelonly"
    out_channels : int
        number of channels in output function
    coord_dim : int
        dimension of domain on which x and y are defined
    radius : float
        radius in which to search for neighbors
    use_open3d_neighbor_search : _type_, optional
        _description_, by default None
    channel_mlp : nn.Module, optional
        ChannelMLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        ChannelMLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the 
        kernel integral.
    channel_mlp_layers : List[int], optional
        list of layer widths to dynamically construct
        LinearChannelMLP network to parameterize kernel k, by default None
    channel_mlp_non_linearity : torch.nn function, optional
        activation function for ChannelMLPLinear above, by default F.gelu
    transform_type : str, optional
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b) [DEFAULT]
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_open3d_neighbor_search: bool, optional
    use_torch_scatter_reduce : bool, optional
        whether to reduce in integral computation using a function
        provided by the extra dependency torch_scatter or the slower
        native PyTorch implementation, by default True

    References
    -----------
    _[1]. Neural Operator: Graph Kernel Network for Partial Differential Equations.
        Zongyi Li, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, 
        Anima Anandkumar. ArXiV, 2020 
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 coord_dim: int,
                 radius: float,
                 n_layers: int=None,
                 channel_mlp: nn.Module=None,
                 channel_mlp_layers: List[int]=None,
                 channel_mlp_non_linearity=F.gelu,
                 transform_type="linear",
                 use_open3d_neighbor_search: bool=True,
                 use_torch_scatter_reduce=True,):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim

        self.radius = radius
        self.n_layers = n_layers

        # Create in-to-out nb search module
        if use_open3d_neighbor_search:
            assert self.coord_dim == 3, f"Error: open3d is only designed for 3d data, \
                GNO instantiated for dim={coord_dim}"
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d_neighbor_search)

        # create proper kernel input channel dim
        # if nonlinear of either type, add in_features dim
        # otherwise just add x and y dim
        kernel_in_dim = self.coord_dim * 2
        kernel_in_dim_str = "dim(y) + dim(x)"
        if transform_type == "nonlinear" or transform_type == "nonlinear_kernelonly":
            kernel_in_dim += self.in_channels
            kernel_in_dim_str += " + dim(f_y)"
        if channel_mlp:
            assert channel_mlp.in_channels == kernel_in_dim, f"Error: expected ChannelMLP to take\
                  input with {kernel_in_dim} channels (feature channels={kernel_in_dim_str}),\
                      got {channel_mlp.in_channels}."
            assert channel_mlp.out_channels == out_channels, f"Error: expected ChannelMLP to have\
                 {out_channels=} but got {channel_mlp.in_channels=}."
            self.channel_mlp = channel_mlp
        if channel_mlp_layers:
            if channel_mlp_layers[0] != kernel_in_dim:
                channel_mlp_layers = [kernel_in_dim] + channel_mlp_layers
            if channel_mlp_layers[-1] != self.out_channels:
                channel_mlp_layers.append(self.out_channels)
            self.channel_mlp = LinearChannelMLP(layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity)

        # Create integral transform module
        self.integral_transform = IntegralTransform(
            channel_mlp=self.channel_mlp,
            transform_type=transform_type,
            use_torch_scatter=use_torch_scatter_reduce
        )
    
    def forward(self, y, x, f_y=None, weights=None):
        """Compute a GNO neighbor search and kernel integral transform.

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
            If batched, these must remain constant
            over the whole batch so no batch dim is needed.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """
        
        neighbors_dict = self.neighbor_search(data=y, queries=x, radius=self.radius)
        out_features = self.integral_transform(y=y,
                                               x=x,
                                               neighbors=neighbors_dict,
                                               f_y=f_y)
        
        return out_features

