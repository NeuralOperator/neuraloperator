import torch
from tensorly.decomposition import tucker
from tensorly import tenalg 
from torch.utils.checkpoint import checkpoint

class TensorGaLoreProjector:
    """TensorGaLoreProjector implements low-rank gradient projection [1]_
    for higher-order tensors as described in [2]_. The original tensor is projected
    into a low-rank subspace using low-rank mode-wise factors obtained by Tucker decomposition.
    The parameters are optimized in this space to save memory and are then projected back into
    the full-rank space for use in a model. 

        Parameters
        ----------
        rank : float, int or int tuple
            Goal rank of the transformed gradient tensor.
            Either a float corresponding to a percentage of params
            to preserve, or an list of int ranks corresponding to
            each mode of the tensor. If a single int is given, it is used 
            for all modes.
        update_proj_gap : int, optional
            How often (number of iterations) to update the projection
            tensor used to project gradients to low rank, by default 200
        scale : float, optional
            An additional LR-like scalar factor by which the learned
            low-rank gradients are multiplied after projection back to full rank
            before being added to the original weights, by default 1.0
        warm_restart : bool, optional
            If True, uses the last computed projection tensor as the starting point
            for High-Order Iteration (HOI) computation of the new projection tensor,
            by default False
        activation_checkpoint : bool, optional
            Whether to use activation checkpointing for Tucker decomposition
            for maximum memory savings, by default False
        
        References
        ----------
        .. _[1] : Zhao, J. et al. (2024). GaLore: Memory-Efficient LLM Training
            by Gradient Low-Rank Projection. ICML 2024, https://arxiv.org/abs/2403.03507.
        
        .. _[2] : George, R.J. et al. (2024). Tensor-GaLore: Memory-Efficient
            Training via Gradient Tensor Decomposition. NeurIPS 2024 OPT workshop, 
            https://openreview.net/pdf?id=sBaUZzZXJN.
        """
    def __init__(self, rank, 
                 update_proj_gap: int=200, 
                 scale: float=1.0, 
                 tucker_n_iter_max: int=10,
                 warm_restart: bool=False,
                 activation_checkpoint=False, 
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.warm_restart = warm_restart
        self.tucker_n_iter_max = tucker_n_iter_max
        self.activation_checkpoint = activation_checkpoint

        self.proj_tensor = None

        
    def project(self, full_rank_grad, iter):
        if self.proj_tensor is None and iter % self.update_proj_gap == 0:                    
            self.proj_tensor = self.get_projection_tensor(full_rank_grad, self.rank) 
        self.proj_tensor = [factor.to(full_rank_grad.device) for factor in self.proj_tensor]
        transformed_low_rank = self.transform(self.proj_tensor, full_rank_grad)
        return transformed_low_rank

    def project_back(self, low_rank_grad):
        full_rank_grad = self.inverse_transform(self.proj_tensor, low_rank_grad)     
        return full_rank_grad * self.scale
            
    # Tucker decomp: higher-order SVD
    def get_projection_tensor(self, weights, rank):
        if torch.is_complex(weights.data) and weights.data.dtype != torch.cfloat:
            matrix = weights.data.cfloat()
        else:
            matrix = weights.data

        # if warm_restart is True, initialize with 
        # existing projection tensor if it exists
        if self.warm_restart and self.proj_tensor is not None:
            init = self.proj_tensor
        else:
            init = 'svd' # default setting
        if self.activation_checkpoint:
            core, factors = checkpoint(tucker, matrix, rank, init)
        else:
            core, factors = tucker(matrix, rank=rank, init=init)
        del core
        return factors

    def transform(self, factors, x):
        # unpack/drop core
        if self.activation_checkpoint:
            return checkpoint(tenalg.multi_mode_dot, x, factors, Transpose=True)
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        if self.activation_checkpoint:
            return checkpoint(tenalg.multi_mode_dot, x, factors)
        return tenalg.multi_mode_dot(x, factors)
            

