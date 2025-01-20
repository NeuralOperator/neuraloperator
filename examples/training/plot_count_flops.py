"""
Using `torchtnt` to count FLOPS
================================

In this example, we demonstrate how to use torchtnt to estimate the number of floating-point
operations per second (FLOPS) required for a model's forward and backward pass. 

We will use the FLOP computation to compare the resources used by a base FNO.

"""

# %%
# 
from copy import deepcopy
import torch
from torchtnt.utils.flops import FlopTensorDispatchMode

from neuralop.models import FNO

device = 'cpu'

fno = FNO(n_modes=(64,64), 
          in_channels=1, 
          out_channels=1, 
          hidden_channels=64, 
          projection_channel_ratio=1)

batch_size = 4
model_input = torch.randn(batch_size, 1, 128, 128)


with FlopTensorDispatchMode(fno) as ftdm:
    # count forward flops
    res = fno(model_input).mean()
    fno_forward_flops = deepcopy(ftdm.flop_counts)
    
    ftdm.reset()
    res.backward()
    fno_backward_flops = deepcopy(ftdm.flop_counts)
# %%
# This output is organized as a defaultdict object that counts the FLOPS used in each submodule. 
print(fno_forward_flops)
# %%
# To check the maximum FLOPS used during the forward pass, let's create a recursive function to search the nested dict:
from collections import defaultdict
def get_max_flops(flop_count_dict, max_value = 0):
    for _, value in flop_count_dict.items():
        # if not nested, compare leaf value to max
        if isinstance(value, int):
            max_value = max(max_value, value)
        
        # otherwise compute recursive max value below node
        elif isinstance(value, defaultdict):
            new_val = get_max_flops(value, max_value)
            max_value = max(max_value, new_val)
    return max_value

print(f"Max FLOPS required for FNO.forward: {get_max_flops(fno_forward_flops)}")
print(f"Max FLOPS required for FNO.backward: {get_max_flops(fno_backward_flops)}")
# %%
#