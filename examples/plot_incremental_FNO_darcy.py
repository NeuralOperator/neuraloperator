"""
Training a neural operator on Darcy-Flow - Author Robert Joseph
========================================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package on Incremental FNO
"""

# %%
# 

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
device = 'cpu'

 
# %%
# Loading the Navier-Stokes dataset in 128x128 resolution
train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)

# %%
# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Set up the incremental FNO model
# We start with 2 modes in each dimension
# We choose to update the modes by the incremental gradient explained algorithm
starting_modes = (2, 2)
incremental = True

# %%
# set up model
model = FNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, incremental_n_modes=starting_modes)
model = model.to(device)
n_params = count_params(model)

# %%
# Set up the optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# %%
# Set up the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)
train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}
print('\n### N PARAMS ###\n', n_params)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print('\n### INCREMENTAL GRADIENT EXPLAINED ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()

# %%
# Set up the trainer
# other options include setting incremental_loss_gap = True
# If one wants to use incremental resolution set it to True
# In this example we only update the modes and not the resolution
trainer = Trainer(model, n_epochs=20, device=device, mg_patching_levels=0, wandb_log=False, log_test_interval=3, use_distributed=False, verbose=True, incremental = incremental, dataset_name="SmallDarcy")

# %% 
# Train the model
trainer.train(train_loader, test_loaders, output_encoder, model, optimizer, scheduler, regularizer=False, training_loss=train_loss, eval_losses=eval_losses)

# %%
# Check that the number of modes has dynamically increased (Atleast for these settings on this dataset it should increase)
assert model.convs.incremental_n_modes > starting_modes

# %%
# Plot the prediction, and compare with the ground-truth 
# Note that we trained on a very small resolution for
# a very small number of epochs
# In practice, we would train at larger resolution, on many more samples.
# 
# However, for practicity, we created a minimal example that
# i) fits in just a few Mb of memory
# ii) can be trained quickly on CPU
#
# In practice we would train a Neural Operator on one or multiple GPUs

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()