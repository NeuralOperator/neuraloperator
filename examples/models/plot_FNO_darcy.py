"""
Training an FNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small `Darcy-Flow example <../auto_examples/plot_darcy_flow.html>`_ we ship with the package
to train a Fourier Neural Operator. 

Note that this dataset is much smaller than one we would use in practice. The small Darcy-flow is an example built to
be trained on a CPU in a few seconds, whereas normally we would train on one or multiple GPUs. 

"""

# %%
# 

import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import FNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

device = 'cpu'


# %%
# Let's load the small Darcy-flow dataset. 
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
)
data_processor = data_processor.to(device)


# %%
# We create a simple FNO model

model = FNO(n_modes=(16, 16),
             in_channels=1, 
             out_channels=1,
             hidden_channels=32, 
             projection_channel_ratio=2)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

# %%
# Training setup
# ----------------

# %%
#Create the optimizer
optimizer = AdamW(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


# %%
# Then create the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%
# Training the model
# ---------------------

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer:
trainer = Trainer(model=model, n_epochs=20,
                  device=device,
                  data_processor=data_processor,
                  wandb_log=False,
                  eval_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Then train the model on our small Darcy-Flow dataset:

trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

# %%
# .. _plot_preds :
# Visualizing predictions
# ------------------------
# Let's take a look at what our model's predicted outputs look like. 
# Again note that in this example, we train on a very small resolution for
# a very small number of epochs.
# In practice, we would train at a larger resolution, on many more samples.

test_samples = test_loaders[16].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
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

fig.suptitle('Inputs, ground-truth output and prediction (16x16).', y=0.98)
plt.tight_layout()
fig.show()


# %%
# .. zero_shot :
# Zero-shot super-evaluation
# ---------------------------
# In addition to training and making predictions on the same input size, 
# the FNO's invariance to the discretization of input data means we 
# can natively make predictions on higher-resolution inputs and get higher-resolution outputs.

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    data = data_processor.preprocess(data, batched=False)
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

fig.suptitle('Inputs, ground-truth output and prediction (32x32).', y=0.98)
plt.tight_layout()
fig.show()

# %%
# We only trained the model on data at a resolution of 16x16, and with no modifications 
# or special prompting, we were able to perform inference on higher-resolution input data 
# and get higher-resolution predictions! In practice, we often want to evaluate neural operators
# at multiple resolutions to track a model's zero-shot super-evaluation performance throughout 
# training. That's why many of our datasets, including the small Darcy-flow we showcased,
# are parameterized with a list of `test_resolutions` to choose from. 
#
# However, as you can see, these predictions are noisier than we would expect for a model evaluated 
# at the same resolution at which it was trained. Leveraging the FNO's discretization-invariance, there
# are other ways to scale the outputs of the FNO to train a true super-resolution capability. 