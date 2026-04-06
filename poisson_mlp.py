# %% [markdown]
# # Learning Neural Operators: A Example of DeepONet
#
# Unlike traditional solvers that solve a PDE for a specific parameter set, SciML uses Neural Operators to learn the general mapping from input functions (e.g., source terms) to solution functions.
#
# ## 1. Problem Introduction: Steady-State Heat Equation
#
# We approximate the solution to the Poisson equation on $\Omega = [0, 1]^2$:
#
# $$\kappa \nabla^2 u(x, y) + u_0(x, y) = 0, \quad u|_{\partial \Omega} = 0$$
#
# - **Input ($u_0$):** Source term (Gaussian Random Field).
# - **Output ($u$):** Temperature field.
# - **Parameters:** $\kappa = 0.01$.

# %% [markdown]
# ## 2. Environment Setup and Data Loading
#
# We mount Google Drive to access the dataset and set the working directory to the webinar folder.

# %%
from torch.utils.data import DataLoader, TensorDataset
import time
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from google.colab import drive
import os
import sys

# 1. Mount the user's Google Drive
drive.mount('/content/drive')

# 2. Define the path to the shared webinar folder
# Note: If the trainee named the shortcut differently, they change this string.
WEBINAR_FOLDER = '/content/drive/MyDrive/NCSA_sciML_webinar_2026'

# 3. Change the working directory so we can use relative paths like './data/...'
if os.path.exists(WEBINAR_FOLDER):
    os.chdir(WEBINAR_FOLDER)
    print(f"Successfully moved to: {os.getcwd()}")
else:
    print("Error: Folder not found. Did you add the shortcut to your Drive?")

# 4. Verify data is visible
!ls ./data/

# %% [markdown]
# We load the training and testing datasets, which consist of source terms ($u_0$), temperature fields ($s$), and the spatial grid coordinates ($xy$).

# %%
# Load the datasets
num_samples = 1000  # use a subset of the data for faster training in this demo, max=5000
u0_train = np.load(
    './data/data_u0_train_ht.npy').astype(np.float32)[:num_samples]
u0_testing = np.load('./data/data_u0_testing_ht.npy').astype(np.float32)
s_train = np.load(
    './data/data_s_train_ht.npy').astype(np.float32)[:num_samples]
s_testing = np.load('./data/data_s_testing_ht.npy').astype(np.float32)
xy_coords = np.load('./data/xy_train_test_ht.npy').astype(np.float32)

print(f"Input (u0) Shape: {u0_train.shape}")   # (num_samples, num_grid_points)
print(f"Output (s) Shape: {s_train.shape}")    # (num_samples, num_grid_points)
print(f"Coordinate Shape: {xy_coords.shape}")  # (num_grid_points, 2)

# %% [markdown]
# Visualization of a sample source term and its corresponding solution from the training set:

# %%
# Visualize one example from the training data
sample_idx = 0

# Reshape the data to 2D grid (128x128 based on 16384 points)
grid_size = int(np.sqrt(u0_train.shape[1]))
u0_sample = u0_train[sample_idx].reshape(grid_size, grid_size)
s_sample = s_train[sample_idx].reshape(grid_size, grid_size)
x_grid = xy_coords[:, 0].reshape(grid_size, grid_size)
y_grid = xy_coords[:, 1].reshape(grid_size, grid_size)


# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Plot input source term
im1 = axes[0].contourf(x_grid, y_grid, u0_sample,
                       cmap='viridis', origin='lower')
axes[0].set_title('Input Source Term $u_0(x, y)$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0])

# Plot solution
im2 = axes[1].contourf(x_grid, y_grid, s_sample,
                       cmap='viridis', origin='lower')
axes[1].set_title('Solution (Temperature) $u(x, y)$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. DeepONet Architecture
#
# High-level structure:
# - **Branch Net**: Encodes $u_0$ (evaluated at grid points) into a feature vector.
# - **Trunk Net**: Encodes coordinates $(x, y)$ into a feature vector.
# - **Output**: Dot product of branch and trunk features plus a bias: $G(u_0)(x, y) \approx \sum_{k=1}^{p} b_k(u_0) \cdot t_k(x, y) + b_0$.
#
# In this implementation, both sub-networks use fully connected layers (MLPs) with ReLU activations.
#
# %% [markdown]
# ### DeepONet Implementation Detail
#
# The `DeepONet` class handles the dual network passes:
# 1. **Branch**: Maps `branch_dim` (16384 grid points) to `latent_dim` (128) via multiple hidden layers (default: 5 layers of 100).
# 2. **Trunk**: Maps `trunk_dim` (2D coordinates) to `latent_dim` (128) via similar MLP structure.
# 3. **Forward**: Uses `torch.einsum` to compute the batch-wise inner product between branch and trunk signals.

# %%


class DeepONet(nn.Module):
    def __init__(self, branch_dim, trunk_dim, latent_dim, br_hid_layers=[100]*5, tr_hid_layers=[100]*5):
        super(DeepONet, self).__init__()

        # Branch Network: Takes u0(x, y) values
        branch_layers = []
        input_dim = branch_dim
        for hidden_dim in br_hid_layers:
            branch_layers.append(nn.Linear(input_dim, hidden_dim))
            branch_layers.append(nn.ReLU())
            input_dim = hidden_dim
        branch_layers.append(nn.Linear(input_dim, latent_dim))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk Network: Takes (x, y) coordinates
        trunk_layers = []
        input_dim = trunk_dim
        for hidden_dim in tr_hid_layers:
            trunk_layers.append(nn.Linear(input_dim, hidden_dim))
            trunk_layers.append(nn.ReLU())
            input_dim = hidden_dim
        trunk_layers.append(nn.Linear(input_dim, latent_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u0, xy):
        # u0: [batch, branch_dim]
        # xy: [N, 2] (coordinates)
        branch_out = self.branch(u0)  # [batch, latent_dim]
        trunk_out = self.trunk(xy)  # [N, latent_dim]

        # Dot product of the two outputs
        res = torch.einsum('bi,ni->bn', branch_out, trunk_out) + self.bias
        return res


don_model = DeepONet(branch_dim=16384, trunk_dim=2, latent_dim=128)

# %% [markdown]
# ## 4. Data Preparation
#
# We normalize the source terms and solutions using training set statistics and create `DataLoader` instances with batch sizes 16 (train) and 128 (test).

# %%
u0_mean, u0_std = np.mean(u0_train), np.std(u0_train)
s_mean, s_std = np.mean(s_train), np.std(s_train)
u0_train_norm = (u0_train - u0_mean) / u0_std
s_train_norm = (s_train - s_mean) / s_std
u0_testing_norm = (u0_testing - u0_mean) / u0_std
s_testing_norm = (s_testing - s_mean) / s_std


# %%
# [num_samples, num_grid_points]
u0_train_norm = torch.from_numpy(u0_train_norm)
# [num_samples, num_grid_points]
s_train_norm = torch.from_numpy(s_train_norm)
xy_tensor = torch.from_numpy(xy_coords)  # [num_grid_points, 2]
# [num_samples, num_grid_points]
u0_testing_norm = torch.from_numpy(u0_testing_norm)
# [num_samples, num_grid_points]
s_testing_norm = torch.from_numpy(s_testing_norm)
xy_tensor = torch.from_numpy(xy_coords)  # [num_grid_points, 2]
train_dataset = TensorDataset(u0_train_norm, s_train_norm)
testing_dataset = TensorDataset(u0_testing_norm, s_testing_norm)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
testing_loader = DataLoader(testing_dataset, batch_size=128, shuffle=False)

# %% [markdown]
# ## 5. Training Loop
#
# We optimize the model using the Adam optimizer ($lr=5 \times 10^{-4}$) and MSE loss. The training runs for 500 epochs, evaluating test loss at each step.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to the appropriate device (GPU if available, otherwise CPU)
don_model = don_model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(don_model.parameters(), lr=5e-4)


# %%
# Move coordinates to the same device as the model
xy_tensor = xy_tensor.to(device)
epochs = 500
history = {"train_loss": [], "test_loss": []}
for epoch in range(epochs):
    don_model.train()
    total_loss = 0
    s_timep = time.time()
    for u0_batch, s_batch in train_loader:
        u0_batch = u0_batch.to(device)
        s_batch = s_batch.to(device)

        optimizer.zero_grad()
        s_pred = don_model(u0_batch, xy_tensor)  # Predict for the entire grid
        # Compute loss against the true solution
        loss = criterion(s_pred, s_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    test_loss = 0
    don_model.eval()
    for u0_batch, s_batch in testing_loader:
        u0_batch = u0_batch.to(device)
        s_batch = s_batch.to(device)
        with torch.no_grad():
            s_pred = don_model(u0_batch, xy_tensor)
            test_loss += criterion(s_pred, s_batch).item()
    e_timep = time.time()
    history["train_loss"].append(total_loss/len(train_loader))
    history["test_loss"].append(test_loss/len(testing_loader))
    print(f"Epoch {epoch+1}/{epochs} took {e_timep - s_timep:.2f}s, Loss: {total_loss/len(train_loader)}, Test Loss: {test_loss/len(testing_loader)}")

# %% [markdown]
# Visualizing the convergence of training and testing loss:

# %%
plt.figure(figsize=(4.8, 3.6))
plt.plot(history["train_loss"], label='Train Loss')
plt.plot(history["test_loss"], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.title('Training and Testing Loss Curves')
plt.legend()

# %% [markdown]
# ## 6. Performance Evaluation
#
# After training, we evaluate accuracy using the L2 relative error on the test dataset. We then visualize predictions for the best, median, and worst-performing samples to check for physical consistency.

# %%
# We first compute the predictions of the DeepONet on the test set
don_model.eval()  # Set the model to evaluation mode
pred_testing = don_model(u0_testing_norm.to(device), xy_tensor)
# We convert the normalized predictions back to the original scale
# We also move the prediction result to CPU for further analysis
pred_testing = pred_testing.cpu().detach().numpy() * s_std+s_mean

# %%
l2_error = np.linalg.norm(pred_testing - s_testing,
                          axis=1) / np.linalg.norm(s_testing, axis=1)
mean, std = np.mean(l2_error), np.std(l2_error)
print(f"Mean L2 relative error of test dataset: {mean}, std: {std}")
fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_ = ax.hist(l2_error*100, bins=20,
            color="skyblue", edgecolor="black")
ax.set_xlabel("L2 relative error (%)")
ax.set_ylabel("Frequency")
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# %%
idx_sorted = np.argsort(l2_error)

visual_samples = {"Best": idx_sorted[0], "25%": idx_sorted[len(idx_sorted) // 4],
                  "Median": idx_sorted[len(idx_sorted) // 2], "75%": idx_sorted[3 * len(idx_sorted) // 4],
                  "Worst": idx_sorted[-1]}

visu_key = "Best"  # change this key to visualize different samples
pred_sample = pred_testing[visual_samples[visu_key]]
true_sample = s_testing[visual_samples[visu_key]]
abs_error = np.abs(pred_sample-true_sample)

fig = plt.figure(figsize=(3*4, 3))
# best sample
ax = fig.add_subplot(1, 3, 1)
im_true = ax.contourf(x_grid, y_grid, true_sample.reshape(
    grid_size, grid_size), cmap='viridis', origin='lower')
ax.set_title('True Solution $u(x, y)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im_true, ax=ax)
ax = fig.add_subplot(1, 3, 2)
im_pred = ax.contourf(x_grid, y_grid, pred_sample.reshape(
    grid_size, grid_size), cmap='viridis', origin='lower')
ax.set_title('Predicted Solution $\\hat{u}(x, y)$')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im_pred, ax=ax)
ax = fig.add_subplot(1, 3, 3)
im = ax.contourf(x_grid, y_grid, abs_error.reshape(
    grid_size, grid_size), cmap='inferno', origin='lower')
ax.set_title('Absolute Error $|u - \\hat{u}|$')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax)
plt.tight_layout()
