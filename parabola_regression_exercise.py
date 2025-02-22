import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set_theme('notebook', style='whitegrid')
# Torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# Libraries for data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
# Metrics
from torchmetrics import MeanAbsoluteError

# Set random seed for reproducibility
torch.manual_seed(42)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

if torch.cuda.is_available():
    x = torch.ones(1, device=device)
    print(x)
    
    # GPU operations have a separate seed we also want to set
    torch.cuda.manual_seed(42)
    
    # Some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Print CUDA availability and version
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

print("Gpu check: completed")
print("Starting the regression exercise")
# Consider the following dataset
x, y = make_regression(n_samples=1000, n_features=1, noise=0.2)
y = np.power(y,2)
plt.plot(x, y, 'x')
plt.show()
print("Data visualization: completed")

print("Starting the data preprocessing")
# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
print("Splitting the data: completed")

print("Starting the data check and transformation")
# Transform the data to PyTorch tensors
x_tor = torch.from_numpy(x_train).float()
y_tor = torch.from_numpy(y_train).float()


# Check if tensor contains NaN values
if torch.isnan(x_tor).any() or torch.isnan(y_tor).any():
    print("Tensor contains NaN values")
else:
    print("Tensor does not contain NaN values")
print("Data check and transformation: completed")

print("Starting the model definition")
# Define the model
class MLPRegressor(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_inputs, 64),  # Increase neurons
            nn.ReLU(),
            nn.Linear(64, 64),  # Another hidden layer
            nn.ReLU(),
            nn.Linear(64, 32),  # Decreasing neurons progressively
            nn.ReLU(),
            nn.Linear(32, 1)  # Output layer
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

model = MLPRegressor(1).to(device)
print(model)
print("Model definition: completed")

print("Creating the dataset and dataloader")
# Create the dataset class
class RegressionData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Create the DataLoader
X = x_tor.reshape(-1,1)
train_data = RegressionData(X, y_tor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

print("Dataset and dataloader creation: completed")

print("Starting the training")
# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define the training loop
def train_model(model, optimizer, data_loader, loss_function, num_epochs=200):
    # Set model to train mode
    model.train() 
    
    # Training loop
    for epoch in range(num_epochs):
        for data_inputs, data_labels in data_loader:
            
            ## Step 1: Move input data to device
            data_inputs = data_inputs.to(device)
            data_labels = data_labels.to(device)
            
            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1], we flatten it to [Batch size]
            
            ## Step 3: Calculate the loss
            loss = loss_function(preds, data_labels)
            
            # Check if loss is NaN
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch}. Stopping training.")
                return

            ## Step 4: Perform backpropagation
            # Before calculating the gradients, we need to ensure that they are all zero
            # The gradients would not be overwritten, but actually added to the existing ones
            optimizer.zero_grad() 
            # Perform backpropagation
            loss.backward()
            
            ## Step 5: Update the parameters
            optimizer.step()

        # Give some feedback after each 5th pass through the data
        if epoch % 10 == 0:
            print(f"loss: {loss}")

# Train the model
train_model(model, optimizer, train_loader, loss_fn, num_epochs=200)

print("Training: completed")

print("Starting the validation")
# Make predictions on the validation set
model.eval()
X_val_tor = torch.from_numpy(x_val).float()
y_val_tor = torch.from_numpy(y_val).float()

with torch.no_grad():
    x_val_tor = X_val_tor.to(device)
    preds = model(x_val_tor)

# Evaltuate the model
preds = preds.squeeze(dim=1).cpu()
x_val_tor = x_val_tor.cpu()
y_val_tor = y_val_tor.cpu()
mae = MeanAbsoluteError()
print(mae(preds, y_val_tor))

print("Validation: completed")

print("Starting the results visualization")
# Plot the results
plt.plot(x_val, y_val, 'x', label='Ground truth')
plt.plot(x_val, preds, 'o', label='Predictions')
plt.legend()
plt.show()
print("Results visualization: completed")