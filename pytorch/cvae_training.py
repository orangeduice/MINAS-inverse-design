import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import helper

# Assuming cvae_pytorch_mlp.py is in the same directory
from cvae_training import CVAE_MLP



# --- PyTorch Training Equivalents ---

class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops improving.
    Equivalent to keras.callbacks.EarlyStopping.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def calculate_beta(epoch, params):
    """
    Calculates the KL annealing beta weight for a given epoch.
    Equivalent to the KLAnnealingCallback.
    """
    if epoch < params['annealing_epochs']:
        if params['shape'] == 'linear':
            return params['start_beta'] + (params['end_beta'] - params['start_beta']) * (epoch / params['annealing_epochs'])
        elif params['shape'] == 'sigmoid':
            sigmoid_val = 1 / (1 + np.exp(-10 * (epoch / params['annealing_epochs'] - 0.5)))
            return params['start_beta'] + (params['end_beta'] - params['start_beta']) * sigmoid_val
    return params['end_beta']


def plot_history(history, save_path=None):
    """Plots the training and validation loss from the history dictionary."""
    plt.figure(figsize=(10, 6))
    
    ax1 = plt.gca()
    ax1.plot(history['train_loss'], label='train_loss')
    ax1.plot(history['val_loss'], label='val_loss')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log Loss')
    ax1.legend(loc='upper left')

    if 'beta' in history:
        ax2 = ax1.twinx()
        ax2.plot(history['beta'], label='beta', color='green', linestyle='--')
        ax2.set_ylabel('Beta')
        ax2.legend(loc='upper right')

    plt.title('Model Training History')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def save_experiment(tracker, model, hyperparameters, wavelengths, history, model_folder="models_pytorch"):
    """Logs experiment data and saves the PyTorch model state."""
    # Log metadata
    train_id = tracker.create_training_run(
        model_type="cVAE_PyTorch",
        hyperparameters=hyperparameters,
        wavelengths=wavelengths.tolist(),
        train_loss=history['train_loss'],
        val_loss=history['val_loss'],
    )

    # Save model state dictionary
    run_folder = os.path.join(model_folder, str(train_id))
    os.makedirs(run_folder, exist_ok=True)
    model_file = os.path.join(run_folder, "cvae_mlp.pt")
    torch.save(model.state_dict(), model_file)

    print(f"\nLogged new training run. Train ID: {train_id}")
    print(f"Experiment data saved in: {tracker.training_dir}/{train_id}.json")
    print(f"Model saved in: {run_folder}")


# --- NEW: Function to generate designs from a given spectral condition ---
def generate_designs(model, condition_slice, n_samples=1, device='cpu'):
    """
    Generates new designs using the decoder part of the cVAE.

    Args:
        model (CVAE_MLP): The trained cVAE model with weights loaded.
        condition_slice (torch.Tensor): A tensor representing the spectral slice of interest.
                                        Shape should be (num_wavelength_points, 1).
        n_samples (int): Number of designs to generate for the given condition.
        device (str): The device to run inference on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The generated design(s) as a tensor.
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        # Get latent dimension from the model's architecture
        latent_dim = model.encoder.fc_mean.out_features
        
        # The condition needs a batch dimension for the model, so we add one
        if condition_slice.dim() == 2:
            condition_slice = condition_slice.unsqueeze(0)
        
        # If generating multiple samples, repeat the condition for each
        condition_batch = condition_slice.repeat(n_samples, 1, 1).to(device)

        # Sample random latent vectors from a standard normal distribution
        z = torch.randn(n_samples, latent_dim).to(device)
        
        # Use the decoder to generate designs from the latent vectors and conditions
        generated_designs = model.decoder(z, condition_batch)
        
        return generated_designs


# %%

"""Main function to run the CVAE training pipeline with PyTorch."""
# --- Configuration ---
CVAE_HYPERPARAMS = {
    "latent_dim": 2,
}

TRAINING_PARAMS = {
    "learning_rate": 0.0002,
    "epochs": 100,
    "batch_size": 32,
    "reduce_lr_factor": 1/3,
    "reduce_lr_patience": 5,
    "early_stopping_patience": 10,
    "kl_annealing": {
        "start_beta": 1e-6,
        "end_beta": 0.001,
        "annealing_epochs": 25,
        "shape": "sigmoid"
    }
}

# --- NEW: Define the wavelength range of interest for the condition ---
GENERATION_PARAMS = {
    "wavelength_min": 500, # Example: 500nm
    "wavelength_max": 600, # Example: 600nm
}

DATA_PATH = 'datasets/pisa_data.h5'

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Data ---
print("Loading data...")
x_data, _, y_data, _, wavelengths = help.load_reflection_spectra_data(
    DATA_PATH, save_scalers=True
)

# --- NEW: Slice the condition data (spectrum) to the region of interest ---
print(f"Original condition shape: {y_data.shape}")
wl_min = GENERATION_PARAMS["wavelength_min"]
wl_max = GENERATION_PARAMS["wavelength_max"]

# Find the indices of the wavelengths within the desired range
interest_indices = np.where((wavelengths >= wl_min) & (wavelengths <= wl_max))[0]

if len(interest_indices) == 0:
    raise ValueError(f"Wavelength range [{wl_min}, {wl_max}] is not valid for the given data.")

# Use these indices to slice the spectral data (y_data)
y_data_sliced = y_data[:, interest_indices, :]
print(f"Sliced condition shape for wavelengths {wl_min}-{wl_max}nm: {y_data_sliced.shape}")

# Convert numpy arrays to torch tensors using the SLICED condition data
x_tensor = torch.from_numpy(x_data)
y_tensor = torch.from_numpy(y_data_sliced) # Use the sliced data!

# Create dataset and dataloaders
dataset = TensorDataset(x_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=TRAINING_PARAMS['batch_size'], shuffle=False)

print(f"x_train shape: {x_tensor.shape}")
print(f"y_train (sliced) shape: {y_tensor.shape}")

# --- Build Model ---
print("Building model...")
design_shape = x_tensor.shape[1:]
# The condition_shape is now taken from the sliced data
condition_shape = y_tensor.shape[1:]

model = CVAE_MLP(
    design_shape=design_shape,
    condition_shape=condition_shape,
    latent_dim=CVAE_HYPERPARAMS['latent_dim']
).to(device)

print(model)

# --- Setup Optimizer, Scheduler, and Early Stopping ---
optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_PARAMS['learning_rate'])
scheduler = ReduceLROnPlateau(
    optimizer,
    'min',
    factor=TRAINING_PARAMS['reduce_lr_factor'],
    patience=TRAINING_PARAMS['reduce_lr_patience'],
    verbose=True
)
early_stopper = EarlyStopping(
    patience=TRAINING_PARAMS['early_stopping_patience'],
    verbose=True
)

# --- Training Loop ---
print("Starting training...")
history = {'train_loss': [], 'val_loss': [], 'beta': []}

for epoch in range(TRAINING_PARAMS['epochs']):
    # -- Training --
    model.train()
    train_loss_epoch = 0
    beta = calculate_beta(epoch, TRAINING_PARAMS['kl_annealing'])
    history['beta'].append(beta)
    
    for batch_idx, (design, condition) in enumerate(train_loader):
        design, condition = design.to(device), condition.to(device)
        
        optimizer.zero_grad()
        recon_design, mu, logvar = model(design, condition)
        losses = model.loss_function(recon_design, design, mu, logvar, beta)
        loss = losses['loss']
        loss.backward()
        optimizer.step()
        
        train_loss_epoch += loss.item()

    avg_train_loss = train_loss_epoch / len(train_loader)
    history['train_loss'].append(avg_train_loss)

    # -- Validation --
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for batch_idx, (design, condition) in enumerate(val_loader):
            design, condition = design.to(device), condition.to(device)
            recon_design, mu, logvar = model(design, condition)
            losses = model.loss_function(recon_design, design, mu, logvar, beta)
            val_loss_epoch += losses['loss'].item()
    
    avg_val_loss = val_loss_epoch / len(val_loader)
    history['val_loss'].append(avg_val_loss)

    print(f"Epoch {epoch+1}/{TRAINING_PARAMS['epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Beta: {beta:.6f}")
    
    # Update LR scheduler and check for early stopping
    scheduler.step(avg_val_loss)
    early_stopper(avg_val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping triggered")
        break

# Load best model weights saved by EarlyStopping
model.load_state_dict(torch.load('checkpoint.pt'))

# %%
# --- Plot, Save, and Generate ---
plot_history(history, save_path="loss_curve_pytorch.png")

# %%
tracker = MockExperimentTracker()
all_hyperparams = {**CVAE_HYPERPARAMS, **TRAINING_PARAMS, **GENERATION_PARAMS}

save_experiment(tracker, model, all_hyperparams, wavelengths, history)

# --- NEW: Generate a sample design using the trained model ---
print("\n--- Generating sample designs ---")
# Take the first validation sample's condition as an example
sample_condition_slice = val_dataset[0][1] # This is already a tensor

# Generate 5 different designs for the same spectral condition
generated_designs = generate_designs(model, sample_condition_slice, n_samples=5, device=device)

print(f"Generated {generated_designs.shape[0]} designs.")
print(f"Shape of one generated design: {generated_designs[0].shape}")
# You can now save or visualize `generated_designs.cpu().numpy()`
