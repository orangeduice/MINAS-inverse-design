# %%
import os
import json
import pickle
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming these are local modules in your working directory
import model_architecture as model_arch
import helper

# %%
# Modern Dynamic GPU memory allocation (Replaces tf.compat.v1 code)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

data_path = 'datasets/2026_04_15_full_pisa_data.h5'

# Load data and scalers
x_train, x_test, y_train, y_test, wavelengths = helper.load_data_from_h5(
    data_path,
    condition_name='R',
    input_name='geo',
    save_scalers=True) # Save scalers for first time

# Slicing the data as originally defined
y_train = y_train[:, 35:110]
y_test = y_test[:, 35:110]
wavelengths = wavelengths[35:110]

print("Wavelengths:", wavelengths)
print("Wavelengths shape:", wavelengths.shape)

scaler_geo = pickle.load(
    open('{}_scalers.pkl'.format(os.path.splitext(data_path)[0]), 'rb'))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# %%
keras.backend.clear_session()

# cVAE Configuration
latent_dim = 4    # for interpretability
N_blocks = 2      # Nr of resblocks between upsamplings

# ==========================================
# 1. ENCODER
# ==========================================
condition_target_spec_input = keras.layers.Input(shape=y_train.shape[1:], name='condition_in')
design_in = keras.Input(shape=x_train.shape[1:], name='design_in')

# spectrum path
x = condition_target_spec_input
x = model_arch.resblock_sequence_down(x, N_filter=32, N_blocks=N_blocks)   # 64 --> 32
x = model_arch.resblock_sequence_down(x, N_filter=64, N_blocks=N_blocks)   # 32 --> 16
x = model_arch.resblock_sequence_down(x, N_filter=128, N_blocks=N_blocks)  # 16 --> 8
x = model_arch.resblock_sequence_down(x, N_filter=256, N_blocks=N_blocks)  # 8 --> 4
x_spec = keras.layers.Flatten()(x)

# geometry path
x = design_in
x = model_arch.resblock_sequence_id(x, N_filter=128, N_blocks=N_blocks)    # 5 --> 2
x_design = keras.layers.Flatten()(x)

# merge both paths
x = keras.layers.Concatenate()([x_spec, x_design])
x = keras.layers.Dense(256)(x)
x = keras.layers.LeakyReLU(alpha=1e-1)(x)

# output: latent sampling
z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
z_var_log = keras.layers.Dense(latent_dim, name="z_var_log")(x)
z = model_arch.SamplingRNG()([z_mean, z_var_log])

encoder = keras.Model(
    [design_in, condition_target_spec_input], 
    [z_mean, z_var_log, z], 
    name="encoder"
)
encoder.summary()

# ==========================================
# 2. DECODER
# ==========================================
latent_in = keras.Input(shape=(latent_dim,), name='latent_in')

# spectrum path
x = condition_target_spec_input
x = model_arch.resblock_sequence_down(x, N_filter=32, N_blocks=N_blocks)   # 64 --> 32
x = model_arch.resblock_sequence_down(x, N_filter=64, N_blocks=N_blocks)   # 32 --> 16
x = model_arch.resblock_sequence_down(x, N_filter=128, N_blocks=N_blocks)  # 16 --> 8
x = model_arch.resblock_sequence_down(x, N_filter=256, N_blocks=N_blocks)  # 8 --> 4
x_spec = keras.layers.Flatten()(x)

# merge with latent
x = keras.layers.Concatenate()([x_spec, latent_in])

# geometry generator path
x = keras.layers.Dense(6 * 128)(x)
x = keras.layers.Reshape((6, 128))(x)
x = model_arch.resblock_sequence_id(x, N_filter=128, N_blocks=N_blocks)

# output: number of channels from design
out_design = keras.layers.Conv1D(
    filters=x_train.shape[-1], kernel_size=1, name='design_out')(x)

decoder = keras.Model(
    [latent_in, condition_target_spec_input],
    out_design, name="decoder"
)
decoder.summary()

# ==========================================
# 3. cVAE ASSEMBLY & METRICS
# ==========================================

class CVAEModel(keras.Model):
    def __init__(self, encoder, decoder, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = float(latent_dim)
        
        # Make beta a tf.Variable so it can be dynamically updated by the callback
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32, name='beta')
        
        # Setup metric trackers
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstr_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        z_mean, z_var_log, z = self.encoder(inputs)
        return self.decoder([z, inputs[1]])

    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data 
        else:
            x = data
            
        design_in = x[0]
        condition_in = x[1]

        with tf.GradientTape() as tape:
            # 1. Forward pass
            z_mean, z_var_log, z = self.encoder([design_in, condition_in])
            design_pred = self.decoder([z, condition_in])

            # 2. Compute losses natively
            flat_in = keras.layers.Flatten()(design_in)
            flat_pred = keras.layers.Flatten()(design_pred)

            reconstruction_loss = keras.ops.mean(keras.losses.mae(flat_in, flat_pred))
            kl_loss = keras.ops.sum(
                1 + z_var_log - keras.ops.square(z_mean) - keras.ops.exp(z_var_log), axis=-1
            )
            # Use the dynamic self.beta
            kl_loss = self.beta * keras.ops.mean(-0.5 / self.latent_dim * kl_loss)
            
            total_loss = reconstruction_loss + kl_loss

        # 3. Apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # 4. Track metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstr_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            
        design_in = x[0]
        condition_in = x[1]
        # 1. Forward pass (No GradientTape here)
        z_mean, z_var_log, z = self.encoder([design_in, condition_in])
        design_pred = self.decoder([z, condition_in])

        # 2. Compute losses
        flat_in = keras.layers.Flatten()(design_in)
        flat_pred = keras.layers.Flatten()(design_pred)

        reconstruction_loss = keras.ops.mean(keras.losses.mae(flat_in, flat_pred))
        kl_loss = keras.ops.sum(
            1 + z_var_log - keras.ops.square(z_mean) - keras.ops.exp(z_var_log), axis=-1
        )
        kl_loss = self.beta * keras.ops.mean(-0.5 / self.latent_dim * kl_loss)
        
        total_loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstr_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Instantiate model without passing a static beta
cVAE = CVAEModel(encoder, decoder, latent_dim=latent_dim)

checkpoint_dir = f"geomerty_cVAE/model_latent{latent_dim}_blocks{N_blocks}/checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# ==========================================
# CUSTOM KL ANNEALING CALLBACK
# ==========================================
class KLAnnealingCallback(keras.callbacks.Callback):
    def __init__(self, max_beta, total_epochs):
        super().__init__()
        self.max_beta = max_beta
        self.total_epochs = total_epochs
        # Global epoch tracks progress across multiple .fit() calls
        self.global_epoch = 0 

    def on_epoch_begin(self, epoch, logs=None):
        # Calculate linear progress (0.0 up to 1.0)
        progress = min(1.0, self.global_epoch / max(1, self.total_epochs - 1))
        new_beta = self.max_beta * progress
        
        # Update the tf.Variable in the model
        self.model.beta.assign(new_beta)
        print(f"\n[KL Annealing] Global Epoch {self.global_epoch+1}/{self.total_epochs} - Setting beta to {new_beta:.5f}")
        
        self.global_epoch += 1

# ==========================================
# SPECTRAL MISMATCH CALLBACK
# ==========================================
class SpectralMismatchCallback(keras.callbacks.Callback):
    def __init__(self, decoder, surrogate_model, y_val, latent_dim, num_samples=3):
        super().__init__()
        self.decoder = decoder
        self.surrogate_model = surrogate_model
        self.y_val = y_val  # Target spectra
        self.latent_dim = int(latent_dim)
        self.num_samples = num_samples

    def on_epoch_end(self, epoch, logs=None):
        batch_size = len(self.y_val)
        
        # 1. Sample z directly from prior N(0, I)
        z_prior = np.random.normal(size=(batch_size, self.latent_dim))
        
        # 2. Decode using the prior samples and target spectra
        # decoder expects [latent_in, condition_target_spec_input]
        generated_geometry = self.decoder.predict([z_prior, self.y_val], verbose=0)
        
        # 3. Simulate the spectra of the generated geometries using the surrogate
        simulated_spectra = self.surrogate_model.predict(generated_geometry, verbose=0)
        
        # 4. Measure spectral mismatch (MSE between requested spectra and simulated spectra)
        # Use np.squeeze to align shapes (e.g., stripping the trailing 1 from (490, 75, 1))
        mismatch = np.mean(np.square(np.squeeze(self.y_val) - np.squeeze(simulated_spectra)))
        
        print(f"\n[Figure of Merit] Epoch {epoch+1} Prior-Sample Spectral Mismatch: {mismatch:.6f}")
        
        if logs is not None:
            logs['val_spectral_mismatch'] = mismatch

# %%
# ==========================================
# TRAINING LOOP
# ==========================================
lr = 0.00001

optimizer = keras.optimizers.AdamW(learning_rate=lr)
cVAE.compile(optimizer=optimizer) 

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.66, patience=5, verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
)

# Calculate total expected epochs for the annealer
epochs_round_1 = int(25 * 1)
epochs_round_2 = int(25 * 0.5)
total_expected_epochs = epochs_round_1 + epochs_round_2

# Setup Annealer
kl_annealer = KLAnnealingCallback(max_beta=0.5, total_epochs=total_expected_epochs)

# Load the pre-trained surrogate model
try:
    surrogate_model = keras.models.load_model("surrogate_model/surrogate_geo_to_spec.keras")
    print("\nLoaded pre-trained forward surrogate model successfully.")
except Exception as e:
    print(f"\nWarning: Could not load surrogate model. Did you run train_surrogate.py? Error: {e}")
    surrogate_model = None

# Initialize mismatch callback (using test set conditions as validation)
if surrogate_model:
    mismatch_tracker = SpectralMismatchCallback(
        decoder=decoder, 
        surrogate_model=surrogate_model, 
        y_val=y_test, # Using y_test to evaluate unseen conditions
        latent_dim=latent_dim
    )
    all_callbacks = [reduce_lr, kl_annealer, mismatch_tracker]
else:
    all_callbacks = [reduce_lr, kl_annealer]

hist = None
for i, o in zip([1, 2], [1, 0.5]):
    batch_size = 8 * 2**i
    epochs_this_round = int(25 * o)
    print(f"\n--- Starting Training Round {i} with batch_size={batch_size} ---")
    
    _h = cVAE.fit(
        x=[x_train, y_train], 
        y=x_train, 
        validation_split=.3,
        batch_size=batch_size,
        epochs=epochs_this_round,
        callbacks=all_callbacks 
    )

    if hist is None:
        hist = _h
    else:
        for k in hist.history:
            hist.history[k] = np.concatenate([hist.history[k], _h.history[k]])

    if cVAE.stop_training:
        print(f"Early stopping triggered in round {i}. Halting training.")
        break

# ==========================================
# PLOTTING
# ==========================================
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot total loss on the primary y-axis (ax1)
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss (log scale)', color=color)
ax1.plot(hist.history['loss'], color=color, linestyle='-', label='Total Loss (Train)')
ax1.plot(hist.history['val_loss'], color=color, linestyle='--', label='Total Loss (Val)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')

# Create a second y-axis (ax2) that shares the same x-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Component Losses')
ax2.set_yscale('log')

# Plot reconstruction loss on the secondary y-axis
color_recon = 'tab:blue'
ax2.plot(hist.history['reconstr_loss'], color=color_recon, linestyle='-', label='Reconstruction Loss (Train)')
ax2.plot(hist.history['val_reconstr_loss'], color=color_recon, linestyle='--', label='Reconstruction Loss (Val)')

# Plot KL loss on the secondary y-axis
color_kl = 'tab:green'
ax2.plot(hist.history['kl_loss'], color=color_kl, linestyle='-', label='KL Loss (Train)')
ax2.plot(hist.history['val_kl_loss'], color=color_kl, linestyle='--', label='KL Loss (Val)')

# Plot Spectral Mismatch on the secondary y-axis (if tracked)
if 'val_spectral_mismatch' in hist.history:
    color_spec = 'tab:purple'
    # Using a distinct dash-dot line style and thicker weight to make it stand out
    ax2.plot(hist.history['val_spectral_mismatch'], color=color_spec, linestyle='-.', linewidth=2.5, label='Spectral Mismatch (Val)')

ax2.tick_params(axis='y')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Training and Validation Losses')
plt.savefig(f"geomerty_cVAE/model_latent{latent_dim}_blocks{N_blocks}/training_losses.png", dpi=300, bbox_inches='tight')
plt.show()
# %%

def analyze_latent_space_from_arrays(encoder, x_test, y_test, batch_size=128):
    """
    Computes and plots per-dimension KL divergence and variance of posterior means
    directly from NumPy arrays.
    
    Args:
        encoder: Your Keras encoder model taking [x, y] and returning [mu, log_var, z]
        x_test: NumPy array of input features
        y_test: NumPy array of target conditions
        batch_size: Size of mini-batches to process at a time
    """
    all_mus = []
    all_log_vars = []
    
    num_samples = len(x_test)
    
    # 1. Process data in mini-batches to prevent memory overload
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        
        # Get predictions from encoder
        # (Pass inputs as a list if your model takes multiple inputs: [x, y])
        mu, log_var, _ = encoder.predict([x_batch, y_batch], verbose=0)
        
        all_mus.append(mu)
        all_log_vars.append(log_var)
        
    # Concatenate all batches back into full arrays
    mus = np.concatenate(all_mus, axis=0)
    log_vars = np.concatenate(all_log_vars, axis=0)
    
    latent_dim = mus.shape[1]
    
    # 2. Calculate Per-Dimension KL Divergence
    kl_per_sample_per_dim = -0.5 * (1 + log_vars - np.square(mus) - np.exp(log_vars))
    mean_kl_per_dim = np.mean(kl_per_sample_per_dim, axis=0)
    
    # 3. Calculate Variance of Posterior Means
    var_of_means_per_dim = np.var(mus, axis=0)
    
    # 4. Plot the diagnostics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    dimensions = np.arange(latent_dim)
    
    # Left Plot: Mean KL Divergence
    ax1.bar(dimensions, mean_kl_per_dim, color='royalblue', alpha=0.8, edgecolor='black')
    ax1.set_title("Mean KL Divergence per Latent Dimension")
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("KL Divergence")
    ax1.set_xticks(dimensions)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Right Plot: Variance of Posterior Means
    ax2.bar(dimensions, var_of_means_per_dim, color='darkorange', alpha=0.8, edgecolor='black')
    ax2.axhline(y=0.01, color='red', linestyle='--', label='Active Threshold (0.01)')
    ax2.set_title("Variance of Posterior Means ($\sigma^2(\mu_i)$)")
    ax2.set_xlabel("Latent Dimension")
    ax2.set_ylabel("Variance")
    ax2.set_xticks(dimensions)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Summary Output
    active_units = np.sum(var_of_means_per_dim > 0.01)
    print("Diagnostics Summary:")
    print("--------------------")
    print(f"Total latent dimensions: {latent_dim}")
    print(f"Active dimensions (variance > 0.01): {active_units} / {latent_dim}")
    print(f"Inactive/Collapsed dimensions: {latent_dim - active_units}")

# %%


analyze_latent_space_from_arrays(encoder, x_test, y_test)


# %%

def test_decoder_neglect(encoder, decoder, x_test, y_test, batch_size=128):
    """
    Compares reconstruction error using posterior z ~ q(z|x,y) 
    versus prior z ~ N(0,I) to check if decoder relies on z.
    """
    mse_posterior_list = []
    mse_prior_list = []
    
    num_samples = len(x_test)
    latent_dim = encoder.output[0].shape[-1]  # infers latent dimension
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        current_batch_size = end_idx - start_idx
        
        # 1. Posterior path: Sample z from encoder q(z|x,y)
        mu, log_var, z_posterior = encoder.predict([x_batch, y_batch], verbose=0)
        
        # Decode using posterior z
        reconstruction_posterior = decoder.predict([z_posterior, y_batch], verbose=0)
        
        # 2. Prior path: Sample random z from standard normal N(0, I)
        z_prior = np.random.normal(size=(current_batch_size, latent_dim))
        
        # Decode using prior z (same condition y)
        reconstruction_prior = decoder.predict([z_prior, y_batch], verbose=0)
        
        # Compute Mean Squared Error against true targets (assuming target is x_batch)
        mse_post = np.mean(np.square(x_batch - reconstruction_posterior))
        mse_pr = np.mean(np.square(x_batch - reconstruction_prior))
        
        mse_posterior_list.append(mse_post)
        mse_prior_list.append(mse_pr)
        
    avg_mse_posterior = np.mean(mse_posterior_list)
    avg_mse_prior = np.mean(mse_prior_list)
    
    print("--- Decoder Reliance Test Results ---")
    print(f"MSE with Encoder z  (q(z|x,y)): {avg_mse_posterior:.6f}")
    print(f"MSE with Random z   (N(0, I)) : {avg_mse_prior:.6f}")
    print(f"Absolute Gap                  : {abs(avg_mse_prior - avg_mse_posterior):.6f}")
    
    # Interpretation logic
    ratio = avg_mse_prior / (avg_mse_posterior + 1e-8)
    if ratio < 1.05:
        print("\nConclusion: The Decoder is IGNORING z!")
        print("Swapping z for random noise barely changes the prediction.")
    else:
        print(f"\nConclusion: The Decoder uses z (Error increases by {ratio:.2f}x when using random z).")

# %%

test_decoder_neglect(encoder, decoder, x_test, y_test)

# %%
import umap
def visualize_latent_space(model, x_data, y_data, batch_size=64, 
                           method='raw', dims=(0, 1), color_values=None, 
                           save_path=None, cmap='viridis'):
    
    latents = []
    print("Extracting latent representations...")
    
    total_samples = len(x_data)
    
    # Process all samples dynamically in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        
        batch_x = tf.convert_to_tensor(x_data[start_idx:end_idx], dtype=tf.float32)
        batch_y = tf.convert_to_tensor(y_data[start_idx:end_idx], dtype=tf.float32)
        
        # Encoder forward pass
        encode_outputs = model([batch_x, batch_y], training=False)
        mu = encode_outputs[0] if isinstance(encode_outputs, (list, tuple)) else encode_outputs
        
        latents.append(mu.numpy())
        
    latents = np.concatenate(latents, axis=0)
    
    print(f"Plotting using method: {method.upper()}")
    
    if method == 'raw':
        # Safely extract specific dimensions
        max_dim = latents.shape[1] - 1
        d1 = min(dims[0], max_dim)
        d2 = min(dims[1], max_dim)
        latents_2d = latents[:, [d1, d2]]
        title_prefix = f"Latent Dims {d1} & {d2} of "
        xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
        
    elif method == 'umap' and latents.shape[1] > 2:
        reducer = umap.UMAP(random_state=42)
        latents_2d = reducer.fit_transform(latents)
        title_prefix = "UMAP of "
        xlabel, ylabel = "UMAP 1", "UMAP 2"
        
    elif method == 'pca' and latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
        title_prefix = "PCA of "
        xlabel, ylabel = "PCA 1", "PCA 2"
        
    else:
        # Fallback if invalid method or missing libraries
        print(f"Fallback to raw dimensions {dims} (Dim reductions failed or latent size <= 2)")
        d1 = min(dims[0], latents.shape[1] - 1)
        d2 = min(dims[1], latents.shape[1] - 1)
        latents_2d = latents[:, [d1, d2]]
        title_prefix = f"Latent Dims {d1} & {d2} of "
        xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
    # ... [dimensionality reduction logic: UMAP / PCA / raw] ...
    
    plt.figure(figsize=(9, 7))
    
    # Tuned scatter settings for ~10,000 points
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1], 
        c=color_values if color_values is not None else '#1f77b4', 
        cmap=cmap if color_values is not None else None, 
        alpha=0.8,      # Lower alpha highlights density differences
        s=8,             # Smaller points reduce visual crowding
        edgecolors='none'
    )
    
    if color_values is not None:
        plt.colorbar(scatter, label="Color Value")
        
    plt.title(f"{title_prefix}cVAE Latent Space (N={len(latents)})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
# %%

def generate_min_index_colors(y_data):
    """
    Generates a color array based on the index of the minimum value 
    in the condition data (y_data) for each sample.
    
    Args:
        y_data: NumPy array of condition data (e.g., shape: [N, 161, 1]).
        
    Returns:
        A 1D NumPy array of length N containing the index of the minimum value.
    """
    # Reshape to 2D [N, sequence_length] to safely handle the trailing dimension
    y_reshaped = y_data.reshape(y_data.shape[0], -1)
    
    # Find the index of the minimum value along axis 1 (the sequence)
    min_indices = np.argmin(y_reshaped, axis=1)
    
    return min_indices

# %%



x_full = np.concatenate([x_train, x_test], axis=0)
y_full = np.concatenate([y_train, y_test], axis=0)

min_indices = wavelengths[generate_min_index_colors(y_full)]

split_mask = np.array([0] * len(x_train) + [1] * len(x_test))
# h_pill, d_pill, w_pill, d_sep, w_sep, angle
geom_param_h_pill = x_full.squeeze()[:, 0]
geom_param_d_pill = x_full.squeeze()[:, 1]
geom_param_w_pill = x_full.squeeze()[:, 2]
geom_param_d_sep = x_full.squeeze()[:, 3]
geom_param_w_sep = x_full.squeeze()[:, 4]
geom_param_angle = x_full.squeeze()[:, 5]


# %%
visualize_latent_space(
    encoder, 
    x_full, 
    y_full, 
    method='umap', 
    # dims=(1, 2), 
    color_values=geom_param_h_pill,
    # cmap='Dark2'
    )
# %%
cVAE.decoder.save("decoder.keras")


# %%
