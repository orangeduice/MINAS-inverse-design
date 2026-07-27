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
latent_dim = 8    # for interpretability
N_blocks = 1     # Nr of resblocks between upsamplings

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

# output: number of  channels from design
out_design = keras.layers.Conv1D(
    filters=x_train.shape[-1], kernel_size=1, name='design_out')(x)

decoder = keras.Model(
    [latent_in, condition_target_spec_input],
    out_design, name="decoder"
)
decoder.summary()

# ==========================================
# 3. cVAE ASSEMBLY & METRICS (Keras 3 Subclass)
# ==========================================
beta = 0.01   # weight of KL loss. If too high, reconstruction will suffer.

class CVAEModel(keras.Model):
    def __init__(self, encoder, decoder, beta, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.latent_dim = float(latent_dim)
        
        # Setup metric trackers
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstr_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        # By listing these here, Keras will automatically reset them each epoch
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        # Used for inference
        z_mean, z_var_log, z = self.encoder(inputs)
        return self.decoder([z, inputs[1]])

    def train_step(self, data):
        # Unpack data depending on what is passed to .fit()
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

            reconstruction_loss = keras.ops.mean(keras.losses.mse(flat_in, flat_pred))
            kl_loss = keras.ops.sum(
                1 + z_var_log - keras.ops.square(z_mean) - keras.ops.exp(z_var_log), axis=-1
            )
            kl_loss = self.beta * keras.ops.mean(-0.5 / self.latent_dim * kl_loss)
            
            total_loss = reconstruction_loss + kl_loss

        # 3. Apply gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # 4. Track metrics (Keras auto-prepends 'val_' during validation!)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstr_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        # This handles the validation_split automatically
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

        reconstruction_loss = keras.ops.mean(keras.losses.mse(flat_in, flat_pred))
        kl_loss = keras.ops.sum(
            1 + z_var_log - keras.ops.square(z_mean) - keras.ops.exp(z_var_log), axis=-1
        )
        kl_loss = self.beta * keras.ops.mean(-0.5 / self.latent_dim * kl_loss)
        
        total_loss = reconstruction_loss + kl_loss

        # 3. Track validation metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstr_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

cVAE = CVAEModel(encoder, decoder, beta=beta, latent_dim=latent_dim)

checkpoint_dir = f"geomerty_cVAE/model_latent{latent_dim}_blocks{N_blocks}/checkpoints/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# %%
# ==========================================
# TRAINING LOOP
# ==========================================
lr = 0.00001

# compile the model
optimizer = keras.optimizers.AdamW(learning_rate=lr)
# No need to specify a loss function here since we are using cVAE.add_loss()
cVAE.compile(optimizer=optimizer) 

# Instantiate standard callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.66, patience=5, verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', patience=15, verbose=1, restore_best_weights=True
)

hist = None
for i, o in zip([1, 2], [1, 0.5]):
    batch_size = 8 * 2**i
    print(f"\n--- Starting Training Round {i+1} with batch_size={batch_size} ---")
    
    _h = cVAE.fit(
        x=[x_train, y_train], 
        y=x_train, 
        validation_split=.3,
        batch_size=batch_size,
        epochs=int(25 * o),
        # callbacks=[reduce_lr, early_stopping] # Re-activated!
    )

    # History concatenation logic
    if hist is None:
        hist = _h
    else:
        for k in hist.history:
            hist.history[k] = np.concatenate([hist.history[k], _h.history[k]])

    # Crucially, we must check if early stopping was triggered
    if cVAE.stop_training:
        print(f"Early stopping triggered in round {i+1}. Halting training.")
        break


# %%
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
ax2.tick_params(axis='y')

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
plt.savefig(f"geomerty_cVAE/model_latent{latent_dim}_blocks{N_blocks}/training_losses.png", dpi=300, bbox_inches='tight')
plt.title('Training and Validation Losses')
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
        alpha=0.35,      # Lower alpha highlights density differences
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
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# try:
#     from sklearn.decomposition import PCA
#     HAS_SKLEARN = True
# except ImportError:
#     HAS_SKLEARN = False

# try:
#     import umap
#     HAS_UMAP = False
# except ImportError:
#     HAS_UMAP = False

# def visualize_latent_space(model, x_data, y_data, batch_size=32, num_batches=20, 
#                            method='raw', dims=(0, 1), color_values=None, 
#                            save_path=None, cmap='viridis'):
#     """
#     Plots the latent space.
#     Supports plotting raw dimensions or dimensionality reduction via 'umap' or 'pca'.
    
#     Args:
#         model: A Keras model representing the Conditional VAE encoder.
#         x_data: NumPy array of input data (e.g., x_test, shape: [N, 6, 1]).
#         y_data: NumPy array of condition data (e.g., y_test, shape: [N, 161, 1]).
#         batch_size: Number of samples per batch.
#         num_batches: The maximum number of batches to process.
#         method: Dimensionality reduction method ('raw', 'umap', or 'pca').
#         dims: Tuple of (dim_x, dim_y) to plot if method='raw'. Default is (0, 1).
#         color_values: Optional array of values to color the points. If None, uses a single uniform color.
#         save_path: Optional path to save the generated plot.
#         cmap: Matplotlib colormap string.
#     """
#     latents = []
    
#     print("Extracting latent representations...")
    
#     total_samples = len(x_data)
#     max_possible_batches = int(np.ceil(total_samples / batch_size))
#     actual_batches = min(num_batches, max_possible_batches)
    
#     for i in range(actual_batches):
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, total_samples)
        
#         batch_data = x_data[start_idx:end_idx]
#         batch_condition = y_data[start_idx:end_idx]
            
#         batch_data_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)
#         batch_condition_tf = tf.convert_to_tensor(batch_condition, dtype=tf.float32)
        
#         # Call the encoder directly
#         encode_outputs = model([batch_data_tf, batch_condition_tf], training=False)
#         mu = encode_outputs[0]
            
#         latents.append(mu.numpy())
        
#     # Concatenate the lists into full arrays
#     latents = np.concatenate(latents, axis=0)
    
#     print(f"Plotting using method: {method.upper()}")
    
#     if method == 'raw':
#         # Safely extract specific dimensions
#         max_dim = latents.shape[1] - 1
#         d1 = min(dims[0], max_dim)
#         d2 = min(dims[1], max_dim)
#         latents_2d = latents[:, [d1, d2]]
#         title_prefix = f"Latent Dims {d1} & {d2} of "
#         xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
        
#     elif method == 'umap' and HAS_UMAP and latents.shape[1] > 2:
#         reducer = umap.UMAP(random_state=42)
#         latents_2d = reducer.fit_transform(latents)
#         title_prefix = "UMAP of "
#         xlabel, ylabel = "UMAP 1", "UMAP 2"
        
#     elif method == 'pca' and HAS_SKLEARN and latents.shape[1] > 2:
#         pca = PCA(n_components=2)
#         latents_2d = pca.fit_transform(latents)
#         title_prefix = "PCA of "
#         xlabel, ylabel = "PCA 1", "PCA 2"
        
#     else:
#         # Fallback if invalid method or missing libraries
#         print(f"Fallback to raw dimensions {dims} (Dim reductions failed or latent size <= 2)")
#         d1 = min(dims[0], latents.shape[1] - 1)
#         d2 = min(dims[1], latents.shape[1] - 1)
#         latents_2d = latents[:, [d1, d2]]
#         title_prefix = f"Latent Dims {d1} & {d2} of "
#         xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
        
#     # Handle coloring logic
#     if color_values is not None:
#         # Ensure the color array matches the number of latents extracted
#         c = np.array(color_values)[:len(latents)]
#         cmap_to_use = cmap
#     else:
#         c = '#1f77b4' # Default standard blue
#         cmap_to_use = None

#     plt.figure(figsize=(9, 7))
#     scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
#                           c=c, cmap=cmap_to_use, alpha=0.8, 
#                           s=20, edgecolors='none')
    
#     if color_values is not None:
#         plt.colorbar(scatter, label="Color Value")
        
#     plt.title(f"{title_prefix}cVAE Latent Space")
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True, alpha=0.3, linestyle='--')
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#         print(f"--- Latent space plot saved to {save_path} ---")
        
#     plt.show()
# # try:
#     from sklearn.decomposition import PCA
#     HAS_SKLEARN = True
# except ImportError:
#     HAS_SKLEARN = False

# try:
#     import umap
#     HAS_UMAP = True # BUGFIX: This was incorrectly set to False in the original code
# except ImportError:
#     HAS_UMAP = False

# def visualize_latent_space(model, x_data, y_data, batch_size=32, num_batches=20, 
#                            method='raw', dims=(0, 1), color_by='y_first', 
#                            x_col_idx=0, scaler_geo=None, save_path=None, 
#                            full_model=None, cmap='viridis', log_scale_color=False):
#     """
#     Plots the latent space colored by a specified metric from x_data or y_data.
#     Supports plotting raw dimensions or dimensionality reduction via 'umap' or 'pca'.
    
#     Args:
#         model: A Keras model representing the Conditional VAE encoder.
#         x_data: NumPy array of input data (e.g., x_test, shape: [N, 6, 1]).
#         y_data: NumPy array of condition data (e.g., y_test, shape: [N, 161, 1]).
#         batch_size: Number of samples per batch.
#         num_batches: The maximum number of batches to process.
#         method: Dimensionality reduction method ('raw', 'umap', or 'pca').
#         dims: Tuple of (dim_x, dim_y) to plot if method='raw'. Default is (0, 1).
#         color_by: String dictating color mapping ('y_first', 'y_avg', 'x_col', or 'mse').
#         x_col_idx: Integer specifying which x_data column to use if color_by='x_col'.
#         scaler_geo: Optional scaler object with an `inverse_transform` method to convert x_data back to physical units.
#         save_path: Optional path to save the generated plot.
#         full_model: The complete cVAE model, required if color_by='mse'.
#         cmap: Matplotlib colormap string.
#         log_scale_color: Boolean to apply logarithmic scaling to the color bar.
#     """
#     latents = []
#     condition_vals = []
    
#     if color_by == 'y_first':
#         color_label = "Condition (y) Feature 0"
#         title_suffix = "y Feature 0"
#     elif color_by == 'y_avg':
#         color_label = "Condition (y) Average"
#         title_suffix = "y Average"
#     elif color_by == 'x_col':
#         color_label = f"Design (x) Col {x_col_idx}" + (" (Physical Units)" if scaler_geo else "")
#         title_suffix = f"x Col {x_col_idx}"
#     elif color_by == 'mse':
#         if full_model is None:
#             raise ValueError("full_model must be provided if color_by='mse'")
#         color_label = "Reconstruction MSE"
#         title_suffix = "Reconstruction MSE"
#     else:
#         raise ValueError("color_by must be 'y_first', 'y_avg', 'x_col', or 'mse'")
    
#     print("Extracting latent representations...")
    
#     total_samples = len(x_data)
#     max_possible_batches = int(np.ceil(total_samples / batch_size))
#     actual_batches = min(num_batches, max_possible_batches)
    
#     for i in range(actual_batches):
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, total_samples)
        
#         batch_data = x_data[start_idx:end_idx]
#         batch_condition = y_data[start_idx:end_idx]
            
#         batch_data_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)
#         batch_condition_tf = tf.convert_to_tensor(batch_condition, dtype=tf.float32)
        
#         # Call the encoder directly
#         encode_outputs = model([batch_data_tf, batch_condition_tf], training=False)
#         mu = encode_outputs[0]
            
#         latents.append(mu.numpy())
        
#         if color_by == 'y_first':
#             condition_vals.append(batch_condition[:, 0, 0])
    #     elif color_by == 'y_avg':
    #         condition_vals.append(np.mean(batch_condition, axis=(1, 2)))
    #     elif color_by == 'x_col':
    #         if scaler_geo is not None:
    #             batch_data_2d = batch_data[:, :, 0]
    #             batch_physical = scaler_geo.inverse_transform(batch_data_2d)
    #             condition_vals.append(batch_physical[:, x_col_idx])
    #         else:
    #             condition_vals.append(batch_data[:, x_col_idx, 0])
    #     elif color_by == 'mse':
    #         reconstructed = full_model([batch_data_tf, batch_condition_tf], training=False)
    #         mse = np.mean(np.square(batch_data - reconstructed.numpy()), axis=tuple(range(1, batch_data.ndim)))
    #         condition_vals.append(mse)
        
    # # Concatenate the lists into full arrays
    # latents = np.concatenate(latents, axis=0)
    # condition_vals = np.concatenate(condition_vals, axis=0)
    
    # print(f"Plotting using method: {method.upper()}")
    
    # if method == 'raw':
    #     # Safely extract specific dimensions
    #     max_dim = latents.shape[1] - 1
    #     d1 = min(dims[0], max_dim)
    #     d2 = min(dims[1], max_dim)
    #     latents_2d = latents[:, [d1, d2]]
    #     title_prefix = f"Latent Dims {d1} & {d2} of "
    #     xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
        
    # elif method == 'umap' and HAS_UMAP and latents.shape[1] > 2:
    #     reducer = umap.UMAP(random_state=42)
    #     latents_2d = reducer.fit_transform(latents)
    #     title_prefix = "UMAP of "
    #     xlabel, ylabel = "UMAP 1", "UMAP 2"
        
    # elif method == 'pca' and HAS_SKLEARN and latents.shape[1] > 2:
    #     pca = PCA(n_components=2)
    #     latents_2d = pca.fit_transform(latents)
    #     title_prefix = "PCA of "
    #     xlabel, ylabel = "PCA 1", "PCA 2"
        
    # else:
    #     # Fallback if invalid method or missing libraries
    #     print(f"Fallback to raw dimensions {dims} (Dim reductions failed or latent size <= 2)")
    #     d1 = min(dims[0], latents.shape[1] - 1)
    #     d2 = min(dims[1], latents.shape[1] - 1)
    #     latents_2d = latents[:, [d1, d2]]
    #     title_prefix = f"Latent Dims {d1} & {d2} of "
    #     xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
        
    # norm = None
    # if log_scale_color:
    #     min_val = np.min(condition_vals)
    #     if min_val <= 0:
    #         print("Warning: LogNorm requested but color values contain zero/negatives. Using SymLogNorm.")
    #         norm = mcolors.SymLogNorm(linthresh=1e-3, vmin=min_val, vmax=np.max(condition_vals))
    #     else:
    #         norm = mcolors.LogNorm(vmin=min_val, vmax=np.max(condition_vals))

    # plt.figure(figsize=(9, 7))
    # scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
    #                       c=condition_vals, cmap=cmap, alpha=0.8, 
    #                       s=20, norm=norm, edgecolors='none')
    
    # plt.colorbar(scatter, label=color_label)
    # plt.title(f"{title_prefix}cVAE Latent Space (Colored by {title_suffix})")
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.grid(True, alpha=0.3, linestyle='--')
    # plt.tight_layout()
    
    # if save_path:
    #     plt.savefig(save_path, dpi=300)
    #     print(f"--- Latent space plot saved to {save_path} ---")
        
    # plt.show()

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
    color_values=min_indices,
    cmap='Dark2'
    )

# %%


# try:
#     from sklearn.decomposition import PCA
#     HAS_SKLEARN = True
# except ImportError:
#     HAS_SKLEARN = False

# try:
#     import umap
#     HAS_UMAP = False
# except ImportError:
#     HAS_UMAP = False

# def visualize_latent_space(model, x_data, y_data, batch_size=32, num_batches=20, method='umap', color_by='y_first', x_col_idx=0, scaler_geo=None, save_path=None, full_model=None):
#     """
#     Plots the latent space colored by a specified metric from x_data or y_data.
#     Supports dimensionality reduction via 'umap' or 'pca'.
    
#     Args:
#         model: A Keras model representing the Conditional VAE.
#         x_data: NumPy array of input data (e.g., x_test, shape: [N, 6, 1]).
#         y_data: NumPy array of condition data (e.g., y_test, shape: [N, 161, 1]).
#         batch_size: Number of samples per batch.
#         num_batches: The maximum number of batches to process.
#         method: Dimensionality reduction method ('umap' or 'pca').
#         color_by: String dictating color mapping ('y_first', 'y_avg', 'x_col', or 'mse').
#         x_col_idx: Integer specifying which x_data column to use if color_by='x_col'.
#         scaler_geo: Optional scaler object with an `inverse_transform` method to convert x_data back to physical units.
#         save_path: Optional path to save the generated plot.
#         full_model: The complete cVAE model, required if color_by='mse'.
#     """
#     latents = []
#     condition_vals = [] # Track the chosen variable to color the plot
    
#     if color_by == 'y_first':
#         color_label = "Condition (y) Feature 0"
#         title_suffix = "y Feature 0"
#     elif color_by == 'y_avg':
#         color_label = "Condition (y) Average"
#         title_suffix = "y Average"
#     elif color_by == 'x_col':
#         color_label = f"Design (x) Col {x_col_idx}" + (" (Physical Units)" if scaler_geo else "")
#         title_suffix = f"x Col {x_col_idx}"
#     elif color_by == 'mse':
#         if full_model is None:
#             raise ValueError("full_model must be provided if color_by='mse'")
#         color_label = "Reconstruction MSE"
#         title_suffix = "Reconstruction MSE"
#     else:
#         raise ValueError("color_by must be 'y_first', 'y_avg', 'x_col', or 'mse'")
    
#     print("Extracting latent representations...")
    
#     # Calculate how many batches we can actually process
#     total_samples = len(x_data)
#     max_possible_batches = int(np.ceil(total_samples / batch_size))
#     actual_batches = min(num_batches, max_possible_batches)
    
#     # Iterate over the numpy arrays in batches
#     for i in range(actual_batches):
#         start_idx = i * batch_size
#         end_idx = min((i + 1) * batch_size, total_samples)
        
#         batch_data = x_data[start_idx:end_idx]
#         batch_condition = y_data[start_idx:end_idx]
            
#         # In Keras, models handle device placement internally most of the time,
#         # especially if you are just passing tensors.
#         # Ensure your encode method handles training=False if 0it has layers like Dropout/BatchNorm
#         # Convert numpy arrays to tensors to be safe, though Keras often handles this automatically
#         batch_data_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)
#         batch_condition_tf = tf.convert_to_tensor(batch_condition, dtype=tf.float32)
        
#         # Call the encoder directly using the standard __call__ method since it's a keras.Model
#         # Pass the inputs as a list to match the [design_in, condition_target_spec_input] definition
#         encode_outputs = model([batch_data_tf, batch_condition_tf], training=False)
        
#         # Unpack based on your encoder definition: [z_mean, z_var_log, z]
#         mu = encode_outputs[0]
            
#         latents.append(mu.numpy())
        
#         if color_by == 'y_first':
#             # Grab the 0th feature of the condition sequence (first element)
#             condition_vals.append(batch_condition[:, 0, 0])
#         elif color_by == 'y_avg':
#             # Average across the sequence length (axis 1) and features (axis 2)
#             condition_vals.append(np.mean(batch_condition, axis=(1, 2)))
#         elif color_by == 'x_col':
#             if scaler_geo is not None:
#                 # Reshape to 2D (batch_size, num_features) for the scaler
#                 batch_data_2d = batch_data[:, :, 0]
#                 batch_physical = scaler_geo.inverse_transform(batch_data_2d)
#                 condition_vals.append(batch_physical[:, x_col_idx])
#             else:
#                 # Grab the specific column from the scaled input design data
#                 condition_vals.append(batch_data[:, x_col_idx, 0])
#         elif color_by == 'mse':
#             # Generate reconstructed designs using the full cVAE model
#             reconstructed = full_model([batch_data_tf, batch_condition_tf], training=False)
#             # Calculate Mean Squared Error per sample
#             mse = np.mean(np.square(batch_data - reconstructed.numpy()), axis=tuple(range(1, batch_data.ndim)))
#             condition_vals.append(mse)
        
#     latents = np.concatenate(latents, axis=0)
#     condition_vals = np.concatenate(condition_vals, axis=0)
    
#     # Dimensionality Reduction if latent_dim > 2
#     if latents.shape[1] > 2:
#         if method == 'umap' and HAS_UMAP:
#             print("Running UMAP on latent space...")
#             reducer = umap.UMAP(random_state=42)
#             latents_2d = reducer.fit_transform(latents)
#             title_prefix = "UMAP of "
#         elif HAS_SKLEARN:
#             print("Running PCA on latent space...")
#             pca = PCA(n_components=2)
#             latents_2d = pca.fit_transform(latents)
#             title_prefix = "PCA of "
#         else:
#             latents_2d = latents[:, :2]
#             title_prefix = "First 2 Dims of "
#     else:
#         latents_2d = latents
#         title_prefix = ""
        
#     import matplotlib.colors as mcolors
#     norm = mcolors.LogNorm() if True else None

#     plt.figure(figsize=(9, 7))
#     scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=condition_vals, cmap='Blues', alpha=1, s=20, norm=norm)
#     plt.colorbar(scatter, label=color_label)
#     plt.title(f"{title_prefix}cVAE Latent Space (Colored by {title_suffix})")
#     plt.xlabel(f"{method.upper()} 1" if method in ['umap', 'pca'] else "Latent Dim 1")
#     plt.ylabel(f"{method.upper()} 2" if method in ['umap', 'pca'] else "Latent Dim 2")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"--- Latent space plot saved to {save_path} ---")
        
#     plt.show()
#     plt.close()

# %%

# visualize_latent_space(encoder, x_test, y_test, color_by='mse', scaler_geo=scaler_geo, full_model=cVAE, method ="a")
# %%
def visualize_latent_space_with_highlights(model, x_data, y_data, N=5, batch_size=32, num_batches=20, method='umap', color_by='y_first', x_col_idx=0, scaler_geo=None, save_path=None):
    """
    Plots the latent space colored by a specified metric, and highlights N randomly selected 
    examples from the dataset with distinct markers and labels.
    
    Args:
        model: A Keras model representing the Conditional VAE encoder.
        x_data: NumPy array of input data.
        y_data: NumPy array of condition data.
        N: Number of examples to randomly sample and highlight.
        batch_size: Number of samples per batch for latent extraction.
        num_batches: Maximum number of batches to process for the background plot.
        method: Dimensionality reduction method ('umap' or 'pca').
        color_by: String dictating color mapping ('y_first', 'y_avg', or 'x_col').
        x_col_idx: Integer specifying which x_data column to use if color_by='x_col'.
        scaler_geo: Optional scaler object to convert x_data back to physical units.
        save_path: Optional path to save the generated plot.
    """
    latents = []
    condition_vals = []
    
    if color_by == 'y_first':
        color_label = "Condition (y) Feature 0"
        title_suffix = "y Feature 0"
    elif color_by == 'y_avg':
        color_label = "Condition (y) Average"
        title_suffix = "y Average"
    elif color_by == 'x_col':
        color_label = f"Design (x) Col {x_col_idx}" + (" (Physical Units)" if scaler_geo else "")
        title_suffix = f"x Col {x_col_idx}"
    else:
        raise ValueError("color_by must be 'y_first', 'y_avg', or 'x_col'")
    
    print("Extracting latent representations for background...")
    
    total_samples = len(x_data)
    max_possible_batches = int(np.ceil(total_samples / batch_size))
    actual_batches = min(num_batches, max_possible_batches)
    processed_samples = actual_batches * batch_size
    
    # Ensure N isn't larger than the number of samples we are plotting
    N = min(N, processed_samples)
    
    for i in range(actual_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_samples)
        
        batch_data = x_data[start_idx:end_idx]
        batch_condition = y_data[start_idx:end_idx]
            
        batch_data_tf = tf.convert_to_tensor(batch_data, dtype=tf.float32)
        batch_condition_tf = tf.convert_to_tensor(batch_condition, dtype=tf.float32)
        
        encode_outputs = model([batch_data_tf, batch_condition_tf], training=False)
        mu = encode_outputs[0]
            
        latents.append(mu.numpy())
        
        if color_by == 'y_first':
            condition_vals.append(batch_condition[:, 0, 0])
        elif color_by == 'y_avg':
            condition_vals.append(np.mean(batch_condition, axis=(1, 2)))
        elif color_by == 'x_col':
            if scaler_geo is not None:
                batch_data_2d = batch_data[:, :, 0]
                batch_physical = scaler_geo.inverse_transform(batch_data_2d)
                condition_vals.append(batch_physical[:, x_col_idx])
            else:
                condition_vals.append(batch_data[:, x_col_idx, 0])
        
    latents = np.concatenate(latents, axis=0)
    condition_vals = np.concatenate(condition_vals, axis=0)
    
    # We select indices from the subset of data we actually processed
    highlight_indices = np.random.choice(len(latents), size=N, replace=False)
    
    reducer_obj = None # Keep track of the reducer to project new points if needed
    
    if latents.shape[1] > 2:
        if method == 'umap' and HAS_UMAP:
            print("Running UMAP on latent space...")
            reducer_obj = umap.UMAP(random_state=42)
            latents_2d = reducer_obj.fit_transform(latents)
            title_prefix = "UMAP of "
        elif HAS_SKLEARN:
            print("Running PCA on latent space...")
            reducer_obj = PCA(n_components=2)
            latents_2d = reducer_obj.fit_transform(latents)
            title_prefix = "PCA of "
        else:
            latents_2d = latents[:, :2]
            title_prefix = "First 2 Dims of "
    else:
        latents_2d = latents
        title_prefix = ""
        
    plt.figure(figsize=(10, 8))
    
    # Plot the background points
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=condition_vals, cmap='viridis', alpha=0.5, s=20)
    plt.colorbar(scatter, label=color_label)
    
    # Highlight the N selected points
    highlight_colors = plt.cm.rainbow(np.linspace(0, 1, N))
    
    for i, idx in enumerate(highlight_indices):
        x_val = latents_2d[idx, 0]
        y_val = latents_2d[idx, 1]
        
        # Plot a larger, distinct marker with an edge
        plt.scatter(x_val, y_val, color=highlight_colors[i], marker='o', s=100, edgecolor='black', linewidth=1.5, zorder=5, label=f'Sample {idx}')
        
        # Add a text label next to the marker
        plt.annotate(f'S{idx}', (x_val, y_val), xytext=(5, 5), textcoords='offset points', 
                     fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8), zorder=6)

    plt.title(f"{title_prefix}cVAE Latent Space with {N} Highlighted Samples\n(Background Colored by {title_suffix})")
    plt.xlabel(f"{method.upper()} 1" if method in ['umap', 'pca'] else "Latent Dim 1")
    plt.ylabel(f"{method.upper()} 2" if method in ['umap', 'pca'] else "Latent Dim 2")
    
    # Add a legend for the highlighted points
    if N > 0:
        plt.legend()

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"--- Latent space plot saved to {save_path} ---")
        
    plt.show()
    plt.close()
    
    return highlight_indices, latents_2d, condition_vals 


# %%

samples, latents_2d, condition_vals = visualize_latent_space_with_highlights(encoder, x_test, y_test, N = 9)
# %%




model_folder = f"geomerty_cVAE/model_latent{latent_dim}_blocks{N_blocks}/"

directory_base_latents = model_folder + "latents/"

directory_base_inference = model_folder + "inference_data/"

if not os.path.exists(directory_base_inference):
    os.makedirs(directory_base_inference, exist_ok=True)

if not os.path.exists(directory_base_latents):
    os.makedirs(directory_base_latents, exist_ok=True)

print(latents_2d.shape)
print(y_test.squeeze().shape)
print(x_test.squeeze().shape)

print(x_test[0])
print(scaler_geo.inverse_transform(x_test.squeeze())[0])
print(y_test.squeeze()[0])

np.savetxt(f'{directory_base_latents}latents_2d.csv', latents_2d, delimiter=',', header='Latent_Dim_1,Latent_Dim_2', comments='#')
np.savetxt(f'{directory_base_latents}test_condition.csv', y_test.squeeze(), delimiter=',')

header = 'h_pill, d_pill, w_pill, d_sep, w_sep, angle\num, um, um, um, um, rad'
np.savetxt(f'{directory_base_latents}test_input.csv', scaler_geo.inverse_transform(x_test.squeeze()), delimiter=',', header=header, comments='#')

#%%

def save_inference(inference_id, inference_params, inference_output, target, target_id, wavelengths, scaler_geo, base_dir="inference_data"):
    """Logs the inference run and saves results using the unified ID."""
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Find the highest existing run ID and increment it
    existing_runs = [
        int(d.replace('run_', '')) for d in os.listdir(results_dir)
        if d.startswith('run_') and d.replace('run_', '').isdigit()
    ]
    run_id = max(existing_runs) + 1 if existing_runs else 1

    run_folder = os.path.join(results_dir, f"run_{run_id}")
    run_folder = os.path.join(results_dir, f"run_{inference_id}")
    os.makedirs(run_folder, exist_ok=True)

    designs_predict_physical = scaler_geo.inverse_transform(inference_output)

    print("\n--- Physical Designs ---")
    print(designs_predict_physical)

    target = np.expand_dims(target, axis=1)
    wavelengths_cut = np.expand_dims(wavelengths, axis=1)
    target_spectra = np.hstack((wavelengths_cut, target))

    print(f"Saving predicted designs with full run ID: {run_id}")
    print(f"Saving predicted designs with full inference ID: {inference_id}")

    np.savetxt(os.path.join(run_folder, f'{inference_id}_suggested_designs.csv'), inference_output, delimiter=',')
    np.savetxt(os.path.join(run_folder, f'{inference_id}_suggested_designs_physical.csv'), designs_predict_physical, delimiter=',')
    np.savetxt(os.path.join(run_folder, f'{inference_id}_target_spectra.csv'), target_spectra , delimiter=',')

    header = 'h_pill, d_pill, w_pill, d_sep, w_sep, angle\num, um, um, um, um, rad'
    np.savetxt(os.path.join(run_folder, f'{inference_id}_params.csv'), designs_predict_physical, delimiter=',', header=header, comments='')

    metadata = {
        "inference_run_id": inference_id,
        "target_id": target_id,
        "inference_params": inference_params
    }

    with open(os.path.join(run_folder, f'{inference_id}_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Data saved in: {run_folder}")
    return run_id


#%%

def make_infrerence(decoder, z_sample, condition_vector):
    """Generates designs from the decoder given latent samples and condition vectors."""
    condition_vector_tf = tf.expand_dims(tf.convert_to_tensor(condition_vector),1)
    condition_vector_tf = tf.repeat(tf.expand_dims(condition_vector_tf, axis=0), repeats=z_sample.shape[0], axis=0)

    decoder_in = [z_sample, condition_vector_tf]
    designs_predict = tf.squeeze(decoder.predict(decoder_in, verbose=0))
    return designs_predict


# %%

id_base = "10001"

num_pred = 10

z_sample =  np.random.randn(num_pred, latent_dim)


for idx in samples:
    plt.plot(y_test[idx, :], label = f"Sample {idx}")

    target_spec = y_test[idx, :].numpy().flatten() if hasattr(y_test[idx, :], 'numpy') else y_test[idx, :].flatten()

    
    inference_id = int(f"{id_base}{idx}")
    

    decoder_out = make_infrerence(decoder, z_sample, target_spec)


    decoder_out_physical = scaler_geo.inverse_transform(decoder_out)


    inference_parameters = {
    "latent_sampling_details": "np.random.randn(num_pred, latent_dim) ",
    "num_predictions": int(num_pred),
    "z_samples": z_sample.tolist(),
    }

    save_inference(
        inference_id=str(inference_id),
        inference_params=inference_parameters,
        inference_output=decoder_out.numpy() if hasattr(decoder_out, 'numpy') else decoder_out,
        target=target_spec,
        target_id=str(idx),
        wavelengths=wavelengths,
        scaler_geo=scaler_geo,
        base_dir=directory_base_inference
        )

    plt.ylim([0,1])
    plt.legend()
    plt.show()


# %%

import slurm_job_submitter
import time


for idx in samples:
    
    inference_id = f"{id_base}{idx}"

    inference_data_folder = directory_base_inference + f"results/run_{inference_id}" # os.path.join(f"{cVAE_type}_cVAE", f"inference_data/results/run_{inference_id}")

    local_param_file = os.path.join(inference_data_folder, f"{inference_id}_params.csv")

    remote_path = "/home/okcj1g19/micropillar_resim/" # Location on Iridis where files will be copied to

    job_script = "js_micropillar_resim_dis.sh"

    slurm_job_submitter.run_submission(inference_id, local_param_file, remote_path, job_script)#
    
    time.sleep(1)
# %%

from helper import read_comsol_csv, plot_comsol

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

samples = [462, 325, 53, 197, 86, 439, 214, 87, 243]

for idx in samples:
    
    ID = f"10001{idx}"


    fig, axes = plt.subplots(1, 1, figsize=(12, 6))


    # ID = "7"

    run_folder = f"geomerty_cVAE/model_latent8_blocks1/inference_data/results/run_{ID}/"

    # df = read_comsol_csv(run_folder / "micro_pillar_resim.csv")
    
    try:
        df = read_comsol_csv(run_folder + (ID + "_dataout.csv"))

    except FileNotFoundError:
        print(f"File not found for ID: {ID}")
        continue

    target = np.loadtxt(run_folder + (ID + "_target_spectra.csv"), delimiter=",")

    axes.plot(target[:, 0], target[:, 1], label = "Target", linestyle='--', color='red')

    plot_comsol(
        df,
        axes,
        ["Reflection (1)"], # ["abs(emw.S11)^2 (1)"], #
        xdata_indx=6,
        resim=True
    )

    # axes.set_title("Re-simulation of test dataset sample")
    axes.set_title("Re-simulation of suggested designs for test sample " + str(idx))
    axes.set_xlabel("Frequency (THz)")
    axes.set_ylabel("Reflectance")
    axes.legend(title="h_pill, d_pill, w_pill, d_sep, w_sep, angle\num, um, um, um, um, rad")
    #plt.savefig(run_folder / "testdata_micro_pillar_resim.png", dpi=300)
    plt.savefig(run_folder + ID + "_test_index_resim.png", dpi=300)
    plt.show()

# %%
