# %%

import pickle
import os

from experiment_tracker import ExperimentTracker

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import model_architecture as model_arch
import helper

# %%
# As the original model_architecture module was not provided,
# we define the SamplingRNG layer directly in the script.
class SamplingRNG(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# %% --- Data Loading and Preparation ---

# Dynamic GPU memory allocation: For local prototyping, don't block GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

data_path = 'datasets/pisa_data_expanded_h.h5'

# load data and scalers
x_train, x_test, y_train, y_test, wavelengths = helper.load_reflection_spectra_data(
    data_path, save_scalers=True) # Save scalers for first time

print(wavelengths)
print(wavelengths.shape)

scaler_geo = pickle.load(
    open('{}_scalers.pkl'.format(os.path.splitext(data_path)[0]), 'rb'))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)


# %% --- cVAE Model Definition ---

keras.backend.clear_session()

# --- HYPERPARAMETERS ---
latent_dim = 4      # z_dim
hidden_size = 64
hidden_num = 2
learning_rate = 0.002   # Reduced learning rate
batch_size = 64
validation_split = 0.1
beta = 0.01           # alpha - Drastically reduced to prioritize reconstruction
max_epochs = 250
dropout_rate = 0.01     # Added dropout for regularization

# --- SYMMETRICAL ARCHITECTURE CONFIGURATION ---
# Dynamically create the architecture based on hyperparameters
core_architecture = [hidden_size] * hidden_num

# --- Input Layers ---
condition_target_spec_input = keras.layers.Input(shape=y_train.shape[1:], name='condition_in')
design_in = keras.Input(shape=x_train.shape[1:], name='design_in')

# --- ENCODER MODEL ---
spec_flat = keras.layers.Flatten()(condition_target_spec_input)
design_flat = keras.layers.Flatten()(design_in)
# Concatenate inputs immediately
x = keras.layers.Concatenate()([spec_flat, design_flat])

# Pass through the core encoder architecture with dropout
for num_neurons in core_architecture:
    x = keras.layers.Dense(num_neurons, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)

x = keras.layers.LeakyReLU(alpha=1e-1)(x)
z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
z_var_log = keras.layers.Dense(latent_dim, name='z_log_var')(x)
z = SamplingRNG()([z_mean, z_var_log])
encoder = keras.Model([design_in, condition_target_spec_input], [z_mean, z_var_log, z], name="encoder")
print("\n--- Encoder Summary ---")
encoder.summary()


# --- DECODER MODEL ---
latent_in = keras.Input(shape=(latent_dim,), name='latent_in')
spec_flat_d = keras.layers.Flatten()(condition_target_spec_input)
# Concatenate latent vector and condition immediately
x = keras.layers.Concatenate()([latent_in, spec_flat_d])

# Pass through the mirrored core architecture with dropout
decoder_architecture = [hidden_size] * hidden_num
for num_neurons in decoder_architecture:
    x = keras.layers.Dense(num_neurons, activation='relu')(x)
    x = keras.layers.Dropout(dropout_rate)(x)

output_dim = np.product(x_train.shape[1:])
x = keras.layers.Dense(output_dim, activation='relu')(x)
out_design = keras.layers.Reshape(x_train.shape[1:], name='design_out')(x)
decoder = keras.Model([latent_in, condition_target_spec_input], out_design, name="decoder")
print("\n--- Decoder Summary ---")
decoder.summary()

# --- FULL cVAE MODEL ---
z_mean_out, z_var_log_out, z_rnd_out = encoder([design_in, condition_target_spec_input])
design_pred = decoder([z_rnd_out, condition_target_spec_input])
cVAE = keras.Model([design_in, condition_target_spec_input], design_pred, name="cVAE")
print("\n--- cVAE Summary ---")
cVAE.summary()


# %% --- Loss Functions and Compilation ---
reconstruction_loss = keras.losses.mse(keras.backend.flatten(design_in), keras.backend.flatten(design_pred))
cVAE.add_loss(reconstruction_loss)
cVAE.add_metric(reconstruction_loss, name='reconstr_loss', aggregation='mean')
kl_loss = keras.backend.sum(1 + z_var_log_out - keras.backend.square(z_mean_out) - keras.backend.exp(z_var_log_out), axis=-1)
kl_loss *= -0.5 / latent_dim
kl_loss = beta * keras.backend.mean(kl_loss)
cVAE.add_loss(kl_loss)
cVAE.add_metric(kl_loss, name='kl_loss', aggregation='mean')
optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
cVAE.compile(optimizer=optimizer)

# %% --- Model Training ---
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=50, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)

print(f"\n--- Starting Training ---")
history = cVAE.fit(
    x=[x_train, y_train],
    y=[x_train],
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=max_epochs,
    callbacks=[reduce_lr, early_stopping],
    verbose=2
)
hist_dict = history.history
print("\n--- Training Finished ---")

# %%

print("\n--- Plotting Results ---")

# 1. Plot Training History
plt.figure(figsize=(8, 6))
plt.plot(hist_dict['loss'], label='Training Loss')
plt.plot(hist_dict['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

#%%
# %%
# Save everything

tracker = ExperimentTracker()

cvae_hyperparams = {
        "misc": "expanded h, dense nn test, 15x4 [2, 2, 3, 3] 32 * (2**round_num)"
    }

train_id = tracker.create_training_run(
        model_type="cVAE_nn",
        hyperparameters=cvae_hyperparams,
        wavelengths=wavelengths.tolist(),
        # test_loss=hist.history['loss'],
        # val_loss=hist.history['val_loss'],
    )

model_folder = "models"

full_model_file = os.path.join(model_folder, str(train_id), "pisa_cVAE_full.h5")
encoder_file = os.path.join(model_folder, str(train_id), "pisa_cVAE_enc.h5")
decoder_file = os.path.join(model_folder, str(train_id), "pisa_cVAE_dec.h5")

cVAE.save(full_model_file)
encoder.save(encoder_file)
decoder.save(decoder_file)

print(f"Logged new training run. Train ID: {train_id}")
print(f"Data saved in: {tracker.training_dir}/{train_id}.json")
print("saved models to ", full_model_file, encoder_file, decoder_file)
# %%
# %% --- INFERENCE EXAMPLES ---
print("\n--- Performing Inference ---")

def to_physical(scaled_data, scaler):
    """Reshapes, inverse transforms, and reshapes back."""
    original_shape = scaled_data.shape
    # Reshape from (n_samples, geo_size, 1) to (n_samples, geo_size) for scaler
    scaled_2d = scaled_data.reshape(original_shape[0], -1)
    physical_2d = scaler.inverse_transform(scaled_2d)
    # Reshape back to the original 3D shape
    return physical_2d.reshape(original_shape)

# 1. GENERATION: Create new designs by sampling from the latent space
num_predictions = 5
random_latent_vectors = np.random.randn(num_predictions, latent_dim)
condition_spectra = y_test[:num_predictions]
generated_designs_scaled = decoder.predict([random_latent_vectors, condition_spectra])
generated_designs_physical = to_physical(generated_designs_scaled, scaler_geo)

print(f"\n--- {num_predictions} Generated Designs (Physical Units) ---")
for i in range(num_predictions):
    print(f"Design {i+1}: {generated_designs_physical[i].flatten()}")

# 2. RECONSTRUCTION: See how well the model can reproduce a known design
design_to_reconstruct_scaled = x_test[:1]
spectrum_for_reconstruction = y_test[:1]
z_mean_orig, _, z_orig = encoder.predict([design_to_reconstruct_scaled, spectrum_for_reconstruction])
reconstructed_design_scaled = decoder.predict([z_orig, spectrum_for_reconstruction])
reconstructed_design_physical = to_physical(reconstructed_design_scaled, scaler_geo)
original_design_physical = to_physical(design_to_reconstruct_scaled, scaler_geo)

print("\n--- Reconstruction Example (Physical Units) ---")
print("Original Design:      ", original_design_physical.flatten())
print("Reconstructed Design: ", reconstructed_design_physical.flatten())


def create_reflectance_target(freq, center_freq, width, depth, order=2.0):
    """Creates a target reflectance spectrum with a super-Gaussian dip."""
    # This formula creates a dip with controllable flatness via the 'order' parameter.
    sigma_equivalent = width / (2 * np.sqrt(2 * np.log(2)))
    exponent = ((freq - center_freq) / sigma_equivalent) ** order
    reflectance = 1.0 - depth * np.exp(-exponent)
    return reflectance


# 3. LATENT SPACE VISUALIZATION (for 2D latent space)
if latent_dim == 2:
    print("\n--- Visualizing Latent Space with Custom Target Spectrum ---")

    # --- User-defined Target Spectrum ---
    # To make the target visible, we center it in the placeholder wavelength range
    target_frequency = 26.0# np.mean(wavelengths)
    # The width is set to be 1/5th of the total wavelength range
    dip_width = 0.1#(wavelengths[-1] - wavelengths[0]) / 5.0
    dip_depth = 1.0
    flatness_order = 16.0

    target_spectra_1d = create_reflectance_target(
        wavelengths,
        target_frequency,
        dip_width,
        dip_depth,
        order=flatness_order
    )

    # Reshape the 1D target spectrum to the shape the model expects: (1, spec_size, 1)
    viz_condition_spectrum = target_spectra_1d.reshape(1, -1, 1).astype('float32')
    # --- End of Custom Spectrum ---

    n = 15
    grid_x, grid_y = np.linspace(-2, 2, n), np.linspace(-2, 2, n)
    viz_conditions = np.repeat(viz_condition_spectrum, n*n, axis=0)
    grid_vectors = np.array([[xi, yi] for yi in grid_y for xi in grid_x])
    generated_designs_grid_scaled = decoder.predict([grid_vectors, viz_conditions])
    generated_designs_grid_physical = to_physical(generated_designs_grid_scaled, scaler_geo)
    print(f"\nGenerated {len(generated_designs_grid_physical)} designs for latent space visualization.")


# %% --- PLOTTING RESULTS ---
# 2. Plot Reconstruction Comparison
original_params = original_design_physical.flatten()
recon_params = reconstructed_design_physical.flatten()
param_indices = np.arange(len(original_params))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(param_indices - bar_width/2, original_params, bar_width, label='Original')
rects2 = ax.bar(param_indices + bar_width/2, recon_params, bar_width, label='Reconstructed')

ax.set_ylabel('Physical Value')
ax.set_title('Original vs. Reconstructed Design Parameters')
ax.set_xticks(param_indices)
ax.set_xticklabels([f'Param {i+1}' for i in param_indices])
ax.legend()
fig.tight_layout()
plt.show()

# 3. Plot Latent Space Visualization
if latent_dim == 2:
    # First, plot the target spectrum that was used as the condition
    plt.figure(figsize=(8, 6))
    plt.plot(wavelengths, target_spectra_1d, label='Target Spectrum Condition', color='r', linewidth=2)
    plt.title('Target Spectrum for Latent Space Visualization')
    plt.xlabel('Wavelength / Frequency')
    plt.ylabel('Reflectance')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Now, plot the heatmaps of the generated parameters
    geo_size = generated_designs_grid_physical.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Latent Space Visualization of Geometry Parameters', fontsize=16)
    
    axes = axes.flatten()

    for i in range(geo_size):
        ax = axes[i]
        param_values = generated_designs_grid_physical[:, i, 0]
        
        sc = ax.scatter(grid_vectors[:, 0], grid_vectors[:, 1], c=param_values, cmap='viridis')
        ax.set_title(f'Parameter {i+1}')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        fig.colorbar(sc, ax=ax, label=f'Param {i+1} Value')

    for j in range(geo_size, len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %%
