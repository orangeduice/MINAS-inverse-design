# %%
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.decomposition import PCA
import umap

# Assuming these are local modules in your working directory
import model_architecture as model_arch
import helper


# %%
# ==========================================
# SETUP & UTILS
# ==========================================
def setup_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s).")
        except RuntimeError as e:
            print(e)

# %%
# ==========================================
# ARCHITECTURE BUILDERS
# ==========================================
def build_encoder(input_shape_design, input_shape_condition, latent_dim, n_blocks):
    condition_target_spec_input = keras.layers.Input(shape=input_shape_condition, name='condition_in')
    design_in = keras.Input(shape=input_shape_design, name='design_in')

    # Spectrum path
    x = condition_target_spec_input
    x = model_arch.resblock_sequence_down(x, N_filter=32, N_blocks=n_blocks)   
    x = model_arch.resblock_sequence_down(x, N_filter=64, N_blocks=n_blocks)   
    x = model_arch.resblock_sequence_down(x, N_filter=128, N_blocks=n_blocks)  
    x = model_arch.resblock_sequence_down(x, N_filter=256, N_blocks=n_blocks)  
    x_spec = keras.layers.Flatten()(x)

    # Geometry path
    x = design_in
    x = model_arch.resblock_sequence_id(x, N_filter=128, N_blocks=n_blocks)    
    x_design = keras.layers.Flatten()(x)

    # Merge paths
    x = keras.layers.Concatenate()([x_spec, x_design])
    x = keras.layers.Dense(256)(x)
    x = keras.layers.LeakyReLU(alpha=1e-1)(x)

    # Output: latent sampling
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_var_log = keras.layers.Dense(latent_dim, name="z_var_log")(x)
    z = model_arch.SamplingRNG()([z_mean, z_var_log])

    encoder = keras.Model(
        [design_in, condition_target_spec_input], 
        [z_mean, z_var_log, z], 
        name="encoder"
    )
    return encoder

def build_decoder(latent_dim, input_shape_condition, output_channels, n_blocks):
    latent_in = keras.Input(shape=(latent_dim,), name='latent_in')
    condition_target_spec_input = keras.layers.Input(shape=input_shape_condition, name='condition_in_dec')

    # Spectrum path
    x = condition_target_spec_input
    x = model_arch.resblock_sequence_down(x, N_filter=32, N_blocks=n_blocks)   
    x = model_arch.resblock_sequence_down(x, N_filter=64, N_blocks=n_blocks)   
    x = model_arch.resblock_sequence_down(x, N_filter=128, N_blocks=n_blocks)  
    x = model_arch.resblock_sequence_down(x, N_filter=256, N_blocks=n_blocks)  
    x_spec = keras.layers.Flatten()(x)

    # Merge with latent
    x = keras.layers.Concatenate()([x_spec, latent_in])

    # Geometry generator path
    x = keras.layers.Dense(6 * 128)(x)
    x = keras.layers.Reshape((6, 128))(x)
    x = model_arch.resblock_sequence_id(x, N_filter=128, N_blocks=n_blocks)

    # Output
    out_design = keras.layers.Conv1D(
        filters=output_channels, kernel_size=1, name='design_out')(x)

    decoder = keras.Model(
        [latent_in, condition_target_spec_input],
        out_design, name="decoder"
    )
    return decoder

# %%
# ==========================================
# CVAE MODEL CLASS
# ==========================================
class CVAEModel(keras.Model):
    def __init__(self, encoder, decoder, latent_dim, beta=1.0, surrogate_model=None, spectral_weight=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = float(latent_dim)
        self.beta = float(beta) # Static beta replacing tf.Variable
        self.surrogate_model = surrogate_model
        self.spectral_weight = float(spectral_weight)
        
        # Setup metric trackers
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstr_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.spectral_loss_tracker = keras.metrics.Mean(name="spec_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.spectral_loss_tracker,
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
            z_mean, z_var_log, z = self.encoder([design_in, condition_in])
            design_pred = self.decoder([z, condition_in])

            flat_in = keras.layers.Flatten()(design_in)
            flat_pred = keras.layers.Flatten()(design_pred)

            reconstruction_loss = keras.ops.mean(keras.losses.mae(flat_in, flat_pred))
            kl_loss = keras.ops.sum(
                1 + z_var_log - keras.ops.square(z_mean) - keras.ops.exp(z_var_log), axis=-1
            )
            # Use the static self.beta
            kl_loss = self.beta * keras.ops.mean(-0.5 / self.latent_dim * kl_loss)
            
            # Forward pass through surrogate for spectral mismatch loss
            spectral_loss = 0.0
            if self.surrogate_model is not None and self.spectral_weight > 0.0:
                spec_pred = self.surrogate_model(design_pred, training=False)
                flat_cond = keras.layers.Flatten()(condition_in)
                flat_spec_pred = keras.layers.Flatten()(spec_pred)
                spectral_loss = keras.ops.mean(keras.losses.mse(flat_cond, flat_spec_pred))
                
            total_loss = reconstruction_loss + kl_loss + (self.spectral_weight * spectral_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.spectral_loss_tracker.update_state(spectral_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstr_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "spec_loss": self.spectral_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        else:
            x = data
            
        design_in = x[0]
        condition_in = x[1]
        
        z_mean, z_var_log, z = self.encoder([design_in, condition_in])
        design_pred = self.decoder([z, condition_in])

        flat_in = keras.layers.Flatten()(design_in)
        flat_pred = keras.layers.Flatten()(design_pred)

        reconstruction_loss = keras.ops.mean(keras.losses.mae(flat_in, flat_pred))
        kl_loss = keras.ops.sum(
            1 + z_var_log - keras.ops.square(z_mean) - keras.ops.exp(z_var_log), axis=-1
        )
        kl_loss = self.beta * keras.ops.mean(-0.5 / self.latent_dim * kl_loss)
        
        # Forward pass through surrogate for spectral mismatch loss
        spectral_loss = 0.0
        if self.surrogate_model is not None and self.spectral_weight > 0.0:
            spec_pred = self.surrogate_model(design_pred, training=False)
            flat_cond = keras.layers.Flatten()(condition_in)
            flat_spec_pred = keras.layers.Flatten()(spec_pred)
            spectral_loss = keras.ops.mean(keras.losses.mse(flat_cond, flat_spec_pred))
        
        total_loss = reconstruction_loss + kl_loss + (self.spectral_weight * spectral_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.spectral_loss_tracker.update_state(spectral_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstr_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "spec_loss": self.spectral_loss_tracker.result(),
        }











# %%
# ==========================================
# EVALUATION & VISUALIZATION UTILS
# ==========================================
def analyze_latent_space_from_arrays(encoder, x_test, y_test, batch_size=128):
    all_mus = []
    all_log_vars = []
    num_samples = len(x_test)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        mu, log_var, _ = encoder.predict([x_batch, y_batch], verbose=0)
        
        all_mus.append(mu)
        all_log_vars.append(log_var)
        
    mus = np.concatenate(all_mus, axis=0)
    log_vars = np.concatenate(all_log_vars, axis=0)
    latent_dim = mus.shape[1]
    
    kl_per_sample_per_dim = -0.5 * (1 + log_vars - np.square(mus) - np.exp(log_vars))
    mean_kl_per_dim = np.mean(kl_per_sample_per_dim, axis=0)
    var_of_means_per_dim = np.var(mus, axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    dimensions = np.arange(latent_dim)
    
    ax1.bar(dimensions, mean_kl_per_dim, color='royalblue', alpha=0.8, edgecolor='black')
    ax1.set_title("Mean KL Divergence per Latent Dimension")
    ax1.set_xlabel("Latent Dimension")
    ax1.set_ylabel("KL Divergence")
    ax1.set_xticks(dimensions)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
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
    
    active_units = np.sum(var_of_means_per_dim > 0.01)
    print("Diagnostics Summary:")
    print("--------------------")
    print(f"Total latent dimensions: {latent_dim}")
    print(f"Active dimensions (variance > 0.01): {active_units} / {latent_dim}")
    print(f"Inactive/Collapsed dimensions: {latent_dim - active_units}")

def test_decoder_neglect(encoder, decoder, x_test, y_test, batch_size=128):
    mse_posterior_list, mse_prior_list = [], []
    num_samples = len(x_test)
    latent_dim = encoder.output[0].shape[-1] 
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        x_batch = x_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        current_batch_size = end_idx - start_idx
        
        mu, log_var, z_posterior = encoder.predict([x_batch, y_batch], verbose=0)
        reconstruction_posterior = decoder.predict([z_posterior, y_batch], verbose=0)
        
        z_prior = np.random.normal(size=(current_batch_size, latent_dim))
        reconstruction_prior = decoder.predict([z_prior, y_batch], verbose=0)
        
        mse_posterior_list.append(np.mean(np.square(x_batch - reconstruction_posterior)))
        mse_prior_list.append(np.mean(np.square(x_batch - reconstruction_prior)))
        
    avg_mse_posterior = np.mean(mse_posterior_list)
    avg_mse_prior = np.mean(mse_prior_list)
    
    print("--- Decoder Reliance Test Results ---")
    print(f"MSE with Encoder z  (q(z|x,y)): {avg_mse_posterior:.6f}")
    print(f"MSE with Random z   (N(0, I)) : {avg_mse_prior:.6f}")
    print(f"Absolute Gap                  : {abs(avg_mse_prior - avg_mse_posterior):.6f}")
    
    ratio = avg_mse_prior / (avg_mse_posterior + 1e-8)
    if ratio < 1.05:
        print("\nConclusion: The Decoder is IGNORING z! (Error barely changes)")
    else:
        print(f"\nConclusion: The Decoder uses z (Error increases by {ratio:.2f}x with random z).")

def visualize_latent_space(model, x_data, y_data, batch_size=64, 
                           method='raw', dims=(0, 1), color_values=None, 
                           save_path=None, cmap='viridis', sample = False, N=10):
    latents = []
    print("Extracting latent representations...")
    total_samples = len(x_data)
    
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_x = tf.convert_to_tensor(x_data[start_idx:end_idx], dtype=tf.float32)
        batch_y = tf.convert_to_tensor(y_data[start_idx:end_idx], dtype=tf.float32)
        
        encode_outputs = model([batch_x, batch_y], training=False)
        mu = encode_outputs[0] if isinstance(encode_outputs, (list, tuple)) else encode_outputs
        latents.append(mu.numpy())
        
    latents = np.concatenate(latents, axis=0)
    print(f"Plotting using method: {method.upper()}")
    
    if method == 'raw':
        max_dim = latents.shape[1] - 1
        d1, d2 = min(dims[0], max_dim), min(dims[1], max_dim)
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
        d1, d2 = min(dims[0], latents.shape[1] - 1), min(dims[1], latents.shape[1] - 1)
        latents_2d = latents[:, [d1, d2]]
        title_prefix = f"Latent Dims {d1} & {d2} of "
        xlabel, ylabel = f"Latent Dim {d1}", f"Latent Dim {d2}"
    
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(
        latents_2d[:, 0], latents_2d[:, 1], 
        c=color_values if color_values is not None else '#1f77b4', 
        cmap=cmap if color_values is not None else None, 
        alpha=0.8, s=8, edgecolors='none'
    )
    if color_values is not None:
        plt.colorbar(scatter, label="Color Value")


    if sample:
        # We select indices from the subset of data we actually processed
        highlight_indices = np.random.choice(len(latents), size=N, replace=False)

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

        #plt.title(f"{title_prefix}cVAE Latent Space with {N} Highlighted Samples\n(Background Colored by {title_suffix})")
        #plt.xlabel(f"{method.upper()} 1" if method in ['umap', 'pca'] else "Latent Dim 1")
        #plt.ylabel(f"{method.upper()} 2" if method in ['umap', 'pca'] else "Latent Dim 2")

    
    plt.title(f"{title_prefix}cVAE Latent Space (N={len(latents)})")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.show()

    return highlight_indices if sample else None

def generate_min_index_colors(y_data):
    y_reshaped = y_data.reshape(y_data.shape[0], -1)
    return np.argmin(y_reshaped, axis=1)

# %%
# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'data_path': 'datasets/2026_04_15_full_pisa_data.h5',
    'latent_dim': 6,
    'n_blocks': 2,
    'static_beta': 0.005, # Static beta replaces KL Annealing
    'spectral_loss_weight': 0.0, # Weight for spectral mismatch loss in training
    'learning_rate': 0.0001,
    'batch_size_rounds': [
        {'multiplier': 1, 'epoch_multiplier': 1.0},
        {'multiplier': 2, 'epoch_multiplier': 0.5}
    ],
    'base_epochs': 25,
    'base_batch_size': 8,
    'surrogate_path': "surrogate_model/surrogate_geo_to_spec.keras",
    'checkpoint_base_dir': "geomerty_cVAE"
}

# %%
# ==========================================
# MAIN EXECUTION
# ==========================================
setup_gpus()
keras.backend.clear_session()


# 1. Load Data
print("Loading data...")
x_train, x_test, y_train, y_test, wavelengths = helper.load_data_from_h5(
    CONFIG['data_path'], condition_name='R', input_name='geo', save_scalers=True)

#y_train = y_train[:, 35:110]
#y_test = y_test[:, 35:110]
#wavelengths = wavelengths[35:110]

scaler_path = f"{os.path.splitext(CONFIG['data_path'])[0]}_scalers.pkl"
scaler_geo = pickle.load(open(scaler_path, 'rb'))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# %%
# 2. Build Model Architecture
print("Building models...")
encoder = build_encoder(
    input_shape_design=x_train.shape[1:], 
    input_shape_condition=y_train.shape[1:], 
    latent_dim=CONFIG['latent_dim'], 
    n_blocks=CONFIG['n_blocks']
)
decoder = build_decoder(
    latent_dim=CONFIG['latent_dim'], 
    input_shape_condition=y_train.shape[1:], 
    output_channels=x_train.shape[-1], 
    n_blocks=CONFIG['n_blocks']
)

encoder.summary()
decoder.summary()

try:
    surrogate_model = keras.models.load_model(CONFIG['surrogate_path'])
    surrogate_model.trainable = False # Lock weights so CVAE training doesn't alter it
    print("\nLoaded pre-trained forward surrogate model successfully.")
except Exception as e:
    print(f"\nWarning: Could not load surrogate model. Error: {e}")
    surrogate_model = None

cVAE = CVAEModel(
    encoder=encoder, 
    decoder=decoder, 
    latent_dim=CONFIG['latent_dim'], 
    beta=CONFIG['static_beta'],
    surrogate_model=surrogate_model,
    spectral_weight=CONFIG['spectral_loss_weight']
)

checkpoint_dir = f"{CONFIG['checkpoint_base_dir']}/model_latent{CONFIG['latent_dim']}_blocks{CONFIG['n_blocks']}/checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

# %%
# 3. Setup Callbacks
optimizer = keras.optimizers.AdamW(learning_rate=CONFIG['learning_rate'])
cVAE.compile(optimizer=optimizer) 

callbacks = [
    #ReduceLROnPlateau(monitor='val_loss', factor=0.66, patience=5, verbose=1),
    #EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
]

# if surrogate_model is not None:
#     callbacks.append(SpectralMismatchCallback(
#         decoder=decoder, surrogate_model=surrogate_model, 
#         y_val=y_test, latent_dim=CONFIG['latent_dim']
#     ))

# %%
# 4. Training Loop
hist = None
for round_cfg in CONFIG['batch_size_rounds']:
    batch_size = CONFIG['base_batch_size'] * (2 ** round_cfg['multiplier'])
    epochs_this_round = int(CONFIG['base_epochs'] * round_cfg['epoch_multiplier'])
    
    print(f"\n--- Starting Training Round (batch_size={batch_size}, epochs={epochs_this_round}) ---")
    _h = cVAE.fit(
        x=[x_train, y_train], 
        y=x_train, 
        validation_split=0.3,
        batch_size=batch_size,
        epochs=epochs_this_round,
        #callbacks=callbacks 
    )

    if hist is None:
        hist = _h
    else:
        for k in hist.history:
            hist.history[k] = np.concatenate([hist.history[k], _h.history[k]])

    if cVAE.stop_training:
        print("Early stopping triggered. Halting training.")
        break

# %%
# 5. Plotting Training Curves
print("Generating training plots...")
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss (log scale)', color=color)
ax1.plot(hist.history['loss'], color=color, linestyle='-', label='Total Loss (Train)')
ax1.plot(hist.history['val_loss'], color=color, linestyle='--', label='Total Loss (Val)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')

ax2 = ax1.twinx()
ax2.set_ylabel('Component Losses')
ax2.set_yscale('log')

color_recon = 'tab:blue'
ax2.plot(hist.history['reconstr_loss'], color=color_recon, linestyle='-', label='Recon Loss (Train)')
ax2.plot(hist.history['val_reconstr_loss'], color=color_recon, linestyle='--', label='Recon Loss (Val)')

color_kl = 'tab:green'
ax2.plot(hist.history['kl_loss'], color=color_kl, linestyle='-', label='KL Loss (Train)')
ax2.plot(hist.history['val_kl_loss'], color=color_kl, linestyle='--', label='KL Loss (Val)')

if 'spec_loss' in hist.history and sum(hist.history['spec_loss']) > 0:
    color_specloss = 'tab:orange'
    ax2.plot(hist.history['spec_loss'], color=color_specloss, linestyle='-', label='Surrogate Loss (Train)')
    ax2.plot(hist.history['val_spec_loss'], color=color_specloss, linestyle='--', label='Surrogate Loss (Val)')

if 'val_spectral_mismatch' in hist.history:
    color_spec = 'tab:purple'
    ax2.plot(hist.history['val_spectral_mismatch'], color=color_spec, linestyle='-.', 
                linewidth=2.5, label='Spectral Mismatch (Val)')

ax2.tick_params(axis='y')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('Training and Validation Losses')
plot_path = f"{CONFIG['checkpoint_base_dir']}/model_latent{CONFIG['latent_dim']}_blocks{CONFIG['n_blocks']}/training_losses.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
# %%

# 6. Evaluation & Latent Diagnostics
print("\nRunning Diagnostics...")
analyze_latent_space_from_arrays(encoder, x_test, y_test)
test_decoder_neglect(encoder, decoder, x_test, y_test)

x_full = np.concatenate([x_train, x_test], axis=0)
y_full = np.concatenate([y_train, y_test], axis=0)

color_values = {}

frequency_of_min = wavelengths[generate_min_index_colors(y_full)]
color_values['frequency of minimum reflectance'] = frequency_of_min

split_mask = np.array([0] * len(x_train) + [1] * len(x_test))
color_values['train-test split'] = split_mask

# h_pill, d_pill, w_pill, d_sep, w_sep, angle
geom_param_h_pill = x_full.squeeze()[:, 0]
color_values['h_pill'] = geom_param_h_pill
geom_param_d_pill = x_full.squeeze()[:, 1]
color_values['d_pill'] = geom_param_d_pill
geom_param_w_pill = x_full.squeeze()[:, 2]
color_values['w_pill'] = geom_param_w_pill
geom_param_d_sep = x_full.squeeze()[:, 3]
color_values['d_sep'] = geom_param_d_sep
geom_param_w_sep = x_full.squeeze()[:, 4]
color_values['w_sep'] = geom_param_w_sep
geom_param_angle = x_full.squeeze()[:, 5]
color_values['theta_pill'] = geom_param_angle

# %%
for key, value in color_values.items():
    print(f"\nVisualizing latent space colored by: {key}")

    visualize_latent_space(
        encoder, 
        x_full, 
        y_full, 
        method='umap', 
        color_values=value,
        save_path=f"{CONFIG['checkpoint_base_dir']}/model_latent{CONFIG['latent_dim']}_blocks{CONFIG['n_blocks']}/latent_umap_{key.replace(' ', '_')}.png"
    )


# %%
# 7. Save models
decoder.save("decoder.keras")
print("Training pipeline complete.")

# %%
#8. Running inference on a random sample from the test set


# %%
