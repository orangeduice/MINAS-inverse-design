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

# Dynamic GPU memory allocation: For local prototyping, don't block GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

data_path = 'datasets/pisa_data.h5'

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

# %%

keras.backend.clear_session()

# some config for the cVAE
latent_dim = 2   # for interpretability
N_blocks = 3     # Nr of resblocks between upsamplings


# input layers
condition_target_spec_input = keras.layers.Input(
    shape=y_train.shape[1:], name='condition_in')
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
z_mean = keras.layers.Dense(latent_dim)(x)
z_var_log = keras.layers.Dense(latent_dim)(x)
z = model_arch.SamplingRNG()([z_mean, z_var_log])


encoder = keras.Model([design_in, condition_target_spec_input], [
                    z_mean, z_var_log, z], name="encoder")
encoder.summary()

# %%

# condition_target_spec_input
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
x = keras.layers.Dense(4*128)(x)
x = keras.layers.Reshape((4, 128))(x)
x = model_arch.resblock_sequence_id(x, N_filter=128, N_blocks=N_blocks)

# output: number of  channels from design
out_design = keras.layers.Conv1D(
    filters=x_train.shape[-1], kernel_size=1, name='design_out')(x)


decoder = keras.Model(
    [latent_in, condition_target_spec_input],
    out_design, name="decoder")
decoder.summary()

# %%

z_mean_out, z_var_log_out, z_rnd_out = encoder([design_in, condition_target_spec_input])
design_pred = decoder([z_rnd_out, condition_target_spec_input])   # use randomized latent

cVAE = keras.Model([design_in, condition_target_spec_input], design_pred, name="cVAE")
cVAE.summary()

# %%

input_dim = np.product(x_train.shape[1:])
latent_dim = np.product(z.shape[1:])
beta = 0.001   # weight of KL loss. If too high, reconstruction will suffer.

# print(input_dim)

# --- add multiple losses to the model
# reconstruction loss
# reconstruction_loss = keras.losses.mse(keras.backend.flatten(design_in), keras.backend.flatten(design_pred))*0.01 + keras.losses.mae(keras.backend.flatten(design_in), keras.backend.flatten(design_pred))
reconstruction_loss = keras.losses.mse(keras.backend.flatten(design_in), keras.backend.flatten(design_pred))
cVAE.add_loss(reconstruction_loss)
cVAE.add_metric(reconstruction_loss, name='reconstr_loss', aggregation='mean')

# KL loss
kl_loss = keras.backend.sum(
    1 + z_var_log_out - keras.backend.square(z_mean_out) - keras.backend.exp(z_var_log_out), axis=-1)
kl_loss = beta * keras.backend.mean(-0.5 / latent_dim * kl_loss)
cVAE.add_loss(kl_loss)
cVAE.add_metric(kl_loss, name='kl_loss', aggregation='mean')

# %%

lr = 0.0002

# compile the model
optimizer = keras.optimizers.AdamW(learning_rate=lr)
cVAE.compile(optimizer=optimizer)

reduce_lr_factor = 1/3
reduce_lr_patience = 5

## ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patience, verbose=1)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)


# Choose a single, reasonable batch size
BATCH_SIZE = 32

# Train in a single, long run. Let the callbacks do their job.
# Set a high number of epochs; early stopping will find the right time to halt.
hist = cVAE.fit(
    x=[x_train, y_train],
    y=[x_train],
    validation_split=.2,
    batch_size=BATCH_SIZE,
    epochs=90,  # A large number
    callbacks=[reduce_lr, early_stopping]
)

# # fit the model with BN --> increasing batchsize schdedule
# hist = None  # global history after BS incrase
# for i in range(4):  # 4x50 epochs, doubling batchsize
#     _h = cVAE.fit(x=[x_train, y_train], y=[x_train],
#                 validation_split=.2,
#                 batch_size=16 * 2**i, epochs=30,
#                 callbacks = [reduce_lr, early_stopping])
#     if hist is None:
#         hist = _h
#     else:
#         for k in hist.history:
#             hist.history[k] = np.concatenate([hist.history[k], _h.history[k]])

# %%
# plot loss
plt.figure()
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.yscale('log')
plt.legend()
plt.savefig("modified_lossFun_loss_curve.png", dpi = 300)
plt.show()

# %%
# Save everything

tracker = ExperimentTracker()

cvae_hyperparams = {
        "latent_dim": int(latent_dim),
        "encoder_resnet_blocks": int(N_blocks),
        "decoder_resnet_blocks": int(N_blocks),
        "learning_rate": lr,
        "epochs": "4 X 30",
        "batch_size_schedule": "16*2^i, i=0..3",
        "dataset": "MINAS_0",
        "beta_kl_loss": beta,
        "reduce_lr_factor": reduce_lr_factor,
        "reduce_lr_patience": reduce_lr_patience,
        "y_dim": "64",
        "misc": "reconstruction loss mse, with early stopping, no br scheduling"
    }

train_id = tracker.create_training_run(
        model_type="cVAE",
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
