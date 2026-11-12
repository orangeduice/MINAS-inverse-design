# %%
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.subplots as subplots
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Assuming these are local modules in your working directory
import model_architecture as model_arch
import helper

# %%
# Modern Dynamic GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(e)

data_path = 'datasets/2026_04_15_full_pisa_data.h5'

# Load data and scalers
x_train, x_test, y_train, y_test, wavelengths = helper.load_data_from_h5(
    data_path,
    condition_name='R',
    input_name='geo',
    save_scalers=True)

# Slicing the data as originally defined
y_train = y_train[:, 35:110]
y_test = y_test[:, 35:110]
wavelengths = wavelengths[35:110]

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# %%
keras.backend.clear_session()

# WGAN Configuration
latent_dim = 8    # Same as cVAE for fair comparison
N_blocks = 1      # Nr of resblocks between upsamplings
CRITIC_STEPS = 5  # Train Critic 5 times per Generator step
GP_WEIGHT = 10.0  # Weight for the gradient penalty
BATCH_SIZE = 64

# ==========================================
# 1. GENERATOR (Identical to cVAE Decoder)
# ==========================================
latent_in = keras.Input(shape=(latent_dim,), name='latent_in')
condition_target_spec_input = keras.layers.Input(shape=y_train.shape[1:], name='condition_in')

# spectrum path
x = condition_target_spec_input
x = model_arch.resblock_sequence_down(x, N_filter=32, N_blocks=N_blocks)   
x = model_arch.resblock_sequence_down(x, N_filter=64, N_blocks=N_blocks)   
x = model_arch.resblock_sequence_down(x, N_filter=128, N_blocks=N_blocks)  
x = model_arch.resblock_sequence_down(x, N_filter=256, N_blocks=N_blocks)  
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

generator = keras.Model(
    [latent_in, condition_target_spec_input],
    out_design, name="generator"
)
generator.summary()

# ==========================================
# 2. CRITIC (Adapted from cVAE Encoder)
# ==========================================
design_in = keras.Input(shape=x_train.shape[1:], name='design_in')
condition_in = keras.Input(shape=y_train.shape[1:], name='critic_condition_in')

# spectrum path
x_c = condition_in
x_c = model_arch.resblock_sequence_down(x_c, N_filter=32, N_blocks=N_blocks)
x_c = model_arch.resblock_sequence_down(x_c, N_filter=64, N_blocks=N_blocks)
x_c = model_arch.resblock_sequence_down(x_c, N_filter=128, N_blocks=N_blocks)
x_c_spec = keras.layers.Flatten()(x_c)

# geometry path
x_d = design_in
x_d = model_arch.resblock_sequence_id(x_d, N_filter=128, N_blocks=N_blocks)
x_d_design = keras.layers.Flatten()(x_d)

# Merge and score (Must be linear output for WGAN)
x = keras.layers.Concatenate()([x_c_spec, x_d_design])
x = keras.layers.Dense(256)(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
x = keras.layers.Dense(128)(x)
x = keras.layers.LeakyReLU(alpha=0.2)(x)
# No activation function - output is a raw Wasserstein distance score
critic_out = keras.layers.Dense(1, name="critic_score")(x)

critic = keras.Model([design_in, condition_in], critic_out, name="critic")
critic.summary()


# ==========================================
# 3. WGAN-GP MODEL CLASS
# ==========================================
class CWGAN_GP(keras.Model):
    def __init__(self, critic, generator, latent_dim, critic_steps=5, gp_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

        self.c_loss_tracker = keras.metrics.Mean(name="c_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.c_loss_tracker, self.g_loss_tracker]

    def compile(self, c_optimizer, g_optimizer, **kwargs):
        super().compile(**kwargs)
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_designs, fake_designs, conditions):
        # 1. Get the interpolated designs
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_designs - real_designs
        interpolated = real_designs + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 2. Get the critic's output for the interpolated design
            pred = self.critic([interpolated, conditions], training=True)

        # 3. Calculate gradients of the predictions wrt the interpolated images
        grads = gp_tape.gradient(pred, [interpolated])[0]
        
        # 4. Calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        
        # 5. Return penalty: (norm - 1)^2
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        real_designs, conditions = data
        batch_size = tf.shape(real_designs)[0]

        # =================================================================
        # Train the Critic (runs multiple times per generator step)
        # =================================================================
        for _ in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            with tf.GradientTape() as tape:
                # Generate fake designs
                fake_designs = self.generator([random_latent_vectors, conditions], training=True)
                
                # Get the critic scores
                fake_logits = self.critic([fake_designs, conditions], training=True)
                real_logits = self.critic([real_designs, conditions], training=True)

                # Calculate Wasserstein Critic Loss: D(fake) - D(real) + GP
                c_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_designs, fake_designs, conditions)
                c_loss = c_cost + gp * self.gp_weight

            # Apply Critic Gradients
            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        # =================================================================
        # Train the Generator (runs once)
        # =================================================================
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_designs = self.generator([random_latent_vectors, conditions], training=True)
            gen_logits = self.critic([generated_designs, conditions], training=True)
            
            # Calculate Generator Loss: -D(fake)
            g_loss = -tf.reduce_mean(gen_logits)

        # Apply Generator Gradients
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # Update metrics
        self.c_loss_tracker.update_state(c_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {
            "c_loss": self.c_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result()
        }

# %%
# ==========================================
# TRAINING SETUP & DATASET
# ==========================================
# Using tf.data.Dataset is highly recommended for custom GAN loops
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE, drop_remainder=True)

cwgan = CWGAN_GP(
    critic=critic,
    generator=generator,
    latent_dim=latent_dim,
    critic_steps=CRITIC_STEPS,
    gp_weight=GP_WEIGHT
)

# WGANs often require lower learning rates than standard models (Adam betas are also often tweaked)
c_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

cwgan.compile(
    c_optimizer=c_optimizer,
    g_optimizer=g_optimizer
)

# %%
# ==========================================
# TRAINING EXECUTION
# ==========================================
EPOCHS = 100
print(f"--- Starting cWGAN-GP Training for {EPOCHS} Epochs ---")

history = cwgan.fit(
    train_dataset,
    epochs=EPOCHS
)

# %%
# ==========================================
# PLOTTING
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(history.history['c_loss'], label='Critic Loss (c_loss)', color='tab:blue')
plt.plot(history.history['g_loss'], label='Generator Loss (g_loss)', color='tab:red')
plt.title('WGAN-GP Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# NOTE: In GANs, these loss lines will not converge to 0. 
# A stable GAN usually shows the losses fluctuating around an equilibrium.