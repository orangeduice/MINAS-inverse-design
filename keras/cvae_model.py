import tensorflow as tf
from tensorflow import keras
import numpy as np
import model_architecture as model_arch # Assuming this file exists with your resblock/sampling functions

def build_encoder(design_shape, condition_shape, latent_dim, n_blocks, architecture_type='conv'):
    """
    Builds the Encoder part of the cVAE.

    Args:
        design_shape (tuple): Shape of the design input (e.g., x_train.shape[1:]).
        condition_shape (tuple): Shape of the condition input (e.g., y_train.shape[1:]).
        latent_dim (int): Dimensionality of the latent space.
        n_blocks (int): Number of residual blocks in sequence. (Only for 'conv' architecture)
        architecture_type (str): Type of architecture to use, 'conv' or 'mlp'.

    Returns:
        keras.Model: The compiled Encoder model.
    """
    # Input layers
    design_in = keras.Input(shape=design_shape, name='design_in')
    condition_target_spec_input = keras.layers.Input(shape=condition_shape, name='condition_in')

    if architecture_type == 'conv':
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

    elif architecture_type == 'mlp':
        # Flatten inputs for MLP
        flat_design = keras.layers.Flatten()(design_in)
        flat_condition = keras.layers.Flatten()(condition_target_spec_input)

        # Merge flattened inputs
        x = keras.layers.Concatenate()([flat_design, flat_condition])
        x = keras.layers.Dense(512)(x)
        x = keras.layers.LeakyReLU(alpha=1e-1)(x)
        x = keras.layers.Dense(256)(x)
        x = keras.layers.LeakyReLU(alpha=1e-1)(x)

    else:
        raise ValueError("Unsupported architecture_type. Choose 'conv' or 'mlp'.")


    # Latent space sampling
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_var_log = keras.layers.Dense(latent_dim, name='z_var_log')(x)
    z = model_arch.SamplingRNG()([z_mean, z_var_log])

    encoder = keras.Model(
        [design_in, condition_target_spec_input],
        [z_mean, z_var_log, z],
        name="encoder"
    )
    return encoder

def build_decoder(latent_dim, condition_shape, output_shape, n_blocks, architecture_type='conv'):
    """
    Builds the Decoder part of the cVAE.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        condition_shape (tuple): Shape of the condition input.
        output_shape (tuple): Shape of the final design output.
        n_blocks (int): Number of residual blocks in sequence. (Only for 'conv' architecture)
        architecture_type (str): Type of architecture to use, 'conv' or 'mlp'.

    Returns:
        keras.Model: The compiled Decoder model.
    """
    latent_in = keras.Input(shape=(latent_dim,), name='latent_in')
    condition_target_spec_input = keras.layers.Input(shape=condition_shape, name='condition_in')

    if architecture_type == 'conv':
        # Spectrum path
        x = condition_target_spec_input
        x = model_arch.resblock_sequence_down(x, N_filter=32, N_blocks=n_blocks)
        x = model_arch.resblock_sequence_down(x, N_filter=64, N_blocks=n_blocks)
        x = model_arch.resblock_sequence_down(x, N_filter=128, N_blocks=n_blocks)
        x = model_arch.resblock_sequence_down(x, N_filter=256, N_blocks=n_blocks)
        x_spec = keras.layers.Flatten()(x)

        # Merge with latent vector
        x = keras.layers.Concatenate()([x_spec, latent_in])

        # Geometry generator path
        x = keras.layers.Dense(4 * 128)(x)
        x = keras.layers.Reshape((4, 128))(x)
        x = model_arch.resblock_sequence_id(x, N_filter=128, N_blocks=n_blocks)

        # Output layer
        out_design = keras.layers.Conv1D(
            filters=output_shape[-1], kernel_size=1, name='design_out'
        )(x)

    elif architecture_type == 'mlp':
        # Flatten condition input
        flat_condition = keras.layers.Flatten()(condition_target_spec_input)
        
        # Merge with latent vector
        x = keras.layers.Concatenate()([latent_in, flat_condition])
        x = keras.layers.Dense(256)(x)
        x = keras.layers.LeakyReLU(alpha=1e-1)(x)
        x = keras.layers.Dense(512)(x)
        x = keras.layers.LeakyReLU(alpha=1e-1)(x)

        # Output layer
        output_units = np.prod(output_shape)
        x = keras.layers.Dense(output_units)(x)
        out_design = keras.layers.Reshape(output_shape, name='design_out')(x)
        
    else:
        raise ValueError("Unsupported architecture_type. Choose 'conv' or 'mlp'.")


    decoder = keras.Model(
        [latent_in, condition_target_spec_input],
        out_design,
        name="decoder"
    )
    return decoder

def build_cvae(encoder, decoder, design_shape, condition_shape, initial_beta=0.001):
    """
    Builds and compiles the full cVAE model with custom losses and support for KL annealing.

    Args:
        encoder (keras.Model): The encoder model.
        decoder (keras.Model): The decoder model.
        design_shape (tuple): Shape of the design input.
        condition_shape (tuple): Shape of the condition input.
        initial_beta (float): The initial weight for the KL divergence loss term.

    Returns:
        keras.Model: The compiled cVAE model.
    """
    design_in = keras.Input(shape=design_shape, name='design_in')
    condition_in = keras.Input(shape=condition_shape, name='condition_in')

    z_mean, z_var_log, z = encoder([design_in, condition_in])
    design_pred = decoder([z, condition_in])

    cVAE = keras.Model([design_in, condition_in], design_pred, name="cVAE")

    # Make beta a tf.Variable so it can be updated by a callback
    cVAE.beta = tf.Variable(initial_beta, trainable=False, name='kl_beta')

    # Add reconstruction loss
    reconstruction_loss = keras.losses.mse(
        keras.backend.flatten(design_in),
        keras.backend.flatten(design_pred)
    )
    cVAE.add_loss(reconstruction_loss)
    cVAE.add_metric(reconstruction_loss, name='reconstr_loss', aggregation='mean')

    # Add KL divergence loss, weighted by the dynamic beta
    kl_loss = -0.5 * keras.backend.sum(
        1 + z_var_log - keras.backend.square(z_mean) - keras.backend.exp(z_var_log),
        axis=-1
    )
    # The loss function now multiplies by the model's beta attribute
    kl_loss = cVAE.beta * keras.backend.mean(kl_loss)
    cVAE.add_loss(kl_loss)
    cVAE.add_metric(kl_loss, name='kl_loss', aggregation='mean')

    return cVAE



