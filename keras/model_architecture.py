import pickle
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def resblock_sequence_down(x_in, N_filter, N_blocks):
    x = x_in
    for i in range(N_blocks):
        x = residual_block(x, N_filter, kernel_size=3, strides=1, with_BN=True)
    # down: use strides=2 for downsampling (more flexible, since trainable).
    x = residual_block(x, N_filter, kernel_size=3, strides=2)
    return x


def resblock_sequence_up(x_in, N_filter, N_blocks):
    x = x_in
    for i in range(N_blocks):
        x = residual_block(x, N_filter, kernel_size=3, strides=1, with_BN=True)
    # up: use upsampling (more robust than transpose convolutions with stride 2)
    x = keras.layers.UpSampling1D()(x)
    return x


def resblock_sequence_id(x_in, N_filter, N_blocks):
    # identity block sequence: input shape = output shape
    x = x_in
    for i in range(N_blocks):
        x = residual_block(x, N_filter, kernel_size=3, strides=1, with_BN=True)
    return x


def residual_block(x_in, N_filter, kernel_size=3, strides=1,
                   conv_layer=keras.layers.Conv1D, alpha=0.3,
                   with_BN=False):
    """resnet block, default: 1D convolutions"""

    # residual connection
    if x_in.shape[-1] != N_filter or strides != 1:
        # if input!=output dimension: add BN/ReLU/conv. into shortcut
        conv_shortcut = conv_layer(
            filters=N_filter, kernel_size=1, strides=strides, padding='same')(x_in)
    else:
        # if input==output dimension: use bare input as shortcut
        conv_shortcut = x_in

    # convolutional path
    x = x_in

    x = conv_layer(filters=N_filter, kernel_size=1, strides=1,
                   padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)

    x = conv_layer(filters=N_filter, kernel_size=kernel_size,
                   strides=strides, padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha)(x)

    x = conv_layer(filters=N_filter, kernel_size=1, strides=1,
                   padding='same', use_bias=not with_BN)(x)
    if with_BN:
        x = keras.layers.BatchNormalization()(x)

    # add residual and main and apply a further activation
    x = keras.layers.Add()([x, conv_shortcut])
    x = keras.layers.LeakyReLU(alpha)(x)

    return x


class SamplingRNG(keras.layers.Layer):
    """layer to sample a normal distributed random numbers z with mean `z_mean` and logarithmic variance `z_log_var`"""

    def call(self, inputs):
        z_mean, z_var_log = inputs
        epsilon = keras.backend.random_normal(shape=tf.shape(z_mean)[1:])
        z = z_mean + keras.backend.exp(0.5 * z_var_log) * epsilon
        return z


