import tensorflow as tf
from tensorflow import keras
import numpy as np

class KLAnnealingCallback(keras.callbacks.Callback):
    """
    Callback to implement KL Annealing for a VAE.

    This callback gradually increases the weight of the KL divergence loss term (beta)
    from a starting value to an end value over a specified number of epochs.

    Args:
        start_beta (float): The initial value of beta at epoch 0.
        end_beta (float): The final value of beta to anneal towards.
        annealing_epochs (int): The number of epochs over which to anneal beta.
        annealing_shape (str): The shape of the annealing function ('linear' or 'sigmoid').
    """
    def __init__(self, start_beta, end_beta, annealing_epochs, annealing_shape='linear'):
        super(KLAnnealingCallback, self).__init__()
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.annealing_epochs = annealing_epochs
        self.annealing_shape = annealing_shape

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch."""
        # Calculate the new beta value based on the current epoch
        if epoch < self.annealing_epochs:
            if self.annealing_shape == 'linear':
                new_beta = self.start_beta + (self.end_beta - self.start_beta) * (epoch / self.annealing_epochs)
            elif self.annealing_shape == 'sigmoid':
                # Sigmoid annealing from https://arxiv.org/abs/1511.06349
                slope = 10 / self.annealing_epochs
                midpoint = self.annealing_epochs / 2
                new_beta = self.end_beta / (1 + np.exp(-slope * (epoch - midpoint)))
            else:
                 raise ValueError("Unsupported annealing_shape. Choose 'linear' or 'sigmoid'.")
        else:
            # Once annealing is complete, keep beta at its final value
            new_beta = self.end_beta

        # Set the new beta value on the model
        if hasattr(self.model, 'beta'):
             tf.keras.backend.set_value(self.model.beta, new_beta)
        else:
            raise ValueError("Model must have a 'beta' attribute (tf.Variable) for KLAnnealingCallback to work.")

    def on_epoch_end(self, epoch, logs=None):
        """Logs the current beta value at the end of each epoch."""
        if logs is not None:
            logs['beta'] = tf.keras.backend.get_value(self.model.beta)
