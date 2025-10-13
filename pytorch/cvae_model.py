import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class EncoderMLP(nn.Module):
    """
    PyTorch implementation of the MLP Encoder.
    This corresponds to the `architecture_type='mlp'` block in the `build_encoder` function.
    """
    def __init__(self, design_dim, condition_dim, latent_dim):
        """
        Args:
            design_dim (int): The flattened dimension of the design input.
            condition_dim (int): The flattened dimension of the condition input.
            latent_dim (int): The dimensionality of the latent space.
        """
        super(EncoderMLP, self).__init__()

        input_dim = design_dim + condition_dim

        # Keras equivalent: Concatenate -> Dense(512) -> LeakyReLU -> Dense(256) -> LeakyReLU
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1)
        )

        # Output layers for mean and log variance of the latent space
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)

    def forward(self, design_input, condition_input):
        """
        Forward pass for the encoder.

        Args:
            design_input (torch.Tensor): The design tensor.
            condition_input (torch.Tensor): The condition tensor.

        Returns:
            tuple: A tuple containing z_mean and z_log_var.
        """
        # Flatten inputs and concatenate them, similar to Keras Flatten and Concatenate
        flat_design = design_input.view(design_input.size(0), -1)
        flat_condition = condition_input.view(condition_input.size(0), -1)
        x = torch.cat([flat_design, flat_condition], dim=1)

        # Pass through the MLP
        x = self.mlp(x)

        # Calculate mean and log variance
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)

        return z_mean, z_log_var

class DecoderMLP(nn.Module):
    """
    PyTorch implementation of the MLP Decoder.
    This corresponds to the `architecture_type='mlp'` block in the `build_decoder` function.
    """
    def __init__(self, latent_dim, condition_dim, output_shape):
        """
        Args:
            latent_dim (int): Dimensionality of the latent space.
            condition_dim (int): The flattened dimension of the condition input.
            output_shape (tuple): The desired shape of the output design tensor.
        """
        super(DecoderMLP, self).__init__()
        self.output_shape = output_shape
        
        input_dim = latent_dim + condition_dim
        output_dim = int(np.prod(output_shape)) # Calculate total units for the output layer

        # Keras equivalent: Concatenate -> Dense(256) -> LeakyReLU -> Dense(512) -> LeakyReLU -> Dense(output_units)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, output_dim)
        )

    def forward(self, latent_input, condition_input):
        """
        Forward pass for the decoder.

        Args:
            latent_input (torch.Tensor): The latent vector from the encoder.
            condition_input (torch.Tensor): The condition tensor.

        Returns:
            torch.Tensor: The reconstructed design tensor.
        """
        # Flatten condition input and concatenate with latent vector
        flat_condition = condition_input.view(condition_input.size(0), -1)
        x = torch.cat([latent_input, flat_condition], dim=1)

        # Pass through the MLP to get the flattened output design
        reconstructed_flat = self.mlp(x)
        
        # Reshape to the target output shape
        # The first dimension is the batch size, which we keep dynamic (-1)
        reconstructed_design = reconstructed_flat.view(-1, *self.output_shape)

        return reconstructed_design

class CVAE_MLP(nn.Module):
    """
    A complete Conditional Variational Autoencoder (cVAE) model using an MLP architecture.
    This class combines the Encoder and Decoder and handles the VAE-specific logic
    like reparameterization and loss calculation.
    """
    def __init__(self, design_shape, condition_shape, latent_dim):
        super(CVAE_MLP, self).__init__()
        
        design_dim = int(np.prod(design_shape))
        condition_dim = int(np.prod(condition_shape))

        self.encoder = EncoderMLP(design_dim, condition_dim, latent_dim)
        self.decoder = DecoderMLP(latent_dim, condition_dim, design_shape)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        This corresponds to the `SamplingRNG` layer in the Keras code.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, design_input, condition_input):
        """
        Defines the forward pass of the cVAE.
        """
        z_mean, z_log_var = self.encoder(design_input, condition_input)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed_design = self.decoder(z, condition_input)
        return reconstructed_design, z_mean, z_log_var

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        """
        Calculates the VAE loss, which is a sum of reconstruction loss and KL divergence.
        This logic is equivalent to the loss added in the `build_cvae` Keras function.

        Args:
            recon_x (torch.Tensor): The reconstructed design from the decoder.
            x (torch.Tensor): The original input design.
            mu (torch.Tensor): The latent space mean from the encoder.
            logvar (torch.Tensor): The latent space log variance from the encoder.
            beta (float): The weight for the KL divergence term (for KL annealing).

        Returns:
            dict: A dictionary containing total loss, reconstruction loss, and KL loss.
        """
        # Reconstruction Loss (MSE)
        # We flatten the tensors to match the Keras `keras.backend.flatten` behavior.
        recon_loss = F.mse_loss(recon_x.view(-1), x.view(-1), reduction='mean')

        # KL Divergence Loss
        # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Average KL loss over the batch
        kl_loss = kl_loss / x.size(0)

        # Total Loss
        total_loss = recon_loss + beta * kl_loss

        return {
            'loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }