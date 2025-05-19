import torch
import torch.nn as nn
import torch.nn.functional as F

from parameter import *
torch.cuda.empty_cache()
torch.manual_seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Custom Variational Autoencoder Model for property vector encoding and reconstruction
class vaeModel(nn.Module):
    def __init__(self):
        super(vaeModel, self).__init__()

        # ---- Encoder Network ----
        # Maps input vectors to latent distribution (mu and log_var)
        x_hidden_dim = [Comp_vec_dim, 128, 128, 128, latent_dim]

        # Feed-forward layers for encoding
        self.en_x_fc1 = nn.Linear(x_hidden_dim[0], x_hidden_dim[1])
        self.en_x_fc2 = nn.Linear(x_hidden_dim[1], x_hidden_dim[2])
        self.en_x_fc3 = nn.Linear(x_hidden_dim[2], x_hidden_dim[3])
        self.en_x_fc4 = nn.Linear(x_hidden_dim[3], x_hidden_dim[4])

        # Latent space parameters (mean and log variance)
        self.x_fc_mu = nn.Linear(latent_dim,  latent_dim)
        self.x_fc_std = nn.Linear(latent_dim,  latent_dim)

        # ---- Decoder Network ----
        # Maps latent vector z back to the original input space
        de_x_hidden_dim = [latent_dim, 128, 128, 128, Comp_vec_dim]

        # Feed-forward layers for decoding
        self.de_x_fc1 = nn.Linear(de_x_hidden_dim[0], de_x_hidden_dim[1])
        self.de_x_fc2 = nn.Linear(de_x_hidden_dim[1], de_x_hidden_dim[2])
        self.de_x_fc3 = nn.Linear(de_x_hidden_dim[2], de_x_hidden_dim[3])
        self.de_x_fc4 = nn.Linear(de_x_hidden_dim[3], de_x_hidden_dim[4])

        self.activation = nn.ReLU()

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z from N(mu, sigma^2) using
        a standard normal distribution.
        """
        sigma = torch.exp(logvar)
        eps = torch.randn_like(sigma)  # Sample epsilon from N(0, I)
        return mu + sigma * eps

    def encoder(self, x):
        """
        Encodes input vector into latent distribution parameters.
        """
        x = self.activation(self.en_x_fc1(x))
        x = self.activation(self.en_x_fc2(x))
        x = self.activation(self.en_x_fc3(x))
        x = self.en_x_fc4(x)

        x_mu = self.x_fc_mu(x)
        x_log_var = self.x_fc_std(x)

        x_z = self.reparameterize(x_mu, x_log_var)
        return x_z, x_mu, x_log_var

    def decoder(self, z):
        """
        Decodes the latent vector z back to the input space.
        """
        x = self.activation(self.de_x_fc1(z))
        x = self.activation(self.de_x_fc2(x))
        x = self.activation(self.de_x_fc3(x))
        x = self.de_x_fc4(x)

        # Apply softmax to produce normalized probability-like output
        return torch.softmax(x, dim=-1)


class pModel(nn.Module):
    def __init__(self):
        super(pModel, self).__init__()
        self.dropout_p = 0.0

        p_hidden_dim = [latent_dim, 128, 128, 128]  # Easy to modify

        # Define the layers explicitly
        self.e_fc1 = nn.Linear(latent_dim, p_hidden_dim[0], bias=False)
        self.e_relu1 = nn.ReLU()

        self.e_fc2 = nn.Linear(p_hidden_dim[0], p_hidden_dim[1], bias=False)
        self.e_relu2 = nn.ReLU()
        self.e_dropout2 = nn.Dropout(self.dropout_p)

        self.e_fc3 = nn.Linear(p_hidden_dim[1], p_hidden_dim[2], bias=False)
        self.e_relu3 = nn.ReLU()
        self.e_dropout3 = nn.Dropout(self.dropout_p)

        self.e_fc4 = nn.Linear(p_hidden_dim[2], p_hidden_dim[3], bias=False)
        self.e_relu4 = nn.ReLU()
        self.e_dropout4 = nn.Dropout(self.dropout_p)

        self.de_out = nn.Linear(p_hidden_dim[3], param_Dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.e_relu1(self.e_fc1(z))
        x = self.e_relu2(self.e_fc2(x))
        x = self.e_dropout2(x)
        x = self.e_relu3(self.e_fc3(x))
        x = self.e_dropout3(x)
        x = self.e_relu4(self.e_fc4(x))
        x = self.e_dropout4(x)
        x = self.de_out(x)
        x = self.sigmoid(x)
        return x


def kld_loss(mu, logvar):
    """
    Compute the Kullback-Leibler Divergence (KLD) loss between a Gaussian
    posterior with mean `mu` and log-variance `logvar`, and the standard normal prior.
    """
    # Apply the closed-form expression for KLD between two Gaussians
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def weights_init(m):
    """
    Initialize weights of layers using Xavier uniform initialization.
    Bias terms (if present) are initialized to zero.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        # Initialize weights with Xavier uniform distribution
        torch.nn.init.xavier_uniform_(m.weight.data)
        # Zero-initialize biases if they exist
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def compute_total_mse(predicted, target):
    """
    Compute the total mean square error (MSE) between predicted and target properties.
    
    Parameters:
    predicted (torch.Tensor): The predicted values with shape [batch_size, 37].
    target (torch.Tensor): The target values with shape [batch_size, 37].
    
    Returns:
    total_mse (torch.Tensor): The total mean square error (sum of all errors).
    """
    # Ensure the tensors are of the same shape
    assert predicted.shape == target.shape, "Shape mismatch between predicted and target tensors"

    # Compute the total mean square error (sum of all errors)
    total_mse = F.mse_loss(predicted, target, reduction='mean')
    
    return total_mse


def refine_composition(composition_tensor, threshold=1e-5):
    """
    Refines a composition tensor by:
    - Zeroing out small values below the threshold.
    - Adding the removed mass to the component with the highest concentration.

    Args:
        composition_tensor (torch.Tensor): shape (..., num_components)
        threshold (float): Values below this will be removed.

    Returns:
        torch.Tensor: Refined composition tensor, same shape as input.
    """
    refined = composition_tensor.clone()

    small_mask = refined < threshold
    small_sum = refined[small_mask].sum(dim=-1, keepdim=True)

    refined[small_mask] = 0.0

    max_indices = torch.argmax(refined, dim=-1, keepdim=True)

    one_hot = torch.zeros_like(refined)
    one_hot.scatter_(-1, max_indices, 1.0)

    refined += small_sum * one_hot

    return refined
