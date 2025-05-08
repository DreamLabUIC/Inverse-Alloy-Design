import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility: Extract timestep-specific values from precomputed arrays
# -----------------------------
def extract_value(arr, time_indices, ref_shape):
    """
    Selects values from `arr` using `time_indices`, reshaping to match the shape of the input tensor.
    
    Args:
        arr (Tensor): 1D tensor of shape (T,) containing precomputed diffusion coefficients.
        time_indices (Tensor): Tensor of shape (B,) with timestep indices for each batch.
        ref_shape (tuple): Shape of the reference tensor (e.g., input shape).

    Returns:
        Tensor: Extracted values reshaped to broadcast with input data.
    """
    batch_sz = time_indices.shape[0]
    out = arr.gather(-1, time_indices.cpu())
    return out.reshape(batch_sz, *((1,) * (len(ref_shape) - 1))).to(time_indices.device)


# -----------------------------
# Forward Diffusion Step: q(x_t | x_0)
# -----------------------------
def diffuse_step(x_init, time_steps, sqrt_cumprod_alpha, sqrt_cumprod_one_minus_alpha, epsilon=None):
    """
    Adds noise to the clean input `x_init` to simulate diffusion at timestep `t`.

    Args:
        x_init (Tensor): Clean input data x₀ of shape (B, D).
        time_steps (Tensor): Timestep tensor of shape (B,).
        sqrt_cumprod_alpha (Tensor): Precomputed sqrt of cumulative alpha products.
        sqrt_cumprod_one_minus_alpha (Tensor): Precomputed sqrt of (1 - cumulative alpha products).
        epsilon (Tensor, optional): Optional noise to use; if None, sampled from N(0, 1).

    Returns:
        Tensor: Noisy input x_t.
    """
    if epsilon is None:
        epsilon = torch.randn_like(x_init)

    alpha_t = extract_value(sqrt_cumprod_alpha, time_steps, x_init.shape)
    one_minus_alpha_t = extract_value(sqrt_cumprod_one_minus_alpha, time_steps, x_init.shape)

    return alpha_t * x_init + one_minus_alpha_t * epsilon


# -----------------------------
# Denoising Loss Function
# -----------------------------
def diffusion_loss(model, x_init, time_steps, cond_vec, sqrt_cumprod_alpha, sqrt_cumprod_one_minus_alpha, epsilon=None, loss_mode="l1"):
    """
    Computes the loss between true noise and predicted noise from a denoising model.

    Args:
        model (nn.Module): Denoising neural network.
        x_init (Tensor): Clean input data x₀.
        time_steps (Tensor): Timesteps (B,).
        cond_vec (Tensor): Conditioning vector.
        sqrt_cumprod_alpha (Tensor): Precomputed alpha schedule.
        sqrt_cumprod_one_minus_alpha (Tensor): Precomputed one-minus-alpha schedule.
        epsilon (Tensor, optional): Optional true noise.
        loss_mode (str): One of ['l1', 'l2', 'huber'].

    Returns:
        Tensor: Loss value.
    """
    if epsilon is None:
        epsilon = torch.randn_like(x_init)

    x_t = diffuse_step(x_init, time_steps, sqrt_cumprod_alpha, sqrt_cumprod_one_minus_alpha, epsilon=epsilon)
    epsilon_hat = model(x_t, time_steps, cond_vec)

    if loss_mode == 'l1':
        loss = F.l1_loss(epsilon, epsilon_hat)
    elif loss_mode == 'l2':
        loss = F.mse_loss(epsilon, epsilon_hat)
    elif loss_mode == "huber":
        loss = F.smooth_l1_loss(epsilon, epsilon_hat)
    else:
        raise NotImplementedError("Unsupported loss type")

    return loss


# -----------------------------
# Positional Encoding for Timestep Embeddings
# -----------------------------
class TimePositionalEmbedding(nn.Module):
    """
    Generates sinusoidal positional embeddings for timesteps.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, timesteps):
        """
        Args:
            timesteps (Tensor): Integer timestep tensor of shape (B,)

        Returns:
            Tensor: Positional embeddings of shape (B, emb_dim)
        """
        device = timesteps.device
        half_dim = self.emb_dim // 2
        scale = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -scale)
        angle = timesteps[:, None] * freq[None, :]
        return torch.cat([angle.sin(), angle.cos()], dim=-1)


# -----------------------------
# Denoising Neural Network
# -----------------------------

class DenoisingModel(nn.Module):
    """
    Denoising MLP conditioned on timestep and optional auxiliary vector.
    """
    def __init__(self, z_dim, hidden_dim, depth, cond_in, cond_dim):
        """
        Args:
            z_dim (int): Dimensionality of noisy input.
            hidden_dim (int): Hidden layer size.
            depth (int): Number of MLP layers.
            cond_in (int): Conditioning input dimension.
            cond_dim (int): Projected conditioning vector dimension.
        """
        super(DenoisingModel, self).__init__()
        self.depth = depth
        self.cond_in = cond_in

        # Project conditioning vector (was: cond_project)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_in, 128),
            nn.ReLU(),
            nn.Linear(128, cond_dim),
        )

        # Timestep embedding (was: time_embedding)
        self.time_mlp = nn.Sequential(
            TimePositionalEmbedding(70),
            nn.Linear(70, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # MLP backbone (was: mlp_layers)

        net = [nn.Linear(z_dim + cond_dim, hidden_dim)] + [nn.Linear(hidden_dim+cond_dim, hidden_dim) for i in range(depth-2)]
        net.append(nn.Linear(hidden_dim, z_dim))
        self.mlp = nn.ModuleList(net)

        # Normalization layers (was: norm_layers)
        self.bn = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(depth - 1)])

        self.activation = nn.ReLU()

    def forward(self, x, t, cond):
        """
        Forward pass.

        Args:
            x (Tensor): Noisy input at time t.
            t (Tensor): Timesteps.
            cond (Tensor): Conditioning vector.

        Returns:
            Tensor: Predicted noise ε̂.
        """
        cond = cond.reshape(-1, self.cond_in)
        cond = torch.nan_to_num(cond, nan=-100.0)
        cond = self.cond_mlp(cond)
        t_emb = self.time_mlp(t)

        for i in range(self.depth - 1):
            x = torch.cat([x, cond], dim=1)
            x = self.activation(self.mlp[i](x)) + t_emb
            x = self.bn[i](x)

        x = self.mlp[self.depth - 1](x)
        return x



# -----------------------------
# Reverse Diffusion Step (p(x_{t-1} | x_t))
# -----------------------------
@torch.no_grad()
def reverse_diffusion_step(model, x_t, t, cond_vec, time_idx, beta_sched):
    """
    Computes one denoising step: p(x_{t-1} | x_t)

    Args:
        model (nn.Module): Trained denoising model.
        x_t (Tensor): Noisy input at time t.
        t (Tensor): Timesteps.
        cond_vec (Tensor): Conditioning vector.
        time_idx (int): Integer value of timestep t.
        beta_sched (Tensor): Beta schedule used in diffusion.

    Returns:
        Tensor: Denoised sample x_{t-1}
    """
    alphas = 1. - beta_sched
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

    sqrt_recip_alpha = torch.sqrt(1.0 / alphas)
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_cumprod = torch.sqrt(1. - alpha_cumprod)
    posterior_var = beta_sched * (1. - alpha_cumprod_prev) / (1. - alpha_cumprod)

    beta_t = extract_value(beta_sched, t, x_t.shape)
    sqrt_one_minus_alpha_t = extract_value(sqrt_one_minus_cumprod, t, x_t.shape)
    sqrt_recip_alpha_t = extract_value(sqrt_recip_alpha, t, x_t.shape)

    # Compute predicted mean using model's noise prediction
    pred_mean = sqrt_recip_alpha_t * (x_t - beta_t * model(x_t, t, cond_vec) / sqrt_one_minus_alpha_t)

    if time_idx == 0:
        return pred_mean
    else:
        post_var_t = extract_value(posterior_var, t, x_t.shape)
        z = torch.randn_like(x_t)
        return pred_mean + torch.sqrt(post_var_t) * z


# -----------------------------
# Full Reverse Diffusion Process
# -----------------------------
@torch.no_grad()
def reverse_diffusion_loop(model, cond_vec, n_steps, beta_sched, shape):
    """
    Runs the full reverse diffusion process starting from pure noise.

    Args:
        model (nn.Module): Trained denoising model.
        cond_vec (Tensor): Conditioning vector.
        n_steps (int): Number of diffusion steps (T).
        beta_sched (Tensor): Beta schedule.
        shape (tuple): Shape of the output (B, z_dim).

    Returns:
        List[Tensor]: Sequence of generated samples.
    """
    device = next(model.parameters()).device
    batch_sz = shape[0]
    z = torch.randn(shape, device=device)
    seq = []

    for i in reversed(range(n_steps)):
        z = reverse_diffusion_step(model, z, torch.full((batch_sz,), i, device=device, dtype=torch.long), cond_vec, i, beta_sched)
        seq.append(z)
    return seq


# -----------------------------
# Sampling API
# -----------------------------
@torch.no_grad()
def sample_from_model(model, cond_vec, z_dim, n_steps, beta_sched, batch_sz):
    """
    Samples from the diffusion model using reverse diffusion.

    Args:
        model (nn.Module): Trained denoising model.
        cond_vec (Tensor): Conditioning vector.
        z_dim (int): Latent dimension.
        n_steps (int): Number of diffusion steps.
        beta_sched (Tensor): Beta schedule.
        batch_sz (int): Number of samples.

    Returns:
        List[Tensor]: Sequence of generated samples (denoised from noise).
    """
    return reverse_diffusion_loop(model, cond_vec, n_steps, beta_sched, shape=(batch_sz, z_dim))



def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for beta values as proposed in the paper:
    "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672)

    Args:
        timesteps (int): Number of timesteps in the diffusion process.
        s (float): Small offset to prevent division by zero and improve numerical stability.

    Returns:
        torch.Tensor: A tensor of beta values clipped between 0.0001 and 0.9999.
    """
    steps = timesteps + 1  # One more than timesteps to define intervals
    x = torch.linspace(0, timesteps, steps)  # Linearly spaced time steps
    # Compute cumulative product of alphas using a cosine-based schedule
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # Normalize so that first value is 1
    # Derive betas from the ratio of consecutive alpha products
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)  # Clip to avoid extreme values


def linear_beta_schedule(timesteps):
    """
    Linear schedule for beta values from a small start to a higher end value.

    Args:
        timesteps (int): Number of timesteps in the diffusion process.

    Returns:
        torch.Tensor: A linearly spaced tensor of beta values.
    """
    beta_start = 0.0001  # Starting value of beta
    beta_end = 0.02      # Ending value of beta
    return torch.linspace(beta_start, beta_end, timesteps)  # Linearly spaced betas


def quadratic_beta_schedule(timesteps):
    """
    Quadratic schedule for beta values: square of linearly spaced square roots.

    Args:
        timesteps (int): Number of timesteps in the diffusion process.

    Returns:
        torch.Tensor: A tensor of beta values increasing quadratically.
    """
    beta_start = 0.0001
    beta_end = 0.02
    # Linearly interpolate between square roots and then square to get quadratic growth
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """
    Sigmoid schedule for beta values: slow growth at beginning and end, fast in middle.

    Args:
        timesteps (int): Number of timesteps in the diffusion process.

    Returns:
        torch.Tensor: A tensor of beta values shaped like a sigmoid curve.
    """
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)  # Spread betas over [-6, 6] for sigmoid shape
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start  # Scale to desired range



