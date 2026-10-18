"""
Restricted Boltzmann Machine (RBM) Implementation in PyTorch.

Mathematical Foundations:
- Energy Function:
    E(v, h) = - v^T W h - a^T v - b^T h
- Conditional Distributions:
    P(h_j = 1 | v) = sigmoid(b_j + sum_i v_i W_{ij})
    P(v_i = 1 | h) = sigmoid(a_i + sum_j W_{ij} h_j)
- Free Energy:
    F(v) = - a^T v - sum_j softplus(v W_{*,j} + b_j)
- Contrastive Divergence (CD-k):
    Minimizes free energy difference: L = F(v^(0)) - F(v^(k))
    whose analytical gradient matches Hinton's CD update rule:
    grad_W = - <v^(0) h^(0)> + <v^(k) h^(k)>
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine with Bernoulli visible and hidden units.
    """

    def __init__(
        self,
        visible_dim: int,
        hidden_dim: int,
        k_cd: int = 1,
        learning_rate: float = 0.05,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        initial_weights_std: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize RBM parameters.
        
        Args:
            visible_dim: Dimension of visible layer (e.g. 784 for MNIST)
            hidden_dim: Dimension of latent hidden layer
            k_cd: Number of Gibbs sampling steps in Contrastive Divergence
            learning_rate: Base learning rate for parameter updates
            momentum: Momentum factor for velocity updates
            weight_decay: L2 regularization penalty
            initial_weights_std: Std dev for Gaussian weight initialization
            device: torch.device (CPU or CUDA)
        """
        super(RBM, self).__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k_cd = k_cd
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Trainable parameters
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * initial_weights_std)
        self.v_bias = nn.Parameter(torch.zeros(visible_dim))
        self.h_bias = nn.Parameter(torch.zeros(hidden_dim))

        # Optimizer with SGD + momentum and weight decay
        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        if device is not None:
            self.to(device)

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute hidden unit activation probabilities P(h=1|v) and draw Bernoulli samples.

        Args:
            v: Visible batch tensor of shape (batch_size, visible_dim)

        Returns:
            p_h: Activation probabilities (batch_size, hidden_dim)
            h_sample: Binary sample tensor in {0, 1}
        """
        linear = torch.matmul(v, self.W) + self.h_bias
        p_h = torch.sigmoid(linear)
        h_sample = torch.bernoulli(p_h)
        return p_h, h_sample

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visible unit reconstruction probabilities P(v=1|h) and draw Bernoulli samples.

        Args:
            h: Hidden batch tensor of shape (batch_size, hidden_dim)

        Returns:
            p_v: Reconstruction probabilities (batch_size, visible_dim)
            v_sample: Binary sample tensor in {0, 1}
        """
        linear = torch.matmul(h, self.W.t()) + self.v_bias
        p_v = torch.sigmoid(linear)
        v_sample = torch.bernoulli(p_v)
        return p_v, v_sample

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a visible vector v:
        F(v) = - v^T a - sum_j softplus(v W_{*,j} + b_j)

        Args:
            v: Batch tensor (batch_size, visible_dim)

        Returns:
            Per-sample free energy of shape (batch_size,)
        """
        vbias_term = torch.matmul(v, self.v_bias)
        linear = torch.matmul(v, self.W) + self.h_bias
        # softplus(x) = log(1 + exp(x)), numerically stable
        hidden_term = torch.sum(F.softplus(linear), dim=1)
        return -vbias_term - hidden_term

    def gibbs_step(
        self,
        v: torch.Tensor,
        sample_visible_binary: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform one alternating Gibbs sampling step: v -> h -> v' -> h'.

        Args:
            v: Input visible states
            sample_visible_binary: Whether to sample binary visible units or use probabilities

        Returns:
            (p_h0, h0_sample, p_vk, vk_sample)
        """
        p_h0, h0 = self.sample_h(v)
        p_vk, vk_sample = self.sample_v(h0)
        vk = vk_sample if sample_visible_binary else p_vk
        p_hk, hk_sample = self.sample_h(vk)
        return p_h0, h0, p_vk, vk

    def gibbs_k_steps(
        self,
        v_start: torch.Tensor,
        k: Optional[int] = None,
        sample_visible_binary: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run k-step Markov chain starting from visible batch v_start.

        Args:
            v_start: Initial visible states (batch_size, visible_dim)
            k: Number of Gibbs steps (defaults to self.k_cd)
            sample_visible_binary: Whether intermediate visible states are binarized

        Returns:
            v_k: Reconstructed visible states at step k
            p_vk: Probabilities of visible states at step k
        """
        steps = k if k is not None else self.k_cd
        v_current = v_start
        p_vk = None

        for _ in range(steps):
            _, h = self.sample_h(v_current)
            p_vk, v_sample = self.sample_v(h)
            v_current = v_sample if sample_visible_binary else p_vk

        return v_current, p_vk

    def contrastive_divergence(
        self,
        v_pos: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Execute one step of Contrastive Divergence (CD-k) parameter update.

        Args:
            v_pos: Positive phase visible batch from real data (batch_size, visible_dim)
            k: CD step count (defaults to self.k_cd)

        Returns:
            loss_val: Surrogate loss value F(v_0) - F(v_k)
            recon_error: Mean squared error between v_pos and reconstruction p_v
        """
        steps = k if k is not None else self.k_cd
        self.optimizer.zero_grad()

        # Step 1: Positive phase
        v0 = v_pos.detach()

        # Step 2: Negative phase via k-step Gibbs chain
        with torch.no_grad():
            vk, p_vk = self.gibbs_k_steps(v0, k=steps, sample_visible_binary=False)

        # Step 3: Contrastive surrogate loss: L = F(v0) - F(vk)
        # Note: vk is detached, so autograd computes:
        # grad = dF(v0)/dTheta - dF(vk)/dTheta
        # which exactly matches Hinton's Contrastive Divergence gradient!
        loss = torch.mean(self.free_energy(v0) - self.free_energy(vk.detach()))

        loss.backward()
        self.optimizer.step()

        # Reconstruction diagnostic MSE
        recon_error = F.mse_loss(p_vk, v0).item()

        return loss.item(), recon_error

    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """
        One-step deterministic/probability reconstruction for diagnostic visualization.

        Args:
            v: Input visible vectors

        Returns:
            Reconstructed visible probabilities in [0, 1]
        """
        with torch.no_grad():
            p_h, _ = self.sample_h(v)
            p_v, _ = self.sample_v(p_h)
        return p_v
