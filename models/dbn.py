"""
Deep Belief Network (DBN) Implementation in PyTorch.

Composed of stacked Restricted Boltzmann Machines (RBMs) trained via greedy
layer-wise pretraining (Hinton, Osindero, Teh, 2006).

Architecture & Generative Strategy:
1. Greedy Layer-Wise Pretraining:
   - Train RBM 1 on raw normalized MNIST vectors (784 -> d1).
   - Use activations P(h1=1 | v) to train RBM 2 (d1 -> d2).
   - Use activations P(h2=1 | h1) to train RBM 3 (d2 -> d3).
2. Probabilistically Justified Generation:
   - The top layer (RBM 3) acts as an undirected associative memory.
   - Alternating Gibbs sampling is run in the top layer for M steps
     to sample an equilibrium state (h2, h3) from the joint prior P(h2, h3).
   - Once equilibrium is reached, downward directed ancestral sampling passes
     h2 down through RBM 2 to generate h1, then through RBM 1 to generate
     the visible pixel vector v in [0, 1].
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from .rbm import RBM


class DeepBeliefNetwork(nn.Module):
    """
    Stacked Deep Belief Network composed of multiple Restricted Boltzmann Machines.
    """

    def __init__(
        self,
        layer_dims: List[int],
        k_cd: int = 1,
        learning_rate: float = 0.05,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            layer_dims: List of layer dimensions, e.g. [784, 500, 250, 100]
            k_cd: Number of CD steps per RBM
            learning_rate: Base learning rate
            momentum: Momentum for weight updates
            weight_decay: L2 penalty
            device: torch.device (CPU or CUDA)
        """
        super(DeepBeliefNetwork, self).__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.device = device if device is not None else torch.device("cpu")

        self.rbms = nn.ModuleList()
        for i in range(self.num_layers):
            rbm = RBM(
                visible_dim=layer_dims[i],
                hidden_dim=layer_dims[i + 1],
                k_cd=k_cd,
                learning_rate=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                device=self.device
            )
            self.rbms.append(rbm)

        self.to(self.device)

    def upward_pass(
        self,
        x: torch.Tensor,
        up_to_layer: Optional[int] = None,
        use_probabilities: bool = True
    ) -> torch.Tensor:
        """
        Propagate inputs bottom-up through the stacked RBMs.

        Args:
            x: Input visible batch (batch_size, visible_dim)
            up_to_layer: Layer index (0-indexed) up to which to propagate.
                         If None, propagates all the way to top layer.
            use_probabilities: Whether to propagate continuous probabilities
                              or stochastic binary samples.

        Returns:
            Latent representation at requested layer.
        """
        max_layer = self.num_layers if up_to_layer is None else (up_to_layer + 1)
        current = x

        for i in range(max_layer):
            p_h, h_sample = self.rbms[i].sample_h(current)
            current = p_h if use_probabilities else h_sample

        return current

    def downward_pass(
        self,
        top_representation: torch.Tensor,
        from_layer: Optional[int] = None,
        sample_stochastic: bool = False
    ) -> torch.Tensor:
        """
        Propagate activations top-down through the directed generative connections.

        Args:
            top_representation: Hidden activation at starting layer
            from_layer: Layer index where top_representation resides (defaults to top RBM)
            sample_stochastic: Whether intermediate downward activations are binary sampled

        Returns:
            Synthesized visible images (batch_size, 784)
        """
        start_layer = (self.num_layers - 1) if from_layer is None else from_layer
        current = top_representation

        # Traverse downwards: RBM_(start_layer) down to RBM_0
        for i in range(start_layer, -1, -1):
            p_v, v_sample = self.rbms[i].sample_v(current)
            # Visible layer (layer 0) is kept as continuous grayscale probabilities [0, 1]
            if i == 0 or not sample_stochastic:
                current = p_v
            else:
                current = v_sample

        return current

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full round-trip reconstruction: upward through all layers, then downward to visible.
        Used for evaluating end-to-end representation learning fidelity.

        Args:
            x: Input batch (batch_size, 784)

        Returns:
            Reconstructed batch (batch_size, 784)
        """
        with torch.no_grad():
            top_h = self.upward_pass(x, up_to_layer=self.num_layers - 1, use_probabilities=True)
            # Reconstruct top layer using top RBM associative memory
            p_top_v, _ = self.rbms[-1].sample_v(top_h)
            # Downward pass through lower directed layers
            recon_v = self.downward_pass(p_top_v, from_layer=self.num_layers - 2 if self.num_layers > 1 else 0)
        return recon_v

    def generate(
        self,
        num_samples: int = 25,
        gibbs_steps: int = 100,
        random_seed: Optional[int] = None,
        sample_stochastic_downward: bool = False
    ) -> torch.Tensor:
        """
        Mathematically justified DBN image generation procedure:
        1. Initialize top associative memory layer with random Bernoulli states.
        2. Run alternating Gibbs sampling in top RBM for `gibbs_steps` to reach
           equilibrium on the learned latent digit prior distribution.
        3. Perform downward directed ancestral pass through lower RBMs to project
           latent states into the 784-dimensional pixel space.

        Args:
            num_samples: Number of synthetic images to generate
            gibbs_steps: Number of Gibbs sampling iterations in top associative RBM
            random_seed: Optional seed for reproducibility
            sample_stochastic_downward: Whether intermediate directed layers sample binary units

        Returns:
            Generated synthetic image batch of shape (num_samples, 28, 28) with values in [0, 1]
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)

        top_rbm = self.rbms[-1]

        with torch.no_grad():
            # Step 1: Initialize visible state of top RBM randomly
            # (corresponding to latent dimension of the second-to-last layer)
            v_top = torch.bernoulli(torch.full((num_samples, top_rbm.visible_dim), 0.5, device=self.device))

            # Step 2: Alternating Gibbs sampling in top RBM (Associative Memory)
            for _ in range(gibbs_steps):
                p_h, h_sample = top_rbm.sample_h(v_top)
                p_v, v_sample = top_rbm.sample_v(h_sample)
                v_top = v_sample  # Keep binary chain for proper Markov mixing

            # At equilibrium, v_top is a sample from the joint prior over top features
            # Step 3: Directed downward ancestral propagation through lower layers
            if self.num_layers > 1:
                gen_flat = self.downward_pass(
                    top_representation=v_top,
                    from_layer=self.num_layers - 2,
                    sample_stochastic=sample_stochastic_downward
                )
            else:
                p_v, _ = top_rbm.sample_v(v_top)
                gen_flat = p_v

            # Reshape flattened 784 vectors into (num_samples, 28, 28) images
            images = gen_flat.view(num_samples, 28, 28).cpu().clamp(0.0, 1.0)

        return images

    def generate_trajectory(
        self,
        num_samples: int = 5,
        milestones: Optional[List[int]] = None,
        random_seed: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Record intermediate digit states during Gibbs sampling to visualize
        the MCMC chain evolving from pure Bernoulli noise to a coherent digit.

        Args:
            num_samples: Number of parallel chains
            milestones: Steps at which to record intermediate reconstructions (e.g. [0, 5, 15, 30, 60, 100])
            random_seed: Reproducibility seed

        Returns:
            Dictionary mapping step_number -> Tensor of shape (num_samples, 28, 28)
        """
        if milestones is None:
            milestones = [0, 5, 15, 30, 60, 100]
        milestones = sorted(list(set(milestones)))
        max_steps = max(milestones)

        if random_seed is not None:
            torch.manual_seed(random_seed)

        top_rbm = self.rbms[-1]
        trajectory: Dict[int, torch.Tensor] = {}

        def _decode_to_visible(v_state: torch.Tensor) -> torch.Tensor:
            if self.num_layers > 1:
                gen_flat = self.downward_pass(
                    top_representation=v_state,
                    from_layer=self.num_layers - 2,
                    sample_stochastic=False
                )
            else:
                p_v, _ = top_rbm.sample_v(v_state)
                gen_flat = p_v
            return gen_flat.view(num_samples, 28, 28).cpu().clamp(0.0, 1.0)

        with torch.no_grad():
            v_top = torch.bernoulli(torch.full((num_samples, top_rbm.visible_dim), 0.5, device=self.device))

            if 0 in milestones:
                trajectory[0] = _decode_to_visible(v_top)

            for step in range(1, max_steps + 1):
                p_h, h_sample = top_rbm.sample_h(v_top)
                p_v, v_sample = top_rbm.sample_v(h_sample)
                v_top = v_sample

                if step in milestones:
                    trajectory[step] = _decode_to_visible(v_top)

        return trajectory

