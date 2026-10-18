"""
Visualization and Plotting Utilities for DBN Digit Generation.

Includes:
- Training diagnostic curves (Loss & Reconstruction Error per RBM)
- Grid displays of synthetic handwritten digits
- Side-by-side Real vs Generated comparisons
- Pixel intensity distribution histograms
- PNG export using Pillow
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_training_curves(
    history: Dict[str, Any],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot reconstruction errors across pretraining epochs for all stacked RBMs.

    Args:
        history: Dict mapping "rbm_{i}" -> {"epochs": [...], "loss": [...], "recon_error": [...]}
        save_path: Optional path to save the generated figure

    Returns:
        matplotlib Figure object
    """
    num_rbms = len(history)
    fig, axes = plt.subplots(1, max(num_rbms, 1), figsize=(5 * max(num_rbms, 1), 4), squeeze=False)
    axes = axes[0]

    for idx, (rbm_key, metrics) in enumerate(history.items()):
        ax = axes[idx]
        epochs = metrics.get("epochs", list(range(1, len(metrics.get("recon_error", [])) + 1)))
        recon_errors = metrics.get("recon_error", [])
        losses = metrics.get("loss", [])

        if recon_errors:
            ax.plot(epochs, recon_errors, marker="o", color="#1f77b4", label="Recon MSE")
        if losses:
            ax.plot(epochs, losses, marker="s", color="#ff7f0e", linestyle="--", label="CD Surrogate Loss")

        ax.set_title(f"{rbm_key.upper()} Pretraining", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Diagnostic Metric", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_digit_grid(
    images: torch.Tensor,
    titles: Optional[List[str]] = None,
    cols: Optional[int] = None,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Display a batch of 28x28 digit images in an organized rectangular grid.

    Args:
        images: Float tensor of shape (N, 28, 28) with values in [0, 1]
        titles: Optional per-image captions (e.g. classifier predicted label)
        cols: Number of columns (auto-computed if None)
        save_path: Optional path to save the grid figure

    Returns:
        matplotlib Figure object
    """
    images_np = images.detach().cpu().numpy()
    n_images = len(images_np)

    if cols is None:
        cols = min(10, math.ceil(math.sqrt(n_images)))
    rows = math.ceil(n_images / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6), squeeze=False)

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        if i < n_images:
            ax.imshow(images_np[i], cmap="gray", vmin=0.0, vmax=1.0)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=8)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_real_vs_generated(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    num_samples: int = 10,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Side-by-side comparison: Top row real MNIST digits, Bottom row DBN synthetic digits.

    Args:
        real_images: (N, 28, 28) tensor
        gen_images: (N, 28, 28) tensor
        num_samples: Number of pairs to plot (default 10)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    k = min(num_samples, len(real_images), len(gen_images))
    real_np = real_images[:k].detach().cpu().numpy()
    gen_np = gen_images[:k].detach().cpu().numpy()

    fig, axes = plt.subplots(2, k, figsize=(k * 1.4, 3.2), squeeze=False)

    for i in range(k):
        # Real row
        axes[0, i].imshow(real_np[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Real", fontsize=11, fontweight="bold", loc="left")

        # Synthetic row
        axes[1, i].imshow(gen_np[i], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("DBN Synthetic", fontsize=11, fontweight="bold", loc="left")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_pixel_distribution(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Histogram comparison of pixel intensity distributions between real and synthetic images.
    Highlights how DBN learns the bimodal nature of MNIST (dominant 0 background vs strokes).

    Args:
        real_images: Real MNIST images (N, 28, 28)
        gen_images: Synthetic images (M, 28, 28)
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    real_flat = real_images.detach().cpu().numpy().flatten()
    gen_flat = gen_images.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.0, 1.0, 50)

    ax.hist(real_flat, bins=bins, alpha=0.6, label="Real MNIST Pixels", color="#1f77b4", density=True)
    ax.hist(gen_flat, bins=bins, alpha=0.6, label="DBN Generated Pixels", color="#ff7f0e", density=True)

    ax.set_title("Pixel Intensity Distribution: Real vs. Generated", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel Intensity [0.0 = Background, 1.0 = Stroke]", fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def save_images_to_png(
    images: torch.Tensor,
    output_dir: Path,
    prefix: str = "digit"
) -> List[Path]:
    """
    Save each 28x28 image in the batch as a separate grayscale PNG.

    Args:
        images: Float tensor (N, 28, 28) in [0, 1]
        output_dir: Directory where PNG files will be written
        prefix: Filename prefix

    Returns:
        List of Paths of saved image files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    images_np = (images.detach().cpu().numpy() * 255.0).astype(np.uint8)

    for i, arr in enumerate(images_np):
        pil_img = Image.fromarray(arr, mode="L")
        out_path = output_dir / f"{prefix}_{i+1:03d}.png"
        pil_img.save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def plot_gibbs_trajectory(
    trajectory: Dict[int, torch.Tensor],
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot the evolution of synthetic digits across MCMC Gibbs sampling milestones.
    Rows = individual Markov chains / samples.
    Columns = milestone Gibbs steps (e.g. Step 0 -> Step 5 -> Step 15 -> Step 30 -> Step 60 -> Step 100).

    Args:
        trajectory: Dict mapping step_int -> Tensor of shape (num_samples, 28, 28)
        save_path: Optional output path to save the strip plot figure

    Returns:
        matplotlib Figure
    """
    sorted_steps = sorted(list(trajectory.keys()))
    num_steps = len(sorted_steps)
    first_tensor = trajectory[sorted_steps[0]]
    num_samples = len(first_tensor)

    fig, axes = plt.subplots(num_samples, num_steps, figsize=(num_steps * 1.8, num_samples * 1.8), squeeze=False)

    for col_idx, step in enumerate(sorted_steps):
        step_images = trajectory[step].detach().cpu().numpy()
        for row_idx in range(num_samples):
            ax = axes[row_idx, col_idx]
            ax.imshow(step_images[row_idx], cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")

            # Column header on first row
            if row_idx == 0:
                step_label = f"Step {step}"
                if step == 0:
                    step_label += "\n(Noise)"
                elif col_idx == num_steps - 1:
                    step_label += "\n(Final)"
                ax.set_title(step_label, fontsize=10, fontweight="bold")

            # Row label on first column
            if col_idx == 0:
                ax.text(-0.35, 0.5, f"Chain {row_idx+1}", transform=ax.transAxes,
                        verticalalignment="center", fontsize=9, fontweight="bold", rotation=90)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig

