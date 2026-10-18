"""
Synthetic Handwritten Digit Generation Module for Deep Belief Network.

Executes mathematically justified DBN sampling:
1. Loads pre-trained DBN weights from checkpoint.
2. Initializes top-layer associative memory.
3. Runs alternating Gibbs sampling for `gibbs_steps` to reach equilibrium.
4. Performs downward directed ancestral pass through lower RBM layers.
5. Saves synthetic digit images to outputs/generated_digits/.
6. Creates and saves a consolidated grid plot.
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import torch

from config.config import get_config, PathConfig, DeviceConfig
from models.dbn import DeepBeliefNetwork
from utils.checkpoint import load_dbn_checkpoint, get_latest_checkpoint
from utils.visualization import plot_digit_grid, save_images_to_png


def generate_digits(
    checkpoint_path: Optional[Path] = None,
    num_samples: int = 25,
    gibbs_steps: int = 100,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, List[Path], Path]:
    """
    Generate synthetic handwritten digit images using trained DBN.

    Args:
        checkpoint_path: Path to .pt checkpoint. If None, loads latest checkpoint.
        num_samples: Number of images to generate (10, 25, 50, 100)
        gibbs_steps: Top RBM alternating Gibbs sampling iterations
        random_seed: Seed for reproducible generation
        output_dir: Folder to save PNGs
        device: CPU or CUDA device

    Returns:
        (generated_images_tensor, list_of_png_paths, grid_plot_path)
    """
    paths = PathConfig()
    paths.ensure_directories()
    target_device = device if device is not None else DeviceConfig().device

    # Find checkpoint
    if checkpoint_path is None:
        ckpt = get_latest_checkpoint(paths.CHECKPOINT_DIR)
        if ckpt is None:
            raise FileNotFoundError(
                f"No DBN checkpoint found in {paths.CHECKPOINT_DIR}. "
                f"Please run model training first via 'python -m models.train' or the web app."
            )
        checkpoint_path = ckpt
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"Loading checkpoint from: {checkpoint_path}")
    model, metadata = load_dbn_checkpoint(checkpoint_path, device=target_device)
    print(f"DBN Architecture: {model.layer_dims} (Pretrained with {metadata.get('training_mode', 'standard')} mode)")

    # Execute probabilistic sampling
    print(f"Generating {num_samples} synthetic digits with {gibbs_steps} Gibbs steps (Seed={random_seed})...")
    gen_images = model.generate(
        num_samples=num_samples,
        gibbs_steps=gibbs_steps,
        random_seed=random_seed
    )

    out_folder = Path(output_dir) if output_dir is not None else paths.GENERATED_DIR
    out_folder.mkdir(parents=True, exist_ok=True)

    # Save individual PNGs
    saved_pngs = save_images_to_png(gen_images, output_dir=out_folder, prefix="synthetic_digit")

    # Save consolidated grid
    grid_path = out_folder / f"grid_{num_samples}_samples.png"
    plot_digit_grid(gen_images, save_path=grid_path)
    print(f"Generated {len(saved_pngs)} PNG images saved to: {out_folder}")
    print(f"Grid preview saved to: {grid_path}")

    return gen_images, saved_pngs, grid_path


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic handwritten digits with trained DBN.")
    parser.add_argument("--samples", type=int, default=25, choices=[10, 25, 50, 100],
                        help="Number of images to generate")
    parser.add_argument("--gibbs_steps", type=int, default=100, help="Number of top RBM Gibbs steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    args = parser.parse_args()

    generate_digits(
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
        gibbs_steps=args.gibbs_steps,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
