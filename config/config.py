"""
Global configuration settings for DBN Digit Generation.
Supports both CPU and GPU execution with dual configurations:
1. Quick Demo Mode: Optimized for rapid laptop CPU testing.
2. Full Training Mode: Full-capacity architecture for academic research and quality.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import os
import torch


@dataclass
class PathConfig:
    """Directory paths for datasets, checkpoints, outputs, and documentation."""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    CHECKPOINT_DIR: Path = OUTPUTS_DIR / "checkpoints"
    GENERATED_DIR: Path = OUTPUTS_DIR / "generated_digits"
    PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
    DOCS_DIR: Path = BASE_DIR / "docs"

    def ensure_directories(self):
        """Create all required output and data folders if they do not exist."""
        for p in [self.DATA_DIR, self.OUTPUTS_DIR, self.CHECKPOINT_DIR, self.GENERATED_DIR, self.PLOTS_DIR, self.DOCS_DIR]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class DeviceConfig:
    """Device selection with automatic fallback to CPU."""
    force_cpu: bool = False
    
    @property
    def device(self) -> torch.device:
        if not self.force_cpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def device_name(self) -> str:
        if not self.force_cpu and torch.cuda.is_available():
            return f"GPU: {torch.cuda.get_device_name(0)}"
        return "CPU (Standard Processor)"


@dataclass
class ModelConfig:
    """Architecture configuration for Stacked RBMs forming the DBN."""
    visible_dim: int = 784  # 28x28 MNIST pixels flattened
    layer_dims: List[int] = field(default_factory=lambda: [784, 500, 250, 100])
    k_cd: int = 1  # Contrastive Divergence k steps
    initial_weights_std: float = 0.01  # Gaussian initialization standard deviation


@dataclass
class TrainingConfig:
    """Hyperparameters for layer-wise pretraining."""
    mode: str = "demo"  # "demo" or "full"
    batch_size: int = 64
    learning_rate: float = 0.05
    momentum: float = 0.5
    weight_decay: float = 0.0001
    epochs_per_rbm: int = 5
    max_train_samples: int = 5000  # Subset size for demo mode (or None for all 60000)
    seed: int = 42


@dataclass
class GenerationConfig:
    """Sampling settings for synthetic digit generation."""
    default_num_samples: int = 25
    gibbs_steps: int = 100
    temperature: float = 1.0
    random_seed: int = 42


# Pre-defined Configurations
DEMO_MODE_CONFIG = {
    "name": "Quick Demo Mode (Laptop CPU Friendly)",
    "layer_dims": [784, 256, 128, 64],
    "epochs_per_rbm": 5,
    "batch_size": 64,
    "learning_rate": 0.05,
    "max_train_samples": 6000,
    "k_cd": 1,
}

FULL_MODE_CONFIG = {
    "name": "Full Training Mode (Deep Multi-layer Architecture)",
    "layer_dims": [784, 500, 250, 100],
    "epochs_per_rbm": 15,
    "batch_size": 64,
    "learning_rate": 0.05,
    "max_train_samples": 30000,
    "k_cd": 1,
}


def get_config(mode: str = "demo"):
    """
    Factory function returning coordinated config objects for paths, device, model, and training.
    """
    paths = PathConfig()
    paths.ensure_directories()
    device_cfg = DeviceConfig()

    preset = DEMO_MODE_CONFIG if mode == "demo" else FULL_MODE_CONFIG

    model_cfg = ModelConfig(
        visible_dim=784,
        layer_dims=preset["layer_dims"],
        k_cd=preset["k_cd"]
    )

    training_cfg = TrainingConfig(
        mode=mode,
        batch_size=preset["batch_size"],
        learning_rate=preset["learning_rate"],
        epochs_per_rbm=preset["epochs_per_rbm"],
        max_train_samples=preset["max_train_samples"],
        seed=42
    )

    gen_cfg = GenerationConfig()

    return paths, device_cfg, model_cfg, training_cfg, gen_cfg
