"""
Model Checkpointing and Parameter Persistence.

Enables saving and loading trained Deep Belief Network checkpoints along with
hyperparameters, layer configurations, and training histories so that users
can load pre-trained weights without waiting for full retraining.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, TYPE_CHECKING
import json
import torch

if TYPE_CHECKING:
    from models.dbn import DeepBeliefNetwork


def save_dbn_checkpoint(
    model: Any,
    save_path: Path,
    training_mode: str = "demo",
    history: Optional[Dict[str, Any]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save DBN model weights, architecture configuration, and training metrics.

    Args:
        model: Trained DeepBeliefNetwork instance
        save_path: Destination filepath (.pt)
        training_mode: "demo" or "full"
        history: Epoch-by-epoch loss and reconstruction diagnostic dict
        extra_metadata: Additional arbitrary metadata

    Returns:
        Path to the saved checkpoint file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "layer_dims": model.layer_dims,
        "num_layers": model.num_layers,
        "training_mode": training_mode,
        "history": history or {},
        "extra": extra_metadata or {}
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata
    }

    torch.save(checkpoint, str(save_path))

    # Also save an easily human-readable JSON summary alongside the binary weights
    meta_path = save_path.with_suffix(".json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        pass

    return save_path


def load_dbn_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None
) -> Tuple[DeepBeliefNetwork, Dict[str, Any]]:
    """
    Load a saved DeepBeliefNetwork from disk.

    Args:
        checkpoint_path: Path to .pt file
        device: Device to place the loaded model on (defaults to CPU)

    Returns:
        (model, metadata)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    target_device = device if device is not None else torch.device("cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=target_device)

    metadata = checkpoint.get("metadata", {})
    layer_dims = metadata.get("layer_dims", [784, 256, 128, 64])

    from models.dbn import DeepBeliefNetwork
    model = DeepBeliefNetwork(
        layer_dims=layer_dims,
        device=target_device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, metadata


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the most recently modified DBN .pt checkpoint in the given folder.
    Excludes independent classifier checkpoints like eval_classifier.pt.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # Specifically filter for DBN checkpoints
    pts = [p for p in checkpoint_dir.glob("*.pt") if not p.name.startswith("eval_")]
    if not pts:
        return None

    # Sort by modification time, newest first
    pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0]

