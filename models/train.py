"""
Greedy Layer-Wise Pretraining Pipeline for Deep Belief Network.

Orchestrates:
1. Loading & preprocessing MNIST.
2. Training RBM 1 on normalized pixel vectors.
3. Propagating activations to form inputs for RBM 2.
4. Training RBM 2.
5. Propagating activations to form inputs for RBM 3.
6. Training RBM 3.
7. Model checkpointing and training diagnostic plots.
8. Training evaluation classifier.

Can be executed directly from CLI or invoked by the Streamlit web interface with
live progress callbacks.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import torch
from torch.utils.data import DataLoader, TensorDataset

from config.config import get_config, PathConfig, DeviceConfig
from models.dbn import DeepBeliefNetwork
from models.classifier import train_classifier, save_classifier
from utils.data_loader import load_mnist
from utils.checkpoint import save_dbn_checkpoint
from utils.visualization import plot_training_curves


def extract_hidden_representations(
    rbm_model,
    dataloader: DataLoader,
    device: torch.device
) -> TensorDataset:
    """
    Pass dataset through an RBM to obtain continuous hidden probabilities P(h=1|v),
    returning a new TensorDataset ready for the next RBM layer.
    """
    rbm_model.eval()
    hidden_features = []

    with torch.no_grad():
        for batch in dataloader:
            # batch is (data, target) or (data,)
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            p_h, _ = rbm_model.sample_h(x)
            hidden_features.append(p_h.cpu())

    all_features = torch.cat(hidden_features, dim=0)
    return TensorDataset(all_features)


def train_dbn_pipeline(
    mode: str = "demo",
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    learning_rate_override: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, int, float, float, str], None]] = None,
    checkpoint_dir: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Full training pipeline for Stacked RBMs forming a Deep Belief Network.

    Args:
        mode: "demo" (fast CPU subset) or "full"
        epochs_override: Optional custom epochs per RBM
        batch_size_override: Optional custom batch size
        learning_rate_override: Optional custom learning rate
        progress_callback: Optional fn(rbm_idx, epoch, total_epochs, loss, recon_error, msg)
        checkpoint_dir: Optional custom folder for checkpoints
        seed: Random seed for initialization

    Returns:
        Summary dict containing trained model, history, paths, and metadata
    """
    torch.manual_seed(seed)
    paths, device_cfg, model_cfg, training_cfg, gen_cfg = get_config(mode=mode)
    device = device_cfg.device

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else paths.CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs_per_rbm = epochs_override if epochs_override is not None else training_cfg.epochs_per_rbm
    batch_size = batch_size_override if batch_size_override is not None else training_cfg.batch_size
    learning_rate = learning_rate_override if learning_rate_override is not None else training_cfg.learning_rate

    print("=" * 70)
    print(" DEEP BELIEF NETWORK PRETRAINING PIPELINE")
    print(f" Mode: {mode.upper()} | Device: {device_cfg.device_name}")
    print(f" Layer Dimensions: {model_cfg.layer_dims}")
    print(f" Epochs Per RBM: {epochs_per_rbm} | Batch Size: {batch_size} | LR: {learning_rate}")
    print("=" * 70)

    # Step 1: Load Dataset
    print("\n[Stage 1/6] Loading MNIST dataset...")
    train_loader, test_loader, meta = load_mnist(
        data_dir=paths.DATA_DIR,
        batch_size=batch_size,
        subset_size=training_cfg.max_train_samples,
        binarize=False,
        seed=seed
    )
    print(f"  Training samples used: {meta['active_train_samples']} / {meta['total_train_available']}")

    # Step 2: Initialize DBN
    dbn = DeepBeliefNetwork(
        layer_dims=model_cfg.layer_dims,
        k_cd=model_cfg.k_cd,
        learning_rate=learning_rate,
        device=device
    )

    history: Dict[str, Any] = {}
    current_loader = train_loader
    start_total_time = time.time()

    # Step 3: Greedy Layer-Wise Pretraining
    for rbm_idx in range(dbn.num_layers):
        rbm = dbn.rbms[rbm_idx]
        rbm_name = f"rbm_{rbm_idx + 1}"
        in_dim = model_cfg.layer_dims[rbm_idx]
        out_dim = model_cfg.layer_dims[rbm_idx + 1]

        print(f"\n[Stage {rbm_idx + 2}/6] Training {rbm_name.upper()} ({in_dim} -> {out_dim})...")
        history[rbm_name] = {
            "epochs": [],
            "loss": [],
            "recon_error": []
        }

        rbm.train()
        for epoch in range(1, epochs_per_rbm + 1):
            epoch_loss = 0.0
            epoch_recon = 0.0
            num_batches = 0

            for batch in current_loader:
                v = batch[0] if isinstance(batch, (list, tuple)) else batch
                v = v.to(device)

                loss_val, recon_val = rbm.contrastive_divergence(v)
                epoch_loss += loss_val
                epoch_recon += recon_val
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_recon = epoch_recon / max(num_batches, 1)

            history[rbm_name]["epochs"].append(epoch)
            history[rbm_name]["loss"].append(avg_loss)
            history[rbm_name]["recon_error"].append(avg_recon)

            msg = f"  [{rbm_name.upper()}] Epoch {epoch}/{epochs_per_rbm} - Loss: {avg_loss:.4f} | Recon MSE: {avg_recon:.4f}"
            print(msg)

            if progress_callback:
                progress_callback(rbm_idx, epoch, epochs_per_rbm, avg_loss, avg_recon, msg)

        # Prepare representations for next RBM layer if not the last layer
        if rbm_idx < dbn.num_layers - 1:
            print(f"  Transforming dataset activations for RBM {rbm_idx + 2}...")
            next_dataset = extract_hidden_representations(rbm, current_loader, device)
            current_loader = DataLoader(
                next_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            )

    elapsed_train = time.time() - start_total_time
    print(f"\nTraining completed in {elapsed_train:.2f} seconds.")

    # Step 4: Save Model Checkpoint
    checkpoint_filename = f"dbn_mnist_{mode}.pt"
    ckpt_path = ckpt_dir / checkpoint_filename
    saved_path = save_dbn_checkpoint(
        model=dbn,
        save_path=ckpt_path,
        training_mode=mode,
        history=history,
        extra_metadata={
            "elapsed_seconds": elapsed_train,
            "samples_used": meta["active_train_samples"],
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs_per_rbm": epochs_per_rbm
        }
    )
    print(f"\n[Stage 5/6] Model checkpoint saved to: {saved_path}")

    # Step 5: Save Training Diagnostic Plot
    plot_file = paths.PLOTS_DIR / f"training_curves_{mode}.png"
    try:
        plot_training_curves(history, save_path=plot_file)
        print(f"  Diagnostic training curve saved to: {plot_file}")
    except Exception as e:
        print(f"  Warning: Could not save training plot: {e}")

    # Step 6: Train Evaluation Classifier (Quick 2 epochs)
    print("\n[Stage 6/6] Training independent evaluation classifier (MNIST CNN)...")
    classifier_path = ckpt_dir / "eval_classifier.pt"
    classifier = train_classifier(train_loader, device=device, epochs=2)
    save_classifier(classifier, classifier_path)
    print(f"  Evaluation classifier saved to: {classifier_path}")

    return {
        "model": dbn,
        "history": history,
        "checkpoint_path": saved_path,
        "plot_path": plot_file,
        "classifier_path": classifier_path,
        "elapsed_seconds": elapsed_train,
        "mode": mode,
        "metadata": meta
    }


def main():
    parser = argparse.ArgumentParser(description="Train Deep Belief Network on MNIST.")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full"],
                        help="demo (fast laptop mode) or full (deep architecture)")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs per RBM layer")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    args = parser.parse_args()

    train_dbn_pipeline(
        mode=args.mode,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        learning_rate_override=args.lr
    )


if __name__ == "__main__":
    main()
