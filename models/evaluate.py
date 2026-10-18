"""
Evaluation and Comparison Module for Deep Belief Network.

Provides comprehensive diagnostic metrics:
1. End-to-end Reconstruction MSE on real test digits.
2. Statistical pixel distribution comparisons (Mean intensity, variance, background sparsity).
3. External CNN Classifier recognizability evaluation (proxy metric for feature realism).
4. Side-by-side visual comparison artifact generation.
5. Pixel distribution histogram generation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch

from config.config import get_config, PathConfig, DeviceConfig
from models.dbn import DeepBeliefNetwork
from models.classifier import load_classifier, evaluate_recognizability
from utils.checkpoint import load_dbn_checkpoint, get_latest_checkpoint
from utils.data_loader import load_mnist, get_sample_real_images
from utils.visualization import plot_real_vs_generated, plot_pixel_distribution


def evaluate_dbn(
    checkpoint_path: Optional[Path] = None,
    num_samples: int = 25,
    gibbs_steps: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run full qualitative and quantitative evaluation of trained DBN.

    Returns:
        Evaluation report dictionary with metrics and plot filepaths.
    """
    paths = PathConfig()
    paths.ensure_directories()
    device = DeviceConfig().device

    # 1. Load DBN Model
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint(paths.CHECKPOINT_DIR)
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint found. Please train DBN first.")
    else:
        checkpoint_path = Path(checkpoint_path)

    model, metadata = load_dbn_checkpoint(checkpoint_path, device=device)

    # 2. Load Real MNIST Samples
    real_images_flat, real_labels = get_sample_real_images(
        data_dir=paths.DATA_DIR,
        num_samples=num_samples,
        seed=seed
    )
    # real_images_flat is shape (N, 28, 28)

    # 3. Compute DBN Reconstruction MSE on real samples
    with torch.no_grad():
        flat_input = real_images_flat.view(num_samples, 784).to(device)
        recon_flat = model.reconstruct(flat_input)
        reconstruction_mse = torch.nn.functional.mse_loss(recon_flat, flat_input).item()

    # 4. Generate Synthetic Samples
    gen_images = model.generate(
        num_samples=num_samples,
        gibbs_steps=gibbs_steps,
        random_seed=seed
    )

    # 5. Statistical Pixel Distribution Metrics
    real_np = real_images_flat.numpy()
    gen_np = gen_images.cpu().numpy()

    real_mean_intensity = float(real_np.mean())
    gen_mean_intensity = float(gen_np.mean())
    real_var_intensity = float(real_np.var())
    gen_var_intensity = float(gen_np.var())

    # Sparsity: percentage of background pixels (< 0.1)
    real_background_ratio = float((real_np < 0.1).mean() * 100)
    gen_background_ratio = float((gen_np < 0.1).mean() * 100)

    # 6. Classifier Recognizability Evaluation (if available)
    classifier_path = paths.CHECKPOINT_DIR / "eval_classifier.pt"
    classifier = load_classifier(classifier_path, device=device)
    classifier_results = None
    if classifier is not None:
        classifier_results = evaluate_recognizability(
            classifier=classifier,
            generated_images=gen_images,
            confidence_threshold=0.70
        )

    # 7. Generate and Save Comparison Plots
    comp_plot_path = paths.PLOTS_DIR / "real_vs_generated_comparison.png"
    plot_real_vs_generated(real_images_flat, gen_images, num_samples=min(10, num_samples), save_path=comp_plot_path)

    dist_plot_path = paths.PLOTS_DIR / "pixel_intensity_distribution.png"
    plot_pixel_distribution(real_images_flat, gen_images, save_path=dist_plot_path)

    report = {
        "checkpoint_used": str(checkpoint_path.name),
        "architecture": model.layer_dims,
        "num_samples_evaluated": num_samples,
        "reconstruction_mse": reconstruction_mse,
        "pixel_statistics": {
            "real_mean_intensity": real_mean_intensity,
            "gen_mean_intensity": gen_mean_intensity,
            "real_variance": real_var_intensity,
            "gen_variance": gen_var_intensity,
            "real_background_sparsity_pct": real_background_ratio,
            "gen_background_sparsity_pct": gen_background_ratio,
        },
        "classifier_evaluation": classifier_results,
        "plot_comparison_path": comp_plot_path,
        "plot_distribution_path": dist_plot_path,
        "real_images": real_images_flat,
        "generated_images": gen_images,
    }

    return report


def main():
    report = evaluate_dbn(num_samples=25, gibbs_steps=100)
    print("=" * 60)
    print(" DBN EVALUATION REPORT")
    print("=" * 60)
    print(f"Checkpoint: {report['checkpoint_used']}")
    print(f"Architecture: {report['architecture']}")
    print(f"End-to-End Reconstruction MSE: {report['reconstruction_mse']:.4f}")
    print("\nPixel Statistics:")
    print(f"  Real Mean Intensity: {report['pixel_statistics']['real_mean_intensity']:.4f}")
    print(f"  Gen Mean Intensity:  {report['pixel_statistics']['gen_mean_intensity']:.4f}")
    print(f"  Real Background %:   {report['pixel_statistics']['real_background_sparsity_pct']:.1f}%")
    print(f"  Gen Background %:    {report['pixel_statistics']['gen_background_sparsity_pct']:.1f}%")

    if report["classifier_evaluation"]:
        c = report["classifier_evaluation"]
        print("\nClassifier Recognizability Assessment:")
        print(f"  Recognizable Ratio (conf >= 0.7): {c['recognizable_ratio'] * 100:.1f}% ({c['confident_count']}/{c['total_samples']})")
        print(f"  Average Confidence: {c['average_confidence'] * 100:.1f}%")
        print(f"  Digit Breakdown: {c['digit_distribution']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
