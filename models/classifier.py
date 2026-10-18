"""
Lightweight MNIST Digit Classifier for Synthetic Evaluation.

Provides:
1. A compact PyTorch CNN classifier trained independently on MNIST.
2. Recognizability scoring for DBN-generated images (predicted classes & confidences).
3. Clear architectural and statistical separation:
   - This classifier is strictly an external evaluation diagnostic.
   - It is never used in the DBN loss function or training loop.
   - Classifier confidence is documented as a heuristic proxy, not human realism.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DigitClassifier(nn.Module):
    """
    Lightweight 2-stage CNN for MNIST digit classification (0-9).
    """

    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # 28x28 -> pool1 (14x14) -> pool2 (7x7) -> 32 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, 28, 28) or (B, 28, 28) or (B, 784)
        if x.dim() == 2 and x.size(1) == 784:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def train_classifier(
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 2,
    lr: float = 0.001
) -> DigitClassifier:
    """
    Quickly train or fine-tune the lightweight evaluation classifier.
    2 epochs achieves ~96-98% accuracy on MNIST.
    """
    model = DigitClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def save_classifier(model: DigitClassifier, path: Path) -> Path:
    """Save trained classifier checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    return path


def load_classifier(path: Path, device: torch.device) -> Optional[DigitClassifier]:
    """Load classifier weights if existing."""
    path = Path(path)
    if not path.exists():
        return None
    model = DigitClassifier().to(device)
    model.load_state_dict(torch.load(str(path), map_location=device))
    model.eval()
    return model


def evaluate_recognizability(
    classifier: DigitClassifier,
    generated_images: torch.Tensor,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Evaluate synthetic images using the trained classifier.

    Args:
        classifier: Trained DigitClassifier instance
        generated_images: Tensor of shape (N, 28, 28) in range [0, 1]
        confidence_threshold: Softmax threshold to declare a sample 'confidently recognized'

    Returns:
        Dict containing:
            - predicted_labels: List of int classes
            - confidences: List of float confidence values
            - recognizable_ratio: Fraction of images exceeding confidence_threshold
            - digit_distribution: Count of each digit (0-9) generated
            - average_confidence: Mean softmax confidence across all generated samples
            - disclaimer: Academic statement on the limits of proxy classifiers
    """
    device = next(classifier.parameters()).device
    classifier.eval()

    with torch.no_grad():
        x = generated_images.to(device)
        logits = classifier(x)
        probs = F.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

    preds_list = preds.cpu().tolist()
    confs_list = confidences.cpu().tolist()

    confident_count = sum(1 for c in confs_list if c >= confidence_threshold)
    recognizable_ratio = confident_count / len(confs_list) if confs_list else 0.0

    digit_distribution = {str(d): preds_list.count(d) for d in range(10)}

    return {
        "predicted_labels": preds_list,
        "confidences": confs_list,
        "recognizable_ratio": recognizable_ratio,
        "confident_count": confident_count,
        "total_samples": len(confs_list),
        "average_confidence": sum(confs_list) / len(confs_list) if confs_list else 0.0,
        "confidence_threshold": confidence_threshold,
        "digit_distribution": digit_distribution,
        "disclaimer": (
            "Academic Note: A high classifier confidence indicates that synthetic pixel "
            "activations align with features learned from real MNIST digits. It is an "
            "objective proxy metric and does NOT indicate that the generated sample is "
            "visually indistinguishable from human handwriting."
        )
    }
