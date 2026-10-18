"""
Deep Belief Network models package.
"""
from .rbm import RBM
from .dbn import DeepBeliefNetwork
from .classifier import DigitClassifier, load_classifier, evaluate_recognizability

__all__ = [
    "RBM",
    "DeepBeliefNetwork",
    "DigitClassifier",
    "load_classifier",
    "evaluate_recognizability",
    "train_dbn_pipeline",
    "generate_digits",
    "evaluate_dbn"
]


def __getattr__(name: str):
    if name == "train_dbn_pipeline":
        from .train import train_dbn_pipeline
        return train_dbn_pipeline
    if name == "generate_digits":
        from .generate import generate_digits
        return generate_digits
    if name == "evaluate_dbn":
        from .evaluate import evaluate_dbn
        return evaluate_dbn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
