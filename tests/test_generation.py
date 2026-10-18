"""
Unit tests for DBN synthetic handwritten digit generation.
"""

import pytest
import torch
from models.dbn import DeepBeliefNetwork


def test_generation_shape_and_range():
    layer_dims = [784, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)

    num_samples = 10
    images = dbn.generate(num_samples=num_samples, gibbs_steps=10, random_seed=42)

    assert images.shape == (num_samples, 28, 28)
    assert images.dtype == torch.float32
    assert images.min().item() >= 0.0
    assert images.max().item() <= 1.0


def test_generation_reproducibility_with_seed():
    layer_dims = [784, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)

    images_run1 = dbn.generate(num_samples=5, gibbs_steps=15, random_seed=123)
    images_run2 = dbn.generate(num_samples=5, gibbs_steps=15, random_seed=123)

    assert torch.allclose(images_run1, images_run2, atol=1e-6)


def test_generation_different_seeds_produce_variation():
    layer_dims = [784, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)

    images_seed1 = dbn.generate(num_samples=5, gibbs_steps=10, random_seed=10)
    images_seed2 = dbn.generate(num_samples=5, gibbs_steps=10, random_seed=99)

    assert not torch.equal(images_seed1, images_seed2)


def test_generation_trajectory_milestones():
    layer_dims = [784, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)

    milestones = [0, 5, 20]
    traj = dbn.generate_trajectory(num_samples=4, milestones=milestones, random_seed=42)

    assert set(traj.keys()) == {0, 5, 20}
    for step in milestones:
        assert traj[step].shape == (4, 28, 28)
        assert traj[step].min().item() >= 0.0
        assert traj[step].max().item() <= 1.0

