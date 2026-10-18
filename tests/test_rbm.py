"""
Unit tests for Restricted Boltzmann Machine (RBM) mathematical implementation.
"""

import pytest
import torch
from models.rbm import RBM


def test_rbm_initialization():
    visible_dim = 784
    hidden_dim = 256
    rbm = RBM(visible_dim=visible_dim, hidden_dim=hidden_dim)

    assert rbm.W.shape == (visible_dim, hidden_dim)
    assert rbm.v_bias.shape == (visible_dim,)
    assert rbm.h_bias.shape == (hidden_dim,)
    assert rbm.visible_dim == visible_dim
    assert rbm.hidden_dim == hidden_dim


def test_sample_h_dimensions_and_values():
    rbm = RBM(visible_dim=784, hidden_dim=128)
    v = torch.rand(16, 784)

    p_h, h_sample = rbm.sample_h(v)

    assert p_h.shape == (16, 128)
    assert h_sample.shape == (16, 128)
    # Probabilities must be strictly bounded in [0, 1]
    assert (p_h >= 0.0).all() and (p_h <= 1.0).all()
    # Bernoulli samples must only contain 0 or 1
    unique_vals = set(h_sample.flatten().tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_sample_v_dimensions_and_values():
    rbm = RBM(visible_dim=784, hidden_dim=128)
    h = torch.rand(16, 128)

    p_v, v_sample = rbm.sample_v(h)

    assert p_v.shape == (16, 784)
    assert v_sample.shape == (16, 784)
    assert (p_v >= 0.0).all() and (p_v <= 1.0).all()
    unique_vals = set(v_sample.flatten().tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_free_energy_shape():
    rbm = RBM(visible_dim=100, hidden_dim=50)
    v = torch.rand(8, 100)

    fe = rbm.free_energy(v)
    assert fe.shape == (8,)
    assert torch.isfinite(fe).all()


def test_contrastive_divergence_updates_parameters():
    torch.manual_seed(42)
    rbm = RBM(visible_dim=64, hidden_dim=32, learning_rate=0.1)

    initial_W = rbm.W.clone()
    initial_v_bias = rbm.v_bias.clone()
    initial_h_bias = rbm.h_bias.clone()

    dummy_data = (torch.rand(32, 64) > 0.5).float()
    loss, recon = rbm.contrastive_divergence(dummy_data, k=1)

    assert isinstance(loss, float)
    assert isinstance(recon, float)
    assert recon >= 0.0

    # Weights and biases must be updated by the optimizer
    assert not torch.equal(rbm.W, initial_W)
    assert not torch.equal(rbm.v_bias, initial_v_bias)
    assert not torch.equal(rbm.h_bias, initial_h_bias)
