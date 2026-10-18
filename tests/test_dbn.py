"""
Unit and integration tests for Deep Belief Network (DBN) architecture and checkpointing.
"""

from pathlib import Path
import tempfile
import pytest
import torch

from models.dbn import DeepBeliefNetwork
from utils.checkpoint import save_dbn_checkpoint, load_dbn_checkpoint


def test_dbn_initialization():
    layer_dims = [784, 256, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)

    assert dbn.num_layers == 3
    assert len(dbn.rbms) == 3
    assert dbn.rbms[0].W.shape == (784, 256)
    assert dbn.rbms[1].W.shape == (256, 128)
    assert dbn.rbms[2].W.shape == (128, 64)


def test_dbn_upward_pass():
    layer_dims = [784, 256, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)
    x = torch.rand(10, 784)

    # Pass to layer 1
    h1 = dbn.upward_pass(x, up_to_layer=0)
    assert h1.shape == (10, 256)

    # Pass to layer 2
    h2 = dbn.upward_pass(x, up_to_layer=1)
    assert h2.shape == (10, 128)

    # Pass to top layer
    h3 = dbn.upward_pass(x, up_to_layer=2)
    assert h3.shape == (10, 64)


def test_dbn_downward_pass():
    layer_dims = [784, 256, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)
    top_h = torch.rand(8, 64)

    recon = dbn.downward_pass(top_h, from_layer=2)
    assert recon.shape == (8, 784)
    assert (recon >= 0.0).all() and (recon <= 1.0).all()


def test_dbn_roundtrip_reconstruction():
    layer_dims = [784, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)
    x = torch.rand(4, 784)

    recon = dbn.reconstruct(x)
    assert recon.shape == (4, 784)
    assert (recon >= 0.0).all() and (recon <= 1.0).all()


def test_dbn_checkpoint_save_and_load():
    layer_dims = [784, 128, 64]
    dbn = DeepBeliefNetwork(layer_dims=layer_dims)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_checkpoint.pt"
        save_dbn_checkpoint(
            model=dbn,
            save_path=tmp_path,
            training_mode="test",
            history={"rbm_1": {"epochs": [1], "loss": [0.5], "recon_error": [0.1]}}
        )

        assert tmp_path.exists()

        # Load back
        loaded_model, metadata = load_dbn_checkpoint(tmp_path)
        assert loaded_model.layer_dims == layer_dims
        assert metadata["training_mode"] == "test"
        assert "rbm_1" in metadata["history"]

        # Ensure parameters match
        for p1, p2 in zip(dbn.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)
