"""Testy architektur MLP i autoencodera."""

import torch

from src.model import Autoencoder, MLPClassifier


def test_mlp_forward_shape():
    model = MLPClassifier(input_dim=12, hidden_dims=[8, 4], num_classes=2)
    x = torch.randn(5, 12)
    out = model(x)
    assert out.shape == (5, 2)


def test_autoencoder_reconstructs_same_shape():
    model = Autoencoder(input_dim=12, hidden_dims=[8, 4], latent_dim=2)
    x = torch.randn(5, 12)
    recon = model(x)
    assert recon.shape == x.shape


def test_autoencoder_state_dict_has_encoder_and_decoder_keys():
    model = Autoencoder(input_dim=10, hidden_dims=[6], latent_dim=3)
    keys = list(model.state_dict().keys())
    assert any(k.startswith("encoder.") for k in keys)
    assert any(k.startswith("decoder.") for k in keys)
