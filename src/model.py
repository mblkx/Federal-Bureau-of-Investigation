"""Architektury PyTorch: MLP (klasyfikacja) oraz autoencoder (detekcja anomalii)."""

from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Perceptron wielowarstwowy: cechy → logity klas (normal / atak)."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU(), nn.Dropout(0.1)])
            prev = hidden
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Autoencoder(nn.Module):
    """Autoencoder rekonstrukcyjny: uczy się ruchu normalnego (M2.3).

    Enkoder: ``input → hidden_dims → latent``.
    Dekoder: ``latent → reversed(hidden_dims) → input`` (wyjście liniowe —
    cechy po standaryzacji mogą być ujemne). Bez Dropout, żeby rekonstrukcja
    była stabilna.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty for the autoencoder")

        enc: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            enc.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        enc.extend([nn.Linear(prev, latent_dim), nn.ReLU()])
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = []
        prev = latent_dim
        for hidden in reversed(hidden_dims):
            dec.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Zwraca rekonstrukcję ``x`` (ten sam kształt co wejście)."""
        return self.decoder(self.encoder(x))
