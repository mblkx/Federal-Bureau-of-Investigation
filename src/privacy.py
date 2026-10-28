"""Sprawdzenia, czy odpowiedź ``fit`` zawiera tylko wagi modelu (bez macierzy cech)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def model_parameter_shapes(reference_parameters: Sequence[np.ndarray]) -> tuple[tuple[int, ...], ...]:
    """Zwraca kształty tensorów wag wzorcowego modelu.

    Kroki:
    1. Dla każdego ``ndarray`` w ``reference_parameters`` odczytaj wymiary.
    2. Zwróć krotkę krotek (jeden wpis na tensor warstwy MLP).
    """
    return tuple(tuple(int(d) for d in arr.shape) for arr in reference_parameters)


def assert_fit_payload_is_model_only(
    parameters: Sequence[np.ndarray],
    reference_parameters: Sequence[np.ndarray],
    *,
    forbidden_feature_shape: tuple[int, ...] | None = None,
) -> None:
    """Weryfikuje, że payload ``fit`` nie wygląda jak wyciek macierzy cech.

    Kroki:
    1. Sprawdź, czy liczba tensorów equals liczbie wag modelu.
    2. Dla każdego indeksu porównaj kształt z wzorcem — muszą się zgadzać dokładnie.
    3. Jeśli podano ``forbidden_feature_shape`` (np. ``X_train.shape``), upewnij się,
       że żaden tensor nie ma tego kształtu.
    4. W razie naruszenia rzuć ``ValueError`` (używane w testach privacy smoke).
    """
    if len(parameters) != len(reference_parameters):
        raise ValueError(
            f"expected {len(reference_parameters)} parameter tensors, got {len(parameters)}"
        )
    ref_shapes = model_parameter_shapes(reference_parameters)
    for index, (param, ref_shape) in enumerate(zip(parameters, ref_shapes, strict=True)):
        if tuple(param.shape) != ref_shape:
            raise ValueError(
                f"parameter tensor {index} shape {param.shape} != model shape {ref_shape}"
            )
    if forbidden_feature_shape is not None:
        for param in parameters:
            if tuple(param.shape) == forbidden_feature_shape:
                raise ValueError(
                    f"payload matches training feature matrix shape {forbidden_feature_shape}"
                )


def estimate_payload_bytes(parameters: Sequence[np.ndarray]) -> int:
    """Szacuje rozmiar payloadu wag w bajtach (suma ``nbytes`` wszystkich tensorów).

    Kroki:
    1. Dla każdego tensora dodaj ``arr.nbytes`` do sumy.
    2. Zwróć łączny rozmiar (porównanie z rozmiarem shardu w testach).
    """
    return sum(int(arr.nbytes) for arr in parameters)
