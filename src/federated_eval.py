"""Trening FedAvg w jednym procesie oraz końcowa ocena modelu globalnego na KDDTest+."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from flwr.common import Parameters, parameters_to_ndarrays

from src.evaluate import evaluate, save_results
from src.fedavg import fed_avg
from src.preprocess import load_prepared_arrays
from src.sharding import load_shard
from src.training import LocalModel, load_project_config, set_seed, shard_path


def save_global_weights(path: str | Path, parameters: Sequence[np.ndarray]) -> None:
    """Zapisuje wagi modelu globalnego do skompresowanego pliku NPZ.

    Kroki:
    1. Utwórz katalog nadrzędny ścieżki docelowej, jeśli nie istnieje.
    2. Każdy tensor wag zapisz pod kluczem ``p0``, ``p1``, … (kolejność warstw MLP).
    3. Zapisz archiwum przez ``np.savez_compressed``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = list(parameters)
    payload = {f"p{i}": arr for i, arr in enumerate(arrays)}
    np.savez_compressed(path, **payload)


def load_global_weights(path: str | Path) -> list[np.ndarray]:
    """Odczytuje wagi zapisane przez ``save_global_weights``.

    Kroki:
    1. Otwórz archiwum NPZ.
    2. Posortuj klucze ``p*`` numerycznie, żeby odtworzyć kolejność tensorów.
    3. Zwróć listę tablic w tej kolejności.
    """
    with np.load(Path(path)) as archive:
        keys = sorted(archive.files, key=lambda name: int(name[1:]))
        return [archive[key] for key in keys]


def parameters_to_ndarray_list(parameters: Parameters) -> list[np.ndarray]:
    """Konwertuje obiekt Flower ``Parameters`` na listę ``numpy.ndarray``.

    Kroki:
    1. Wywołaj ``parameters_to_ndarrays`` z biblioteki Flower.
    2. Zwróć wynik (gotowy do ``LocalModel.set_parameters``).
    """
    return parameters_to_ndarrays(parameters)


def evaluate_global_parameters(
    cfg: dict[str, Any],
    parameters: Sequence[np.ndarray],
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Ocenia model globalny na oficjalnym zbiorze ``KDDTest+``.

    Kroki:
    1. Wczytaj przetworzone macierze testowe i kategorie ataków z ``processed_dir``.
    2. Załaduj wagi do ``LocalModel`` i policz P(atak) dla każdego wiersza testu.
    3. Wywołaj ``evaluate()`` (próg, F1, macierz pomyłek, metryki per kategoria).
    4. Dołącz metadane runu (model, split, partycja shardów, liczba rund) oraz
       ``meets_project_criterion`` względem ``evaluation.f1_threshold`` z configu.
    """
    data_cfg = cfg["data"]
    processed = load_prepared_arrays(data_cfg["processed_dir"])
    local = LocalModel.from_config(cfg)
    local.set_parameters(parameters)
    scores = local.predict_attack_proba(processed["X_test"])
    categories = processed.get("cat_test")
    result: dict[str, object] = {
        "model": "fedavg_global",
        "split": "KDDTest+",
        "shard_partition": data_cfg.get("shard_partition"),
        "num_rounds": int(cfg["federated"]["num_rounds"]),
        **evaluate(processed["y_test"], scores, categories, threshold=threshold),
    }
    eval_cfg = cfg.get("evaluation") or {}
    f1_threshold = float(eval_cfg.get("f1_threshold", 0.89))
    result["f1_threshold"] = f1_threshold
    result["meets_project_criterion"] = float(result["f1"]) > f1_threshold
    return result


def run_inprocess_fedavg(config_path: str | Path) -> list[np.ndarray]:
    """Symuluje pętlę FedAvg lokalnie (bez sieci), zgodnie z logiką klientów Flower.

    Kroki (dla każdej rundy):
    1. Pobierz aktualne wagi modelu globalnego.
    2. Dla każdego klienta: załaduj shard, skopiuj wagi globalne, wytrenuj lokalnie,
       zapisz ``state_dict`` i liczbę próbek (waga w ``fed_avg``).
    3. Zagreguj wagi przez ``fed_avg`` i załaduj je z powrotem do modelu globalnego.
    Po wszystkich rundach zwróć finalną listę ``ndarray`` wag.
    """
    cfg = load_project_config(config_path)
    set_seed(int(cfg["random_seed"]))
    data_cfg = cfg["data"]
    num_clients = int(cfg["num_clients"])
    num_rounds = int(cfg["federated"]["num_rounds"])
    partition = data_cfg["shard_partition"]

    global_model = LocalModel.from_config(cfg)
    for _round in range(num_rounds):
        global_params = global_model.get_parameters()
        state_dicts: list[OrderedDict[str, torch.Tensor]] = []
        sample_counts: list[int] = []

        for client_id in range(num_clients):
            shard = load_shard(shard_path(data_cfg["shards_dir"], partition, client_id))
            client = LocalModel.from_config(cfg)
            client.set_parameters(global_params)
            num_examples = client.train_on(shard["X"], shard["y"])
            state_dicts.append(OrderedDict(client.model.state_dict()))
            sample_counts.append(num_examples)

        aggregated = fed_avg(state_dicts, sample_counts)
        global_model.model.load_state_dict(aggregated, strict=True)

    return global_model.get_parameters()


def write_federated_results(
    cfg: dict[str, Any],
    parameters: Sequence[np.ndarray],
    metrics_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Zapisuje opcjonalnie wagi i zawsze JSON z metrykami końcowymi.

    Kroki:
    1. Jeśli podano ``weights_path`` — zapisz parametry do NPZ.
    2. Policz słownik metryk przez ``evaluate_global_parameters``.
    3. Zapisz JSON przez ``save_results`` i zwróć ten sam słownik.
    """
    if weights_path is not None:
        save_global_weights(weights_path, parameters)
    result = evaluate_global_parameters(cfg, parameters, threshold=threshold)
    save_results(result, metrics_path)
    return result
