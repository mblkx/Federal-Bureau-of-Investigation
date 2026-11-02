"""Serwer Flower: weighted FedAvg, logi metryk po rundzie, opcjonalny zapis modelu globalnego."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg

from src.federated_eval import write_federated_results
from src.training import LocalModel, load_project_config, set_seed

logger = logging.getLogger(__name__)


def weighted_evaluate_metrics(
    metrics: Sequence[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Scala metryki z fazy ``evaluate`` wszystkich klientów w jednej rundzie.

    Kroki:
    1. Jeśli brak wpisów lub suma ``num_examples`` jest zerowa — zwróć pusty słownik.
    2. Dla każdego klucza metryki (np. ``f1``) zbierz ważoną sumę:
       ``sum_i (n_i * metric_i)``, gdzie ``n_i`` to liczba próbek z evaluate klienta ``i``.
    3. Podziel każdą sumę przez łączną liczbę próbek — dostajesz jedną wartość na metrykę.
    """
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples <= 0:
        return {}

    keys = metrics[0][1].keys()
    aggregated: dict[str, float] = {}
    for key in keys:
        weighted_sum = 0.0
        for num_examples, client_metrics in metrics:
            if key in client_metrics:
                weighted_sum += num_examples * float(client_metrics[key])
        aggregated[key] = weighted_sum / total_examples
    return aggregated


def log_weighted_evaluate_metrics(
    metrics: Sequence[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    """Callback Flower: agreguje metryki rundy i zapisuje je w logu serwera.

    Kroki:
    1. Wywołaj ``weighted_evaluate_metrics`` na liście od klientów.
    2. Jeśli wynik niepusty — zaloguj posortowane metryki na poziomie INFO.
    3. Zwróć ten sam słownik (Flower zapisuje go też w historii strategii).
    """
    aggregated = weighted_evaluate_metrics(metrics)
    if aggregated:
        parts = ", ".join(f"{name}={value:.4f}" for name, value in sorted(aggregated.items()))
        logger.info("Federated evaluate (weighted): %s", parts)
    return aggregated


def initial_parameters_from_config(
    config_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> Parameters:
    """Buduje deterministyczne wagi globalne na początek federacji.

    Kroki:
    1. Wczytaj YAML (jeśli ``cfg`` nie został podany z góry).
    2. Ustaw seed z configu i utwórz ``LocalModel`` — ta sama architektura co u klientów.
    3. Pobierz listę ``ndarray`` wag i opakuj w typ ``Parameters`` dla strategii FedAvg.
    """
    if cfg is None:
        cfg = load_project_config(config_path)
    set_seed(int(cfg["random_seed"]))
    local = LocalModel.from_config(cfg)
    return ndarrays_to_parameters(local.get_parameters())


class FedAvgRecordingFinal(FedAvg):
    """Strategia FedAvg z zapamiętaniem ostatnich zagregowanych wag po ``fit``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Inicjalizuje strategię Flower i zeruje bufor ``final_parameters``."""
        super().__init__(*args, **kwargs)
        self.final_parameters: Parameters | None = None

    def aggregate_fit(self, server_round: int, results, failures):  # type: ignore[no-untyped-def]
        """Agreguje odpowiedzi ``fit`` klientów i aktualizuje bufor wag globalnych.

        Kroki:
        1. Deleguj uśrednianie wag do ``FedAvg.aggregate_fit``.
        2. Jeśli agregacja się udała — nadpisz ``final_parameters`` (po ostatniej rundzie
           zostaje model końcowy).
        3. Zwróć parę ``(zagregowane_wagi, metryki_fit)`` zgodnie z API Flower.
        """
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.final_parameters = aggregated
        return aggregated, metrics


def build_fedavg_strategy(
    config_path: str | Path,
    *,
    num_clients: int = 3,
) -> FedAvgRecordingFinal:
    """Tworzy strategię weighted FedAvg pod NSL-KDD (wszyscy klienci w każdej rundzie).

    Kroki:
    1. Wczytaj config i ustal ``n`` — liczbę klientów (z YAML lub argumentu domyślnego).
    2. Zbuduj ``initial_parameters`` z tego samego configu (spójny start rundy 0).
    3. Zwróć ``FedAvgRecordingFinal`` z ``fraction_* = 1.0``, ``min_*_clients = n`` oraz
       callbackiem agregacji metryk evaluate.
    """
    cfg = load_project_config(config_path)
    n = int(cfg.get("num_clients", num_clients))
    initial_parameters = initial_parameters_from_config(config_path, cfg=cfg)
    return FedAvgRecordingFinal(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n,
        min_evaluate_clients=n,
        min_available_clients=n,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=log_weighted_evaluate_metrics,
    )


def main() -> None:
    """Uruchamia serwer gRPC Flower i opcjonalnie zapisuje model oraz metryki na KDDTest+.

    Kroki:
    1. Skonfiguruj logowanie i parsuj argumenty CLI (config, adres, liczba klientów, ścieżki zapisu).
    2. Zbuduj strategię i wywołaj ``start_server`` — proces czeka na ``n`` klientów i ``num_rounds`` rund.
    3. Gdy podano ``--metrics-out``: odczytaj ``final_parameters``, zapisz wagi (domyślnie
       ``results/global_model.npz``) i policz metryki przez ``write_federated_results``.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description="Start Flower FedAvg server for NSL-KDD FL.")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--server-address", default="0.0.0.0:8080")
    ap.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Minimum clients per round when config omits num_clients",
    )
    ap.add_argument(
        "--metrics-out",
        default=None,
        help="After FL, evaluate global model on KDDTest+ and write JSON (e.g. results/metrics.json)",
    )
    ap.add_argument(
        "--weights-out",
        default=None,
        help="After FL, save global weights (e.g. results/global_model.npz); used with --metrics-out",
    )
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    num_rounds = int(cfg["federated"]["num_rounds"])
    strategy = build_fedavg_strategy(args.config, num_clients=args.num_clients)

    logger.info(
        "Starting FedAvg server at %s (%s rounds, min %s clients)",
        args.server_address,
        num_rounds,
        strategy.min_fit_clients,
    )
    start_server(
        server_address=args.server_address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if args.metrics_out:
        if strategy.final_parameters is None:
            raise RuntimeError("No aggregated global weights recorded; did federated rounds complete?")
        parameters = parameters_to_ndarrays(strategy.final_parameters)
        weights_path = args.weights_out or "results/global_model.npz"
        result = write_federated_results(
            cfg,
            parameters,
            args.metrics_out,
            weights_path=weights_path,
        )
        logger.info(
            "Global KDDTest+ F1=%.4f (criterion > %.2f: %s)",
            float(result["f1"]),
            float(result["f1_threshold"]),
            result["meets_project_criterion"],
        )


if __name__ == "__main__":
    main()
