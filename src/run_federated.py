"""Uruchomienie FedAvg w procesie i zapis ``metrics.json`` / wag globalnych."""

from __future__ import annotations

import argparse
import logging

from src.evaluate import format_report
from src.federated_eval import load_global_weights, run_inprocess_fedavg, write_federated_results
from src.server import build_fedavg_strategy
from src.training import load_project_config

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI: trening FedAvg in-process lub ewaluacja wcześniej zapisanych wag.

    Kroki:
    1. Parsuj argumenty (config, ścieżki wag/metryk, opcjonalny ``--weights-in``, próg).
    2. Zweryfikuj, że config da się zmapować na strategię serwera (``build_fedavg_strategy``).
    3. Albo wczytaj wagi z pliku, albo uruchom ``run_inprocess_fedavg``.
    4. Wywołaj ``write_federated_results``, wypisz raport tekstowy i zaloguj ścieżki plików.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Train FedAvg (in-process) or evaluate saved global weights on KDDTest+.",
    )
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument(
        "--metrics-out",
        default="results/metrics.json",
        help="Final test metrics (evaluate.save_results JSON)",
    )
    ap.add_argument(
        "--weights-out",
        default="results/global_model.npz",
        help="Compressed global model weights after training",
    )
    ap.add_argument(
        "--weights-in",
        default=None,
        help="Skip training; evaluate existing global_model.npz",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Attack score threshold for evaluate.evaluate",
    )
    args = ap.parse_args()

    cfg = load_project_config(args.config)
    _ = build_fedavg_strategy(args.config)

    if args.weights_in:
        parameters = load_global_weights(args.weights_in)
        logger.info("Loaded global weights from %s", args.weights_in)
        weights_out = None
    else:
        logger.info(
            "Running in-process FedAvg (%s rounds, %s clients)",
            cfg["federated"]["num_rounds"],
            cfg["num_clients"],
        )
        parameters = run_inprocess_fedavg(args.config)
        weights_out = args.weights_out

    result = write_federated_results(
        cfg,
        parameters,
        args.metrics_out,
        weights_path=weights_out,
        threshold=args.threshold,
    )
    print(format_report(result, title="FedAvg global model (KDDTest+)"))
    logger.info("Wrote metrics to %s", args.metrics_out)
    if weights_out:
        logger.info("Wrote weights to %s", weights_out)


if __name__ == "__main__":
    main()
