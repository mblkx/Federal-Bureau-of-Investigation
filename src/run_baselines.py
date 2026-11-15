"""M2.4: baseline scentralizowany + wyniki lokalne (MLP i autoencoder) → ``results/baselines/``.

Zgodnie z planem kamienia milowego M2 każdy wynik zawiera:
* metryki walidacyjne (holdout 80/20),
* jednorazową ocenę na ``KDDTest+`` przy zamrożonym progu z walidacji.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.evaluate import evaluate as score_predictions
from src.evaluate import format_report, save_results, stratified_holdout
from src.preprocess import load_prepared_arrays
from src.sharding import load_shard
from src.training import (
    ARCHITECTURE_AUTOENCODER,
    ARCHITECTURE_MLP,
    ARCHITECTURES,
    VAL_FRACTION,
    LocalModel,
    load_project_config,
    shard_path,
    split_shard_holdout,
)

# Domyślne partycje M1 / M2.4 (katalogi pod ``data/shards/<preprocessing>/``).
DEFAULT_PARTITIONS = ("iid", "dirichlet_a0.5", "category")


def make_json_safe(obj: Any) -> Any:
    """Konwersja wyników (numpy / NaN) do typów serializowalnych przez ``json``."""
    if isinstance(obj, dict):
        return {str(key): make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, (np.floating, float)):
        number = float(obj)
        return number if math.isfinite(number) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())
    return obj


def list_available_partitions(shards_dir: str | Path) -> list[str]:
    """Zwróć partycje z ``DEFAULT_PARTITIONS``, które istnieją na dysku."""
    root = Path(shards_dir)
    return [name for name in DEFAULT_PARTITIONS if (root / name).is_dir()]


def _cfg_with_architecture(cfg: dict[str, Any], architecture: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("model", {})["architecture"] = architecture
    return out


def _meets_criterion(cfg: dict[str, Any], f1: float) -> bool:
    threshold = float((cfg.get("evaluation") or {}).get("f1_threshold", 0.89))
    return float(f1) > threshold


def evaluate_on_kddtest(
    local: LocalModel,
    processed: dict[str, np.ndarray],
    *,
    threshold: float,
    cfg: dict[str, Any],
) -> dict[str, object]:
    """Jednorazowy pomiar na ``KDDTest+`` przy zamrożonym ``τ`` (bez strojenia)."""
    scores = local.predict_attack_score(processed["X_test"])
    result: dict[str, object] = {
        "split": "KDDTest+",
        **score_predictions(
            processed["y_test"],
            scores,
            processed.get("cat_test"),
            threshold=threshold,
        ),
    }
    f1_threshold = float((cfg.get("evaluation") or {}).get("f1_threshold", 0.89))
    result["f1_threshold"] = f1_threshold
    result["meets_project_criterion"] = _meets_criterion(cfg, float(result["f1"]))
    return result


def run_centralized_baseline(
    cfg: dict[str, Any],
    *,
    architecture: str = ARCHITECTURE_MLP,
) -> dict[str, object]:
    """Baseline scentralizowany: cały ``KDDTrain+`` (80/20) + ocena na ``KDDTest+``.

    Kroki:
    1. Holdout stratyfikowany na pełnym zbiorze treningowym.
    2. Trening MLP lub AE (AE tylko na ruchu normalnym).
    3. Dobór progu na walidacji (AE) / τ=0,5 (MLP).
    4. Metryki walidacyjne, potem jednorazowy pomiar na teście.
    """
    cfg = _cfg_with_architecture(cfg, architecture)
    data_cfg = cfg["data"]
    processed = load_prepared_arrays(data_cfg["processed_dir"])
    seed = int(cfg["random_seed"])

    train_idx, val_idx = stratified_holdout(processed["cat_train"], VAL_FRACTION, seed)
    X_tr = processed["X_train"][train_idx]
    y_tr = processed["y_train"][train_idx]
    X_va = processed["X_train"][val_idx]
    y_va = processed["y_train"][val_idx]
    cat_va = processed["cat_train"][val_idx]

    local = LocalModel.from_config(cfg)
    n_train = local.train_on(X_tr, y_tr, server_round=0)
    if local.is_autoencoder:
        local.fit_threshold(X_va, y_va)

    val_loss, _ = local.evaluate(X_va, y_va, cat_va)
    validation = local.evaluate_full(X_va, y_va, cat_va)
    test = evaluate_on_kddtest(local, processed, threshold=float(local.threshold), cfg=cfg)

    return make_json_safe(
        {
            "model": "centralized",
            "architecture": local.architecture,
            "client_id": None,
            "shard_partition": None,
            "preprocessing": data_cfg.get("preprocessing"),
            "n_train": n_train,
            "n_val": int(len(y_va)),
            "threshold": float(local.threshold),
            "validation_loss": val_loss,
            "validation": validation,
            "test": test,
        }
    )


def run_local_client_baseline(
    cfg: dict[str, Any],
    client_id: int,
    *,
    architecture: str = ARCHITECTURE_MLP,
    shard_partition: str | None = None,
) -> dict[str, object]:
    """Model lokalny jednego klienta: walidacja na 20% shardu + ``KDDTest+``.

    ``KDDTest+`` jest używany wyłącznie do raportu końcowego (próg z walidacji lokalnej).
    """
    cfg = _cfg_with_architecture(cfg, architecture)
    data_cfg = cfg["data"]
    if shard_partition is not None:
        data_cfg["shard_partition"] = shard_partition
    partition = data_cfg["shard_partition"]

    if client_id < 0 or client_id >= int(cfg["num_clients"]):
        raise ValueError(f"client_id must be between 0 and {cfg['num_clients'] - 1}")

    path = shard_path(data_cfg["shards_dir"], partition, client_id)
    shard = load_shard(path)
    if "cat" not in shard:
        raise KeyError(f"shard {path} lacks 'cat'; re-run sharding / build_data")

    X_tr, y_tr, _cat_tr, X_va, y_va, cat_va = split_shard_holdout(
        shard["X"],
        shard["y"],
        shard["cat"],
        int(cfg["random_seed"]),
    )

    local = LocalModel.from_config(cfg)
    n_train = local.train_on(X_tr, y_tr, server_round=0)
    if local.is_autoencoder:
        local.fit_threshold(X_va, y_va)

    val_loss, _ = local.evaluate(X_va, y_va, cat_va)
    validation = local.evaluate_full(X_va, y_va, cat_va)

    processed = load_prepared_arrays(data_cfg["processed_dir"])
    test = evaluate_on_kddtest(local, processed, threshold=float(local.threshold), cfg=cfg)

    return make_json_safe(
        {
            "model": "local",
            "architecture": local.architecture,
            "client_id": int(client_id),
            "shard_partition": partition,
            "shard_path": str(path),
            "preprocessing": data_cfg.get("preprocessing"),
            "n_train": n_train,
            "n_val": int(len(y_va)),
            "threshold": float(local.threshold),
            "validation_loss": val_loss,
            "validation": validation,
            "test": test,
        }
    )


def run_baselines(
    config_path: str | Path = "src/config.yaml",
    *,
    results_dir: str | Path = "results/m2",
    architectures: Sequence[str] = ARCHITECTURES,
    partitions: Sequence[str] | None = None,
    run_centralized: bool = True,
    run_local: bool = True,
) -> dict[str, object]:
    """Uruchom siatkę M2.4 i zapisz JSON-y pod ``results_dir``.

    Zwraca słownik podsumowania (ścieżki plików + skrócone F1).
    """
    cfg = load_project_config(config_path)
    out_root = Path(results_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    arch_list = [str(a).lower() for a in architectures]
    for arch in arch_list:
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown architecture '{arch}', expected {ARCHITECTURES}")

    if partitions is None:
        part_list = list_available_partitions(cfg["data"]["shards_dir"])
        if not part_list:
            part_list = [cfg["data"]["shard_partition"]]
    else:
        part_list = list(partitions)

    summary: dict[str, object] = {
        "results_dir": str(out_root),
        "architectures": arch_list,
        "partitions": part_list,
        "centralized": [],
        "local": [],
    }

    if run_centralized:
        central_dir = out_root / "centralized"
        central_dir.mkdir(parents=True, exist_ok=True)
        for architecture in arch_list:
            result = run_centralized_baseline(cfg, architecture=architecture)
            path = central_dir / f"{architecture}.json"
            save_results(result, path)
            print(format_report(result["validation"], title=f"centralized {architecture} val"))
            print(format_report(result["test"], title=f"centralized {architecture} KDDTest+"))
            print(f"zapisano {path}")
            summary["centralized"].append(
                {
                    "path": str(path),
                    "architecture": architecture,
                    "val_f1": result["validation"]["f1"],
                    "test_f1": result["test"]["f1"],
                    "meets_project_criterion": result["test"]["meets_project_criterion"],
                }
            )

    if run_local:
        local_root = out_root / "local"
        num_clients = int(cfg["num_clients"])
        for partition in part_list:
            part_dir = local_root / partition
            part_dir.mkdir(parents=True, exist_ok=True)
            for architecture in arch_list:
                for client_id in range(num_clients):
                    result = run_local_client_baseline(
                        cfg,
                        client_id,
                        architecture=architecture,
                        shard_partition=partition,
                    )
                    path = part_dir / f"client_{client_id}_{architecture}.json"
                    save_results(result, path)
                    print(
                        format_report(
                            result["validation"],
                            title=f"local {partition} c{client_id} {architecture} val",
                        )
                    )
                    print(
                        format_report(
                            result["test"],
                            title=f"local {partition} c{client_id} {architecture} KDDTest+",
                        )
                    )
                    print(f"zapisano {path}")
                    summary["local"].append(
                        {
                            "path": str(path),
                            "architecture": architecture,
                            "shard_partition": partition,
                            "client_id": client_id,
                            "val_f1": result["validation"]["f1"],
                            "test_f1": result["test"]["f1"],
                            "meets_project_criterion": result["test"]["meets_project_criterion"],
                        }
                    )

    summary_path = out_root / "summary.json"
    save_results(make_json_safe(summary), summary_path)
    print(f"podsumowanie: {summary_path}")
    return make_json_safe(summary)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="M2.4: baseline scentralizowany i lokalny (MLP + AE) → results/baselines/"
    )
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--results-dir", default="results/baselines")
    ap.add_argument(
        "--architectures",
        nargs="+",
        choices=list(ARCHITECTURES),
        default=list(ARCHITECTURES),
        help="Architektury do uruchomienia (domyślnie obie)",
    )
    ap.add_argument(
        "--partitions",
        nargs="+",
        default=None,
        help="Partycje shardów (domyślnie wykryte: iid, dirichlet_a0.5, category)",
    )
    ap.add_argument(
        "--skip-centralized",
        action="store_true",
        help="Pomiń baseline scentralizowany",
    )
    ap.add_argument(
        "--skip-local",
        action="store_true",
        help="Pomiń modele lokalne klientów",
    )
    args = ap.parse_args()
    run_baselines(
        args.config,
        results_dir=args.results_dir,
        architectures=args.architectures,
        partitions=args.partitions,
        run_centralized=not args.skip_centralized,
        run_local=not args.skip_local,
    )


if __name__ == "__main__":
    main()
