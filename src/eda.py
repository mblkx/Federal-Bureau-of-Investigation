"""Exploratory plots for NSL-KDD (read-only, saves PNG files).

CLI: python -m src.eda  ->  results/eda/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.preprocess import CATEGORIES, load_nsl_kdd


def plot_category_distribution(train: pd.DataFrame, test: pd.DataFrame, out: Path) -> Path:
    """Share of each attack category in train vs test (distribution shift)."""
    tr = train["attack_category"].value_counts(normalize=True).reindex(CATEGORIES, fill_value=0) * 100
    te = test["attack_category"].value_counts(normalize=True).reindex(CATEGORIES, fill_value=0) * 100
    x = np.arange(len(CATEGORIES))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, tr.values, 0.4, label="KDDTrain+")
    ax.bar(x + 0.2, te.values, 0.4, label="KDDTest+")
    ax.set_xticks(x, CATEGORIES)
    ax.set_ylabel("share of records [%]")
    ax.set_title("Attack category distribution: train vs test")
    ax.legend()
    fig.tight_layout()
    path = out / "category_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_protocol_by_category(train: pd.DataFrame, out: Path) -> Path:
    """protocol_type counts per attack category (train)."""
    ct = pd.crosstab(train["attack_category"], train["protocol_type"]).reindex(CATEGORIES, fill_value=0)
    ax = ct.plot(kind="bar", stacked=True, figsize=(7, 4), rot=0)
    ax.set_ylabel("records")
    ax.set_xlabel("")
    ax.set_title("protocol_type by attack category (KDDTrain+)")
    fig = ax.get_figure()
    fig.tight_layout()
    path = out / "protocol_by_category.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_src_bytes(train: pd.DataFrame, out: Path) -> Path:
    """log1p(src_bytes) for normal traffic vs attacks (train)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for flag, name in ((0, "normal"), (1, "attack")):
        vals = np.log1p(train.loc[train["is_attack"] == flag, "src_bytes"])
        ax.hist(vals, bins=50, alpha=0.6, density=True, label=name)
    ax.set_xlabel("log1p(src_bytes)")
    ax.set_ylabel("density")
    ax.set_title("src_bytes: normal vs attack (KDDTrain+)")
    ax.legend()
    fig.tight_layout()
    path = out / "src_bytes_log1p.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_plots(train: pd.DataFrame, test: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        plot_category_distribution(train, test, out),
        plot_protocol_by_category(train, out),
        plot_src_bytes(train, out),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="NSL-KDD exploratory plots.")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--out", default="results/eda")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train = load_nsl_kdd(cfg["data"]["train_path"])
    test = load_nsl_kdd(cfg["data"]["test_path"])
    for p in make_plots(train, test, args.out):
        print("saved", p)


if __name__ == "__main__":
    main()