"""Data quality checks for raw NSL-KDD files (report only, nothing is modified).

Does not import torch, so it runs without PyTorch and in CI.

CLI:
    python -m src.data_checks                    # write results/data_report.json
    python -m src.data_checks --write-checksums  # write docs/data_checksums.json
    python -m src.data_checks --verify           # compare files with stored checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.preprocess import CATEGORICAL, FEATURE_COLUMNS, load_nsl_kdd

DEFAULT_CONFIG = "src/config.yaml"
DEFAULT_REPORT = "results/data_report.json"
DEFAULT_CHECKSUMS = "docs/data_checksums.json"


"""CHECKSUMS"""

def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_facts(path: str | Path) -> dict:
    """SHA-256, size in bytes and number of lines of one data file."""
    p = Path(path)
    with open(p, "rb") as f:
        lines = sum(1 for _ in f)
    return {"sha256": sha256_file(p), "bytes": p.stat().st_size, "lines": lines}


def write_checksums(paths: list[str | Path], out: str | Path) -> dict:
    facts = {Path(p).name: file_facts(p) for p in paths}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8", newline="\n")
    return facts


def verify_checksums(paths: list[str | Path], checksums: str | Path) -> dict[str, bool]:
    """Compare files with stored checksums. Missing file or missing entry -> False."""
    expected = json.loads(Path(checksums).read_text(encoding="utf-8"))
    result = {}
    for p in paths:
        p = Path(p)
        entry = expected.get(p.name)
        result[p.name] = bool(entry) and p.exists() and sha256_file(p) == entry["sha256"]
    return result


"""QUALITY CHECKS (each takes a DataFrame from load_nsl_kdd)"""

def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    """Columns with at least one missing value (empty dict = none)."""
    counts = df[FEATURE_COLUMNS + ["label"]].isna().sum()
    return {str(k): int(v) for k, v in counts[counts > 0].items()}


def constant_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns with a single distinct value."""
    return [c for c in FEATURE_COLUMNS if df[c].nunique(dropna=False) <= 1]


def duplicate_summary(df: pd.DataFrame) -> dict:
    """Exact duplicates (41 features + attack name) and label conflicts.

    A conflict is a group of rows with identical features but different labels.
    Only groups that mix normal traffic with an attack contradict the binary
    task (`mixed_normal_attack_*`).
    """
    exact = int(df.duplicated(FEATURE_COLUMNS + ["label"]).sum())
    g = df.groupby(FEATURE_COLUMNS, sort=False, dropna=False)
    n_labels = g["label"].transform("nunique")
    n_binary = g["is_attack"].transform("nunique")
    conflict = n_labels > 1
    mixed = n_binary > 1
    groups = df.loc[conflict, FEATURE_COLUMNS].drop_duplicates().shape[0]
    mixed_groups = df.loc[mixed, FEATURE_COLUMNS].drop_duplicates().shape[0]
    return {
        "exact_duplicates": exact,
        "label_conflict_groups": int(groups),
        "label_conflict_rows": int(conflict.sum()),
        "mixed_normal_attack_groups": int(mixed_groups),
        "mixed_normal_attack_rows": int(mixed.sum()),
        "mixed_attack_names": sorted(df.loc[mixed & (df["is_attack"] == 1), "label"].unique().tolist()),
    }


def overlap_train_test(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Test rows that also occur in train.

    `rows` / `share_of_test`: record = 41 features + attack name (same label).
    `rows_same_features`: only the 41 features match (label may differ).
    """
    def _count(cols: list[str]) -> int:
        merged = test[cols].merge(train[cols].drop_duplicates(), on=cols, how="left", indicator=True)
        return int((merged["_merge"] == "both").sum())

    n = _count(FEATURE_COLUMNS + ["label"])
    return {
        "rows": n,
        "share_of_test": round(n / len(test), 4) if len(test) else 0.0,
        "rows_same_features": _count(FEATURE_COLUMNS),
    }


def unseen_categories(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, list[str]]:
    """Categorical values present in test but not in train."""
    out = {}
    for c in CATEGORICAL:
        extra = sorted(set(test[c].unique()) - set(train[c].unique()))
        if extra:
            out[c] = extra
    return out


def _describe(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "attack_share": round(float(df["is_attack"].mean()), 4),
        "categories": {str(k): int(v) for k, v in df["attack_category"].value_counts().items()},
        "missing": missing_counts(df),
        "constant_columns": constant_columns(df),
        "duplicates": duplicate_summary(df),
    }


def build_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "train": _describe(train),
        "test": _describe(test),
        "categorical_cardinality": {c: int(train[c].nunique()) for c in CATEGORICAL},
        "train_test_overlap": overlap_train_test(train, test),
        "unseen_categories_in_test": unseen_categories(train, test),
    }


"""MAIN, CLI: python -m src.data_checks"""

def main() -> None:
    ap = argparse.ArgumentParser(description="NSL-KDD data quality report and checksums.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--out", default=DEFAULT_REPORT)
    ap.add_argument("--checksums", default=DEFAULT_CHECKSUMS)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--write-checksums", action="store_true")
    grp.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = [cfg["data"]["train_path"], cfg["data"]["test_path"]]

    if args.write_checksums:
        for name, f in write_checksums(paths, args.checksums).items():
            print(f"{name}: {f['sha256']}")
        return

    if args.verify:
        res = verify_checksums(paths, args.checksums)
        for name, ok in res.items():
            print(f"{name}: {'OK' if ok else 'MISMATCH'}")
        sys.exit(0 if all(res.values()) else 1)

    train, test = load_nsl_kdd(paths[0]), load_nsl_kdd(paths[1])
    report = build_report(train, test)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()