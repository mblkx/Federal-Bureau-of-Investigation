"""NSL-KDD loading and preprocessing (fit on train only).

Client sharding (IID / non-IID) lives in src/sharding.py.
"""

from __future__ import annotations
 
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
 
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


#Dataset from original site was unavailable, https://github.com/Jehuty4949/NSL_KDD
#Columns based on NSL-KDD schema
FEATURE_COLUMNS = [ 
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
] 
assert len(FEATURE_COLUMNS) == 41

COLUMNS = FEATURE_COLUMNS + ["label", "difficulty"] #Additional columns to output file

CATEGORICAL = ["protocol_type", "service", "flag"]
#Heavy-tailed features: log1p before scaling, otherwise a few huge values dominate.
LOG_COLUMNS = ["duration", "src_bytes", "dst_bytes"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL + LOG_COLUMNS]

#Attack name -> category (standard NSL-KDD taxonomy, incl. attacks present only in KDDTest+ <<-- might be too much).
ATTACK_CATEGORY = {
    "normal": "normal",
    #DoS
    **dict.fromkeys(
        ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
         "udpstorm", "processtable", "worm", "mailbomb"], "DoS"),
    #Probe
    **dict.fromkeys(
        ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"], "Probe"),
    #R2L
    **dict.fromkeys(
        ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
         "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
         "httptunnel", "sendmail", "named"], "R2L"),
    #U2R
    **dict.fromkeys(
        ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
         "xterm", "ps"], "U2R"),
}
CATEGORIES = ["normal", "DoS", "Probe", "R2L", "U2R"]


"""LOADING AND LABELS"""

def load_nsl_kdd(path: str | Path) -> pd.DataFrame:
    """Read KDDTrain+.txt / KDDTest+.txt and add `attack_category` and `is_attack`.
 
    `is_attack` is the binary target (0 = normal, 1 = any attack).
    `attack_category` (normal/DoS/Probe/R2L/U2R) is kept for non-IID sharding and
    per-category analysis; it is NOT a model input.
    """
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
 
    unknown = sorted(set(df["label"]) - set(ATTACK_CATEGORY))
    if unknown:
        raise ValueError(f"Unknown attack labels (extend ATTACK_CATEGORY): {unknown}")
 
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY)
    df["is_attack"] = (df["attack_category"] != "normal").astype(np.int64)
    return df


"""PREPROCESSOR (fit on train only)"""

def build_preprocessor() -> ColumnTransformer:
    """Unfitted transformer: log1p+scale | scale | one-hot."""
    log_scale = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("log_num", log_scale, LOG_COLUMNS),
            ("num", StandardScaler(), NUMERIC),
            # unseen categories (e.g. a `service` only in test) -> all-zeros row, no crash
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        remainder="drop",  # drops label / difficulty / targets
        verbose_feature_names_out=False,
    )
 
 
def fit_preprocessor(train_df: pd.DataFrame) -> ColumnTransformer:
    """Fit scaler statistics and one-hot vocabulary on TRAIN data only."""
    return build_preprocessor().fit(train_df[FEATURE_COLUMNS])
 
 
def transform(df: pd.DataFrame, pre: ColumnTransformer) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a fitted preprocessor. Returns (X float32, y int64)."""
    X = pre.transform(df[FEATURE_COLUMNS]).astype(np.float32)
    y = df["is_attack"].to_numpy(dtype=np.int64)
    return X, y
 
 
def save_preprocessor(pre: ColumnTransformer, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, path)
 
 
def load_preprocessor(path: str | Path) -> ColumnTransformer:
    return joblib.load(path)


"""E2E"""

@dataclass
class Prepared:
    X_train: np.ndarray
    y_train: np.ndarray
    cat_train: np.ndarray  # attack_category per train row (for non-IID sharding)
    X_test: np.ndarray
    y_test: np.ndarray
    cat_test: np.ndarray
    preprocessor: ColumnTransformer
 
    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]
 
 
def prepare_data(train_path: str | Path, test_path: str | Path) -> Prepared:
    train_df = load_nsl_kdd(train_path)
    test_df = load_nsl_kdd(test_path)
    pre = fit_preprocessor(train_df)
    X_train, y_train = transform(train_df, pre)
    X_test, y_test = transform(test_df, pre)
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise ValueError("NaN after preprocessing")
    return Prepared(
        X_train, y_train, train_df["attack_category"].to_numpy(dtype=str),
        X_test, y_test, test_df["attack_category"].to_numpy(dtype=str), pre,
    )
 
 
def save_prepared(p: Prepared, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "nsl_kdd_processed.npz",
        X_train=p.X_train, y_train=p.y_train, cat_train=p.cat_train,
        X_test=p.X_test, y_test=p.y_test, cat_test=p.cat_test,
    )
    save_preprocessor(p.preprocessor, out / "preprocessor.joblib")
 
 
def load_prepared_arrays(out_dir: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(out_dir) / "nsl_kdd_processed.npz") as f:
        return {k: f[k] for k in f.files}


"""MAIN, CLI: python -m src.preprocess"""

def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess NSL-KDD and save arrays.")
    ap.add_argument("--config", default="src/config.yaml")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
 
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    p = prepare_data(cfg["data"]["train_path"], cfg["data"]["test_path"])
    save_prepared(p, args.out)
 
    print(f"input_dim = {p.input_dim}  (set model.input_dim from this at runtime)")
    print(f"train: {p.X_train.shape}, attack share = {p.y_train.mean():.3f}")
    print(f"test : {p.X_test.shape}, attack share = {p.y_test.mean():.3f}")
    print("train categories:", {str(k): int(v) for k, v in zip(*np.unique(p.cat_train, return_counts=True))})
    print("test  categories:", {str(k): int(v) for k, v in zip(*np.unique(p.cat_test, return_counts=True))})
 
 
if __name__ == "__main__":
    main()