"""Data check tests on synthetic data (no NSL-KDD download needed, CI-friendly)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data_checks import (
    build_report, constant_columns, duplicate_summary, file_facts, main,
    missing_counts, overlap_train_test, sha256_file, unseen_categories,
    verify_checksums, write_checksums,
)
from src.preprocess import ATTACK_CATEGORY, COLUMNS, FEATURE_COLUMNS, load_nsl_kdd


def _row(label="normal", **over):
    row = {c: 0 for c in FEATURE_COLUMNS}
    row.update(protocol_type="tcp", service="http", flag="SF", label=label, difficulty=10)
    row.update(over)
    return row


def _df(rows):
    df = pd.DataFrame(rows)
    df["attack_category"] = df["label"].map(ATTACK_CATEGORY)
    df["is_attack"] = (df["attack_category"] != "normal").astype(int)
    return df


# --- checksums ---

def test_sha256_known_value(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    assert sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_file_facts(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"x\ny\n")
    facts = file_facts(f)
    assert facts["bytes"] == 4 and facts["lines"] == 2


def test_checksums_roundtrip_and_mismatch(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    cs = tmp_path / "cs.json"
    write_checksums([f], cs)
    assert verify_checksums([f], cs) == {"a.txt": True}
    f.write_text("hello!")
    assert verify_checksums([f], cs) == {"a.txt": False}


def test_verify_missing_entry_or_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    cs = tmp_path / "cs.json"
    cs.write_text("{}")
    assert verify_checksums([f], cs) == {"a.txt": False}
    write_checksums([f], cs)
    f.unlink()
    assert verify_checksums([f], cs) == {"a.txt": False}


# --- quality checks ---

def test_missing_counts():
    df = _df([_row(), _row()])
    assert missing_counts(df) == {}
    df.loc[0, "duration"] = np.nan
    assert missing_counts(df) == {"duration": 1}


def test_constant_columns():
    df = _df([_row(duration=1), _row(duration=2)])
    const = constant_columns(df)
    assert "duration" not in const
    assert "num_outbound_cmds" in const


def test_exact_duplicates():
    df = _df([_row(), _row(), _row(duration=5)])
    assert duplicate_summary(df)["exact_duplicates"] == 1


def test_label_conflict_without_mixing():
    # same features, two different attacks: conflict, but not normal vs attack
    df = _df([_row("neptune"), _row("smurf"), _row("normal", duration=9)])
    s = duplicate_summary(df)
    assert s["label_conflict_groups"] == 1 and s["label_conflict_rows"] == 2
    assert s["mixed_normal_attack_groups"] == 0 and s["mixed_normal_attack_rows"] == 0


def test_mixed_normal_attack():
    df = _df([_row("normal"), _row("snmpgetattack"), _row("normal", duration=9)])
    s = duplicate_summary(df)
    assert s["mixed_normal_attack_groups"] == 1
    assert s["mixed_normal_attack_rows"] == 2
    assert s["mixed_attack_names"] == ["snmpgetattack"]


def test_overlap_requires_same_label_but_reports_features():
    train = _df([_row("normal"), _row("neptune", duration=1)])
    test = _df([_row("normal"), _row("smurf", duration=1), _row("normal", duration=7)])
    o = overlap_train_test(train, test)
    assert o["rows"] == 1
    assert o["rows_same_features"] == 2
    assert o["share_of_test"] == round(1 / 3, 4)


def test_unseen_categories():
    train = _df([_row(service="http")])
    test = _df([_row(service="http"), _row(service="ftp"), _row(flag="REJ")])
    assert unseen_categories(train, test) == {"service": ["ftp"], "flag": ["REJ"]}
    assert unseen_categories(train, train) == {}


def test_build_report_is_json_serializable():
    train = _df([_row("normal"), _row("neptune", duration=1)])
    test = _df([_row("normal")])
    report = build_report(train, test)
    json.dumps(report)
    assert report["train"]["rows"] == 2 and report["test"]["rows"] == 1
    assert report["categorical_cardinality"]["protocol_type"] == 1


# --- CLI ---

def test_cli_report_does_not_modify_data(tmp_path, monkeypatch, capsys):
    train, test = tmp_path / "train.txt", tmp_path / "test.txt"
    rows = [_row("normal"), _row("neptune", duration=1)]
    pd.DataFrame(rows)[COLUMNS].to_csv(train, header=False, index=False)
    pd.DataFrame(rows[:1])[COLUMNS].to_csv(test, header=False, index=False)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"data:\n  train_path: {train.as_posix()}\n  test_path: {test.as_posix()}\n")
    before = train.read_bytes()
    out = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["data_checks", "--config", str(cfg), "--out", str(out)])
    main()
    assert json.loads(out.read_text())["train"]["rows"] == 2
    assert train.read_bytes() == before


def test_cli_verify_exit_code(tmp_path, monkeypatch):
    train, test = tmp_path / "train.txt", tmp_path / "test.txt"
    train.write_text("a")
    test.write_text("b")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"data:\n  train_path: {train.as_posix()}\n  test_path: {test.as_posix()}\n")
    cs = tmp_path / "cs.json"
    base = ["data_checks", "--config", str(cfg), "--checksums", str(cs)]
    monkeypatch.setattr("sys.argv", base + ["--write-checksums"])
    main()
    monkeypatch.setattr("sys.argv", base + ["--verify"])
    main_ok = None
    try:
        main()
    except SystemExit as e:
        main_ok = e.code
    assert main_ok == 0
    test.write_text("changed")
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1