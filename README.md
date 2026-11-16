# Federal-Bureau-of-Investigation

Projekt z Zaawansowanych Technik Kryptografii i Kryptoanalizy oraz Metod Sztucznej Inteligencji w Cyberbezpieczeństwie — rozproszona detekcja anomalii (FedAvg) na NSL-KDD.

Plan realizacji: [docs/PLAN_REALIZACJI.md](docs/PLAN_REALIZACJI.md).  
Protokół metryk: [docs/evaluation_protocol.md](docs/evaluation_protocol.md).  
Trening / FedAvg (szczegóły): [docs/TRAINING.md](docs/TRAINING.md).

Oryginalny dataset nie jest już dostępny: https://www.unb.ca/cic/datasets/nsl.html  
Licencja NSL-KDD wprost pozwala na mirrorowanie. W projekcie wykorzystano mirror: https://github.com/Jehuty4949/NSL_KDD

## Co zrobiono dotychczas

**Dane (M1)** — Preprocessor zamienia surowe pliki NSL-KDD w macierze liczb (one-hot, `log1p`, standaryzacja dopasowana wyłącznie do zbioru treningowego) oraz binarną etykietę atak/normal i kategorię ataku.

**Sharding (M1)** — Trening dzielony na 3 rozłączne shardy (IID, Dirichlet, category), tak aby każdy klient miał ruch normalny i ataki.

**Kontrola danych (M1)** — `python -m src.data_checks` → `results/data_report.json`; `python -m src.eda` → `results/eda/`. Sumy SHA-256: `docs/data_checksums.json` (`python -m src.data_checks --verify`).

**Ekstrakcja cech (M1)** — Metoda nr 2 (`src/features.py`): filtr stałych/skorelowanych + PCA lub selekcja. `input_dim` z `meta.json` (`src/data_config.py`). Jedno polecenie `python -m src.build_data` buduje `full` i `extracted` oraz shardy IID / Dirichlet / category.

**Ewaluacja** — `src/evaluate.py`: F1, precision, recall, accuracy, ROC-AUC, macierz pomyłek, metryki per kategoria. `KDDTest+` tylko do końcowego pomiaru.

**Trening lokalny (M2.1–M2.3)** — `LocalModel` (MLP lub autoencoder): holdout 80/20 na shardzie, ziarno zależne od rundy, reset Adama, AE tylko na ruchu normalnym z progiem z walidacji. Klient Flower **nie ładuje** `KDDTest+`.

**Baseline scentralizowany i lokalny (M2.4)** — `python -m src.run_baselines` zapisuje metryki walidacyjne i `KDDTest+` do `results/baselines/` (centralized + 3 klientów × partycje × MLP/AE).

**Serwer i federacja (M3, szkielet)** — Flower FedAvg (`src/server.py`) lub in-process (`src/run_federated.py`) → `results/metrics.json`.

**Testy** — `pytest`, `pytest -m integration`, privacy smoke: [docs/privacy_protocol.md](docs/privacy_protocol.md).

## Szybki start (surowy dataset → wyniki)

Wymagane pliki: `data/KDDTrain+.txt`, `data/KDDTest+.txt` (mirror NSL-KDD).

```bash
pip install -r requirements.txt

# Opcjonalnie: weryfikacja sum SHA-256
python -m src.data_checks --verify

# M1: obie metody obróbki + 3 partycje shardów
# (ścieżki: data/processed/<full|extracted>/, data/shards/<...>/)
python -m src.build_data --config src/config.yaml

# Testy jednostkowe (bez wolnej integracji FL)
pytest

# M2.4: baseline scentralizowany + lokalny (MLP i AE) → results/baselines/
python -m src.run_baselines

# Pojedynczy klient (tylko walidacja lokalna 80/20)
python -m src.training --cid 0
python -m src.training --cid 0 --architecture autoencoder

# M3 (szkielet): FedAvg in-process + KDDTest+
python -m src.run_federated \
  --metrics-out results/metrics.json \
  --weights-out results/global_model.npz

# Albo: serwer + 3 klienty
# python -m src.server
# python -m src.client --cid 0|1|2 --server-address localhost:8080
```

Konfiguracja: `src/config.yaml` (`data.preprocessing`, `model.architecture`, `shard_partition`, epoki, rundy, …).
