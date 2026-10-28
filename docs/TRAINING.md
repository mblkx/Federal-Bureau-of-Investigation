# Trening lokalny, serwer FedAvg i ewaluacja (M2–M4)

Szczegóły protokołu metryk: [evaluation_protocol.md](evaluation_protocol.md).  
Privacy smoke (T4): [privacy_protocol.md](privacy_protocol.md).

## Wymagane artefakty

1. `data/processed/nsl_kdd_processed.npz` — `python -m src.preprocess`
2. `data/shards/<partition>/client_*.npz` — `python -m src.sharding --partition iid`
3. W `src/config.yaml`: `model.input_dim` (np. 122) oraz `data.shard_partition` zgodny z katalogiem shardów.

## Krok 1 — baseline lokalny (bez serwera)

Jeden klient trenuje wyłącznie na swoim shardzie, ewaluacja na **globalnym** teście NSL-KDD:

```bash
python -m src.training --cid 0
python -m src.training --cid 1
python -m src.training --cid 2
```

## Krok 2 — ten sam baseline przez `client.py`

```bash
python -m src.client --cid 0 --local-only
```

## Krok 3 — federacja (M3): serwer + 3 klienty

### Uruchomienie

**1. Serwer** (jeden terminal, przed klientami):

```bash
python -m src.server
```

Domyślnie: adres `0.0.0.0:8080`, liczba rund z `federated.num_rounds` w configu, `min_*_clients` = `num_clients` (domyślnie 3).

Przykład z jawnym adresem i zapisem wyniku po rundach:

```bash
python -m src.server \
  --server-address 0.0.0.0:8080 \
  --config src/config.yaml \
  --num-clients 3 \
  --metrics-out results/metrics.json \
  --weights-out results/global_model.npz
```

**2. Klienci** (trzy osobne terminale, ten sam `--config` i adres serwera):

```bash
python -m src.client --cid 0 --server-address localhost:8080
python -m src.client --cid 1 --server-address localhost:8080
python -m src.client --cid 2 --server-address localhost:8080
```

Klient w `fit` wysyła **wyłącznie wagi** modelu (`get_parameters`), nie macierz cech `X`.

Po każdej rundzie serwer loguje zagregowane metryki ewaluacji (ważone F1, accuracy, …) na poziomie `INFO`.

### Typowe problemy

| Objaw | Przyczyna | Co zrobić |
|--------|-----------|-----------|
| Klient: błąd połączenia / timeout | Serwer nie działa lub zły adres | Najpierw `python -m src.server`, potem klienty; sprawdź `--server-address` (serwer `0.0.0.0:8080`, klient `localhost:8080`) |
| Serwer „wisi”, rundy się nie zaczynają | Za mało podłączonych klientów | Uruchom **wszystkie** `num_clients` (domyślnie 3). Strategia wymaga `min_available_clients` = `min_fit_clients` = 3 |
| `Address already in use` | Port 8080 zajęty | Zmień port, np. serwer `--server-address 0.0.0.0:9090` i klienci `--server-address localhost:9090` |
| `model.input_dim` / brak plików shardów | Brak preprocessingu lub zła partycja | `python -m src.preprocess`, `python -m src.sharding`, zgodność `data.shard_partition` z katalogiem |
| Po FL brak `metrics.json` | Serwer bez flag zapisu | Użyj `--metrics-out` (i opcjonalnie `--weights-out`) albo [Krok 4](#krok-4--trening--metryka-docelowa-na-kddtest-krok-6-planu-opcja-b) |

## Krok 4 — trening + metryka docelowa na `KDDTest+` (krok 6 planu, opcja B)

Bez ręcznego uruchamiania serwera (ta sama agregacja wag co Flower, in-process):

```bash
python -m src.run_federated --metrics-out results/metrics.json --weights-out results/global_model.npz
```

Ewaluacja zapisanych wag:

```bash
python -m src.run_federated --weights-in results/global_model.npz --metrics-out results/metrics.json
```

Wynik używa `src/evaluate.py` (próg 0,5, klasa atak) i zawiera pole `meets_project_criterion` względem `evaluation.f1_threshold` w configu.

## Testy

```bash
pytest                          # bez testów integracyjnych (domyślnie)
pytest -m integration           # serwer + 3 klienty (wolne)
pytest tests/test_privacy.py    # privacy smoke (T4)
```

Konfiguracja markerów: `pyproject.toml` (`addopts = -m "not integration"`).

## Moduły

| Plik | Rola |
|------|------|
| `src/training.py` | `LocalModel` — pętla SGD, metryki sklearn, serializacja wag |
| `src/client.py` | `NSLKDDClient` (Flower `NumPyClient`) + CLI |
| `src/server.py` | Flower `FedAvg` + `start_server`, logi metryk, opcjonalny zapis wag/metryk |
| `src/run_federated.py` | FedAvg in-process → `results/metrics.json` + `global_model.npz` |
| `src/federated_eval.py` | Ewaluacja globalna na `KDDTest+`, I/O wag |
| `src/privacy.py` | Asercje kształtu payloadu `fit` (brak raw `X`) |
