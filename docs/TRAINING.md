# Trening lokalny, baseline M2.4, serwer FedAvg i ewaluacja (M2–M4)

Szczegóły protokołu metryk: [evaluation_protocol.md](evaluation_protocol.md).  
Privacy smoke (T4): [privacy_protocol.md](privacy_protocol.md).  
Plan etapów: [PLAN_REALIZACJI.md](PLAN_REALIZACJI.md).

## Przepływ end-to-end

```
data/KDDTrain+.txt + data/KDDTest+.txt
        |
        v
python -m src.build_data
        |
        +--> data/processed/<full|extracted>/   (+ meta.json z input_dim)
        +--> data/shards/<...>/<partition>/client_*.npz
        |
        +--> pytest
        +--> python -m src.run_baselines     -> results/baselines/
        +--> python -m src.run_federated     -> results/metrics.json
```

## Wymagane artefakty (po `build_data`)

Przy `data.preprocessing: full` w `src/config.yaml`:

1. `data/processed/full/nsl_kdd_processed.npz` (+ `meta.json`, `preprocessor.joblib`)
2. `data/shards/full/<partition>/client_*.npz` — partycje: `iid`, `dirichlet_a0.5`, `category`
3. `model.input_dim` może zostać `null` — jest uzupełniane z `meta.json` przez `resolve_data_config`

Alternatywa (legacy, bez podkatalogu metody): `python -m src.preprocess` i `python -m src.sharding --partition iid`, ale wtedy usuń lub dostosuj `data.preprocessing` w configu.

Opcjonalnie:

```bash
python -m src.data_checks --verify
python -m src.data_checks
python -m src.eda
```

## Protokół lokalny (M2.1)

- Trening: **80%** shardu (stratyfikacja po kategorii).
- Walidacja: **20%** shardu — strojenie / metryki lokalne.
- Klient Flower **nie ładuje** `KDDTest+`.
- `KDDTest+` tylko do końcowego raportu (baseline M2.4, serwer / `run_federated`).

MLP: próg `τ = 0,5`. Autoencoder: `τ` z walidacji (max F1), trening tylko na ruchu normalnym (`y==0`).

## Krok 1 — jeden klient (szybki check)

```bash
python -m src.training --cid 0
python -m src.training --cid 0 --architecture autoencoder
python -m src.client --cid 0 --local-only
```

Wypisuje metryki **walidacji lokalnej** (nie `KDDTest+`).

## Krok 2 — pełna siatka baseline M2.4

Centralized (cały train) + lokalne (3 klientów × partycje × MLP/AE), walidacja + zamrożony próg + `KDDTest+`:

```bash
python -m src.run_baselines
# domyślny katalog: results/baselines/
```

Podzbiory:

```bash
python -m src.run_baselines --architectures mlp --partitions iid
python -m src.run_baselines --skip-local
python -m src.run_baselines --skip-centralized --partitions category
```

Układ plików:

```
results/baselines/
  centralized/{mlp,autoencoder}.json
  local/{iid,dirichlet_a0.5,category}/client_{0,1,2}_{mlp,autoencoder}.json
  summary.json
```

Każdy JSON ma sekcje `validation` i `test` (w tym `meets_project_criterion`).

## Krok 3 — federacja (M3): serwer + 3 klienty

**1. Serwer** (przed klientami):

```bash
python -m src.server
```

Domyślnie: `0.0.0.0:8080`, `num_rounds` z configu, `min_*_clients` = `num_clients` (3). Serwer przekazuje `server_round` do klientów (ziarno M2.2).

```bash
python -m src.server \
  --server-address 0.0.0.0:8080 \
  --config src/config.yaml \
  --num-clients 3 \
  --metrics-out results/metrics.json \
  --weights-out results/global_model.npz
```

**2. Klienci** (trzy terminale):

```bash
python -m src.client --cid 0 --server-address localhost:8080
python -m src.client --cid 1 --server-address localhost:8080
python -m src.client --cid 2 --server-address localhost:8080
```

Klient w `fit` wysyła **tylko wagi** i liczbę próbek, nie macierz `X`.  
`evaluate` na kliencie dotyczy **lokalnej walidacji** (20% shardu).

### Typowe problemy

| Objaw | Przyczyna | Co zrobić |
|--------|-----------|-----------|
| Klient: błąd połączenia / timeout | Serwer nie działa lub zły adres | Najpierw `src.server`, potem klienty; serwer `0.0.0.0:8080`, klient `localhost:8080` |
| Serwer „wisi” | Za mało klientów | Uruchom wszystkie `num_clients` (domyślnie 3) |
| `Address already in use` | Port 8080 zajęty | Inny port po obu stronach |
| Brak `data/shards/full/...` | Stary layout lub brak `build_data` | `python -m src.build_data`; sprawdź `data.preprocessing` |
| `model.input_dim` unset | Brak `meta.json` | Uruchom `build_data` / preprocess |
| Po FL brak `metrics.json` | Brak flag zapisu | `--metrics-out` / `--weights-out` albo Krok 4 |

## Krok 4 — FedAvg in-process

```bash
python -m src.run_federated \
  --metrics-out results/metrics.json \
  --weights-out results/global_model.npz
```

Ewaluacja zapisanych wag:

```bash
python -m src.run_federated \
  --weights-in results/global_model.npz \
  --metrics-out results/metrics.json
```

Dla MLP domyślny próg to 0,5; dla AE bez `--threshold` próg jest dobierany na holdoucie **treningu** (nie na `KDDTest+`). Wynik zawiera `meets_project_criterion` względem `evaluation.f1_threshold`.

## Testy

```bash
pytest                          # bez @integration (domyślnie)
pytest -m integration           # serwer + 3 klienty (wolne)
pytest tests/test_privacy.py    # privacy smoke (T4)
pytest tests/test_run_baselines.py
```

Markery: `pyproject.toml` (`addopts = -m "not integration"`).

## Moduły

| Plik | Rola |
|------|------|
| `src/build_data.py` | Jedno polecenie: preprocess `full`/`extracted` + 3 partycje |
| `src/data_config.py` | Ścieżki metody + dynamiczny `input_dim` |
| `src/training.py` | `LocalModel` (MLP/AE), seed rundy, Adam reset, holdout 80/20 |
| `src/model.py` | `MLPClassifier`, `Autoencoder` |
| `src/run_baselines.py` | Siatka M2.4 → `results/baselines/` |
| `src/client.py` | Flower `NSLKDDClient` + `--local-only` |
| `src/server.py` | Flower FedAvg, `on_fit_config_fn` (`server_round`) |
| `src/run_federated.py` | FedAvg in-process → metryki / wagi |
| `src/federated_eval.py` | Ocena globalna na `KDDTest+`, I/O wag |
| `src/fedavg.py` | Autorski weighted FedAvg |
| `src/privacy.py` | Asercje payloadu `fit` (brak surowego `X`) |
