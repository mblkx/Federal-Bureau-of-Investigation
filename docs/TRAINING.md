# Trening lokalny i klient Flower (M2)

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

## Krok 3 — klient federacyjny (M3, gdy `server.py` jest gotowy)

Trzy terminale klientów + jeden serwer:

```bash
python -m src.client --cid 0 --server-address localhost:8080
python -m src.client --cid 1 --server-address localhost:8080
python -m src.client --cid 2 --server-address localhost:8080
```

Klient w `fit` wysyła **wyłącznie wagi** modelu (`get_parameters`), nie macierz cech `X`.

## Moduły

| Plik | Rola |
|------|------|
| `src/training.py` | `LocalModel` — pętla SGD, metryki sklearn, serializacja wag |
| `src/client.py` | `NSLKDDClient` (Flower `NumPyClient`) + CLI |
