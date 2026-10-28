# Protokół prywatności (privacy smoke / T4)

Uzupełnienie threat modelu z [PLAN_REALIZACJI.md](PLAN_REALIZACJI.md): serwer koordynujący jest zaufany co do **protokołu**, ale **nie powinien** otrzymywać surowych rekordów treningowych klientów.

## Co trafia do serwera (klient → serwer)

| Element | Zawartość | Uwagi |
|---------|-----------|--------|
| `fit` — parametry | Wyłącznie wagi sieci (`LocalModel.get_parameters()`) | Liczba tensorów i kształty jak w modelu globalnym |
| `fit` — `num_examples` | Liczba próbek na shardzie (skalar) | Metadana do weighted FedAvg, nie wektor cech |
| `fit` — metryki | `{}` (puste) | Brak surowych danych w słowniku metryk |
| `evaluate` | Loss + metryki sklearn (`f1`, …) | Na globalnym teście; nadal **bez** macierzy `X` |

Implementacja: [src/client.py](../src/client.py) (`NSLKDDClient.fit`).

## Czego operator serwera **nie** widzi

- Macierzy cech `X` shardów (`data/shards/.../client_*.npz`).
- Etykiet treningowych `y` jako osobnego payloadu (pozostają lokalnie).
- Batchy wejściowe w rozmiarze `(n_samples, n_features)` w kanale FL.

Dane lokalne są ładowane tylko po stronie klienta (`build_client`); serwer agreguje wyłącznie parametry modelu (Flower `FedAvg`).

## Testy regresji (automatyczne)

- [tests/test_client.py](../tests/test_client.py) — kontrakt `fit` na syntetycznym kliencie.
- [tests/test_privacy.py](../tests/test_privacy.py) — `build_client` + serializacja Flower (`ndarrays_to_parameters` / `parameters_to_ndarrays`).
- Walidacja kształtów: [src/privacy.py](../src/privacy.py) (`assert_fit_payload_is_model_only`).

Uruchomienie:

```bash
pytest tests/test_client.py tests/test_privacy.py
```

## Ograniczenia (świadome)

- Testy **nie** chronią przed odwróceniem gradientu ani atakami na FL — zgodnie z zakresem projektu.
- Nie weryfikują szyfrowania transportu gRPC (Flower domyślnie bez TLS w lokalnym setupie).

## Sekcja do raportu M5 (skrót)

**Operator serwera** ma wgląd w metadane rund Flower, zagregowane wagi globalne i zagregowane metryki ewaluacji (np. F1 z `evaluate`). **Nie** ma dostępu do surowych logów ruchu / wektorów cech klientów — w projekcie symulowanych shardów NSL-KDD. Prywatność tajemnicy przedsiębiorstwa jest spełniona na poziomie **braku przesyłu raw data** w zaimplementowanym protokole klient–serwer.
