# Federal-Bureau-of-Investigation

Projekt z Zaawansowanych Technik Kryptografii i Kryptoanalizy — rozproszona detekcja anomalii (FedAvg) na NSL-KDD.

Plan realizacji: [docs/PLAN_REALIZACJI.md](docs/PLAN_REALIZACJI.md).

Oryginalny dataset nie jest już dostępny: https://www.unb.ca/cic/datasets/nsl.html  
Licencja NSL-KDD wprost pozwala na mirrorowanie. W projekcie wykorzystano mirror: https://github.com/Jehuty4949/NSL_KDD

## Co zrobiono dotychczas

**Dane (M1)** — Preprocessor zamienia surowe pliki NSL-KDD w macierze liczb o jednakowym formacie (one-hot, log1p, standaryzacja dopasowana wyłącznie do zbioru treningowego) i tworzy binarną etykietę atak/normal oraz kategorię ataku.

**Sharding (M1)** — Zbiór treningowy dzielony jest na 3 rozłączne shardy (IID, Dirichlet lub według kategorii ataku), tak aby każdy klient miał ruch normalny i ataki.

**Kontrola danych (M1)** — `python -m src.data_checks` zapisuje raport jakości `results/data_report.json` (liczba rekordów i kolumn, braki, duplikaty, cechy stałe, rekordy testu obecne w treningu), a `python -m src.eda` wykresy do `results/eda/`. Sumy SHA-256 plików danych są w `docs/data_checksums.json` (`python -m src.data_checks --verify` sprawdza zgodność z mirrorem). Wynik: brak braków i duplikatów dokładnych, jedna cecha stała (`num_outbound_cmds`), 610 rekordów testu (2,7%) występuje także w treningu.

**Ewaluacja** — Moduł `src/evaluate.py` liczy F1, precision, recall, accuracy, ROC-AUC, macierz pomyłek i skuteczność wykrycia per kategoria. Protokół: [docs/evaluation_protocol.md](docs/evaluation_protocol.md) (`KDDTest+` tylko do końcowego pomiaru F1 > 0,89).

**Trening lokalny (M2)** — `src/training.py` (`LocalModel`) i klient Flower `src/client.py` (baseline na shardzie, `--local-only`).

**Serwer i federacja (M3)** — `src/server.py`: Flower weighted FedAvg, `num_rounds` z configu, logi metryk po rundzie. Uruchomienie serwera + 3 klientów lub skrypt `src/run_federated.py` (in-process, ten sam FedAvg). Po treningu: ewaluacja globalnego modelu na `KDDTest+` → `results/metrics.json` (pole `meets_project_criterion`). Szczegóły i troubleshooting: [docs/TRAINING.md](docs/TRAINING.md).

**Testy** — Jednostkowe (`pytest`), integracja FL (`pytest -m integration`), privacy smoke (T4): [docs/privacy_protocol.md](docs/privacy_protocol.md).

## Szybki start

```bash
pip install -r requirements.txt
python -m src.preprocess
python -m src.sharding --partition iid

# Baseline lokalny (M2)
python -m src.training --cid 0

# FedAvg + metryka na KDDTest+ (bez 4 terminali)
python -m src.run_federated

# Lub: serwer + 3 klienty (M3)
python -m src.server
# w osobnych terminalach: python -m src.client --cid 0|1|2 --server-address localhost:8080
```
