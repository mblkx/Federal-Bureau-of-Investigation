# Protokół ewaluacji

Dokument ustalany **przed** pierwszymi wynikami. Zmiana zasad po zobaczeniu
wyników wymaga wpisu w sekcji „Zmiany protokołu” na końcu.

## 1. Kryterium sukcesu

Model globalny po FedAvg osiąga **F1 > 0,89** na zbiorze **`KDDTest+`**
(klasa dodatnia = atak, `is_attack = 1`, próg decyzyjny 0,5).

`KDDTest+` ma inny rozkład ataków niż `KDDTrain+` (np. R2L: 0,8% w train,
12,8% w teście), więc to kryterium może być trudne do spełnienia. Jeśli nie
zostanie osiągnięte, raportujemy wynik uczciwie wraz z analizą (rozbicie
na kategorie ataków), a nie zmieniamy zbioru ani progu po fakcie.

## 2. Zbiory danych

| Zbiór | Skąd | Do czego służy |
|---|---|---|
| Trening | 80% każdego shardu klienta (stratyfikowany podział według kategorii ataku) | uczenie lokalnych modeli |
| Walidacja | pozostałe 20% każdego shardu; walidacja globalna = suma walidacji lokalnych | strojenie hiperparametrów, wybór liczby rund, ewentualny próg |
| Test | `KDDTest+` | **wyłącznie** końcowy pomiar |

Zasady:
- `KDDTest+` nie służy do strojenia hiperparametrów, wyboru liczby rund,
  progu decyzyjnego ani wczesnego zatrzymania.
- Preprocessor jest dopasowany tylko na `KDDTrain+` (bez przecieku z testu).
- Podział walidacyjny robi `evaluate.stratified_holdout` ze stałym seedem.
  Dla baseline'u scentralizowanego ten sam podział jest zrobiony na całym
  `KDDTrain+`.

## 3. Metryki

Główna: **F1** (klasa atak). Pomocnicze: precision, recall, accuracy, ROC-AUC,
macierz pomyłek (tn, fp, fn, tp) oraz skuteczność wykrycia dla każdej kategorii
(DoS, Probe, R2L, U2R) i true-negative rate dla ruchu normalnego.

Wszystkie liczy `src/evaluate.py` (`evaluate`, `format_report`, `save_results`).

## 4. Porównywane konfiguracje

1. Baseline scentralizowany (górna granica).
2. Modele tylko lokalne (każdy klient osobno, bez FedAvg).
3. FedAvg: podział `iid`, `dirichlet` (α = 0,5) i `category`.

Wspólne dla wszystkich: ten sam model, ten sam preprocessor, ten sam próg 0,5,
ten sam `KDDTest+`.

## 5. Powtarzalność

- Stałe seedy (podział danych, inicjalizacja modelu, kolejność batchy).
- Co najmniej 3 seedy na konfigurację; raportujemy średnią i odchylenie standardowe.
- Konfiguracja każdego eksperymentu zapisywana razem z wynikiem
  (`experiments/results/`).

## 6. Procedura

1. Stroimy hiperparametry wyłącznie na walidacji.
2. Zamrażamy konfigurację.
3. Uruchamiamy ją na `KDDTest+` i zapisujemy wynik. Bez ponownego strojenia
   po zobaczeniu wyniku testowego.

## 7. Znane ograniczenia

- Preprocessor jest dopasowany na całym zbiorze treningowym, a w prawdziwym FL
  żaden podmiot nie widziałby danych pozostałych (uproszczenie symulacji).
- Walidacja jest wycinkiem danych treningowych, więc nie odzwierciedla
  przesunięcia rozkładu widocznego w `KDDTest+`. Wynik walidacyjny będzie
  zwykle wyższy od testowego.
- Dane pochodzą z mirrora (oficjalna strona zbioru jest niedostępna), co
  odnotowujemy w README.
