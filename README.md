# Federal-Bureau-of-Investigation
Projekt z Zaawansowynych Technik Kryptografii i Kryptoanalizy


Oryginalny dataset nie jest już dostępny: https://www.unb.ca/cic/datasets/nsl.html
Licencja NSL-KDD wprost pozwala na mirrorowanie.
Z tego względu w projekcie został wykorzystany: https://github.com/Jehuty4949/NSL_KDD

# Co zrobiono dotychczas:
Preprocessor zamienia surowe pliki NSL-KDD w macierze liczb o jednakowym formacie (one-hot, log1p, standaryzacja dopasowana wyłącznie do zbioru treningowego) i tworzy binarną etykietę atak/normal oraz kategorię ataku. 

Sharding dzieli zbiór treningowy na 3 rozłączne części dla symulowanych klientów w wariancie IID, Dirichlet lub według kategorii ataku, tak aby każdy klient miał zarówno ruch normalny, jak i ataki.

Trening lokalny (M2): moduł `src/training.py` (`LocalModel`) oraz klient Flower `src/client.py`. Szczegóły uruchomienia: [docs/TRAINING.md](docs/TRAINING.md).

```bash
python -m src.training --cid 0    # baseline na shardzie 0
python -m src.client --cid 0 --local-only
```