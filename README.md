# Federal-Bureau-of-Investigation
Projekt z Zaawansowynych Technik Kryptografii i Kryptoanalizy


Oryginalny dataset nie jest już dostępny: https://www.unb.ca/cic/datasets/nsl.html
Licencja NSL-KDD wprost pozwala na mirrorowanie.
Z tego względu w projekcie został wykorzystany: https://github.com/Jehuty4949/NSL_KDD

# Co zrobiono dotychczas:
Preprocessor zamienia surowe pliki NSL-KDD w macierze liczb o jednakowym formacie (one-hot, log1p, standaryzacja dopasowana wyłącznie do zbioru treningowego) i tworzy binarną etykietę atak/normal oraz kategorię ataku. 

Sharding dzieli zbiór treningowy na 3 rozłączne części dla symulowanych klientów w wariancie IID, Dirichlet lub według kategorii ataku, tak aby każdy klient miał zarówno ruch normalny, jak i ataki.

Evaluation liczy metryki jakości detekcji (F1, precision, recall, accuracy, ROC-AUC, macierz pomyłek) z klasą „atak” jako dodatnią oraz skuteczność wykrycia dla każdej kategorii ataku, i wydziela stratyfikowany zbiór walidacyjny. Protokół zakłada strojenie wyłącznie na walidacji, a `KDDTest+` służy tylko do końcowego pomiaru kryterium F1 > 0,89.