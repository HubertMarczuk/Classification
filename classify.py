from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preparedata import Prepare


# Funkcja przeprowadzająca klasyfikację KNN
def KNN(X_train, X_test, Y_train, Y_test, K):
    # Dopasowanie klasyfikatora KNN dla danych treningowych
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, Y_train)

    # Przewidywanie klasy dla danych testowych
    Y_pred = knn.predict(X_test)

    # Zwrócenie listy z przewidzianaymi klasami dla każdego rekordu
    return Y_pred


# Funkcja przeprowadzająca klasyfikację Random Forest
def RandomForest(X_train, X_test, Y_train, Y_test, Trees):
    # Dopasowanie klasyfikatora Random forest dla danych treningowych
    rf = RandomForestClassifier(n_estimators=Trees)
    rf.fit(X_train, Y_train)

    # Przewidywanie klasy dla danych testowych
    Y_pred = rf.predict(X_test)

    # Zwrócenie listy z przewidzianaymi klasami dla każdego rekordu
    return Y_pred


# Przygotowanie danych do klasyfikacji
X, Y = Prepare("magic04.data")

# Wybór liczby najbliższych sąsiadów oraz liczby drzeww lesie
K = 10
Trees = 50

# Losowy podział zbioru treningowego i testowego
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4)

# Klasyfikowanie
Y_pred_KNN = KNN(X_train, X_test, Y_train, Y_test, K)
Y_pred_RandomForest = RandomForest(X_train, X_test, Y_train, Y_test, Trees)

# Obliczanie wartości macierzy klasyfikacji dla modelu wykorzystującego KNN
TP1, FP1, FN1, TN1 = 0, 0, 0, 0
for i in range(len(Y_pred_KNN)):
    if Y_test[i] == 0 and Y_pred_KNN[i] == 0:
        TN1 += 1
    elif Y_test[i] == 0 and Y_pred_KNN[i] == 1:
        FP1 += 1
    elif Y_test[i] == 1 and Y_pred_KNN[i] == 0:
        FN1 += 1
    else:
        TP1 += 1

# Obliczanie wartości macierzy klasyfikacji dla modelu wykorzystującego Random Forest
TP2, FP2, FN2, TN2 = 0, 0, 0, 0
for i in range(len(Y_pred_RandomForest)):
    if Y_test[i] == 0 and Y_pred_RandomForest[i] == 0:
        TN2 += 1
    elif Y_test[i] == 0 and Y_pred_RandomForest[i] == 1:
        FP2 += 1
    elif Y_test[i] == 1 and Y_pred_RandomForest[i] == 0:
        FN2 += 1
    else:
        TP2 += 1

# Wyświetlenie wyników i obliczenie miar klasyfikacji dla KNN
print("-----Klasyfikator KNN-----")
print("Wartości macierzy klasyfikacji:")
print("TP =", TP1, "   ", "FP =", FP1)
print("FN =", FN1, "   ", "TN =", TN1)
print("Trafność =", (TP1 + TN1) / (TP1 + TN1 + FP1 + FN1))
print("Czułość =", TP1 / (TP1 + FN1))
print("Specyficzność =", TN1 / (TN1 + FP1))
print("-----------------------------")

# Wyświetlenie wyników i obliczenie miar klasyfikacji dla Random Forest
print("-----Klasyfikator Random Forest-----")
print("Wartości macierzy klasyfikacji:")
print("TP =", TP2, "   ", "FP =", FP2)
print("FN =", FN2, "   ", "TN =", TN2)
print("Trafność =", (TP2 + TN2) / (TP2 + TN2 + FP2 + FN2))
print("Czułość =", TP2 / (TP2 + FN2))
print("Specyficzność =", TN2 / (TN2 + FP2))
