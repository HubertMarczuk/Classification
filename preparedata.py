from numpy import sqrt
from pandas import read_csv


# Funkcja przygotowująca zmienne do klasyfikacji
def Prepare(path):
    # Wczytywanie danych z pliku .data
    data = read_csv(path)
    list = data.values.tolist()

    # Tworzenie listy kryterium klasyfikacyjnego z wartościami 0, gdy g oraz 1, gdy h
    Y = []
    for i in range(len(list)):
        if list[i][len(list[0]) - 1] == "h":
            Y.append(1)
        elif list[i][len(list[0]) - 1] == "g":
            Y.append(0)

    # Usuwanie kolumny z kryterium klasyfikacyjnym z wartości cech numerycznych
    for i in range(len(list)):
        list[i].pop(len(list[i]) - 1)

    # Wyliczanie dodatkowych 5 atrybutów
    for i in range(len(list)):
        # stosunek długości osi
        axis_ratio = list[i][1] / list[i][0]
        # stosunek
        pixel_ratio = list[i][3] / list[i][4]
        # różnica
        root_diff_3_moment = list[i][7] - list[i][6]
        # półogniskowa
        half_focal_length = sqrt(pow(list[i][0] / 2, 2) - pow(list[i][1] / 2, 2))
        # mimośród
        eccentricity = half_focal_length / (list[i][0] / 2)
        # Dodawanie wartości atrybutów do tabeli z wartościami kryteriów
        list[i].append(axis_ratio)
        list[i].append(pixel_ratio)
        list[i].append(root_diff_3_moment)
        list[i].append(half_focal_length)
        list[i].append(eccentricity)

    # Normalizacja wszystkich atrybutów
    X = [[0 for i in range(len(list[0]))] for j in range(len(list))]
    for i in range(len(list[0])):
        MIN = min([j[i] for j in list])
        MAX = max([j[i] for j in list])
        for j in range(len(list)):
            tmp = (list[j][i] - MIN) / (MAX - MIN)
            X[j][i] = tmp

    # Zwracanie przygotowanych tabel, gotowych do klasyfikacji
    return X, Y
