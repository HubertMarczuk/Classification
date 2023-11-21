from numpy import sqrt
from pandas import read_csv


def Prepare(path):
    data = read_csv(path)
    list = data.values.tolist()

    Y = []

    for i in range(len(list)):
        if list[i][len(list[0]) - 1] == "h":
            Y.append(1)
        elif list[i][len(list[0]) - 1] == "g":
            Y.append(0)

    for i in range(len(list)):
        list[i].pop(len(list[i]) - 1)

    # 1/0, 3/4, 6/7, sqrt((0/2)^2-(1/2)^2) = półogniskowa, mimośród = półogniskowa/(0/2)
    for i in range(len(list)):
        axis_ratio = list[i][1] / list[i][0]
        pixel_ratio = list[i][3] / list[i][4]
        root_diff_3_moment = list[i][7] - list[i][6]
        half_focal_length = sqrt(pow(list[i][0] / 2, 2) - pow(list[i][1] / 2, 2))
        eccentricity = half_focal_length / (list[i][0] / 2)
        list[i].append(axis_ratio)
        list[i].append(pixel_ratio)
        list[i].append(root_diff_3_moment)
        list[i].append(half_focal_length)
        list[i].append(eccentricity)

    X = [[0 for i in range(len(list[0]))] for j in range(len(list))]
    for i in range(len(list[0])):
        MIN = min([j[i] for j in list])
        MAX = max([j[i] for j in list])
        for j in range(len(list)):
            tmp = (list[j][i] - MIN) / (MAX - MIN)
            X[j][i] = tmp

    return X, Y
