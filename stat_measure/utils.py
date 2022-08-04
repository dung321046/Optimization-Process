from scipy.stats.stats import pearsonr
import numpy as np


def get_avg_dis(X, dis_measure):
    n = len(X)
    avg_dis = 0.0
    for i in range(n):
        min_dis = float("inf")
        for j in range(n):
            if i != j:
                min_dis = min(min_dis, dis_measure(X[i], X[j]))
        avg_dis += min_dis
    return avg_dis / n


def normalize2(p1, p2):
    return np.linalg.norm(p1 - p2)


def coefficent_of_label_n_acc(X, error_arr, marker, dis_type):
    if dis_type == "Euclidean":
        dis_measure = normalize2
    else:
        dis_measure = normalize2
    avg_dis = get_avg_dis(X, dis_measure)
    label_costs = []
    for x in X:
        label_cost = 0
        for i, x2 in enumerate(X):
            if marker[i]:
                label_cost += avg_dis / (avg_dis + dis_measure(x, x2))
        label_costs.append(label_cost)

    print(np.round(label_costs[:20], 2))
    print(np.round(error_arr[:20], 2))
    return pearsonr(label_costs, error_arr)
