import matplotlib.pyplot as plt
import numpy as np


def cost_linear(X, t):
    ans = 0
    for x in X:
        ans += abs(x - t)
    return ans


def draw_cost_1d(X, names, ax):
    costX = []
    best = 0
    for i in range(len(X)):
        costX.append(cost_linear(X, X[i]))
        if costX[best] > costX[-1]:
            best = i
        ax.annotate(names[i], (X[i], costX[i]))
    # for idx in sorted_idx:
    #     print(X[idx], names[idx])
    sorted_idx = np.argsort(X)
    # for i in range(len(X) - 1):
    #     cur_idx = sorted_idx[i]
    #     nxt_idx = sorted_idx[i + 1]
    #     ax.plot(X[cur_idx], costX[cur_idx], X[nxt_idx], costX[nxt_idx], marker='o')
    ax.plot([X[idx] for idx in sorted_idx], [costX[idx] for idx in sorted_idx])
    return X[best]


def find_mean(points):
    X = points[:, 0]
    Y = points[:, 1]
    point_names = ["A", "B", "E", "C", "D"]
    point_names = point_names[:len(X)]
    fig, axs = plt.subplots(2)
    bestX = draw_cost_1d(X, point_names, axs[0])
    bestY = draw_cost_1d(Y, point_names, axs[1])
    fig.savefig("Cost functions")
    print("Answer:", bestX, bestY)
    return [bestX, bestY]


if __name__ == "__main__":
    points = [[0, 0], [2000, 0], [1000, 2000]]
    points = np.asarray(points)
    find_mean(points)
