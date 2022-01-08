import matplotlib.pyplot as plt
import numpy as np

MinV = 0.0
MaxV = 2000.0


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def get_sum_dis(X, p):
    return sum([euclidean_distance(x, p) for x in X])


def update(p, dx, dy, step):
    u = max(MinV, min(MaxV, p[0] + dx * step))
    v = max(MinV, min(MaxV, p[1] + dy * step))
    return (u, v)


def draw_gradient(points, start_point, num_step, eps, ax, runId):
    step = 1.0
    cur_dx = 0
    cur_dy = 0
    cur_p = start_point[:]
    Xpath = [start_point[0]]
    Ypath = [start_point[1]]
    ax.set_xlim(0.0, 2000.0)
    ax.set_ylim(0.0, 2000.0)
    ax.scatter(points[:, 0], points[:, 1])
    for i, p in enumerate(points):
        ax.annotate("P" + str(i + 1), (p[0], p[1]))
    for i in range(num_step):
        bdx, bdy = 0, 0
        b_cost = get_sum_dis(points, cur_p)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                p = (cur_p[0] + dx * step, cur_p[1] + dy * step)
                c = get_sum_dis(points, p)
                if c < b_cost:
                    bdx, bdy = dx, dy
                    b_cost = c
        diff = abs(cur_dx - bdx) + abs(cur_dy - bdy)
        cur_dx, cur_dy = bdx, bdy
        cur_p = update(cur_p, bdx, bdy, step)
        # print(cur_p)
        # print(cur_p, bdx, bdy, step)
        Xpath.append(cur_p[0])
        Ypath.append(cur_p[1])
        if bdx == 0 and bdy == 0:
            step = step * 0.9
        elif diff == 0:
            step *= 1.2
        else:
            step = step * (0.9 ** diff)
        if step < eps:
            break
    ax.plot(Xpath, Ypath)
    ax.annotate("A" + str(runId + 1), cur_p)
    return cur_p


if __name__ == "__main__":
    fig, axs = plt.subplots(2)
    #points = [[0, 0], [2000, 0], [1000, 2000]]
    points = [[2, 3], [1990, 0], [1001, 1999]]
    points = np.asarray(points)

    eps = 10 ^ -6
    num_step = 5000
    start_point = np.asarray([1000, 500])
    ans = draw_gradient(points, start_point, num_step, eps, axs[0], 1)
    print(ans, get_sum_dis(points, ans))
    # start_point = np.asarray([1000, 1000])
    for t in range(10):
        start_point = np.asarray([np.random.random() * 2000, np.random.random() * 2000])
        ans = draw_gradient(points, start_point, num_step, eps, axs[1], t)
        print(ans, get_sum_dis(points, ans))
    fig.savefig("Search")
