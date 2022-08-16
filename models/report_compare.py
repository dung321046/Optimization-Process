import numpy as np
from visualization import plot2d

dataset = "imdb"


def get_active_acc(path):
    predicts = np.load(path + "/predict.npy", mmap_mode='r')
    label = np.load(path + "/label.npy", mmap_mode='r')
    correct = 0.0
    for id, predict in enumerate(predicts):
        if predict == label[id]:
            correct += 1

    return correct / len(label)


random_accs = []
numIter = 11
for t in range(numIter):
    random_accs.append([])
for run in range(3):
    for t in range(numIter):
        path = "./imdb/RandomSampling0" + str(run) + "/"
        random_accs[t].append(get_active_acc(path + str(t)))
entropy_accs = []
for t in range(numIter):
    entropy_accs.append([])
for run in range(3):
    for t in range(numIter):
        path = "./imdb/EntropySampling0" + str(run) + "/"
        entropy_accs[t].append(get_active_acc(path + str(t)))

import matplotlib.pyplot as plt
import statistics

# plot line
# plt.plot(range(10), random_accs, label="Random")
# plt.plot(range(10), entropy_accs, label="Entropy")
plt.errorbar(range(numIter), [statistics.mean(s) for s in random_accs], [statistics.stdev(s) for s in random_accs],
             label='Random', marker='^', alpha=0.5)
plt.errorbar(range(numIter), [statistics.mean(s) for s in entropy_accs], [statistics.stdev(s) for s in entropy_accs],
             label='Entropy', alpha=0.5)
plt.legend()
plt.show()
# from stat_measure.utils import coefficent_of_label_n_acc
#
# print(coefficent_of_label_n_acc(X_feature, error_test, marker, "Euclidean"))
# print(coefficent_of_label_n_acc(X_feature, acc, marker, "Euclidean"))
