import numpy as np
from visualization import plot2d

dataset = "imdb"


def get_test_visual():
    X_feature = np.load("./" + dataset + "/feature.npy", mmap_mode='r')
    marker = np.load("./" + dataset + "/marker.npy", mmap_mode='r')
    predicts = np.load("./" + dataset + "/predict.npy", mmap_mode='r')
    label = np.load("./" + dataset + "/label.npy", mmap_mode='r')
    # acc_test = []
    # for t in error_test:
    #     if t < 0.5:
    #         acc_test.append(1)
    #     else:
    #         acc_test.append(0)
    acc_test = []
    for id, predict in enumerate(predicts):
        if predict == label[id]:
            acc_test.append(1)
        else:
            acc_test.append(0)
    acc_test = np.asarray(acc_test)
    # plot2d.plot2d_density_tsne_label(X_test, acc_test)
    plot2d.plot2d_density_tsne_marker_label(X_feature, marker, acc_test)


get_test_visual()
# from stat_measure.utils import coefficent_of_label_n_acc
#
# print(coefficent_of_label_n_acc(X_feature, error_test, marker, "Euclidean"))
# print(coefficent_of_label_n_acc(X_feature, acc, marker, "Euclidean"))
