import numpy as np
from visualization import plot2d

dataset = "imdb"


def get_test_visual(path, outpath):
    X_feature = np.load(path + "/feature.npy", mmap_mode='r')
    marker = np.load(path + "/marker.npy", mmap_mode='r')
    predicts = np.load(path + "/predict.npy", mmap_mode='r')
    label = np.load(path + "/label.npy", mmap_mode='r')
    probs = np.load(path + "/probs.npy", mmap_mode='r')
    probs = np.squeeze(probs)
    entropies = np.load(path + "/entropies.npy", mmap_mode='r')
    acc_test = []
    for id, predict in enumerate(predicts):
        if predict == label[id]:
            acc_test.append(1)
        else:
            acc_test.append(0)
    acc_test = np.asarray(acc_test)
    # plot2d.plot2d_density_tsne_label(X_test, acc_test)
    X_2d = plot2d.plot2d_density_tsne_marker_label(X_feature, marker, acc_test, outpath + "error_distribution")
    # plot2d.plot2d_weighted_density(X_2d, predicts)
    plot2d.plot2d_weighted_density(X_2d, entropies, acc_test, outpath + "predict_entropies_distribution")


for t in range(10):
    path = "./imdb/EntropySampling01/"
    get_test_visual(path + str(t), path + str(t))
# from stat_measure.utils import coefficent_of_label_n_acc
#
# print(coefficent_of_label_n_acc(X_feature, error_test, marker, "Euclidean"))
# print(coefficent_of_label_n_acc(X_feature, acc, marker, "Euclidean"))
