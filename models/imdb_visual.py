import numpy as np
from visualization import plot2d

dataset = "imdb"


def get_test_visual():
    X_test = np.load("./" + dataset + "/test.npy", mmap_mode='r')
    y_test = np.load("./" + dataset + "/test_label.npy", mmap_mode='r')
    error_test = np.load("./" + dataset + "/error.npy", mmap_mode='r')
    X_feature = np.load("./" + dataset + "/feature.npy", mmap_mode='r')
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    # plot2d.plot2d_density(X_test)
    # plot2d.plot2d_density_tsne(X_test)
    acc_test = []
    for t in error_test:
        if t < 0.5:
            acc_test.append(1)
        else:
            acc_test.append(0)
    acc_test = np.asarray(acc_test)

    # plot2d.plot2d_density_tsne_label(X_test, acc_test)
    plot2d.plot2d_density_tsne_label(X_feature, acc_test)


X_feature = np.load("./" + dataset + "/feature.npy", mmap_mode='r')

idx = np.random.choice(np.arange(len(X_feature)), 500, replace=False)
X_feature = X_feature[idx]
from sklearn.manifold import TSNE

X_feature = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_feature)

error_test = np.load("./" + dataset + "/error.npy", mmap_mode='r')
error_test = error_test[idx]
predicts = np.load("./" + dataset + "/predict.npy", mmap_mode='r')
predicts = predicts[idx]
label = np.load("./" + dataset + "/label.npy", mmap_mode='r')
label = label[idx]
acc = [1 if predicts[i] == label[i] else 0 for i in range(len(label))]
marker = np.load("./" + dataset + "/marker.npy", mmap_mode='r')
marker = marker[idx]
from stat_measure.utils import coefficent_of_label_n_acc

print(coefficent_of_label_n_acc(X_feature, error_test, marker, "Euclidean"))
print(coefficent_of_label_n_acc(X_feature, acc, marker, "Euclidean"))
