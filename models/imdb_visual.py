import numpy as np

dataset = "imdb"
X_test = np.load("./" + dataset + "/test.npy", mmap_mode='r')
y_test = np.load("./" + dataset + "/test_label.npy", mmap_mode='r')
error_test = np.load("./" + dataset + "/error.npy", mmap_mode='r')
X_feature = np.load("./" + dataset + "/feature.npy", mmap_mode='r')
X_test = X_test[:1000]
y_test = y_test[:1000]
from visualization import plot2d

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
