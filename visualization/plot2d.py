import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

import seaborn as sns
import pandas as pd


def plot2d_density(vectors):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)
    # reduced = vectors
    print(reduced)
    # t = reduced.transpose()
    # from scipy.ndimage.filters import gaussian_filter
    import seaborn as sns
    # df3_smooth = gaussian_filter(reduced, sigma=0.001)
    # sns.heatmap(df3_smooth, vmin=np.min(df3_smooth), vmax=np.max(df3_smooth), cmap="coolwarm", cbar=True)
    # ax = sns.kdeplot(x=reduced[:, 0], y=reduced[:, 1], shade=True, cmap="PuBu")
    import pandas as pd
    df = pd.DataFrame(reduced,
                      columns=['x', 'y'])
    ax = sns.kdeplot(data=df, x="x", y="y")
    plt.show()


def plot2d_density_tsne(vectors):
    from sklearn.manifold import TSNE
    reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vectors)
    # reduced = vectors
    print(reduced)
    # t = reduced.transpose()
    # from scipy.ndimage.filters import gaussian_filter
    import seaborn as sns
    # df3_smooth = gaussian_filter(reduced, sigma=0.001)
    # sns.heatmap(df3_smooth, vmin=np.min(df3_smooth), vmax=np.max(df3_smooth), cmap="coolwarm", cbar=True)
    # ax = sns.kdeplot(x=reduced[:, 0], y=reduced[:, 1], shade=True, cmap="PuBu")
    import pandas as pd
    df = pd.DataFrame(reduced,
                      columns=['x', 'y'])
    ax = sns.kdeplot(data=df, x="x", y="y")
    plt.show()


def plot2d_weighted_density(vectors, weights, acc, filename):
    from scipy.interpolate.rbf import Rbf  # radial basis functions
    for id, weight in enumerate(weights):
        if weight == np.nan:
            weights[id] = 0
    rbf_fun = Rbf(vectors[:, 0], vectors[:, 1], weights, function='gaussian')
    minx, maxx = min(vectors[:, 0]), max(vectors[:, 0])
    miny, maxy = min(vectors[:, 1]), max(vectors[:, 1])
    x = np.linspace(minx, maxx, 200)
    y = np.linspace(miny, maxy, 200)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(1, 3)
    z_new = rbf_fun(X.ravel(), Y.ravel()).reshape(X.shape)
    c = ax[0].pcolor(X, Y, z_new, cmap='bwr')
    fig.colorbar(c, ax=ax[0])
    # sns.scatterplot(x=X, y=Y, hue=z_new, ax=ax[0])
    sns.scatterplot(x=vectors[:, 0], y=vectors[:, 1], hue=weights, ax=ax[1])
    sns.scatterplot(x=vectors[:, 0], y=vectors[:, 1], hue=acc, ax=ax[2])
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(filename)
    plt.close()


def plot2d_density_tsne_marker_label(vectors, markers, accuracy, filename):
    from sklearn.manifold import TSNE
    reduced = TSNE(n_components=2, learning_rate='auto', init='random', random_state=27).fit_transform(vectors)
    # reduced = vectors
    print(reduced)

    trainIdx = []
    testIdx = []
    for id, marker in enumerate(markers):
        if marker == 1:
            trainIdx.append(id)
        else:
            testIdx.append(id)
    train = np.concatenate((reduced[trainIdx], accuracy[trainIdx].reshape(-1, 1)), axis=1)
    trainDf = pd.DataFrame(train, columns=['x', 'y', 'acc'])
    testDf = pd.DataFrame(reduced[testIdx], columns=['x', 'y'])
    total = np.concatenate((reduced, markers.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(total,
                      columns=['x', 'y', 'isTrain'])
    total2 = np.concatenate((reduced, accuracy.reshape(-1, 1)), axis=1)
    df2 = pd.DataFrame(total2,
                       columns=['x', 'y', 'acc'])

    fig, ax = plt.subplots(1, 3)
    # sns.kdeplot(data=df, x="x", y="y", ax=ax[0])
    try:
        sns.kdeplot(data=trainDf, x="x", y="y", hue='acc', ax=ax[0])
    except:
        sns.scatterplot(x=trainDf["x"], y=trainDf["y"], hue=accuracy[trainIdx], ax=ax[0])
    ax[0].set_title("Distribution of correct/incorrect train data")
    sns.kdeplot(data=df2, x="x", y="y", hue="acc", kind="kde", ax=ax[1])
    ax[1].set_title("Distribution of correct/incorrect test data")
    # sns.kdeplot(data=testDf, x="x", y="y", ax=ax[3])
    import scipy.stats
    print(reduced[trainIdx])
    kdea = scipy.stats.gaussian_kde(np.transpose(reduced[trainIdx]))
    kdeb = scipy.stats.gaussian_kde(np.transpose(reduced[testIdx]))
    minx, maxx = min(reduced[:, 0]), max(reduced[:, 0])
    miny, maxy = min(reduced[:, 1]), max(reduced[:, 1])
    x = np.linspace(minx, maxx, 200)
    y = np.linspace(miny, maxy, 200)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    print(kdea(positions).shape)
    print(kdeb(positions).shape)
    subtract = kdea(positions) - kdeb(positions)
    Z = np.reshape(subtract.T, X.shape)
    CS = ax[2].imshow(np.rot90(Z), cmap='RdYlGn',
                      extent=[minx, maxx, miny, maxy], aspect="auto")
    ax[2].set_title("Differences between train and test")
    # CS = ax[1].contour(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    #                   extent=[-40, 60, -70, 70], aspect="auto")
    fig.colorbar(CS)
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(filename)
    plt.close()
    return reduced


def plot2d_density_tsne_label(vectors, labels):
    from sklearn.manifold import TSNE
    reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vectors)
    # reduced = vectors
    print(reduced)
    import seaborn as sns
    import pandas as pd
    total = np.concatenate((reduced, labels.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(total,
                      columns=['x', 'y', 'label'])

    fig, ax = plt.subplots(1, 2)
    sns.kdeplot(data=df, x="x", y="y", ax=ax[0])
    sns.kdeplot(data=df, x="x", y="y", hue="label", kind="kde", ax=ax[1])
    plt.show()


def plot2dvectors(vectors, labels):
    pca = PCA(n_components=2)
    # reduced = pca.fit_transform(vectors)
    reduced = vectors
    t = reduced.transpose()
    colors = []
    color_arr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    for l in labels:
        colors.append(color_arr[int(l)])
    plt.scatter(t[0], t[1], c=colors)
    fre_label = dict()
    for i, txt in enumerate(labels):
        if labels[i] not in fre_label:
            fre_label[labels[i]] = 1
        else:
            fre_label[labels[i]] += 1
        if random.uniform(0, 10) > fre_label[labels[i]]:
            plt.annotate(labels[i], (t[0][i], t[1][i]))
    plt.show()


def plot2dvectors_2label(vectors, labels, labels2, title):
    from sklearn.manifold import TSNE
    reduced = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(vectors)
    t = reduced.transpose()
    color_arr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    plt.subplot(1, 2, 1)
    colors = []
    for l in labels:
        colors.append(color_arr[int(l) % 10])
    plt.scatter(t[0], t[1], c=colors)
    fre_label = dict()
    for i, txt in enumerate(labels):
        if labels[i] not in fre_label:
            fre_label[labels[i]] = 1
        else:
            fre_label[labels[i]] += 1
        if random.uniform(0, 10) > fre_label[labels[i]]:
            plt.annotate(labels[i], (t[0][i], t[1][i]))
    plt.subplot(1, 2, 2)
    colors = []
    for l in labels2:
        colors.append(color_arr[int(l)])
    plt.scatter(t[0], t[1], c=colors)
    fre_label = dict()
    for i, txt in enumerate(labels):
        if labels[i] not in fre_label:
            fre_label[labels[i]] = 1
        else:
            fre_label[labels[i]] += 1
        if random.uniform(0, 10) > fre_label[labels[i]]:
            plt.annotate(labels[i], (t[0][i], t[1][i]))
    plt.savefig(title)
    plt.close()
