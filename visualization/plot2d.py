import random

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


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


def plot2dvectors_2label(vectors, labels, labels2):
    t = vectors.transpose()
    color_arr = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    plt.subplot(1, 2, 1)
    colors = []
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
    plt.show()
