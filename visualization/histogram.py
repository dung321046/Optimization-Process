import matplotlib.pyplot as plt


def histogram_uni(x, bins=20, title="", file=None):
    plt.hist(x, bins=bins)
    plt.title(title)
    if file:
        plt.savefig(file)
    else:
        plt.show()
    plt.cla()