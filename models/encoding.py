import numpy as np


def one_hot_encoding(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # Creates an all zero matrix of shape (len(sequences),10K)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # Sets specific indices of results[i] to 1s
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Encoding inputs of several datasets')
    parser.add_argument('--type', type=str, default="onehot", metavar='N', help='dataset(onehot)')
    parser.add_argument('--data', type=str, default="imdb", metavar='N', help='dataset(imdb)')

    args = parser.parse_args()
    if args.data == "imdb":
        path = "./imdb/"
        import os

        if not os.path.exists(path):
            os.mkdir(path)
        from load_datasets.imdb import train_data, train_labels, test_data, test_labels

        X_train = one_hot_encoding(train_data)
        y_train = np.asarray(train_labels).astype('float32')
        X_test = one_hot_encoding(test_data)
        y_test = np.asarray(test_labels).astype('float32')
        with open(path + 'train.npy', 'wb') as f:
            np.save(f, X_train)
        with open(path + 'train_label.npy', 'wb') as f:
            np.save(f, y_train)
        with open(path + 'test.npy', 'wb') as f:
            np.save(f, X_test)
        with open(path + 'test_label.npy', 'wb') as f:
            np.save(f, y_test)
