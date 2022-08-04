import pandas as pd
import numpy as np
def encode_train():
    data = pd.read_csv("train_data.txt")
    data = data.to_numpy()
    data = data[:, 1:86]
    n = len(data)
    m = len(data[0])
    n_c1 = 41
    n_c5 = 10
    removed_atts = [59, 80]
    one_hot_atts = [0, 4]
    feature = len(data[0]) + n_c1 + n_c5 - 2 - 2
    encode_data = np.zeros((n, feature))

    for i in range(n):
        encode_data[i][data[i][0] - 1] = 1
        encode_data[i][n_c1 + data[i][4] - 1] = 1
        start = n_c1 + n_c5
        t = 0
        for j in range(m):
            if j not in one_hot_atts and j not in removed_atts:
                encode_data[i][start + t] = data[i][j]
                t += 1
        print(start + t, feature)
    np.save("encoded_train", encode_data)
def encode_test():
    data = pd.read_csv("test_data.txt")
    data = data.to_numpy()
    data = data[:, 1:86]
    n = len(data)
    m = len(data[0])
    n_c1 = 41
    n_c5 = 10
    removed_atts = [59, 80]
    one_hot_atts = [0, 4]
    feature = len(data[0]) + n_c1 + n_c5 - 2 - 2
    encode_data = np.zeros((n, feature))

    for i in range(n):
        encode_data[i][data[i][0] - 1] = 1
        encode_data[i][n_c1 + data[i][4] - 1] = 1
        start = n_c1 + n_c5
        t = 0
        for j in range(m):
            if j not in one_hot_atts and j not in removed_atts:
                encode_data[i][start + t] = data[i][j]
                t += 1
        print(start + t, feature)
    np.save("encoded_test", encode_data)

encode_test()