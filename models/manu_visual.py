from models.idec import IDEC
import pandas as pd
import torch
import numpy as np
import random

k = 40
idec = IDEC(input_dim=132, z_dim=8, n_clusters=k,
            encodeLayer=[128, 64], decodeLayer=[64, 128], activation="relu", dropout=0.1)
idec.load_model("idec.pt")

data = pd.read_csv("train_data.txt")
data = data.to_numpy()
X_train = np.load("encoded_train.npy")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

y_train = data[:, 86]
idx = []
for i in range(len(y_train)):
    if y_train[i] == 1:
        idx.append(i)
    elif random.random() < 0.1:
        idx.append(i)
X_train = torch.tensor(X_train[idx], dtype=torch.float)
y_train = torch.tensor(y_train[idx], dtype=torch.float)
z = idec.encodeBatch(X_train)

from visualization import plot2d

plot2d.plot2dvectors_2label(z, data[idx, 1], y_train, "Customer Subtype")

plot2d.plot2dvectors_2label(z, data[idx, 5], y_train, "Customer Type")
