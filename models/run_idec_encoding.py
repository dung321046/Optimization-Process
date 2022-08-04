from models.idec import IDEC
import pandas as pd
import numpy as np

data = pd.read_csv("train_data.txt")
data = data.to_numpy()
X_train = np.load("encoded_train.npy")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

y_train = data[:, 86]
k = 40
idec = IDEC(input_dim=X_train.shape[1], z_dim=4, n_clusters=k,
            encodeLayer=[64, 32], decodeLayer=[32, 64], activation="relu", dropout=0.1)
stat = []
import torch

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
train_acc, epo = idec.fit(stat, X_train, y_train, lr=0.01, batch_size=128, num_epochs=200,
                                     update_interval=1, tol=1 * 1e-4)
idec.save_model("idec.pt")
