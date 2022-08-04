from models.idec import IDEC
import pandas as pd

data = pd.read_csv("train_data.txt")

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=27)
train = train.to_numpy()
# X_train = train[:, 1:86]
X_train = train[:, 1:43]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

y_train = train[:, 86]
k = 10
idec = IDEC(input_dim=X_train.shape[1], z_dim=4, n_clusters=k,
            encodeLayer=[64, 8], decodeLayer=[8, 64], activation="relu", dropout=0.1)
stat = []
import torch

X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
train_acc, epo = idec.fit(stat, X_train, y_train, lr=0.001, batch_size=128, num_epochs=200,
                                     update_interval=1, tol=1 * 1e-4)
idec.save_model("idec.pt")
