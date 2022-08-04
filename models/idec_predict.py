from models.idec import IDEC
import torch
import numpy as np

k = 40
idec = IDEC(input_dim=132, z_dim=8, n_clusters=k,
            encodeLayer=[128, 64], decodeLayer=[64, 128], activation="relu", dropout=0.1)
idec.load_model("idec.pt")

X_test = np.load("encoded_test.npy")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float)
manu_values = idec.regressBatch(X_test)
# Return ID of customers => Idx + 1

top_800 = np.argsort(manu_values)[-800:].data.cpu().numpy()
with open("predict.txt", "w") as f:
    for v in top_800:
        f.write(str(v + 1))
        f.write("\n")
