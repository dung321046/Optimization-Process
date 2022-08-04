import pandas as pd

data = pd.read_csv("train_data.txt")
print(data)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2)
train = train.to_numpy()
import torch
from models.torch_models import BinaryClassification, convert_to_train_loader, convert_to_test_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryClassification(85, device)
model.to(device)
print(model)
X_train = train[:, 1:86]
print(X_train)
y_train = train[:, 86]
print(y_train)
print(sum(y_train))
data = convert_to_train_loader(X_train, y_train)
model.fit(data, EPOCHS=500)
torch.save(model.state_dict(), "./model")
test = test.to_numpy()

X_test = test[:, 1:86]
y_test = test[:, 86]
test_loader = convert_to_test_loader(X_test)
model.evaluate(y_test, test_loader)
