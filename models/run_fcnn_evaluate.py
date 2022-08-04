import torch
from models.torch_models import BinaryClassification, convert_to_train_loader, convert_to_test_loader

import numpy as np

dataset = "imdb"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BinaryClassification(10000, device)
model.load_state_dict(torch.load("./imdb/model.pt"))
model.to(device)
model.eval()

X_test = np.load("./" + dataset + "/test.npy", mmap_mode='r')
y_test = np.load("./" + dataset + "/test_label.npy", mmap_mode='r')

X_test = X_test[:1000]
y_test = y_test[:1000]
X_train = np.load("./" + dataset + "/train.npy", mmap_mode='r')
y_train = np.load("./" + dataset + "/train_label.npy", mmap_mode='r')
X_train = X_train[:1000]
y_train = y_train[:1000]

X = np.vstack((X_train, X_test))
y = np.concatenate((y_train, y_test), axis=0)
marker = np.concatenate((np.ones(len(y_train)), np.zeros(len(y_test))), axis=0)

data_loader = convert_to_test_loader(X)
feature = model.get_feature(data_loader)
np.save("./" + dataset + "/feature.npy", feature)
np.save("./" + dataset + "/marker.npy", marker)
np.save("./" + dataset + "/label.npy", y)

predicts = model.evaluate(y, data_loader)
error = model.get_error_arr(y, data_loader)
np.save("./" + dataset + "/error.npy", error)
np.save("./" + dataset + "/predict.npy", predicts)