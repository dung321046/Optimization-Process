import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def buildNetwork(layers, activation="relu", dropout=0.2):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class BinaryClassification(nn.Module):
    def __init__(self, input_size, device):
        super(BinaryClassification, self).__init__()
        self.main_net = buildNetwork([input_size, 16, 64, 1], dropout=0.1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def forward(self, inputs):
        x = self.main_net(inputs)
        return x

    def get_last_layer(self, inputs):
        for layer in self.main_net[:-3]:
            inputs = layer(inputs)
        return inputs

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def fit(self, train_loader, EPOCHS=20):
        # self.cuda(device)
        self.train()
        # optimizer = optim.Adam(self.parameters(), lr=0.01)
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        for e in range(1, EPOCHS + 1):
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()

                y_pred = self.forward(X_batch)

                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                acc = self.binary_acc(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            print(
                f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

    def get_error_arr(self, y_test, test_loader):
        error_list = []
        self.eval()
        n = 0
        with torch.no_grad():
            for X_batch_ in test_loader:
                X_batch = X_batch_[0].to(self.device)
                values = self.forward(X_batch)
                y_prob = torch.sigmoid(values).squeeze().cpu().numpy()
                for id, t in enumerate(y_prob):
                    if y_test[n + id] == 0:
                        error_list.append(t)
                    else:
                        error_list.append(1 - t)
                n += len(y_prob)
        return error_list

    def get_feature(self, test_loader):
        feature_list = []
        self.eval()
        n = 0
        with torch.no_grad():
            for X_batch_ in test_loader:
                X_batch = X_batch_[0].to(self.device)
                values = self.get_last_layer(X_batch).cpu().numpy()
                feature_list.extend(values)
        return feature_list

    def evaluate(self, y_test, test_loader):
        y_pred_list = []
        self.eval()
        with torch.no_grad():
            for X_batch_ in test_loader:
                X_batch = X_batch_[0].to(self.device)
                y_test_pred = self.forward(X_batch)
                y_pred_tag = torch.round(torch.sigmoid(y_test_pred))
                y_pred_list.extend(y_pred_tag.squeeze().cpu().numpy())

        # y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        print(confusion_matrix(y_test, y_pred_list))
        print(classification_report(y_test, y_pred_list, digits=3))
        return y_pred_list


from torch.utils.data import TensorDataset, DataLoader


def convert_to_train_loader(X, y):
    tensor_x = torch.tensor(X, dtype=torch.float)
    tensor_y = torch.tensor(y, dtype=torch.float)

    my_dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(my_dataset, batch_size=128, shuffle=True)


def convert_to_test_loader(X):
    tensor_x = torch.tensor(X, dtype=torch.float)
    my_dataset = TensorDataset(tensor_x)
    return DataLoader(my_dataset, batch_size=128, shuffle=False)
