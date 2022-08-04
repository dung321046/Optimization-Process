import collections
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.autograd import Variable
from torch.nn import Parameter

from models.utils import acc

from sklearn.metrics import confusion_matrix, classification_report


class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target) ** 2)


def buildNetwork(layers, activation="relu", dropout=0):
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


def binarize_arr(y_pred, threshold=0.5):
    arr = []
    for y in y_pred:
        if y > threshold:
            arr.append(1)
        else:
            arr.append(0)
    return arr


def cal_acc(y_pred, y):
    y_pred = binarize_arr(y_pred)
    return np.sum(y_pred == y) / len(y)


class IDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1., gamma=0.1):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._mu_manu = nn.Linear(z_dim, 1)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        xrecon = self._dec(h)
        # compute q -> NxK
        q = self.soft_assign(z)
        return z, q, xrecon

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def encodeBatch(self, X, batch_size=256):
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.cuda()

        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def regressBatch(self, X, batch_size=256):
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _ = self.forward(inputs)
            manu = self._mu_manu(z)
            encoded.append(manu.data)
        encoded = torch.cat(encoded, dim=0).squeeze()
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    def recon_loss(self, x, xrecon):
        recon_loss = torch.mean((xrecon - x) ** 2)
        return recon_loss

    def global_size_loss(self, p, cons_detail):
        m_p = torch.mean(p, dim=0)
        m_p = m_p / torch.sum(m_p)
        return torch.sum((m_p - cons_detail) * (m_p - cons_detail))

    def difficulty_loss(self, q, mask):
        mask = mask.unsqueeze_(-1)
        mask = mask.expand(q.shape[0], q.shape[1])
        mask_q = q * mask
        diff_loss = -torch.norm(mask_q, 2)
        penalty_degree = 0.1
        return penalty_degree * diff_loss

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def predict(self, X, y):
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.cuda()
        latent = self.encodeBatch(X)
        q = self.soft_assign(latent)

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        y = y.data.cpu().numpy()
        if y is not None:
            print("acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
            final_acc = acc(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
        return final_acc, final_nmi

    def fit(self, stat, X, y, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3):
        '''X: tensor data'''
        # use_cuda = torch.cuda.is_available()
        # if use_cuda:
        #     self.cuda()
        print("=====Training IDEC=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(self.n_clusters, n_init=20)
        data = self.encodeBatch(X)
        y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        y_pred_last = y_pred
        self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_))
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        final_acc, final_epoch = 0, 0
        n = len(y)
        y_np = y.data.cpu().numpy()
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data
                manu = self._mu_manu(latent)

                y_pred = manu.squeeze().data.cpu().numpy()
                regess = sum([(y_pred[i] - y_np[i]) ** 2 for i in range(n)]) / n
                final_acc = cal_acc(y_pred, y_np)
                print("acc: %.5f, regess: %.5f" % (final_acc, regess))
                final_epoch = epoch
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                stat.append((final_acc, delta_label))
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                ybatch = y[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch, xrecon = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.recon_loss(inputs, xrecon)
                regess_loss = torch.mean((self._mu_manu(z) - 8 * ybatch) ** 2)
                loss = cluster_loss + recon_loss + regess_loss * 0.5
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                train_loss = cluster_loss_val + recon_loss_val

            print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))
        print(confusion_matrix(y_np, binarize_arr(y_pred)))
        print(classification_report(y_np, binarize_arr(y_pred), digits=3))
        return final_acc, final_epoch
