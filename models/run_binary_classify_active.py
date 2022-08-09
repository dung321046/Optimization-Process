import numpy as np
import os

dataset = "imdb"
X_all = np.load("./" + dataset + "/train.npy")
y_all = np.load("./" + dataset + "/train_label.npy")


def check_gpu():
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    print("Available: %fGB" % (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))


if __name__ == "__main__":
    import torch
    from models.torch_models import BinaryClassification, convert_to_train_loader, convert_to_test_loader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification(10000, device)
    model.to(device)
    print(model)
    N = 1000
    X_all = X_all[:N]
    y_all = y_all[:N]
    markers = np.zeros(N)
    remaining_idx = range(N)
    train_idx = np.random.choice(remaining_idx, size=20, replace=False)
    for id in train_idx:
        markers[id] = 1
    remaining_idx = list(set(remaining_idx) - set(train_idx))
    data = convert_to_train_loader(X_all[train_idx], y_all[train_idx])
    model.fit(data, EPOCHS=100)
    config_path = "/run01"
    main_path = "./" + dataset + config_path
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    for t in range(10):
        local_path = main_path + "/" + str(t)
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        new_idx = np.random.choice(remaining_idx, size=10, replace=False)
        for id in new_idx:
            markers[id] = 1
        train_idx = np.concatenate([train_idx, new_idx])
        remaining_idx = list(set(remaining_idx) - set(train_idx))
        data = convert_to_train_loader(X_all[train_idx], y_all[train_idx])
        model.fit(data, EPOCHS=100)
        torch.save(model.state_dict(), local_path + "/model")
        np.save(local_path + "/marker.npy", markers)
        np.save(local_path + "/label.npy", y_all)
        data_loader = convert_to_test_loader(X_all)

        feature, probs, entropies, predicts = model.get_all_features(y_all, data_loader)
        np.save(local_path + "/feature.npy", feature)
        np.save(local_path + "/entropies.npy", entropies)
        np.save(local_path + "/probs.npy", probs)
        np.save(local_path + "/predict.npy", predicts)
        from sklearn.metrics import confusion_matrix, classification_report

        print(confusion_matrix(y_all, predicts))
        print(classification_report(y_all, predicts, digits=3))
        # predicts = model.evaluate(y_all, data_loader)
