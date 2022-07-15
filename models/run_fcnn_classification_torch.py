import numpy as np

dataset = "imdb"
X_train = np.load("./" + dataset + "/train.npy")
y_train = np.load("./" + dataset + "/train_label.npy")


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

    # Input for Validation
    # X_val = X_train[-1000:]
    partial_X_train = X_train[:10000]

    # Labels for validation
    # y_val = y_train[-1000:]
    partial_y_train = y_train[:10000]

    print(type(partial_X_train))
    print(partial_X_train.shape)
    data = convert_to_train_loader(partial_X_train, partial_y_train)
    model.fit(data, EPOCHS=20)

    X_test = np.load("./" + dataset + "/test.npy", mmap_mode='r')
    y_test = np.load("./" + dataset + "/test_label.npy", mmap_mode='r')

    X_test = X_test[:1000]
    y_test = y_test[:1000]
    test_data_loader = convert_to_test_loader(X_test)
    feature = model.get_feature(test_data_loader)
    np.save("./" + dataset + "/feature.npy", feature)
    predicts = model.evaluate(y_test, test_data_loader)
    error = model.get_error_arr(y_test, test_data_loader)
    np.save("./" + dataset + "/error.npy", error)
