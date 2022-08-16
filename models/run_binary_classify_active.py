import numpy as np
import os
import argparse
import torch
from models.torch_models import BinaryClassification, convert_to_train_loader, convert_to_test_loader

parser = argparse.ArgumentParser("Active Learning Arguments")
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--n_init_labeled', type=int, default=20, help="number of init labeled samples")
parser.add_argument('--n_query', type=int, default=20, help="number of queries per round")
parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
parser.add_argument('--dataset_name', type=str, default="IMDB", choices=["IMDB"], help="dataset")
parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                    choices=["RandomSampling",
                             "LeastConfidence",
                             "MarginSampling",
                             "EntropySampling",
                             "LeastConfidenceDropout",
                             "MarginSamplingDropout",
                             "EntropySamplingDropout",
                             "KMeansSampling",
                             "KCenterGreedy",
                             "BALDDropout",
                             "AdversarialBIM",
                             "AdversarialDeepFool"], help="query strategy")

args = parser.parse_args()
dataset = "imdb"
X_all = np.load("./" + dataset + "/train.npy")
y_all = np.load("./" + dataset + "/train_label.npy")


def check_gpu():
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))
    print("Available: %fGB" % (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024))


if __name__ == "__main__":
    runId = 2
    np.random.seed(runId + 123456)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BinaryClassification(len(X_all[0]), device)
    model.to(device)
    print(model)
    N = 2000
    X_all = X_all[:N]
    y_all = y_all[:N]
    markers = np.zeros(N)
    remaining_idx = range(N)
    train_idx = np.random.choice(remaining_idx, size=args.n_init_labeled, replace=False)
    for id in train_idx:
        markers[id] = 1
    remaining_idx = list(set(remaining_idx) - set(train_idx))
    test_all_loader = convert_to_test_loader(X_all)
    train_loader = convert_to_train_loader(X_all[train_idx], y_all[train_idx])
    model.fit(train_loader, EPOCHS=200)
    config_path = "/" + args.strategy_name + "0" + str(runId)
    main_path = "./" + dataset + config_path
    with open(main_path + "/config.txt", "w") as f:
        f.write(repr(args))
    local_path = main_path + "/" + str(0)
    np.save(local_path + "/marker.npy", markers)
    np.save(local_path + "/label.npy", y_all)
    feature, probs, entropies, predicts = model.get_all_features(y_all, test_all_loader)
    np.save(local_path + "/feature.npy", feature)
    np.save(local_path + "/entropies.npy", entropies)
    np.save(local_path + "/probs.npy", probs)
    np.save(local_path + "/predict.npy", predicts)
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    for t in range(1, args.n_round + 1):
        local_path = main_path + "/" + str(t)
        if not os.path.exists(local_path):
            os.mkdir(local_path)
        if args.strategy_name == "RandomSampling":
            new_idx = np.random.choice(remaining_idx, size=args.n_query, replace=False)
        else:
            # elif args.strategy_name == "entropy":
            order_idx = np.flipud(np.argsort(entropies))
            new_idx = []
            for id in order_idx:
                if id in remaining_idx:
                    new_idx.append(id)
                    if len(new_idx) == args.n_query:
                        break
        for id in new_idx:
            markers[id] = 1
        train_idx = np.concatenate([train_idx, new_idx])
        remaining_idx = list(set(remaining_idx) - set(train_idx))
        train_loader = convert_to_train_loader(X_all[train_idx], y_all[train_idx])
        model.fit(train_loader, EPOCHS=200)
        torch.save(model.state_dict(), local_path + "/model")
        np.save(local_path + "/marker.npy", markers)
        np.save(local_path + "/label.npy", y_all)

        feature, probs, entropies, predicts = model.get_all_features(y_all, test_all_loader)
        np.save(local_path + "/feature.npy", feature)
        np.save(local_path + "/entropies.npy", entropies)
        np.save(local_path + "/probs.npy", probs)
        np.save(local_path + "/predict.npy", predicts)
        from sklearn.metrics import confusion_matrix, classification_report

        print(confusion_matrix(y_all, predicts))
        print(classification_report(y_all, predicts, digits=3))
        # predicts = model.evaluate(y_all, data_loader)
