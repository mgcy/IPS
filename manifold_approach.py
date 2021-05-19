from sklearn.manifold import SpectralEmbedding as SE
import numpy as np
from numpy import linalg as LA
from sklearn.linear_model import LinearRegression as LR
from uji import UJI

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def plot_errors_monthly(accuracies, show=True, save=False):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 4))
    for tech in accuracies.keys():
        aes = []
        for month in accuracies[tech].keys():
            ae = np.mean(accuracies[tech][month])
            aes.append(ae)

        plt.plot(aes, label=tech, linewidth=3)
        plt.title(f"Temporal Degradation of Localization Performance", size=18)

    plt.xlabel("Month", size=18)
    plt.ylabel("Accuracy (meters)", size=18)

    # get the list of months
    months = list(accuracies[list(accuracies.keys())[0]].keys())
    plt.xticks(ticks=list(range(len(months))),
               labels=months, size=14)

    plt.yticks(size=14)
    plt.legend(fontsize=14, loc="upper left",
               bbox_to_anchor=(1, 0, 1, 1.03))
    plt.tight_layout()

    if save:
        plt.savefig("plots/motivation.png")

    if show:
        plt.show()


def k_function(a, b, sigma=50):
    # Gaussian function in paper
    return np.exp(-(LA.norm(a - b) ** 2) / sigma)


def new_embedding(u, xN, ye):
    # u: xu, xN: train_X. ye: train_embedding
    # update the unseen data u using Eq. (6) in the paper
    N = len(xN)
    kuj = 0
    for i in range(N):
        kuj = kuj + k_function(xN[i], u)
    re = 0
    for i in range(N):
        re = re + (k_function(u, xN[i])) / kuj * ye[i]

    return re


def normalize_data(a, b):
    # normalized the whole dataset between 0-1 by:
    # data = data - min(data)
    # data = data / max(data)
    min1 = np.min([np.min(a), np.min(b)])
    a = a - min1
    b = b - min1
    max1 = np.max([np.max(a), np.max(b)])
    a = a / max1
    b = b / max1
    return a, b


def manifold_estimate(d):
    train_X = np.load('train_X.npy', allow_pickle=True)
    train_X = train_X + 100
    train_X = train_X.astype(int)

    test_X = np.load('test_X.npy', allow_pickle=True)
    # test_X = np.load('test_aug_X.npy', allow_pickle=True)
    test_X = test_X + 100

    train_Y = np.load('train_Y.npy', allow_pickle=True)

    test_coor = np.load('test_coordinates.npy', allow_pickle=True)
    train_coor = np.load('train_coordinates.npy', allow_pickle=True)

    train_embedding = SE(n_components=d).fit_transform(train_X)

    plt.scatter(train_embedding[:, 0], train_embedding[:, 1], c=train_Y, cmap=plt.cm.Spectral)
    plt.savefig("Manifold_data")

    co1 = LR()
    co1.fit(train_embedding, train_coor)

    uji = UJI.from_cache("test", 1, cache_dir="uji/db_cache")
    acc = {}

    test_embedding = np.zeros((15, 260, d))

    for month in range(15):
        for i in range(260):
            test_embedding[month, i] = new_embedding(test_X[month, i], train_X, train_embedding)

        # for month in range(15):
        test_tmp = test_embedding[month]

        pred_co = co1.predict(test_tmp)

        dists = uji.compute_distances(pred_co, test_coor[month]).flatten()
        acc[month + 1] = dists

    return acc


def main():
    accuracies = {}
    acc2 = manifold_estimate(d=70)
    accuracies['Manifold 70'] = acc2

    acc50 = manifold_estimate(d=50)
    accuracies['Manifold 50'] = acc50

    plot_errors_monthly(accuracies)

    # train_embedding_normal, test_embedding_normal = normalize_data(train_embedding, test_embedding)

    # np.save('train_embedding_X', train_embedding)
    # np.save('train_embedding_aug_KNN_X', train_embedding)
    # np.save('train_embedding_aug_normal_X', train_embedding_normal)

    # np.save('test_embedding_X', test_embedding_normal)
    # np.save('test_embedding_aug_KNN_X', test_embedding)
    # np.save('test_embedding_aug_normal_X', test_embedding_normal)


if __name__ == "__main__":
    main()
