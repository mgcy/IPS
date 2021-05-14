from sklearn.manifold import SpectralEmbedding as SE
import numpy as np
from numpy import linalg as LA


def k_function(a, b, sigma=50):
    # Gaussian function in paper
    return np.exp(-(LA.norm(a - b) ** 2) / sigma)


def new_embedding(u, xN, ye):
    # u: xu, xN: train_X. ye: train_embedding
    # update the unseen data u using Eq. (6) in the paper
    N = len(train_X)
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


train_X = np.load('train_X.npy', allow_pickle=True)
train_X = train_X + 100
train_X = train_X.astype(int)

test_X = np.load('test_aug_KNN_X.npy', allow_pickle=True)
# test_X = np.load('test_X.npy', allow_pickle=True)
test_X = test_X + 100

d = 60

train_embedding = SE(n_components=d).fit_transform(train_X)

test_embedding = np.zeros((15, 260, d))

for month in range(15):
    for i in range(260):
        test_embedding[month, i] = new_embedding(test_X[month, i], train_X, train_embedding)

train_embedding_normal, test_embedding_normal = normalize_data(train_embedding, test_embedding)

# np.save('train_embedding_X', train_embedding)
# np.save('train_embedding_aug_KNN_X', train_embedding)
np.save('train_embedding_aug_normal_X', train_embedding_normal)

# np.save('test_embedding_X', test_embedding_normal)
# np.save('test_embedding_aug_KNN_X', test_embedding)
np.save('test_embedding_aug_normal_X', test_embedding_normal)
