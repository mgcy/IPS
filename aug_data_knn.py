import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


def euli_distance(a, b):
    return LA.norm(a - b)


def find_neig_mean(k, train_m, test_v, train_all):
    diff_m = train_m - test_v
    err = LA.norm(diff_m, axis=1)

    idx = np.argpartition(err, k)
    select_samples = train_all[idx[:k], :]
    return np.mean(select_samples, 0)


train_X = np.load('train_X.npy', allow_pickle=True)
train_X = train_X.astype(int)
test_X = np.load('test_X.npy', allow_pickle=True)
test_aug_X = np.load('test_X.npy', allow_pickle=True)

for i in range(15):
    for j in range(260):
        miss_index = np.where(test_aug_X[i, j] == -100.0)[0]
        tmp_train = np.delete(train_X, miss_index, 1)
        tmp_test = np.delete(test_aug_X[i, j], miss_index)
        tmp_redeem = find_neig_mean(2, tmp_train, tmp_test, train_X)
        for k in range(131):
            if test_aug_X[i, j, k] == -100.0:
                test_aug_X[i, j, k] = tmp_redeem[k]

np.save('test_aug_KNN_X', test_aug_X)

plt.imshow(test_X[11].T + 100, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
plt.title("Month 12 Before")

plt.show()

plt.imshow(test_aug_X[11].T + 100, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
plt.title("Month 12 After")

plt.show()
