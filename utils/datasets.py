# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import numpy as np


def gen_mog(n=40000, sigma=.05):
    thetas = np.arange(0, 2*np.pi, np.pi / 4)
    x = np.cos(thetas)
    y = np.sin(thetas)
    centres = np.hstack([x[:, None], y[:, None]])

    c_inds = np.random.choice(len(centres), n, replace=True)
    zs = np.random.randn(n, centres.shape[1])
    return centres[c_inds] + zs * sigma


import os


def get_celeba(data_dir='../data', data_path='celeba_aligned/celeba_64.pkl.gz'):
    path = os.path.join(data_dir, data_path)
    from joblib import load
    data = load(path).astype(np.float32)
    data *= (2. / 255.)
    data -= 1.
    import gc
    gc.collect()
    return data


gen_celeba = get_celeba


def get_cifar10(data_dir='../data', data_path='cifar10/cifar10_train.mat'):
    path = os.path.join(data_dir, data_path)

    from scipy.io import loadmat
    Strain = loadmat(path)
    # Stest=loadmat(test_path)

    train_x = Strain['X']
    train_y = Strain['y']
    # test_x = Stest['X']
    # test_y = Stest['y']

    train_x = train_x.astype(np.float32)
    import gc
    gc.collect()
    train_x *= (2. / 255.)
    train_x -= 1.
    train_x = train_x.reshape((-1, 3, 32, 32)).astype(np.float32)
    import gc
    gc.collect()

    return train_x


def get_cifar10_all(data_dir='../data', train_path='cifar10/cifar10_train.mat', test_path='cifar10/cifar10_test.mat'):
    train_path = os.path.join(data_dir, train_path)
    test_path = os.path.join(data_dir, test_path)

    from scipy.io import loadmat
    Strain = loadmat(train_path)
    Stest = loadmat(test_path)

    train_x = Strain['X']
    train_y = Strain['y'].squeeze().astype(np.int8)
    test_x = Stest['X']
    test_y = Stest['y'].squeeze().astype(np.int8)

    train_x = train_x.astype(np.float32)
    import gc
    gc.collect()
    train_x *= (2. / 255.)
    train_x -= 1.
    train_x = train_x.reshape((-1, 3, 32, 32)).astype(np.float32)
    import gc
    gc.collect()

    test_x = test_x.astype(np.float32)
    import gc
    gc.collect()
    test_x *= (2. / 255.)
    test_x -= 1.
    test_x = test_x.reshape((-1, 3, 32, 32)).astype(np.float32)

    return train_x, train_y, test_x, test_y


gen_cifar10 = get_cifar10
gen_cifar10_all = get_cifar10_all


def sample(raw_data, m):
    c_inds = np.random.choice(len(raw_data), m)
    return raw_data[c_inds]

