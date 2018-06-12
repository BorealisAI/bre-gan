# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import os
from PIL import Image
import numpy as np
import sys
from scipy.io import loadmat, savemat
import glob

# path should be sth like '~/cifar-10-batches-py'
data_dir = sys.argv[1]

files = sorted(glob.glob(os.path.join(data_dir, 'data_batch_*.mat')))

DEBUG = 0
all_x = []
all_y = []
for cid, f in enumerate(files):
    S = loadmat(f)
    all_x.append(S['data'])
    all_y.append(S['labels'])

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

if os.path.exists('../data/cifar10/'):
    os.makedirs('../data/cifar10/')

savemat('../data/cifar10/cifar10_train.mat', {'X': all_x, 'y': all_y})

files = glob.glob(os.path.join(data_dir, 'test_batch*.mat'))

DEBUG = 0
all_x = []
all_y = []
for cid, f in enumerate(files):
    S = loadmat(f)
    all_x.append(S['data'])
    all_y.append(S['labels'])

all_x = np.concatenate(all_x)
all_y = np.concatenate(all_y)

savemat('../data/cifar10_test.mat', {'X': all_x, 'y': all_y})

