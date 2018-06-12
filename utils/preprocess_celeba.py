# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


# adapted from:
# https://github.com/soumith/dcgan.torch/blob/master/data/crop_celebA.lua

import os
from PIL import Image
import numpy as np
import sys

# path should be sth like '~/celeba_aligned/img_align_celeba'
data_dir = sys.argv[1]

files = os.listdir(data_dir)

DEBUG = 0
all_arrs = []
for cid, f in enumerate(files):
    img = Image.open(os.path.join(data_dir, f))

    half_w = img.size[0] / 2
    half_h = img.size[1] / 2

    img_new = img.crop((half_w - 64, half_h - 64, half_w + 64, half_h + 64))
    img.close()
    arr = np.asarray(img_new.resize((64, 64)))

    if DEBUG:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as pp
        print arr.shape
        fig = pp.figure()
        ax = fig.add_subplot(111)
        ax.imshow(arr/255., vmin=0., vmax=1.)
        fig.savefig('./DEBUG_%s.png' % f, format='png')
        import ipdb
        ipdb.set_trace()

    all_arrs.append(arr.transpose((2, 0, 1))[None, ...])

    if cid % 1000 == 0:
        print cid, 'of', len(files)
        import gc
        gc.collect()

all_arrs = np.concatenate(all_arrs)

if os.path.exists('../data/celeba_aligned/'):
    os.makedirs('../data/celeba_aligned/')

from joblib import dump
dump(all_arrs, os.path.join('../data/celeba_aligned/',
                            'celeba_64.pkl.gz'), compress=('gzip', 9))

