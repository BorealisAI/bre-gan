# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


from joblib import dump, load
import cPickle as pickle
import numpy as np
from scipy.misc import imsave


def dump_params(param_list, fname, use_joblib=True):
    import cPickle as pickle
    params = [(p.name, p.get_value()) for p in param_list]

    if use_joblib:
        dump(params, fname+'.gz', compress=('gzip', 9))
    else:
        with open(fname, 'wb') as fw:
            pickle.dump(params, fw, pickle.HIGHEST_PROTOCOL)


def load_and_set_params(param_list, fname, raise_on_mismatch=False):

    try:
        params = load(fname)
    except:
        with open(fname, 'rb') as fr:
            params = pickle.load(fr)
    #import ipdb; ipdb.set_trace()
    for p, pv in zip(param_list, params):
        if p.get_value().shape == pv[1].shape:  # p.name == pv[0]:
            p.set_value(pv[1])
        else:
            msg = 'value for {} not found'.format(p.name+' id:'+str(id(p)))
            if raise_on_mismatch:
                raise ValueError(msg)
            else:
                print msg


def best_tile_shape(n):
    n_sqrt = int(np.sqrt(n))
    width = n // n_sqrt

    while (n // width) * width != n:
        width -= 1
    return width, n // width


def tile_imgs(imgs, nx, ny):

    N, nc, x, y = imgs.shape
    canvas = np.zeros((nc, nx*x, ny*y), dtype=imgs.dtype)
    assert N == nx*ny
    for kid, img in enumerate(imgs):
        row_idx = kid // ny
        col_idx = kid % ny

        canvas[:, row_idx*y:(row_idx*y+y), col_idx*x:(col_idx*x+x)] = img

    return canvas.transpose((1, 2, 0))


def tile_and_save_samples(path, samples):
    _w, _h = best_tile_shape(len(samples))
    imsave(path, (tile_imgs(samples, _w, _h) * 255).astype(np.uint8))

# def is_finished_based_on_log(base_hash, results_dir):
#     import os
#     files = filter(lambda f: f.endswith('.log') and (base_hash in f),
#                    os.listdir(results_dir))

#     answer = False
#     if len(files):
#         assert len(files) == 1
#         f = files[0]
#         with open(os.path.join(results_dir,f), 'r') as fr:
#             answer = any(line.strip().endswith('COMPLETED!!!') for line in fr.readlines())

#     return answer

# def is_finished(base_hash, results_dir):
#     return (is_finished_based_on_log(base_hash, results_dir) or
#             is_finished_based_on_gz(base_hash, results_dir))


# def is_finished_based_on_gz(base_hash, results_dir,):
#     import os
#     files = filter(lambda f: base_hash in f, os.listdir(results_dir))

#     d_files = [f for f in files if 'dparams' in f]
#     g_files = [f for f in files if 'gparams' in f]

#     #finished?
#     final_d_file = [f for f in d_files if "_iter" not in f]
#     final_g_file = [f for f in g_files if "_iter" not in f]
#     assert len(final_d_file) == len(final_g_file)

#     return len(final_d_file) == 1

# is_finished = is_finished_based_on_log #is_finished_based_on_gz

def find_last_saved(base_hash, results_dir, only_unfinished=True):
    import os
    files = filter(lambda f: base_hash in f, os.listdir(results_dir))

    #hist_file = [f for f in files if f.endswith('_hist.pkl')]
    # assert len(hist_file) == 1
    # hist_file = hist_file[0]

    d_files = [f for f in files if 'dparams' in f]
    g_files = [f for f in files if 'gparams' in f]

    # finished?
    final_d_file = [f for f in d_files if "_iter" not in f]
    final_g_file = [f for f in g_files if "_iter" not in f]
    assert len(final_d_file) == len(final_g_file)

    if len(final_d_file) and not only_unfinished:
        return final_g_file[0], final_d_file[0], None  # , hist_file
    else:
        d_files = [f for f in d_files if "_iter" in f]
        g_files = [f for f in g_files if "_iter" in f]

        iterations = sorted(
            [int(f.split('_iter')[1].split('.')[0]) for f in d_files])
        if len(iterations):
            if len(g_files) < len(d_files):
                # probably failed saving g previously
                last_iter = iterations[-2]
            else:
                last_iter = iterations[-1]

            final_d_file = [f for f in d_files if "_iter"+str(last_iter) in f]
            final_g_file = [f for f in g_files if "_iter"+str(last_iter) in f]

            return final_g_file[0], final_d_file[0], last_iter  # , hist_file
        else:
            return None, None, None  # , None

# if __name__ == '__main__':
#     import sys
#     func = locals()[sys.argv[1]]
#     args = sys.argv[2:]
#     print func(*args)

    #find_last_saved(base_hash, results_dir)

