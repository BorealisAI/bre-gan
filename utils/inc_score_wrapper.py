# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import sys
import numpy as np
from inceptscore import get_inception_preds
from inceptscore import pred2score
from fid import calculate_fid_with_ref_imgs


def fid_with_ref(filename, ref_imgs):
    with np.load(filename) as data:
        images = data['arr_0']

    images = np.rollaxis(images, 1, 4)

    if ref_imgs.shape[1] == 3:
        ref_imgs = np.rollaxis(ref_imgs, 1, 4)

    if ref_imgs.max() <= 1.:
        ref_imgs *= 255.

    fid_value = calculate_fid_with_ref_imgs(images, ref_imgs, None)
    return fid_value


def inception_score(arr, splits=10):
    images = arr
    images = np.rollaxis(images, 1, 4)
    preds = get_inception_preds(list(images))
    avg, std = pred2score(preds, splits=splits)
    return avg, std


def inception_score_from_file(filename, splits=10):
    with np.load(filename) as data:
        arr = data['arr_0']
    results = inception_score(arr)
    print results
    return results


import sys
if __name__ == '__main__':

    kwargs = dict([tuple(kv.split(':')) for kv in sys.argv[1:]])

    if 'data_file' in kwargs:
        result_file = filename = kwargs['data_file']
        avg, std = inception_score_from_file(filename)
        sys.stdout.write(str((avg, std)))
        sys.stdout.flush()
        with open(result_file+'_inception.csv', 'w') as f:
            f.write('{}, {}\n'.format(avg, std))

    else:
        import cPickle as pkl

        if 'names_stdin' in kwargs:
            import sys
            lines = filter(
                None, map(lambda x: x.strip(), sys.stdin.readlines()))
            file_names = sorted(lines, key=lambda s: int(
                s.split('_iter_')[1].split('_')[0]))
            for f in file_names:
                print f
            notes = []

            result_file = 'stdin_tmp'
        elif 'name_file' in kwargs:

            result_file = name_file = kwargs['name_file']

            with open(name_file, 'r') as fr:
                lines = fr.readlines()

            file_names = [l.split()[0] for l in lines]
            notes = [l.split()[1:] for l in lines]

        elif 'hash_file' in kwargs:

            result_file = hash_file = kwargs['hash_file']
            import glob
            with open(hash_file, 'r') as fr:
                lines = fr.readlines()
                hash_names = [l.split()[0] for l in lines]
                notes = [l.split()[1:] for l in lines]

                file_names = []
                for h in hash_names:
                    paths = glob.glob(
                        '../results_grouped/{}/*N*.npz'.format(h))
                    assert len(paths) == 1
                    file_names.append(paths[0])

        # elif 'hash_file_auto_last' in kwargs:

        #     result_file = hash_file = kwargs['hash_file_auto_last']
        #     import glob
        #     with open(hash_file,'r') as fr:
        #         lines = fr.readlines()
        #         hash_names = [l.split()[0] for l in lines]
        #         notes = [l.split()[1:] for l in lines]

        #         file_names = []
        #         for h in hash_names:
        #             paths = glob.glob('../results_grouped/{}/*N*.npz'.format(h))
        #             assert len(paths) == 1
        #             file_names.append(paths[0])

        else:
            raise ValueError()

        scores = [inception_score_from_file(f) for f in file_names]

        if notes:
            assert len(notes) == len(scores)
            results = zip(file_names, scores, notes)
        else:
            results = zip(file_names, scores)

        with open(result_file+'results.pkl', 'wb') as fw:
            pkl.dump(results, fw, pkl.HIGHEST_PROTOCOL)
            print 'results saved to:', result_file+'results.pkl'

        with open(result_file+'results.csv', 'w') as fw:
            if notes:
                fw.write('filename;\t avg;\t std;\t notes \n')
                for l in results:
                    fw.write('{};\t {};\t {};\t {}\n'.format(
                        l[0], l[1][0], l[1][1], l[2]))
            else:
                fw.write('filename;\t avg;\t std \n')
                for l in results:
                    fw.write('{};\t {};\t {}\n'.format(l[0], l[1][0], l[1][1]))

            print 'results saved to:', result_file+'results.csv'

