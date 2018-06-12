# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import sys
import time
import os
import theano
import lasagne
import numpy as np
import subprocess
from collections import OrderedDict, defaultdict
import gc

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# theano.config.profile = True
# theano.config.profile_memory = True

from utils.datasets import gen_cifar10, gen_celeba

import lasagne.layers as ll
from joblib import dump, load

from utils import batch, registery_factory, \
    dump_params, set_default, dict2hash, find_last_saved, \
    lin_anneal, get_network_str, tile_and_save_samples


from models import GanModel
import logging

sys.setrecursionlimit(10000)

floatX = np.float32
FORCE_NO_RELOAD = 1
############################################################


def config_logging(filename='example.log', format="%(message)s"):
    logging.basicConfig(
        filename=filename,
        format=format,
        level=logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    root.addHandler(ch)


def parse_config():
    cml_kwargs = dict([tuple(kv.split(':')) for kv in sys.argv[1:]])

    kwargs = defaultdict(bool)

    # Default settings
    set_default(kwargs, 'name_str', 'gan')

    set_default(kwargs, 'data_name', 'cifar10')
    set_default(kwargs, 'batch_size', '64')
    set_default(kwargs, 'z_dim', '100')

    set_default(kwargs, 'gan_mode', 'js-gan')
    set_default(kwargs, 'bre_w', '1.')

    set_default(kwargs, 'bre_on_fake', '1')
    set_default(kwargs, 'bre_on_interp', '1')
    set_default(kwargs, 'bre_on_real', '0')

    set_default(kwargs, 'bre_loc_mode', 'lfl1')

    set_default(kwargs, 'gp_weight', '0.')
    set_default(kwargs, 'gp_slope', '1.')

    set_default(kwargs, 'model_func', 'dcgan')

    set_default(kwargs, 'g_bn', 'bn')
    set_default(kwargs, 'd_bn', 'bn')

    set_default(kwargs, 'd_dp', '0')
    set_default(kwargs, 'g_dp', '0')

    set_default(kwargs, 'g_width_factor', '4')
    set_default(kwargs, 'g_depth_factor', '4')
    set_default(kwargs, 'd_width_factor', '4')
    set_default(kwargs, 'd_depth_factor', '4')

    set_default(kwargs, 'd_wdecay', '1e-4')
    set_default(kwargs, 'g_wdecay', '1e-4')

    set_default(kwargs, 'binarizer', 'softsign')

    set_default(kwargs, 'g_lr', '.0002')
    set_default(kwargs, 'd_lr', '.0002')

    set_default(kwargs, 'min_g_lr', str(1e-6))
    set_default(kwargs, 'min_d_lr', str(1e-6))

    set_default(kwargs, 'max_iteration', '40000')

    set_default(kwargs, 'd_beta1', '.5')
    set_default(kwargs, 'g_beta1', '.5')
    set_default(kwargs, 'beta2', '.999')

    set_default(kwargs, 'n_d_steps', '1')
    set_default(kwargs, 'n_g_steps', '1')

    set_default(kwargs, 'eval_iter', '1000')
    set_default(kwargs, 'save_iter', '1000')

    set_default(kwargs, 'outdir', './results/')
    set_default(kwargs, 'data_dir', './data/')

    set_default(kwargs, 'monitor', '1')

    for k in cml_kwargs:
        if k not in kwargs:
            raise ValueError('option {} not recognized'.format(k))

    kwargs.update(cml_kwargs)
    #kwargs = cml_kwargs

    try:
        label = subprocess.check_output(
            ["git", "describe", "--always"]).strip()
    except:
        label = None

    kwargs['git_label'] = label
    setting_hash_code = dict2hash(kwargs)
    return kwargs, setting_hash_code

##############################################################


if __name__ == '__main__':

    kwargs, setting_hash_code = parse_config()

    fname = '{}_{}_{}'.format(kwargs['name_str'],
                              kwargs['data_name'],
                              setting_hash_code)

    data_dir = os.path.expanduser(kwargs['data_dir'])
    outdir = os.path.expanduser(kwargs['outdir'])
    params_outdir = outdir

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    config_logging(filename=os.path.join(outdir, fname + '.log'))

    logging.info('====================================')
    for k in sorted(kwargs.keys()):
        logging.info('%s %s' % (k, kwargs[k]))
    logging.info('====================================')

    with open(os.path.join(outdir, fname + '_settings.txt'), 'w') as wf:
        wf.write(' '.join([k+':'+str(kwargs[k])
                           for k in sorted(kwargs.keys())]))

    logging.info(fname)
    ##############################################################
    # find out if reload
    if FORCE_NO_RELOAD:
        g_saved_path = d_saved_path = last_iter = None
    else:
        if 'reload_hash_code' in kwargs:
            reload_hash_code = kwargs['reload_hash_code'].strip()
        else:
            reload_hash_code = setting_hash_code

        logging.info('reloading from ' + reload_hash_code)
        g_saved_path, d_saved_path, last_iter = find_last_saved(
            reload_hash_code, params_outdir)
    ##############################################################

    data_name = kwargs['data_name']
    batch_size = int(kwargs['batch_size'])
    z_dim = int(kwargs['z_dim'])
    gan_mode = kwargs['gan_mode']
    n_d_steps = int(kwargs['n_d_steps'])
    n_g_steps = int(kwargs['n_g_steps'])
    max_iteration = int(kwargs['max_iteration'])

    all_data = locals()['gen_'+data_name](data_dir=data_dir)
    img_size = all_data.shape[-1]

    logging.info("total train data: %s", len(all_data))

    ##############################################################

    gan_model = GanModel(kwargs['model_func'], img_size=img_size, z_dim=z_dim,
                         batch_size=batch_size, gan_mode=gan_mode,
                         d_bn=kwargs['d_bn'], g_bn=kwargs['g_bn'],
                         d_dp=kwargs['d_dp'], g_dp=kwargs['g_dp'],
                         g_wf=floatX(kwargs['g_width_factor']),
                         d_wf=floatX(kwargs['d_width_factor']),
                         g_df=int(kwargs['g_depth_factor']),
                         d_df=int(kwargs['d_depth_factor']),
                         bre_loc_mode=kwargs['bre_loc_mode'],
                         opt_kwargs=kwargs,
                         logging=logging)

    all_g_layers = ll.get_all_layers(gan_model.g_layers.values())
    all_d_layers = ll.get_all_layers(gan_model.d_layers.values())

    glayer2name = defaultdict(str)
    dlayer2name = defaultdict(str)
    logging.info('~~~~~~~~~~~~~ G model ~~~~~~~~~~~~~~~~~~~~~~')
    glayer2name.update({v: k for k, v in gan_model.g_layers.iteritems()})
    logging.info(get_network_str(all_g_layers, get_network=False,
                                 incomings=True, outgoings=True,
                                 layer2name=glayer2name))

    dlayer2name.update(({v: k for k, v in gan_model.d_layers.iteritems()}))
    logging.info('G total trainable params: %g' %
                 (ll.count_params(gan_model.l_out_g, trainable=True)))
    logging.info('~~~~~~~~~~~~~ D model ~~~~~~~~~~~~~~~~~~~~~~')
    logging.info(get_network_str(all_d_layers, get_network=False,
                                 incomings=True, outgoings=True,
                                 layer2name=dlayer2name))

    logging.info('D total trainable params: %g' %
                 (ll.count_params(gan_model.l_out_d, trainable=True)))

    ##############################################################

    gan_model.build_funcs()
    data_itr = batch(all_data, batch_size)

    tic = None
    hist = defaultdict(list)
    plot_data = []
    d_train_outs_dict = defaultdict(list)
    g_train_outs_dict = defaultdict(list)
    start_iteration = 0  # will start at -1, which shows G at initialization

    for iteration in xrange(start_iteration-1, max_iteration):

        lin_anneal(gan_model.g_sh_lr, floatX(kwargs['g_lr']),
                   max_iteration, min_var=floatX(kwargs['min_g_lr']))

        lin_anneal(gan_model.d_sh_lr, floatX(kwargs['d_lr']),
                   max_iteration, min_var=floatX(kwargs['min_d_lr']))

        if iteration >= 0:

            for _ in xrange(n_d_steps):
                try:
                    real_data_batch = next(data_itr)

                except StopIteration:
                    np.random.shuffle(all_data)
                    all_data = np.ascontiguousarray(all_data)
                    data_itr = batch(all_data, batch_size)
                    real_data_batch = next(data_itr)

                d_cost_values = gan_model.d_train_func(real_data_batch)
                d_cost_v = d_cost_values[0]

                for k, v in zip(gan_model.d_outs, d_cost_values):
                    d_train_outs_dict[gan_model.out_var2name[k]].append(v)

            for _ in xrange(n_g_steps):
                if len(gan_model.g_inputs):
                    g_cost_values = gan_model.g_train_func(real_data_batch)
                else:
                    g_cost_values = gan_model.g_train_func()

                g_cost_v = g_cost_values[0]

                for k, v in zip(gan_model.g_outs, g_cost_values):
                    g_train_outs_dict[gan_model.out_var2name[k]].append(v)

        else:
            real_data_batch = next(data_itr)
            d_cost_v = np.nan
            g_cost_v = np.nan

        ##############################################################
        # monitoring
        if (iteration < 0 or
            iteration % int(kwargs['eval_iter']) == 0 or
                iteration == max_iteration-1):

            logging.info(
                '~~~~~~~~~~~~ Iteration %s ~~~~~~~~~~~~~~~~', iteration)
            logging.info(fname)

            logging.info('epoch %s', iteration * batch_size *
                         n_d_steps // len(all_data))
            logging.info('g lr: %s', gan_model.g_sh_lr.get_value())
            logging.info('d lr: %s', gan_model.d_sh_lr.get_value())

            if int(kwargs.get('monitor', '0')) >= 1:
                gm_outs = gan_model.monitor_func(real_data_batch)

                assert len(gan_model.monitor_stats) == len(gm_outs)
                for k, v in zip(gan_model.monitor_stats.keys(), gm_outs):
                    hist[k].append(v)

            if tic:
                elapsed_time = time.time() - tic
            else:
                elapsed_time = 0.

            hist['elapsed_time'].append(elapsed_time)
            hist['epoch'].append(iteration)
            hist['g_cost'].append(g_cost_v)
            hist['d_cost'].append(d_cost_v)

            for k in sorted(hist.keys()):
                if isinstance(hist[k][-1], np.ndarray) and hist[k][-1].size > 1:
                    logging.info('%s_avg %s', k, '%5.3f' %
                                 (hist[k][-1]).mean())
                else:
                    logging.info('%s %s', k, '%5.3f' % hist[k][-1])
            logging.info('')

            tic = time.time()

            # ~~~~~~~~~~~~~~~~~~~~~~~ sampling and save images
            g_samples = gan_model.g_sample(1024)
            g_samples_cz_v = gan_model.g_sample_cz_func()

            if int(kwargs.get('do_save', '1')):
                _path = os.path.join(
                    outdir, '{}_iter_{}.png'.format(fname, iteration))

                tile_and_save_samples(_path, g_samples)

                _path = os.path.join(
                    outdir, '{}_cz_iter_{}.png'.format(fname, iteration))

                tile_and_save_samples(_path, g_samples_cz_v)

        ##############################################################
        # snapshot
        if (iteration % int(kwargs['save_iter']) == 0
                and int(kwargs.get('do_save', '1'))):

            dump_params(gan_model.all_d_params,
                        os.path.join(params_outdir,
                                     '{}_dparams_iter{}.pkl'.format(fname, iteration)))

            dump_params(gan_model.all_g_params,
                        os.path.join(params_outdir,
                                     '{}_gparams_iter{}.pkl'.format(fname, iteration)))

            dump(hist, os.path.join(outdir, '{}_hist.pkl'.format(fname)),
                 compress=('gzip', 9))

            dump({'g_outs': g_train_outs_dict,
                  'd_outs': d_train_outs_dict},
                 os.path.join(outdir, '{}_outs_dict.pkl'.format(fname)),
                 compress=('gzip', 9))

    gc.collect()
    if int(kwargs.get('do_save', '1')):
        dump_params(gan_model.all_d_params,
                    os.path.join(params_outdir,
                                 '{}_dparams.pkl'.format(fname)))

        dump_params(gan_model.all_g_params,
                    os.path.join(params_outdir,
                                 '{}_gparams.pkl'.format(fname)))

    g_samples = (gan_model.g_sample(len(all_data)) * 255).astype(np.uint8)

    sample_path = os.path.join(outdir,
                               '{}_iter_{}_N_{}.npz'.format(fname,
                                                            iteration,
                                                            len(all_data)))

    np.savez_compressed(sample_path, g_samples)

    logging.info(
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logging.info('final G generated data:' + sample_path)

    del all_data
    if data_name == 'cifar10':
        gc.collect()
        inception_score_code_output = subprocess.check_output(
            ["python2", "utils/inc_score_wrapper.py",
             "data_file:" + sample_path]).strip()

        logging.info('Inception score:')
        logging.info(inception_score_code_output)
        logging.info(
            '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    gc.collect()
    fid_score_code_output = subprocess.check_output(
        ["python2", "utils/fid_score_wrapper.py",
         "data_file:"+sample_path]).strip()

    logging.info('FID score:')
    logging.info(fid_score_code_output)
    logging.info(
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    logging.info(setting_hash_code)
    logging.info('COMPLETED!!!')

