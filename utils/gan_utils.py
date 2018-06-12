# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import lasagne.layers as ll
import lasagne.updates as lu
import lasagne.nonlinearities as ln
import theano.tensor as T
import numpy as np
from lasagne.objectives import binary_crossentropy as bce
from theano.gradient import zero_grad
from theano.sandbox.rng_mrg import MRG_RandomStreams

floatX = np.float32


def build_costs(output_gen, output_data,
                output_gen_det, output_data_det,
                gan_mode, **kwargs):

    output_gen_det = output_gen if output_gen_det is None else output_gen_det
    output_data_det = output_data if output_data_det is None else output_data_det

    # D cost
    if gan_mode in ('wgan', ):
        d_cost_gen = output_gen.mean(axis=0)
        d_cost_real = output_data.mean(axis=0)
        d_cost = -(d_cost_real - d_cost_gen).squeeze()

    elif gan_mode in ('js-gan', 'js-gan-no-flip'):

        label_smoothing = kwargs.get('js_label_smoothing', 0.)
        label_smoothing_det = kwargs.get('js_label_smoothing_det', 0)

        if label_smoothing_det:
            lbl_noise_real = np.float32(1. - label_smoothing)
        else:
            if label_smoothing > 0.:

                rng = np.random.RandomState(1111)
                theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
                lbl_noise_real = 1. - label_smoothing * \
                    theano_rng.uniform(
                        size=(output_data.shape[0],)).dimshuffle((0, 'x'))

            else:
                lbl_noise_real = 1.

        d_cost_real = bce(output_data, lbl_noise_real *
                          T.ones(output_data.shape)).mean()
        d_cost_gen = bce(output_gen, T.zeros(output_gen.shape)).mean()
        d_cost = d_cost_real + d_cost_gen

    else:
        raise NotImplementedError(gan_mode)

    # G cost
    if gan_mode in ('wgan', ):
        g_gan_cost = -(output_gen_det.mean(axis=0)).squeeze()

    elif gan_mode in ('js-gan',):
        g_gan_cost = bce(output_gen_det, T.ones(output_gen.shape)).mean()

    elif gan_mode in ('js-gan-no-flip', ):
        g_gan_cost = - bce(output_gen_det, T.zeros(output_gen.shape)).mean()

    else:
        raise NotImplementedError(gan_mode)

    g_cost = g_gan_cost

    return d_cost, g_cost


def critic_output_layer(l_h, all_critics, register_func=None, **kwargs):
    if register_func is None:
        def register_func(x, y): return x

    if isinstance(all_critics, str):
        all_critics = [all_critics]

    l_outs = []
    for critic in all_critics:

        if critic == 'wgan':
            l_out_d = register_func(ll.DenseLayer(
                l_h, num_units=1, nonlinearity=None), 'last_linear')

        elif critic in ('js-gan', 'js-gan-no-flip'):
            l_out_d = register_func(ll.DenseLayer(l_h, num_units=1, nonlinearity=ln.sigmoid),
                                    'last_linear')

            # l_out_d = ll.NonlinearityLayer(l_h, nonlinearity=)

        else:
            raise NotImplementedError(critic)

        l_outs.append(l_out_d)

    if len(l_outs) == 1:
        l_outs = l_outs[0]

    return l_outs


def grad_penalty(slopes, target_slope=1.):
    return T.mean((slopes - floatX(target_slope))**2)


def dim2margin(d, s=3.):
    _base_var = 1.  # .25 is for 0-1 bernoulli, this is +/- bernoulli
    #print 'DEBUG:_base_var', _base_var
    return s*T.sqrt(_base_var / T.cast(d, 'float32'))


from theano.gradient import zero_grad


def tanh(h, *args, **kwargs):
    return T.tanh(h)


def softsign(h, epsilon=1e-3):
    _mu = abs(h).mean()
    h_epsilon = np.float32(epsilon) * _mu
    act = (h / (abs(h) + zero_grad(h_epsilon)))
    return act


def bre(hs, epsilon=1e-3, binarizer='softsign', s=3.):
    me_value, me_stats = me_term(hs, epsilon=epsilon, binarizer=binarizer)
    ac_value, ac_stats = ac_term(hs, epsilon=epsilon, binarizer=binarizer, s=s)
    return me_value + ac_value, me_value, ac_value, me_stats, ac_stats


def me_term(hs, epsilon=1e-3, binarizer='softsign'):
    terms = []
    stats = []
    binarizer = globals()[binarizer]

    for h in hs:
        h = T.flatten(h, 2)
        act = binarizer(h, epsilon)
        stats = [abs(h).mean()]

        terms.append(T.sqr(act.mean(axis=0)).mean())

    return sum(terms) / np.float32(len(hs)), T.mean(stats)


def ac_term(hs, epsilon=1e-3, s=3., binarizer='softsign'):

    C = 0.
    monitor_C = 0.

    monitor_abs_mean = []
    monitor_saturate9_ratio = []
    monitor_saturate99_ratio = []

    binarizer = globals()[binarizer]

    tot_d = 0.
    for h in hs:
        act = binarizer(h, epsilon)

        monitor_abs_mean.append(abs(act).mean())
        monitor_saturate9_ratio.append(
            T.cast(abs(act) > .9, 'float32').mean())
        monitor_saturate99_ratio.append(
            T.cast(abs(act) > .99, 'float32').mean())

        act = T.flatten(act, 2)

        l_C = T.dot(act, act.T) / T.cast(act.shape[1], 'float32')
        monitor_C += l_C

        assert s >= 1.

        C += T.maximum(0.,  abs(l_C) - dim2margin(act.shape[1], s))

    monitor_abs_mean = sum(monitor_abs_mean) / np.float32(len(hs))
    monitor_saturate9_ratio = sum(
        monitor_saturate9_ratio) / np.float32(len(hs))
    monitor_saturate99_ratio = sum(
        monitor_saturate99_ratio) / np.float32(len(hs))

    C /= np.float32(len(hs))
    C -= T.diag(T.diag(C))

    obj_value = C.mean()

    monitor_C /= np.float32(len(hs))
    monitor_Cmin = monitor_C.min(axis=0)
    monitor_C -= T.diag(T.diag(monitor_C))

    monitor_Cmean = monitor_C.sum(axis=0) / T.cast(monitor_C.shape[1]-1,
                                                   'float32')
    monitor_Cmax = monitor_C.max(axis=0)

    monitor_values = [monitor_Cmin, monitor_Cmean, monitor_Cmax,
                      monitor_abs_mean,
                      monitor_saturate9_ratio,
                      monitor_saturate99_ratio]

    return obj_value, monitor_values

