# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


from collections import OrderedDict
import lasagne.layers as ll
import lasagne.updates as lu
import theano
import theano.tensor as T
from utils import grad_penalty, safe_zip, bre
from utils.gan_utils import build_costs
import lasagne
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from lasagne.regularization import regularize_layer_params
import gan_models

floatX = np.float32


class GanModel(object):

    def __init__(self, gan_model_name, *args, **kwargs):

        self.model_func = getattr(gan_models, gan_model_name)

        self.logging = kwargs.pop('logging')
        self.kwargs = kwargs.pop('opt_kwargs')
        self.batch_size = kwargs['batch_size']
        self.gan_mode = kwargs['gan_mode']

        d_bn_mode = kwargs['d_bn']
        g_bn_mode = kwargs['g_bn']

        d_dp_p = kwargs['d_dp']
        g_dp_p = kwargs['g_dp']

        if kwargs['d_bn'] in ('bn', 'ln'):
            def d_bn(x): return gan_models.batch_norm(
                x, steal_nl=0, axes=d_bn_mode)
        else:
            self.logging.info('no D normalization')

            def d_bn(x): return x

        if kwargs['g_bn'] in ('bn', 'ln'):
            def g_bn(x): return gan_models.batch_norm(
                x, steal_nl=0, axes=g_bn_mode)
        else:
            self.logging.info('no G normalization')

            def g_bn(x): return x

        if floatX(kwargs['d_dp']) != 0.:
            def d_dp(x): return ll.DropoutLayer(x, floatX(d_dp_p))
        else:
            d_dp = None

        if floatX(kwargs['g_dp']) != 0.:
            def g_dp(x): return ll.DropoutLayer(x, floatX(g_dp_p))
        else:
            g_dp = None

        kwargs['d_bn'] = d_bn
        kwargs['g_bn'] = g_bn
        kwargs['d_dp'] = d_dp
        kwargs['g_dp'] = g_dp

        self.l_out_g, self.l_out_d, self.l_data, self.g_layers, self.d_layers = \
            self.model_func(*args, **kwargs)

        rng_data = np.random.RandomState()
        rng = np.random.RandomState()
        self.theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
        lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))

    def get_g_outputs(self):

        self.fake_dat = ll.get_output(self.l_out_g, deterministic=False)

        self.constant_z = T.constant(np.random.randn(
            *self.g_layers['l_z'].input_var.eval().shape).astype(floatX))
        self.fake_dat_cz = ll.get_output(
            self.l_out_g, self.constant_z, deterministic=False)

        return self.fake_dat, self.fake_dat_cz

    def get_d_outputs(self, real_dat=None, fake_dat=None):
        kwargs = self.kwargs

        def df_dx(last_ll, dat):

            gradients = T.grad(T.sum(last_ll, axis=0).squeeze(), dat)
            slopes = T.sqrt(T.sum(gradients**2, axis=(1, 2, 3)))

            return gradients, slopes

        if real_dat is None:
            real_dat = self.real_dat = T.tensor4()

        if fake_dat is None:
            fake_dat = self.fake_dat

        self.alpha = self.theano_rng.uniform(
            (self.batch_size, 1, 1, 1), low=0.0, high=1.0)

        differences = real_dat - fake_dat
        self.interp_dat = interp_dat = real_dat + (self.alpha * differences)

        interp_evaled_layers = fake_evaled_layers = fake_cz_evaled_layers = real_evaled_layers = [
            self.l_out_d]

        preact_layers = [self.d_layers[d]
                         for d in self.d_layers.keys() if d.startswith('preact')]

        if 'last_linear' in self.d_layers:
            interp_evaled_layers.append(self.d_layers['last_linear'])
            fake_evaled_layers.append(self.d_layers['last_linear'])
            real_evaled_layers.append(self.d_layers['last_linear'])
            fake_cz_evaled_layers.append(self.d_layers['last_linear'])

        interp_evaled_layers.extend(preact_layers)
        fake_evaled_layers.extend(preact_layers)
        fake_cz_evaled_layers.extend(preact_layers)
        real_evaled_layers.extend(preact_layers)

        interp_evaled = ll.get_output(
            interp_evaled_layers, self.interp_dat, deterministic=False)
        fake_evaled = ll.get_output(
            fake_evaled_layers, self.fake_dat, deterministic=False)
        fake_cz_evaled = ll.get_output(
            fake_cz_evaled_layers, self.fake_dat_cz, deterministic=False)
        real_evaled = ll.get_output(
            real_evaled_layers, self.real_dat, deterministic=False)

        real_l2v = OrderedDict(safe_zip(real_evaled_layers, real_evaled))
        fake_l2v = OrderedDict(safe_zip(fake_evaled_layers, fake_evaled))
        fake_cz_l2v = OrderedDict(
            safe_zip(fake_cz_evaled_layers, fake_cz_evaled))
        interp_l2v = OrderedDict(safe_zip(interp_evaled_layers, interp_evaled))

        output_gen = fake_l2v[self.l_out_d]
        output_data = real_l2v[self.l_out_d]

        gradients_fake, slopes_fake = df_dx(
            fake_l2v[self.d_layers['last_linear']], self.fake_dat)
        gradients_real, slopes_real = df_dx(
            real_l2v[self.d_layers['last_linear']], self.real_dat)
        gradients_interp, slopes_interp = df_dx(
            interp_l2v[self.d_layers['last_linear']], self.interp_dat)

        gradient_penalty_real = grad_penalty(
            slopes_real, floatX(kwargs['gp_slope']))
        gradient_penalty_fake = grad_penalty(
            slopes_fake, floatX(kwargs['gp_slope']))
        gradient_penalty_interp = grad_penalty(
            slopes_interp, floatX(kwargs['gp_slope']))

        # loss terms
        self.d_cost_adv, self.g_cost_adv = build_costs(
            output_gen, output_data, None, None, self.gan_mode)
        d_cost = self.d_cost_adv
        g_cost = self.g_cost_adv

        if floatX(kwargs['smoothness']):
            smoothness_cost = 0.
            for pl in preact_layers:
                mm = self.theano_rng.binomial(
                    (self.batch_size,) + pl.input_shape[1:],
                    p=.5, dtype='float32')

                zz = self.theano_rng.uniform(
                    (1,) + pl.input_shape[1:],
                    dtype='float32')

                zz = zz * mm
                pl_zh = pl.get_output_for(zz)

                print zz.eval().shape, pl_zh.eval().shape
                smoothness_cost += w_smoothness(pl_zh, self.batch_size)
                # import ipdb
                # ipdb.set_trace()

            d_cost += floatX(kwargs['smoothness']) * smoothness_cost

        if floatX(kwargs['gp_weight']):
            self.logging.info('gp with weight:' + kwargs['gp_weight'])

            d_cost += floatX(kwargs['gp_weight']) * gradient_penalty_interp

        h_layers_fake = [fake_l2v[l] for l in preact_layers]
        h_layers_interp = [interp_l2v[l] for l in preact_layers]
        h_layers_real = [real_l2v[l] for l in preact_layers]

        bre_real, me_real, ac_real, _, ac_stats_real = bre(h_layers_real,
                                                           binarizer=kwargs['binarizer'])

        bre_fake, me_fake, ac_fake, _, ac_stats_fake = bre(h_layers_fake,
                                                           binarizer=kwargs['binarizer'])

        bre_interp, me_interp, ac_interp, _, ac_stats_interp = bre(h_layers_interp,
                                                                   binarizer=kwargs['binarizer'])

        self.bre_w = bre_w = theano.shared(floatX(kwargs['bre_w']))

        if floatX(kwargs['bre_w']):

            bre_loss = 0.
            if floatX(kwargs['bre_on_real']):
                self.logging.info('BRE regularization on real')
                bre_loss += bre_w * bre_real

            if floatX(kwargs['bre_on_fake']):
                self.logging.info('BRE regularization on fake')
                bre_loss += bre_w * bre_fake

            if floatX(kwargs['bre_on_interp']):
                self.logging.info('BRE regularization on interp')
                bre_loss += bre_w * bre_interp

            d_cost += bre_loss

        if floatX(kwargs['monitor']):

            ac_min_fake, ac_mean_fake, ac_max_fake, \
                ac_abs_mean_fake, ac_sat_ratio9_fake, ac_sat_ratio9_fake = ac_stats_fake

            ac_min_interp, ac_mean_interp, ac_max_interp, \
                ac_abs_mean_interp, ac_sat_ratio9_interp, ac_sat_ratio9_interp = ac_stats_interp

            ac_min_real, ac_mean_real, ac_max_real, \
                ac_abs_mean_real, ac_sat_ratio9_real, ac_sat_ratio9_real = ac_stats_real

        all_g_layers = ll.get_all_layers(self.g_layers.values())
        all_d_layers = ll.get_all_layers(self.d_layers.values())

        gen_wdecay = regularize_layer_params(
            all_g_layers, lasagne.regularization.l2)
        disc_wdecay = regularize_layer_params(
            all_d_layers, lasagne.regularization.l2)

        d_cost += floatX(kwargs['d_wdecay']) * disc_wdecay
        g_cost += floatX(kwargs['g_wdecay']) * gen_wdecay

        self.d_cost = d_cost
        self.g_cost = g_cost

        err_data = T.cast(output_data < .5, 'float32').mean()
        err_gen = T.cast(output_gen > .5, 'float32').mean()

        # monitor

        if floatX(kwargs['monitor']):
            monitor_stats = []
            monitor_stats += [err_data, err_gen]

            monitor_stats += [me_fake, me_interp, me_real]
            # absh_mu_fake, absh_mu_interp, absh_mu_real]

            monitor_stats += [gradients_fake,
                              slopes_fake, gradient_penalty_fake]
            monitor_stats += [gradients_real,
                              slopes_real, gradient_penalty_real]
            monitor_stats += [gradients_interp,
                              slopes_interp, gradient_penalty_interp]

            monitor_stats += [ac_fake, ac_real, ac_interp]

            if floatX(kwargs['smoothness']):
                monitor_stats += [smoothness_cost]

            if ac_stats_real:
                monitor_stats += [ac_min_real, ac_mean_real, ac_max_real,
                                  ac_abs_mean_real, ac_sat_ratio9_real, ac_sat_ratio9_real]
            if ac_stats_fake:
                monitor_stats += [ac_min_fake, ac_mean_fake, ac_max_fake,
                                  ac_abs_mean_fake, ac_sat_ratio9_fake, ac_sat_ratio9_fake]
            if ac_stats_interp:
                monitor_stats += [ac_min_interp, ac_mean_interp, ac_max_interp,
                                  ac_abs_mean_interp, ac_sat_ratio9_interp, ac_sat_ratio9_interp]

            _vars = locals()

            def get_name(v):
                for k in _vars:
                    if _vars[k] is v and k != 'v' and k != 'k':
                        return k

            self.monitor_stats = OrderedDict(
                [(get_name(v), v) for v in monitor_stats])

    def build_funcs(self):

        self.get_g_outputs()
        self.get_d_outputs()

        kwargs = self.kwargs

        d_trainable_params = ll.get_all_params(
            ll.get_all_layers(self.d_layers.values()), trainable=True)
        g_trainable_params = ll.get_all_params(
            ll.get_all_layers(self.g_layers.values()), trainable=True)

        self.d_trainable_params = d_trainable_params
        self.g_trainable_params = g_trainable_params

        all_d_params = ll.get_all_params(
            ll.get_all_layers(self.d_layers.values()))
        all_g_params = ll.get_all_params(
            ll.get_all_layers(self.g_layers.values()))

        self.all_d_params = all_d_params
        self.all_g_params = all_g_params

        self.g_sh_lr = theano.shared(
            lasagne.utils.floatX(floatX(kwargs['g_lr'])))
        self.d_sh_lr = theano.shared(
            lasagne.utils.floatX(floatX(kwargs['d_lr'])))

        d_beta1 = floatX(kwargs['d_beta1'])
        g_beta1 = floatX(kwargs['g_beta1'])
        beta2 = floatX(kwargs['beta2'])

        d_updater = lu.adam(self.d_cost, d_trainable_params,
                            self.d_sh_lr, beta1=d_beta1, beta2=beta2)
        g_updater = lu.adam(self.g_cost, g_trainable_params,
                            self.g_sh_lr, beta1=g_beta1, beta2=beta2)

        self.out_var2name = out_var2name = OrderedDict([])

        out_var2name[self.g_cost] = 'g_cost_tot'
        out_var2name[self.g_cost_adv] = 'g_cost_adv'

        out_var2name[self.d_cost] = 'd_cost_tot'
        out_var2name[self.d_cost_adv] = 'd_cost_adv'

        # 0: no, 1: every eval interval, 2: every step
        if int(self.kwargs.get('monitor', '0')) >= 2:
            for k in self.monitor_stats:
                out_var2name[self.monitor_stats[k]] = k
            self.g_inputs = g_inputs = [self.real_dat]
        else:
            self.g_inputs = g_inputs = []

        all_g_updates = g_updater
        all_d_updates = d_updater

        self.g_outs = [self.g_cost, self.g_cost_adv]
        self.g_train_func = theano.function(inputs=g_inputs,
                                            outputs=self.g_outs,
                                            updates=all_g_updates)

        self.d_outs = [self.d_cost, self.d_cost_adv]
        self.d_train_func = theano.function(inputs=[self.real_dat],
                                            outputs=self.d_outs,
                                            updates=all_d_updates)

        self.g_sample_func = theano.function(
            inputs=[], outputs=(self.fake_dat+1.)/2.)

        self.g_sample_cz_func = theano.function(
            inputs=[], outputs=(self.fake_dat_cz+1.)/2.)

        if int(kwargs.get('monitor', '0')) >= 1:
            self.monitor_func = theano.function(inputs=[self.real_dat],
                                                outputs=self.monitor_stats.values())

    def g_sample(self, N_samples):
        g_samples = []

        while sum(x.shape[0] for x in g_samples) <= N_samples:
            success = False
            while not success:
                try:
                    g_data_v = self.g_sample_func()
                    success = True
                except Exception as err:
                    self.logging.info(err)

            g_samples.append(g_data_v)

        g_samples = np.vstack(g_samples)[:N_samples]
        return g_samples

