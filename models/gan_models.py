# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import theano
import numpy as np

import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne.layers as ll
import lasagne.updates as lu
import lasagne.nonlinearities as ln
import lasagne.init as li
import os
from utils import registery_factory
import theano.tensor as T


from collections import OrderedDict
from utils.gan_utils import critic_output_layer

# rng = np.random.RandomState(1111)
# theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
# lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
# from models import build_cnn

_base_hidden_nf = 16


from lasagne import init


class NormalizationLayer(ll.Layer):
    """
    NormalizationLayer(incoming, axes='auto', epsilon=1e-4,
    alpha=0.1, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1),
    mean=lasagne.init.Constant(0), inv_std=lasagne.init.Constant(1), **kwargs)

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its nonlinearity.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """

    def __init__(self, incoming, axes='bn', epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):
        super(NormalizationLayer, self).__init__(incoming, **kwargs)

        if axes == 'bn':
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))

        elif axes == 'ln':
            axes = tuple(range(1, len(self.input_shape)))

        elif isinstance(axes, int):
            axes = (axes,)

        self.axes = axes

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.axes]
        # if any(size is None for size in shape):
        #     raise ValueError("BatchNormalizationLayer needs specified input sizes for "
        #                      "all axes not normalized over.")

        shape = [s if s is not None else 1 for s in shape]

        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)
        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

        # import ipdb; ipdb.set_trace()

    def get_output_for(self, input, deterministic=False,
                       batch_norm_use_averages=False,
                       batch_norm_update_averages=False, **kwargs):
        input_mean = input.mean(self.axes)
        input_inv_std = T.inv(T.sqrt(input.var(self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # import ipdb; ipdb.set_trace()

            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)
        mean = mean.dimshuffle(pattern)
        inv_std = inv_std.dimshuffle(pattern)

        # normalize
        normalized = (input - mean) * (gamma * inv_std) + beta
        return normalized


def batch_norm(layer, steal_nl=True, **kwargs):
    """
    Apply batch normalization to an existing layer. This is a convenience
    function modifying an existing layer to include batch normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`NormalizationLayer` and :class:`NonlinearityLayer` on top.

    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`NormalizationLayer` constructor.

    Returns
    -------
    NormalizationLayer or NonlinearityLayer instance
        A batch normalization layer stacked on the given modified `layer`, or
        a nonlinearity layer stacked on top of both if `layer` was nonlinear.

    Examples
    --------
    Just wrap any layer into a :func:`batch_norm` call on creating it:

    >>> from lasagne.layers import InputLayer, DenseLayer, batch_norm
    >>> from lasagne.nonlinearities import tanh
    >>> l1 = InputLayer((64, 768))
    >>> l2 = batch_norm(DenseLayer(l1, num_units=500, nonlinearity=tanh))

    This introduces batch normalization right before its nonlinearity:

    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'DenseLayer', 'NormalizationLayer', 'NonlinearityLayer']
    """
    if steal_nl:
        nonlinearity = getattr(layer, 'nonlinearity', None)
        if nonlinearity is not None:
            layer.nonlinearity = nonlinearities.identity

    if hasattr(layer, 'b') and layer.b is not None:
        del layer.params[layer.b]
        layer.b = None
    bn_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_bn'))
    layer = NormalizationLayer(layer, name=bn_name, **kwargs)

    if steal_nl:
        if nonlinearity is not None:
            from lasagne.layers import NonlinearityLayer

        nonlin_name = bn_name and bn_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer


class Deconv2DLayer(ll.Layer):
    def __init__(self, incoming, target_shape, filter_size, stride=(2, 2),
                 W=li.GlorotUniform(), b=li.Constant(0.),
                 nonlinearity=ln.rectify, **kwargs):

        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.target_shape = target_shape
        self.nonlinearity = (
            ln.identity if nonlinearity is None else nonlinearity)
        self.filter_size = filter_size if isinstance(
            filter_size, tuple) else (filter_size, filter_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.target_shape = target_shape

        self.W_shape = (
            incoming.output_shape[1], target_shape[1], filter_size[0], filter_size[1])
        self.W = self.add_param(W, self.W_shape, name="W")
        if b is not None:
            self.b = self.add_param(b, (target_shape[1],), name="b")
        else:
            self.b = None

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.target_shape, kshp=self.W_shape,
            subsample=self.stride, border_mode='half')

        activation = op(self.W, input, self.target_shape[2:])

        if self.b is not None:
            activation += self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return self.target_shape


def resnet_g(batch_size, img_size, z_dim=128,
             n_blocks=3, wf=1, block_depth=1,
             g_nl=ln.rectify, fixed_nchan=None,
             g_bn=None, g_dp=None,  **kwargs):

    wf = int(wf)

    if g_bn is None:
        def g_bn(x): return x

    if g_dp is None:
        def g_dp(x): return x

    g_layers = OrderedDict([])
    g_register = registery_factory(g_layers)

    deconv = Deconv2DLayer
    sumlayer = ll.ElemwiseSumLayer

    def deconv_res_block(l_inp, increase_dim=False, fixed_nchan=None, **kwargs):

        def projection(l_inp, fixed_nchan):
            # twice normal channels when projecting!
            l = g_register(ll.Upscale2DLayer(l_inp, 2, mode='repeat'))

            if fixed_nchan:
                n_filters = fixed_nchan
            else:
                n_filters = l_inp.output_shape[1] // 2

            l = g_register(deconv(l, (None,
                                      n_filters,
                                      l_inp.output_shape[2]*2,
                                      l_inp.output_shape[3]*2),
                                  filter_size=(1, 1),
                                  stride=(1, 1), nonlinearity=None, b=None))

            return l

        def filters_increase_dims(l, increase_dims, fixed_nchan):
            in_num_filters = l.output_shape[1]
            map_shape = l.output_shape[2:]
            if increase_dims:
                first_stride = (2, 2)
                if fixed_nchan:
                    out_num_filters = fixed_nchan
                else:
                    out_num_filters = in_num_filters // 2

                map_shape = (map_shape[0]*2, map_shape[1]*2)
            else:
                first_stride = (1, 1)
                if fixed_nchan:
                    out_num_filters = fixed_nchan
                else:
                    out_num_filters = in_num_filters

            return out_num_filters, first_stride, map_shape

        # first figure filters/strides
        n_filters, first_stride, map_shape = filters_increase_dims(
            l_inp, increase_dim, fixed_nchan)

        l = l_inp

        if first_stride == (2, 2):
            print 'first_stride', first_stride
            l = g_register(ll.Upscale2DLayer(l, 2, mode='repeat'))

        l = g_bn(l)

        l = g_register(deconv(l, (None, n_filters, ) + map_shape, filter_size=(3, 3),
                              stride=(1, 1), nonlinearity=g_nl))

        l = g_bn(g_dp(l))
        l = g_register(deconv(l, (None, n_filters, ) + map_shape, filter_size=(3, 3),
                              stride=(1, 1), nonlinearity=g_nl))

        if increase_dim:
            p = projection(l_inp, fixed_nchan)
        else:
            # Identity shortcut
            p = l_inp
        l = sumlayer([l, p])
        return l

    def blockstack(l, n, res_block, nonlinearity, fixed_nchan):
        for _ in range(n):
            l = res_block(l, nonlinearity=nonlinearity,
                          fixed_nchan=fixed_nchan)
        return l

    _reduct_factor = (2 ** n_blocks)
    map_size = img_size // _reduct_factor
    n_channel = fixed_nchan if fixed_nchan is not None else wf * \
        _base_hidden_nf*_reduct_factor

    rng = np.random.RandomState()
    theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))

    noise = theano_rng.normal(size=(batch_size, z_dim))
    l_z = g_register(ll.InputLayer(
        shape=(batch_size, z_dim), input_var=noise), 'l_z')

    l_h = g_register(ll.DenseLayer(l_z, num_units=int(
        n_channel*map_size**2), nonlinearity=None))

    l_h = g_register(ll.ReshapeLayer(l_h, (-1, n_channel, map_size, map_size)))

    for idx in xrange(n_blocks):
        if fixed_nchan:
            n_channel = fixed_nchan
        else:
            n_channel /= 2

        l_h = deconv_res_block(l_h, increase_dim=True,
                               fixed_nchan=None)

        l_h = blockstack(l_h, block_depth-1, deconv_res_block,
                         g_nl, None)

    l_out_g = g_register(Deconv2DLayer(l_h, (None, 3, img_size, img_size),
                                       filter_size=(3, 3), stride=(1, 1),
                                       nonlinearity=ln.tanh))

    return l_out_g, g_layers


def resnet_d(batch_size, img_size, z_dim=128, gan_mode=None,
             n_blocks=3, wf=1, block_depth=1,
             d_nl=ln.rectify, fixed_nchan=None,
             d_bn=None, d_dp=None, bre_loc_mode='',
             end_conv_layer=ll.GlobalPoolLayer, **kwargs):

    wf = int(wf)

    if d_bn is None:
        def d_bn(x): return x

    if d_dp is None:
        def d_dp(x): return x

    d_layers = OrderedDict([])
    d_register = registery_factory(d_layers)

    conv = ll.Conv2DLayer
    sumlayer = ll.ElemwiseSumLayer
    nl_layer = ll.NonlinearityLayer

    def conv_res_block(l_inp, increase_dim=False, fixed_nchan=None,
                       name_base=None, first_block=False, **kwargs):

        def projection(l_inp, fixed_nchan):
            if fixed_nchan:
                n_filters = fixed_nchan
            else:
                n_filters = l_inp.output_shape[1]*2

            l = d_register(conv(l_inp, num_filters=n_filters, filter_size=(1, 1),
                                stride=(2, 2), nonlinearity=None, pad='same', b=None))
            # l = d_bn(l)
            return l

        def filters_increase_dims(l, increase_dims, fixed_nchan):
            in_num_filters = l.output_shape[1]
            map_shape = l.output_shape[2:]
            if increase_dims:
                first_stride = (2, 2)
                if fixed_nchan:
                    out_num_filters = fixed_nchan
                else:
                    out_num_filters = in_num_filters * 2

                map_shape = (map_shape[0]//2, map_shape[1]//2)
            else:
                first_stride = (1, 1)
                if fixed_nchan:
                    out_num_filters = fixed_nchan
                else:
                    out_num_filters = in_num_filters

            return out_num_filters, first_stride, map_shape

        n_filters, first_stride, _ = filters_increase_dims(
            l_inp, increase_dim, fixed_nchan)

        if first_block:  # don't do bn, otherwise can't learn mean and scale properly
            l = l_inp
        else:
            l = d_bn(l_inp)

        # l = l_inp

        name = name_base % len(d_layers) if name_base else None
        l = d_register(conv(l, num_filters=n_filters, filter_size=(3, 3),
                            stride=first_stride, nonlinearity=None, pad='same'), name)
        l = d_register(nl_layer(l, nonlinearity=d_nl))
        l = d_bn(l)

        name = name_base % len(d_layers) if name_base else None
        l = d_register(conv(l, num_filters=n_filters, filter_size=(3, 3),
                            stride=(1, 1), nonlinearity=None, pad='same'), name)
        l = d_register(nl_layer(l, nonlinearity=d_nl))
        # l = d_bn(l)

        if increase_dim:
            p = projection(l_inp, fixed_nchan)
        else:
            p = l_inp

        l = sumlayer([l, p])
        return l

    def blockstack(l, n, res_block, nonlinearity, fixed_nchan, name_base):
        for _ in range(n):
            l = res_block(l, nonlinearity=nonlinearity,
                          fixed_nchan=fixed_nchan, name_base=name_base)
        return l

    n_channel = fixed_nchan if fixed_nchan is not None else int(
        wf*_base_hidden_nf)

    l_data = ll.InputLayer(shape=(None, 3, img_size, img_size))
    l_h = d_register(conv(l_data, num_filters=n_channel,
                          filter_size=(3, 3), stride=1, pad='same', nonlinearity=None))
    l_h = nl_layer(l_h, nonlinearity=d_nl)

    for idx in xrange(n_blocks):

        name_base = layer_bre_name(bre_loc_mode, idx, n_blocks)
        if name_base:
            name_base = name_base + '_layer%d'

        if idx == 0:
            l_h = conv_res_block(l_h, True, None,
                                 name_base, first_block=True)
        elif idx == n_blocks-1:
            l_h = conv_res_block(l_h, False, None, name_base)

        else:
            l_h = conv_res_block(l_h, True, None, name_base)

        l_h = blockstack(l_h, block_depth-1, conv_res_block,
                         d_nl, None, name_base)

    l_h = d_dp((d_register(end_conv_layer(l_h), 'l_h_feat')))
    l_out_d = critic_output_layer(l_h, gan_mode, register_func=d_register)

    return l_out_d, l_data, d_layers


def build_deconv_g_orig(batch_size, img_size, z_dim=100,
                        n_deconv_layer=4,  g_nl=ln.rectify,
                        wf=1, filter_size=5, fixed_nchan=None,
                        g_bn=None, g_dp=None,
                        **kwargs):

    if g_bn is None:
        def g_bn(x): return x

    if g_dp is None:
        def g_dp(x): return x

    deconv = Deconv2DLayer

    _reduct_factor = (2 ** n_deconv_layer)
    map_size = img_size // _reduct_factor

    n_channel = fixed_nchan if fixed_nchan is not None else int(
        wf*_base_hidden_nf*_reduct_factor//2)

    g_layers = OrderedDict([])
    g_register = registery_factory(g_layers)

    rng = np.random.RandomState()
    theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))

    noise = theano_rng.normal(size=(batch_size, z_dim))

    l_z = g_register(ll.InputLayer(
        shape=(batch_size, z_dim), input_var=noise), 'l_z')

    l_h = g_register(ll.DenseLayer(l_z, num_units=int(
        n_channel*map_size**2), nonlinearity=None))

    l_h = ll.ReshapeLayer(l_h, (batch_size, n_channel,
                                int(map_size), int(map_size)))

    for idx in xrange(1, n_deconv_layer):  # last one is already deconv
        if fixed_nchan:
            n_channel = fixed_nchan
        else:
            n_channel /= 2

        map_size *= 2
        l_h = g_register(deconv(l_h, (None, n_channel, map_size, map_size),
                                filter_size=(filter_size, filter_size),
                                nonlinearity=g_nl))

        l_h = g_dp(g_bn(l_h))

    l_out_g = g_register(deconv(l_h, (None, 3, img_size, img_size),
                                filter_size=(filter_size, filter_size),
                                nonlinearity=ln.tanh))

    return l_out_g, g_layers


def build_conv_d_orig(batch_size, img_size, n_conv_layer=4, gan_mode=None,
                      wf=1, filter_size=5, fixed_nchan=None,
                      d_nl=ln.LeakyRectify(.2), end_conv_layer=ll.FlattenLayer,
                      d_bn=None, d_dp=None,  d_recon_reg=0,
                      bre_loc_mode='', **kwargs):

    if d_bn is None:
        def d_bn(x): return x

    if d_dp is None:
        def d_dp(x): return x

    conv = ll.Conv2DLayer
    nl_layer = ll.NonlinearityLayer

    d_layers = OrderedDict([])
    d_register = registery_factory(d_layers)

    l_data = ll.InputLayer(shape=(None, 3, img_size, img_size))

    n_channel = fixed_nchan if fixed_nchan is not None else int(
        wf*_base_hidden_nf)

    name = layer_bre_name(bre_loc_mode, 0, n_conv_layer)

    l_h = d_register(conv(l_data, num_filters=n_channel,
                          filter_size=(filter_size, filter_size),
                          stride=2, pad='same', nonlinearity=None), name)

    l_h = d_dp(d_register(nl_layer(l_h, nonlinearity=d_nl)))

    for idx in xrange(1, n_conv_layer):
        if fixed_nchan:
            n_channel = fixed_nchan
        else:
            n_channel *= 2

        name = layer_bre_name(bre_loc_mode, idx, n_conv_layer)

        l_h = d_register(conv(l_h, num_filters=n_channel,
                              filter_size=(filter_size, filter_size),
                              stride=2, pad='same', nonlinearity=None), name)

        l_h = d_dp(d_bn(d_register(nl_layer(l_h, nonlinearity=d_nl))))

    l_h = d_register(end_conv_layer(l_h), 'l_h_feat')
    l_out_d = critic_output_layer(l_h, gan_mode, register_func=d_register)

    return l_out_d, l_data, d_layers


def build_densenet_d(batch_size, img_size, n_conv_layer=4, gan_mode=None,
                     wf=1, filter_size=3, fixed_nchan=None,
                     d_nl=ln.LeakyRectify(.2), end_conv_layer=ll.FlattenLayer,
                     d_bn=None, d_dp=None,  d_recon_reg=0, bre_loc_mode='', **kwargs):

    if d_bn is None:
        def d_bn(x): return x

    if d_dp is None:
        def d_dp(x): return x

    conv = ll.Conv2DLayer
    nl_layer = ll.NonlinearityLayer
    concat = ll.concat

    d_layers = OrderedDict([])
    d_register = registery_factory(d_layers)

    l_data = ll.InputLayer(shape=(None, 3, img_size, img_size))

    n_channel = fixed_nchan if fixed_nchan is not None else wf*_base_hidden_nf

    name = layer_bre_name(bre_loc_mode, 0, n_conv_layer)

    l_h = d_register(conv(l_data, num_filters=n_channel,
                          filter_size=(5, 5),
                          stride=2, pad='same', nonlinearity=None), name)

    l_h = d_dp(d_register(nl_layer(l_h, nonlinearity=d_nl)))

    prev_layers = [l_h]
    for idx in xrange(1, n_conv_layer):

        name = layer_bre_name(bre_loc_mode, idx, n_conv_layer)

        l_h = d_register(conv(concat(prev_layers, axis=1), num_filters=n_channel,
                              filter_size=(filter_size, filter_size),
                              stride=1, pad='same', nonlinearity=None), name)

        l_h = d_dp(d_bn(d_register(nl_layer(l_h, nonlinearity=d_nl))))

        prev_layers.append(l_h)

    l_h = d_register(end_conv_layer(concat(prev_layers, axis=1)), 'l_h_feat')
    l_out_d = critic_output_layer(l_h, gan_mode, register_func=d_register)

    return l_out_d, l_data, d_layers


def dcgan_orig(batch_size, img_size, z_dim=100, d_df=4, g_df=4,
               gan_mode=None, d_wf=1, g_wf=1,
               d_bn=None, g_bn=None, d_dp=None, g_dp=None,
               d_recon_reg=0, **kwargs):

    l_out_g, g_layers = build_deconv_g_orig(batch_size, img_size, z_dim,
                                            n_deconv_layer=g_df,
                                            wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                            **kwargs)

    l_out_d, l_data, d_layers = build_conv_d_orig(batch_size, img_size,
                                                  n_conv_layer=d_df,
                                                  gan_mode=gan_mode, wf=d_wf,
                                                  d_bn=d_bn, d_dp=d_dp,
                                                  d_recon_reg=d_recon_reg,
                                                  **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def layer_bre_name(loc_mode, layer_idx, tot_n):

    assert loc_mode in ('first_last', 'first', 'el',
                        'all', 'middle', 'last', 'lfl1')
    name = None

    if layer_idx == 0:
        if loc_mode in ('first_last', 'first', 'el', 'all'):
            name = 'preact_h0'

    elif loc_mode in ('first_last', 'last'):
        if layer_idx == tot_n-1:
            name = 'preact_last'

    elif loc_mode == 'middle':
        if layer_idx == tot_n//2:
            name = 'preact_middle'

    elif loc_mode == 'all':
        name = 'preact_h%d' % layer_idx

    elif loc_mode == 'el':  # every even layer
        if layer_idx % 2 == 0:
            name = 'preact_h%d' % layer_idx
        else:
            name = None

    elif loc_mode == 'first' and layer_idx != 0:
        name = None

    elif loc_mode == 'lfl1':
        if layer_idx != tot_n-1:
            name = 'preact_h%d' % layer_idx

    return name


def build_conv_d(batch_size, img_size, n_conv_layer=4, gan_mode=None,
                 wf=1, filter_size=5, fixed_nchan=None,
                 d_nl=ln.LeakyRectify(.2), end_conv_layer=ll.FlattenLayer,
                 d_bn=None, d_dp=None,  d_recon_reg=0, bre_loc_mode='', **kwargs):

    if d_bn is None:
        def d_bn(x): return x

    if d_dp is None:
        def d_dp(x): return x

    conv = ll.Conv2DLayer
    nl_layer = ll.NonlinearityLayer

    d_layers = OrderedDict([])
    d_register = registery_factory(d_layers)

    l_data = ll.InputLayer(shape=(None, 3, img_size, img_size))

    n_channel = fixed_nchan if fixed_nchan is not None else int(
        wf*_base_hidden_nf)

    name = layer_bre_name(bre_loc_mode, 0, n_conv_layer)

    l_h = d_register(conv(l_data, num_filters=n_channel,
                          filter_size=(filter_size, filter_size),
                          stride=1, pad='same', nonlinearity=None), name)
    l_h = d_dp(d_register(nl_layer(l_h, nonlinearity=d_nl)))

    for idx in xrange(1, n_conv_layer):
        if fixed_nchan:
            n_channel = fixed_nchan
        else:
            n_channel *= 2

        print bre_loc_mode

        name = layer_bre_name(bre_loc_mode, idx, n_conv_layer)

        l_h = d_register(conv(l_h, num_filters=n_channel,
                              filter_size=(filter_size, filter_size),
                              stride=2, pad='same', nonlinearity=None), name)

        l_h = d_dp(d_bn(d_register(nl_layer(l_h, nonlinearity=d_nl))))

    l_h = d_register(end_conv_layer(l_h), 'l_h_feat')
    l_out_d = critic_output_layer(l_h, gan_mode, register_func=d_register)

    return l_out_d, l_data, d_layers


def build_deconv_g(batch_size, img_size, z_dim=100,
                   n_deconv_layer=4,  g_nl=ln.rectify,
                   wf=1, filter_size=5, fixed_nchan=None,
                   g_bn=None, g_dp=None, multilayer_noise=False,
                   **kwargs):

    if g_bn is None:
        def g_bn(x): return x

    if g_dp is None:
        def g_dp(x): return x

    deconv = Deconv2DLayer

    _reduct_factor = (2 ** (n_deconv_layer-1))
    map_size = img_size // _reduct_factor

    g_layers = OrderedDict([])
    g_register = registery_factory(g_layers)

    rng = np.random.RandomState()
    theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))

    noise = theano_rng.normal(size=(batch_size, z_dim))

    l_z = g_register(ll.InputLayer(
        shape=(batch_size, z_dim), input_var=noise), 'l_z')

    l_h = g_register(ll.DenseLayer(l_z, num_units=int(
        wf*128*map_size**2), nonlinearity=g_nl))

    l_h = ll.ReshapeLayer(l_h, (batch_size, int(wf*128),
                                int(map_size), int(map_size)))

    n_channel = fixed_nchan if fixed_nchan is not None else int(
        wf*_base_hidden_nf*_reduct_factor)

    for idx in xrange(1, n_deconv_layer):  # last one is already deconv
        if fixed_nchan:
            n_channel = fixed_nchan
        else:
            n_channel /= 2

        map_size *= 2
        l_h = g_register(deconv(l_h, (None, n_channel, map_size, map_size),
                                filter_size=(filter_size, filter_size),
                                nonlinearity=g_nl))

        l_h = g_dp(g_bn(l_h))

    l_out_g = g_register(deconv(l_h, (None, 3, img_size, img_size),
                                stride=(1, 1), filter_size=(3, 3),
                                nonlinearity=ln.tanh))

    return l_out_g, g_layers


def dcgan(batch_size, img_size, z_dim=100, d_df=4, g_df=4, gan_mode=None,
          d_wf=1, g_wf=1, d_bn=None, g_bn=None, d_dp=None, g_dp=None,
          d_recon_reg=0, **kwargs):

    l_out_g, g_layers = build_deconv_g(batch_size, img_size, z_dim,
                                       n_deconv_layer=g_df,
                                       wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                       **kwargs)

    l_out_d, l_data, d_layers = build_conv_d(batch_size, img_size,
                                             n_conv_layer=d_df,
                                             gan_mode=gan_mode, wf=d_wf,
                                             d_bn=d_bn, d_dp=d_dp,
                                             d_recon_reg=d_recon_reg,
                                             **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def dcgan_relu(batch_size, img_size, z_dim=100, d_df=4, g_df=4, gan_mode=None,
               d_wf=1, g_wf=1, d_bn=None, g_bn=None, d_dp=None, g_dp=None,
               d_recon_reg=0, **kwargs):

    l_out_g, g_layers = build_deconv_g(batch_size, img_size, z_dim,
                                       n_deconv_layer=g_df, g_nl=ln.rectify,
                                       wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                       **kwargs)

    l_out_d, l_data, d_layers = build_conv_d(batch_size, img_size,
                                             n_conv_layer=d_df, d_nl=ln.rectify,
                                             gan_mode=gan_mode, wf=d_wf,
                                             d_bn=d_bn, d_dp=d_dp,
                                             d_recon_reg=d_recon_reg,
                                             **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def dcgan_resnet_g_d(batch_size, img_size, z_dim=100, d_df=4, g_df=4,
                     gan_mode=None, d_wf=1, g_wf=1, d_bn=None, g_bn=None,
                     d_dp=None, g_dp=None, d_recon_reg=0, **kwargs):

    l_out_g, g_layers = resnet_g(batch_size, img_size, z_dim,
                                 n_blocks=g_df, block_depth=2,
                                 wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                 fixed_nchan=None,
                                 g_nl=ln.rectify,
                                 **kwargs)

    l_out_d, l_data, d_layers = build_conv_d(batch_size, img_size,
                                             n_conv_layer=d_df,
                                             gan_mode=gan_mode, wf=d_wf,
                                             d_bn=d_bn, d_dp=d_dp,
                                             d_recon_reg=d_recon_reg,
                                             **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def dcgan_g_densenet_d(batch_size, img_size, z_dim=100, d_df=4, g_df=4,
                       gan_mode=None, d_wf=1, g_wf=1, fixed_nchan=64,
                       d_bn=None, g_bn=None, d_dp=None, g_dp=None,
                       d_recon_reg=0, **kwargs):

    l_out_g, g_layers = build_deconv_g(batch_size, img_size, z_dim,
                                       n_deconv_layer=g_df,
                                       wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                       **kwargs)

    l_out_d, l_data, d_layers = build_densenet_d(batch_size, img_size,
                                                 n_conv_layer=d_df,
                                                 gan_mode=gan_mode,
                                                 d_bn=d_bn, d_dp=d_dp,
                                                 fixed_nchan=int(
                                                     fixed_nchan*d_wf),
                                                 d_recon_reg=d_recon_reg,
                                                 end_conv_layer=ll.FlattenLayer,
                                                 **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def resnet_g_densenet_d(batch_size, img_size, z_dim=100, d_df=4, g_df=4,
                        gan_mode=None, d_wf=1, g_wf=1, fixed_nchan=64,
                        d_bn=None, g_bn=None, d_dp=None, g_dp=None,
                        d_recon_reg=0, **kwargs):

    l_out_g, g_layers = resnet_g(batch_size, img_size, z_dim,
                                 n_blocks=int(g_df), block_depth=1,
                                 wf=int(g_wf), g_bn=g_bn, g_dp=g_dp,
                                 fixed_nchan=int(fixed_nchan),
                                 g_nl=ln.rectify,
                                 **kwargs)

    l_out_d, l_data, d_layers = build_densenet_d(batch_size, img_size,
                                                 n_conv_layer=int(d_df),
                                                 gan_mode=gan_mode,
                                                 d_bn=d_bn, d_dp=d_dp,
                                                 fixed_nchan=int(
                                                     fixed_nchan*d_wf),
                                                 d_recon_reg=d_recon_reg,
                                                 end_conv_layer=ll.FlattenLayer,
                                                 **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def dcgan_equal_size(batch_size, img_size, z_dim=100, d_df=4, g_df=4,
                     gan_mode=None, d_wf=1, g_wf=1, fixed_nchan=256,
                     d_bn=None, g_bn=None, d_dp=None, g_dp=None,
                     **kwargs):

    l_out_g, g_layers = build_deconv_g(batch_size, img_size, z_dim,
                                       n_deconv_layer=g_df,
                                       wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                       fixed_nchan=fixed_nchan,
                                       g_nl=ln.rectify,
                                       **kwargs)

    l_out_d, l_data, d_layers = build_conv_d(batch_size, img_size,
                                             n_conv_layer=d_df,
                                             gan_mode=gan_mode, wf=d_wf,
                                             d_bn=d_bn, d_dp=d_dp,
                                             fixed_nchan=fixed_nchan,
                                             d_nl=ln.rectify,
                                             **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers


def resnet(batch_size, img_size, z_dim=100, d_df=3, g_df=3, gan_mode=None,
           d_wf=1, g_wf=1, fixed_nchan=256,
           d_bn=None, g_bn=None, d_dp=None, g_dp=None,  **kwargs):

    l_out_g, g_layers = resnet_g(batch_size, img_size, z_dim,
                                 n_blocks=g_df, block_depth=1,
                                 wf=g_wf, g_bn=g_bn, g_dp=g_dp,
                                 g_nl=ln.rectify,
                                 **kwargs)

    l_out_d, l_data, d_layers = resnet_d(batch_size, img_size,
                                         n_blocks=d_df, block_depth=1,
                                         gan_mode=gan_mode, wf=d_wf,
                                         d_bn=d_bn, d_dp=d_dp,
                                         d_nl=ln.rectify,
                                         **kwargs)

    return l_out_g, l_out_d, l_data, g_layers, d_layers

