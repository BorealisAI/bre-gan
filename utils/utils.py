# Copyright (c) 2018-present, Borealis AI.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Author: Yanshuai Cao


import numpy as np
import theano
import lasagne
import lasagne.layers as ll
from lasagne.layers import get_all_layers
from collections import deque, defaultdict

from io_tools import *
from gan_utils import *


floatX = np.float32


def lin_anneal(sh_var, var0, l, min_var=0., name=''):
    new_var = np.maximum(
        floatX(min_var), sh_var.get_value() - floatX(var0) / floatX(l))
    sh_var.set_value(floatX(new_var))


def batch(iterable, n=1, same_size=True):
    l = len(iterable)
    nb = l // n

    if same_size:
        for ndx in range(0, nb*n, n):
            yield iterable[ndx:ndx+n]
    else:
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx+n, l)]


def registery_factory(layers):
    def register(layer, name=None):
        if isinstance(layer, (list, tuple)):
            for l in layer:
                if name:
                    layers[name + '_' + str(len(layers))] = l
                else:
                    name = 'layer_' + str(len(layers))
                    layers[name] = l
        else:
            if name:
                layers[name] = layer
            else:
                name = 'layer_' + str(len(layers))
                layers[name] = layer

        return layer
    return register


import time
try:
    import queue
except ImportError:
    import Queue as queue


def set_default(d, k, v):
    if k not in d:
        d[k] = v
    return d


def dict2hash(d):
    import hashlib
    m = hashlib.md5()
    m.update(str([(k, d[k]) for k in sorted(d.keys())]))
    return m.hexdigest()


def save_figure_noframe(filename, fig=None, **kwargs):
    ''' Save a Matplotlib figure as an image without borders or frames.
        from http://robotics.usc.edu/~ampereir/wordpress/?p=626
       Args:
            fileName (str): String that ends in .png etc.

            fig (Matplotlib figure instance): figure you want to save as the image
        Keyword Args:
            orig_size (tuple): width, height of the original image used to maintain 
            aspect ratio.
    '''
    from matplotlib import pyplot as plt

    fig_size = fig.get_size_inches()
    print fig_size
    w, h = fig_size[0], fig_size[1]
    fig.patch.set_alpha(0)

    if kwargs.has_key('orig_size'):  # Aspect ratio scaling if required
        w, h = kwargs['orig_size']
        w2, h2 = fig_size[0], fig_size[1]
        fig.set_size_inches([(w2/w)*w, (w2/w)*h])
        fig.set_dpi((w2/w)*fig.get_dpi())
    a = fig.gca()
    a.set_frame_on(False)
    a.set_xticks([])
    a.set_yticks([])

    plt.axis('off')
    plt.xlim(0, h)
    plt.ylim(w, 0)
    fig.savefig(filename, transparent=True, bbox_inches='tight',
                pad_inches=0)


def layer_info(layer, name=''):
    layer_type_name = layer.__class__.__name__

    n_params = sum(int(np.prod(p.get_value().shape))
                   for p in layer.get_params(trainable=True))
    #info_str += '\t\t\t' + str(n_params) + ' params;'

    if hasattr(layer, 'shape'):
        shape = str(layer.shape)
    elif hasattr(layer, 'input_shape'):
        shape = str(layer.input_shape)
    else:
        shape = '??'

    info_str = '{:<25} {:<10} {:<12g}  {:<12}'.format(
        layer_type_name, name, n_params, shape)

    return info_str


def get_network_str(layer, get_network=True, incomings=False,
                    outgoings=False, layer2str=layer_info, layer2name=None):
    """ Returns a string representation of the entire network contained under this layer.

        from Lasagne/Recipes/utils/network_repr.py

        Parameters
        ----------
        layer : Layer or list
            the :class:`Layer` instance for which to gather all layers feeding
            into it, or a list of :class:`Layer` instances.

        get_network : boolean
            if True, calls `get_all_layers` on `layer`
            if False, assumes `layer` already contains all `Layer` instances intended for representation

        incomings : boolean
            if True, representation includes a list of all incomings for each `Layer` instance

        outgoings: boolean
            if True, representation includes a list of all outgoings for each `Layer` instance

        Returns
        -------
        str
            A string representation of `layer`. Each layer is assigned an ID which is it's corresponding index
            in the list obtained from `get_all_layers`.
        """

    # `layer` can either be a single `Layer` instance or a list of `Layer` instances.
    # If list, it can already be the result from `get_all_layers` or not, indicated by the `get_network` flag
    # Get network using get_all_layers if required:
    if get_network:
        network = get_all_layers(layer)
    else:
        network = layer

    # Initialize a list of lists to (temporarily) hold the str representation of each component, insert header
    network_str = deque([])
    network_str = _insert_header(
        network_str, incomings=incomings, outgoings=outgoings)

    # The representation can optionally display incoming and outgoing layers for each layer, similar to adjacency lists.
    # If requested (using the incomings and outgoings flags), build the adjacency lists.
    # The numbers/ids in the adjacency lists correspond to the layer's index in `network`
    if incomings or outgoings:
        ins, outs = _get_adjacency_lists(network)

    from collections import defaultdict
    layer2name = defaultdict(str) if layer2name is None else layer2name
    # For each layer in the network, build a representation and append to `network_str`
    for i, current_layer in enumerate(network):

        # Initialize list to (temporarily) hold str of layer
        layer_str = deque([])

        # First column for incomings, second for the layer itself, third for outgoings, fourth for layer description
        if incomings:
            layer_str.append(ins[i])
        layer_str.append(i)
        if outgoings:
            layer_str.append(outs[i])
        # default representation can be changed by overriding __str__
        layer_str.append(layer2str(current_layer, layer2name[current_layer]))
        network_str.append(layer_str)
    return _get_table_str(network_str)


def _insert_header(network_str, incomings, outgoings):
    """ Insert the header (first two lines) in the representation."""
    line_1 = deque([])
    if incomings:
        line_1.append('In -->')
    line_1.append('Layer')
    if outgoings:
        line_1.append('--> Out')
    line_1.append('Description')
    line_2 = deque([])
    if incomings:
        line_2.append('-------')
    line_2.append('-----')
    if outgoings:
        line_2.append('-------')
    line_2.append('-----------')
    network_str.appendleft(line_2)
    network_str.appendleft(line_1)
    return network_str


def _get_adjacency_lists(network):
    """ Returns adjacency lists for each layer (node) in network.
        Warning: Assumes repr is unique to a layer instance, else this entire approach WILL fail."""
    # ins  is a dict, keys are layer indices and values are lists of incoming layer indices
    # outs is a dict, keys are layer indices and values are lists of outgoing layer indices
    ins = defaultdict(list)
    outs = defaultdict(list)
    lookup = {repr(layer): index for index, layer in enumerate(network)}

    for current_layer in network:
        if hasattr(current_layer, 'input_layers'):
            layer_ins = current_layer.input_layers
        elif hasattr(current_layer, 'input_layer'):
            layer_ins = [current_layer.input_layer]
        else:
            layer_ins = []

        ins[lookup[repr(current_layer)]].extend(
            [lookup[repr(l)] for l in layer_ins])

        for l in layer_ins:
            outs[lookup[repr(l)]].append(lookup[repr(current_layer)])
    return ins, outs


def _get_table_str(table):
    """ Pretty print a table provided as a list of lists."""
    table_str = ''
    col_size = [max(len(str(val)) for val in column) for column in zip(*table)]
    for line in table:
        table_str += '\n'
        table_str += '    '.join('{0:<{1}}'.format(val,
                                                   col_size[i]) for i, val in enumerate(line))
    return table_str


def safe_zip(*args):
    for arr in args[1:]:
        assert len(arr) == len(args[0])
    return zip(*args)

