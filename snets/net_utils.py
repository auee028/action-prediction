"""Utilities for building Inflated 3D ConvNets """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
from snets.scopes import *
slim = tf.contrib.slim

@add_arg_scope
def unit3D(inputs, output_channels,
           kernel_shape=(1, 1, 1),
           strides=(1, 1, 1),
           activation_fn=tf.nn.relu,
           use_batch_norm=True,
           use_bias=False,
           padding='same',
           is_training=True,
           name=None):
  """Basic unit containing Conv3D + BatchNorm + non-linearity."""
  with tf.variable_scope(name, 'unit3D', [inputs]):
    net = tf.layers.conv3d(inputs, filters=output_channels,
                            kernel_size=kernel_shape,
                            strides=strides,
                            padding=padding,
                            use_bias=use_bias)
    if use_batch_norm:
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    if activation_fn is not None:
        net = activation_fn(net)
  return net

