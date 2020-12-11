# -- coding: UTF-8 --
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import tensorflow.contrib.slim as slim
from snets.net_utils import unit3D

from tensorflow.python.training import moving_averages


class ResNet(object):
    """ResNet model."""

    def __init__(self, inputs, batch_size=1, num_classes=2, is_training=True, use_nonlocal=True,
                 final_endpoint='Predictions', dropout_keep_prob=0.5, scope='',
                 filters=[64, 256, 512, 1024, 2048], block_num=[3, 4, 6, 3]):

        self.inputs = inputs
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.is_training = is_training

        self.dropout_keep_prob = dropout_keep_prob
        self.use_nonlocal = use_nonlocal
        self.final_endpoint = final_endpoint
        self.end_points = {}
        self.dropout_keep_prob = dropout_keep_prob

        self.filters = filters
        self.block_num = block_num


    # build_model
    def _build_model(self):

        b, g, r = tf.split(self.inputs, 3, axis=4)
        inputs = tf.squeeze(tf.stack([r, g, b], axis=4), axis=5)

        # (min,max) => (0,255)
        inputs /= 255.0

        with tf.variable_scope('scale1'):
            x = self._conv3d('conv1', inputs, [5, 7, 7], 3, self.filters[0], self._stride_arr([1, 2, 2]))
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
            x = self._relu(x)
        x = tf.nn.max_pool3d(x, ksize=[1, 3, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                             padding='SAME', name='pool1')
        # configs
        activate_before_residual = [True, False, False, False]
        res_func = self._bottleneck_residual

        # res2
        with tf.variable_scope('scale2'):
            with tf.variable_scope('block1'):
                x = res_func(x, self.filters[0], self.filters[1],
                             self._stride_arr([1, 1, 1]),
                             activate_before_residual[0],
                             inflate=True)
            for i in six.moves.range(1, self.block_num[0]):
                with tf.variable_scope('block%d' % (i + 1)):
                    x = res_func(x, self.filters[1], self.filters[1], self._stride_arr([1, 1, 1]), False, inflate=True)

        x = tf.nn.max_pool3d(x, ksize=[1, 3, 1, 1, 1], strides=[1, 2, 1, 1, 1],
                             padding='SAME', name='pool2')
        # res3
        with tf.variable_scope('scale3'):
            with tf.variable_scope('block1'):
                x = res_func(x, self.filters[1], self.filters[2],
                             self._stride_arr([1, 1, 1]),
                             activate_before_residual[1],
                             inflate=True)
            for i in six.moves.range(1, self.block_num[1]):
                with tf.variable_scope('block%d' % (i + 1)):
                    if i % 2:
                        x = res_func(x, self.filters[2], self.filters[2], self._stride_arr([1, 1, 1]), False, inflate=False)
                        if self.use_nonlocal == 'use_nonlocal':
                            x = self._nonlocal(x, out_channels=512, name='NonLocalBlock')
                    else:
                        x = res_func(x, self.filters[2], self.filters[2], self._stride_arr([1, 1, 1]), False, inflate=True)

        # res4
        with tf.variable_scope('scale4'):
            with tf.variable_scope('block1'):
                x = res_func(x, self.filters[2], self.filters[3],
                             self._stride_arr([1, 1, 1]),
                             activate_before_residual[2],
                             inflate=True)
            for i in six.moves.range(1, self.block_num[2]):
                with tf.variable_scope('block%d' % (i + 1)):
                    if i % 2:
                        x = res_func(x, self.filters[3], self.filters[3], self._stride_arr([1, 1, 1]), False, inflate=False)
                        if self.use_nonlocal == 'use_nonlocal':
                            x = self._nonlocal(x, out_channels=1024, name='NonLocalBlock')
                    else:
                        x = res_func(x, self.filters[3], self.filters[3], self._stride_arr([1, 1, 1]), False, inflate=True)

        # res5
        with tf.variable_scope('scale5'):
            with tf.variable_scope('block1'):
                x = res_func(x, self.filters[3], self.filters[4],
                             self._stride_arr([1, 1, 1]),
                             activate_before_residual[3],
                             inflate=False)
            for i in six.moves.range(1, self.block_num[3]):
                with tf.variable_scope('block%d' % (i + 1)):
                    if i % 2:
                        x = res_func(x, self.filters[4], self.filters[4], self._stride_arr([1, 1, 1]), False, inflate=True)
                    else:
                        x = res_func(x, self.filters[4], self.filters[4], self._stride_arr([1, 1, 1]), False, inflate=False)

        end_point = 'FeatureExtraction'
        with tf.variable_scope(end_point):
            x = tf.nn.avg_pool3d(x, ksize=[1, 4, 7, 7, 1], strides=[1, 1, 1, 1, 1],
                                 padding='VALID')

            x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)

            feats = tf.reduce_mean(x, axis=[1,2,3])
            self.end_points[end_point] = feats
            if self.final_endpoint =='FeatureExtraction': return tf.reduce_mean(x, axis=[1,2,3]), self.end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            logits = unit3D(x, self.num_classes,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            is_training=self.is_training,
                            use_batch_norm=False,
                            use_bias=True,
                            name='fc')

            logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')

            averaged_logits = tf.reduce_mean(logits, axis=1)

        self.end_points[end_point] = averaged_logits
        if end_point == self.final_endpoint: return averaged_logits, self.end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averaged_logits)

        self.end_points[end_point] = predictions
        if end_point == self.final_endpoint: return predictions, self.end_points

    def _stride_arr(self, stride):
        return [1, stride[0], stride[1], stride[2], 1]

    def _nonlocal(self, input_x, out_channels, name='NonLocalBlock'):
        batchsize, time, height, width, in_channels = input_x.get_shape().as_list()
        with tf.variable_scope('NonLocalBlock'):
            with tf.variable_scope('g'):
                g = self._conv3d('conv1', input_x, [1, 1, 1], out_channels, out_channels / 2, [1, 1, 1, 1, 1])
            with tf.variable_scope('phi'):
                phi = self._conv3d('conv2', input_x, [1, 1, 1], out_channels, out_channels / 2, [1, 1, 1, 1, 1])
            with tf.variable_scope('theta'):
                theta = self._conv3d('conv3', input_x, [1, 1, 1], out_channels, out_channels / 2, [1, 1, 1, 1, 1])

            g_x = tf.reshape(g, [batchsize, time * height * width, out_channels / 2])
            theta_x = tf.reshape(theta, [batchsize, time * height * width, out_channels / 2])
            phi_x = tf.reshape(phi, [batchsize, time * height * width, out_channels / 2])
            phi_x = tf.transpose(phi_x, [0, 2, 1])

            f = tf.matmul(theta_x, phi_x)
            f_softmax = tf.nn.softmax(f, -1)
            y = tf.matmul(f_softmax, g_x)
            y = tf.reshape(y, [batchsize, time, height, width, out_channels / 2])

            with tf.variable_scope('w'):
                w_y = self._conv3d('conv4', y, [1, 1, 1], out_channels / 2, out_channels, [1, 1, 1, 1, 1])
                w_y = tf.contrib.layers.batch_norm(w_y, is_training=self.is_training)

        z = input_x + w_y
        return z

    # bottleneck resnet block
    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False, inflate=False):
        orig_x = x
        # a
        with tf.variable_scope('a'):
            if inflate:
                x = self._conv3d('conv1', x, [3, 1, 1], in_filter, out_filter / 4, stride)
            else:
                x = self._conv3d('conv1', x, [1, 1, 1], in_filter, out_filter / 4, stride)
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
            x = self._relu(x)

        # b
        with tf.variable_scope('b'):
            if in_filter != out_filter and out_filter != 256:
                x = self._conv3d('conv2', x, [1, 3, 3], out_filter / 4, out_filter / 4, [1, 1, 2, 2, 1])
            else:
                x = self._conv3d('conv2', x, [1, 3, 3], out_filter / 4, out_filter / 4, [1, 1, 1, 1, 1])
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)
            x = self._relu(x)

        # c
        with tf.variable_scope('c'):
            x = self._conv3d('conv3', x, [1, 1, 1], out_filter / 4, out_filter, [1, 1, 1, 1, 1])
            x = tf.contrib.layers.batch_norm(x, is_training=self.is_training)

        # when channels change, shortcut
        with tf.variable_scope('shortcut'):
            if in_filter != out_filter and out_filter != 256:
                orig_x = self._conv3d('project', orig_x, [1, 1, 1], in_filter, out_filter, [1, 1, 2, 2, 1])
                orig_x = tf.contrib.layers.batch_norm(orig_x, is_training=self.is_training)
            elif in_filter != out_filter:
                orig_x = self._conv3d('project', orig_x, [1, 1, 1], in_filter, out_filter, [1, 1, 1, 1, 1])
                orig_x = tf.contrib.layers.batch_norm(orig_x, is_training=self.is_training)
        x += orig_x
        x = self._relu(x)

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    # 3D_conv
    def _conv3d(self, name, x, filter_size, in_filters, out_filters, strides):
        # filter: [filter_depth, filter_height, filter_width]
        # strides: [1, depth_stride, x_stride, y_stride, 1]
        n = filter_size[0] * filter_size[1] * filter_size[2] * out_filters
        kernel = tf.get_variable(
            'weights',
            [filter_size[0], filter_size[1], filter_size[2], in_filters, out_filters],
            tf.float32,
            initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
        return tf.nn.conv3d(x, kernel, strides, padding='SAME')

    # leaky ReLU
    def _relu(self, x):
        return tf.nn.relu(x)

    # fc
    def _fully_connected(self, x, out_dim):
        # reshape
        x = tf.reshape(x, [self.batch_size, -1])
        w = tf.get_variable('weights', [x.get_shape()[1], out_dim],
                            initializer=tf.variance_scaling_initializer(distribution="uniform"))

        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, w, b)
        return x

    # _global_avg_pool
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 5
        return tf.reduce_mean(x, [1, 2, 3], keepdims=True)

