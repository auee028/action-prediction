# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import copy

import model_zoo
# import model_zoo_train

import argparse

import os
import requests


with open('categories.txt') as f:
    lines = map(lambda x: x.strip(), f.readlines())

ix2label = dict(zip(range(len(lines)), lines))

class MultiscaleI3D:
    def __init__(self):
        self.model_path = os.path.join('save_model', 'mtsi3d')

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, 64, 224, 224, 3])
        self.is_training = tf.placeholder(dtype=tf.bool)

        # build multiscaleI3D net
        self.net = model_zoo.multiscaleI3DNet(inps=self.inputs, n_class=len(ix2label), batch_size=1,
                                    pretrained_model_path=None, final_end_point='Logits',
                                    dropout_keep_prob=1.0, is_training=self.is_training, scope='v/SenseTime_I3D')

        # logits from multiscaleI3D net
        out, merge_op = self.net(self.inputs)
        self.softmax = tf.nn.softmax(out)
        self.merge_op = merge_op

        self.pred = tf.argmax(self.softmax, axis=-1)

        # gpu config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # open session
        self.sess = tf.Session(config=config)
        # self.logger = tf.summary.FileWriter('./log', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


        saver = tf.train.Saver()

        ckpt = tf.train.latest_checkpoint(self.model_path)

        if ckpt:
            print('restore from {}...'.format(ckpt))
            saver.restore(self.sess, ckpt)


    def run_demo_wrapper(self, frames):
        summary, predictions, softmax = self.sess.run([self.merge_op, self.pred, self.softmax],
                                                      feed_dict={self.inputs: frames,
                                                                 self.is_training: False})
        top_3 = map(lambda x: ix2label[int(x)], np.argsort(-softmax)[0][:3])

        mask = map(lambda x: int(ix2label[int(x)] != 'Doing other things'), predictions)

        # casting
        mask = np.expand_dims(np.expand_dims(mask, axis=0), 2)

        predicted_label = map(lambda x: ix2label[int(x)], predictions)

        # apply mask to predictions
        softmax_masked = np.mean(mask * softmax, axis=1)

        from collections import Counter
        freq_predictions = Counter(predicted_label).most_common()[0][0]

        str_res = freq_predictions.strip()
        confidence = softmax_masked.max()

        return str_res, confidence, top_3


class LSTM:
    def __init__(self):
        self.model_path = os.path.join('save_model', 'lstm')
        self.n_actions = 15
        self.n_hidden = 128

        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None])

        # build LSTM net
        self.net = model_zoo.LSTMNet(n_class=self.n_actions, n_hidden=self.n_hidden)

        # logits from LSTM net
        out = self.net(self.inputs)
        self.softmax = tf.nn.softmax(out)

        self.pred = tf.argmax(self.softmax, axis=-1)

        # gpu config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # open session
        self.sess = tf.Session(config=config)
        # self.logger = tf.summary.FileWriter('./log', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        ckpt = tf.train.latest_checkpoint(self.model_path)

        if ckpt:
            print('restore from {}...'.format(ckpt))
            saver.restore(self.sess, ckpt)

    def run_demo_wrapper(self, action_seq):

        predictions = self.sess.run(self.pred, feed_dict={self.inputs: action_seq})

        return predictions


