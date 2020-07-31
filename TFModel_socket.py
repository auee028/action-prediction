#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import os
import cv2
import copy
import glob
import natsort
import numpy as np
import math
import tensorflow as tf

import time

from crop_frames import CropFrames

import model_zoo

from darknet.python.darknet import *


with open('categories.txt') as f:
    lines = map(lambda x: x.strip(), f.readlines())

ix2label = dict(zip(range(len(lines)), lines))

cwd = os.getcwd()
# model_path = os.path.join('save_model', 'i3d_ABR_action-finetune')
# model_path = os.path.join('/media/pjh/HDD2/SourceCodes/wonhee-takeover/event_detector', 'save_model', 'i3d_ABR_action-finetune')    # /step-119/200209
# model_path = os.path.join('/media/pjh/HDD2/Dataset/save_model', 'i3d-ABR_action_augmented-{}'.format(5))
# model_path = os.path.join('/media/pjh/HDD2/Dataset/save_model/wonhee-train')
# model_path = os.path.join('/media/pjh/HDD2/Dataset/save_model', 'i3d-ABR_action_augmented-{}'.format(4))
ckpt_num = 24


class Camera:
    def __init__(self):

        self.move_detect = False
        self.frame_diff_thresh = 0.4

        # load yolo v3(tiny) and meta data
        self.yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
        self.meta = load_meta("./darknet/cfg/coco.data")

        # # load yolo v3(tiny) and meta data
        # self.yolo = load_net("./darknet/cfg/yolov3-tiny-voc.cfg", "./darknet/cfg/yolov3-tiny-voc_210000.weights", 0)
        # self.meta = load_meta("./darknet/cfg/voc.data")

        self.frames = []

        self.frame_num = -1
        self.start_frame = -1

    def get_frames(self, frame):
        out_frames = []

        frame = cv2.resize(frame, (224, 224))

        self.frames.append(frame)
        self.frame_num += 1

        r = np_detect(self.yolo, self.meta, frame)

        if len(r) >= 1:
            if len(self.frames) >= 5:
                # calculate the frame differences
                c_frame = self.frames[-1]
                c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
                b_frame = self.frames[-2]
                b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)
                a_frame = self.frames[-3]
                a_frame = cv2.cvtColor(a_frame, cv2.COLOR_BGR2GRAY)

                cb_frame_diff = cv2.absdiff(c_frame, b_frame)
                ba_frame_diff = cv2.absdiff(b_frame, a_frame)

                cba_frame_diff = cv2.absdiff(cb_frame_diff, ba_frame_diff)
                _, cba_frame_diff = cv2.threshold(cba_frame_diff, 30, 255, cv2.THRESH_BINARY)

                cb_diff_mask = np.array(cb_frame_diff > 10, dtype=np.int32)
                ba_diff_mask = np.array(ba_frame_diff > 10, dtype=np.int32)
                cba_diff_mask = np.array(cba_frame_diff > 10, dtype=np.int32)

                try:
                    diff_thresh = float(
                        1.0 * np.sum(cba_diff_mask) / max(np.sum(cb_diff_mask), np.sum(ba_diff_mask)))

                except:
                    diff_thresh = 0
                # print('(threshold : {})'.format(diff_thresh))

                if diff_thresh >= self.frame_diff_thresh and not self.move_detect:
                    self.move_detect = True
                    self.start_frame = self.frame_num - 2
                # elif diff_thresh < 0.3 :
                #     self.move_detect = False
                #     frames = []

                if self.move_detect:
                    if diff_thresh < .1:  # when the movement stops
                        self.frames = self.frames[self.start_frame:]

                        # initialize all members
                        self.move_detect = False
                        self.frame_num = -1
                        self.start_frame = -1

                        out_frames = copy.deepcopy(self.frames)
                        self.frames = []

                        print(self.move_detect, diff_thresh, len(out_frames), len(self.frames))

                        if len(out_frames) > 10:
                            return out_frames
                        else:
                            return self.frames

                    return out_frames  # same as 'return []'

                return out_frames       # same as 'return []'

            return out_frames       # same as 'return []'

        return out_frames       # same as 'return []'


class TFModel:
    def __init__(self,
                 model_path=os.path.join('/media/pjh/HDD2/SourceCodes/wonhee-takeover/event_detector', 'save_model',
                                         'i3d_ABR_action-finetune')):
        self.model_path = model_path

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3])
        self.is_training = tf.placeholder(dtype=tf.bool)

        # build IC3D net
        self.net = model_zoo.I3DNet(inps=self.inputs, n_class=len(ix2label), batch_size=1,
                                    pretrained_model_path=None, final_end_point='SequatialLogits',
                                    dropout_keep_prob=1.0, is_training=self.is_training)

        # logits from IC3D net
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

        ckpt = tf.train.latest_checkpoint(model_path)

        if ckpt:
            print('restore from {}...'.format(ckpt))
            saver.restore(self.sess, ckpt)

        # ckpt_state = tf.train.get_checkpoint_state(model_path)
        #
        # if ckpt_state:
        #     ckpt = ckpt_state.all_model_checkpoint_paths#[ckpt_num]
        #
        #     print(len(ckpt))
        #     print('restore from {}...'.format(ckpt))
        #     saver.restore(self.sess, ckpt)

    def run_demo_wrapper(self, frames):
        tf.reset_default_graph()

        summary, predictions, softmax = self.sess.run([self.merge_op, self.pred, self.softmax],
                                                      feed_dict={self.inputs: frames,
                                                                 self.is_training: False})
        # predictions, softmax = self.sess.run([self.pred, self.softmax], feed_dict={self.inputs: frames,
        #                                                                           self.is_training: False})

        top_3 = map(lambda x: ix2label[int(x)], np.argsort(-softmax)[0][0][:3])

        # for tensorboard
        # self.logger.add_summary(summary)

        mask = map(lambda x: int(ix2label[int(x)] != 'Doing other things'), predictions[0])

        # casting
        mask = np.expand_dims(np.expand_dims(mask, axis=0), 2)

        predicted_label = map(lambda x: ix2label[int(x)], predictions[0])

        # if predicted_label.count('Doing other things') > int(RScam.args.video_length*0.8):
        # if predicted_label.count('Doing other things') > int(35 * 0.8):
        #    return 'Doing other things', 'Null'

        # apply mask to predictions
        softmax_masked = np.mean(mask * softmax, axis=1)

        from collections import Counter
        freq_predictions = Counter(predicted_label).most_common()[0][0]

        str_res = freq_predictions.strip()
        # confidence = softmax_masked.max()

        confidence_3 = np.sort(softmax_masked)[::-1][0][:3]

        return str_res, confidence_3, top_3

