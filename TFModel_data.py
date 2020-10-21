#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import os
import cv2
import copy
import numpy as np


class Camera:
    def __init__(self):

        self.move_detect = False
        self.frame_diff_thresh = 0.4

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

                    out_frames = copy.deepcopy(self.frames)
                    self.frames = []

                    print(self.move_detect, diff_thresh, len(out_frames), len(self.frames))

                    # initialize all members
                    self.move_detect = False
                    self.frame_num = -1
                    self.start_frame = -1

                    if len(out_frames) > 10:
                        return out_frames
                    else:
                        return self.frames      # same as 'return []'

                return out_frames  # same as 'return []'

            return out_frames       # same as 'return []'

        return out_frames       # same as 'return []'
