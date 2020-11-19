#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import cv2
import copy
import numpy as np
import math

import time
import argparse

from crop_frames import CropFrames
from TFModel_mtsi3d import TFModel

import requests

from darknet.python.darknet import *



class Camera:
    def __init__(self):

        self.move_detect = False
        self.frame_diff_thresh = 0.001  #0.4

        # # load yolo v3(tiny) and meta data
        # self.yolo = load_net("./darknet/cfg/yolov3-tiny-voc.cfg", "./darknet/cfg/yolov3-tiny-voc_210000.weights", 0)
        # self.meta = load_meta("./darknet/cfg/voc.data")

        self.frames = []

        self.frame_num = -1
        self.start_frame = -1

    def get_frames(self, frame):
        out_frames = []

        # frame = cv2.resize(frame, (224, 224))

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
                if diff_thresh < 0.001: #.1:  # when the movement stops
                    self.frames = self.frames[self.start_frame:]

                    out_frames = copy.deepcopy(self.frames)
                    self.frames = []

                    # print(self.move_detect, diff_thresh, len(out_frames), len(self.frames))

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


def sampling_frames(input_frames, sampling_num):
    total_num = len(input_frames)

    interval = 1
    if len(input_frames) > sampling_num:
        interval = math.floor(float(total_num) / sampling_num)
    print("sampling interval : {}".format(interval))
    interval = int(interval)

    out_frames = []
    for n in range(min(len(input_frames), sampling_num)):
        out_frames.append(input_frames[n*interval])

    # padding
    if len(out_frames) < sampling_num:
        print("before padding : {}".format(len(out_frames)))
        for k in range(sampling_num - len(out_frames)):
            out_frames.append(input_frames[-1])

    return out_frames

def pred_action(frames):
    result, confidence, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence > 0.7 and result != 'Doing other things':
        #print(result, confidence, top_3)

        return result, confidence, top_3
        # if result == 'working' or result == 'reading' or result == 'coming' or result == 'standing' or result == 'stretching':
        #     # print(frame_num, result, confidence, top_3)
        #     return result, confidence, top_3
        # else:
        #     result = None

    else:
        result = None

    return result, confidence, top_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--caption_video_length', type=int, default=64)
    parser.add_argument('--action_video_length', type=int, default=16)
    parser.add_argument('--action_thresh', type=int, default=20)
    parser.add_argument('--frame_thresh', type=int, default=10)
    parser.add_argument('--frame_diff_thresh', type=int, default=0.4)
    parser.add_argument('--waiting_time', type=int, default=8)

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60

    args = parser.parse_args()

    camera = Camera()

    action_model = TFModel()

    # cap = cv2.VideoCapture('/media/pjh/HDD2/SourceCodes/wonhee-takeover/event_detector/sample/200205/2020-02-05-17-49-01_00_415.avi')
    # cap = cv2.VideoCapture('/media/pjh/HDD2/SourceCodes/wonhee-takeover/event_detector/sample/200206/demo_samples/2020-02-06/10_reading-blowing nose-reading-blowing nose-reading-blowing nose/2020-02-06-15-01-50_00_1024.avi')
    # cap = cv2.VideoCapture(args.cam)
    # cap = cv2.VideoCapture('sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_00_642.avi')
    # cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/youtube/drama_0002.mp4')
    # cap = cv2.VideoCapture('/media/pjh/HDD2/Dataset/ces-demo-4th/trimmed_video/0109/Amin/1/2020-01-09-15-21-07_00_84.avi')    # sitting
    # cap = cv2.VideoCapture('/media/pjh/HDD2/Dataset/ces-demo-4th/trimmed_video/0110/Juhee/9/2020-01-10-17-11-05_00_99.avi')   # coming
    # cap = cv2.VideoCapture('/media/pjh/HDD2/Dataset/ces-demo-4th/trimmed_video/0110/Juhee/4/2020-01-09-17-27-09_00_99.avi')     # reading
    # cap = cv2.VideoCapture('/media/pjh/HDD2/Dataset/ces-demo-4th/trimmed_video/0113/jeongwoo/12/2020-01-10-18-15-12_00_86.avi')
    # cap = cv2.VideoCapture('/home/pjh/Videos/test_vid.avi')
    cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_00_642.avi')
    cap.set(3, args.width)
    cap.set(4, args.height)
    cap.set(5, args.fps)

    # # to save the video result
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # writer = cv2.VideoWriter('/home/pjh/Videos/test-arbitrary.avi', fourcc, 30.0, (args.width, args.height))

    frames = []
    sampled_frames = []
    frame_num = 1
    start_frame = 1
    action_end_frame = 1
    motion_detect = False
    result = None
    action_list = []
    intent_list = []
    intent_result = ''

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while True:
        # prev_time = time.time()
        action_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # display_frame = copy.deepcopy(frame)
        # display_frame = cv2.resize(display_frame, (224, 224))

        # cv2.imshow('frame', display_frame)
        frame = cv2.resize(frame, (224, 224))
        cv2.imshow('frame', frame)
        cv2.waitKey(10)

        '''
        frames.append(frame)
        # print(frame_num, len(frames))
        '''

        frames = camera.get_frames(frame)

        if len(frames) == 0:  # no action yet
            continue

        # action detected !
        sampled_frames = sampling_frames(frames[start_frame:], args.action_video_length)
        print('number of sampled frames : {}'.format(len(sampled_frames)))

        # crop all images
        cropped_frames = np.array(CropFrames(yolo, meta, sampled_frames))

        # zero padding in time-axis
        maxlen = 64
        preprocessed = np.array(cropped_frames.tolist() + [np.zeros_like(cropped_frames[0])] * (maxlen - len(cropped_frames)))

        # result, confidence, top_3 = pred_action(sampled_frames)#pred_action(frames[-args.action_video_length:], frame_num)
        # result, confidence, top_3 = pred_action(cropped_frames)
        result, confidence, top_3 = pred_action(preprocessed)
        print("{}, {}, {}\n".format(result, confidence, top_3))
        # cv2.waitKey(0)

        event_end_frame = frame_num

        action_time = time.time()  # reset action_time

        action_list.append(result)

        motion_detect = False
        sampled_frames = []

        # writer.write(display_frame)
        # cv2.imwrite('/home/pjh/Videos/test-arbitrary_frames/{}.jpg'.format(frame_num), display_frame)

        # print(time.time() - prev_time)
        frame_num = frame_num + 1

        # if len(frames) > args.caption_video_length:
        #     frames.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    # writer.release()
    cv2.destroyAllWindows()

    print('action list: {}\n'.format(action_list))
    # print('action time: {}'.format(start_frame, action_end_frame))
    # action_list = []

    """
    # get RNN result
    # input : action_list
    # output : predicted last action
    
    """

    # try:
    #     # requests.post('http://155.230.24.109:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     requests.get(
    #         'http://192.168.0.4:3001/api/v1/actions/action/{}/{}'.format('home', action_list[-1]))
    #     # requests.post('http://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     print('send action')
    # except:
    #     pass


