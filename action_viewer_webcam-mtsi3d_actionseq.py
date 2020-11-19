#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import cv2
import copy
import glob
import natsort
import numpy as np
import math
import tensorflow as tf

import time
import argparse

from crop_frames import CropFrames
# from TFModel import TFModel
# from TFModel_twoI3D import TFModel# as TFModel_cropped
from TFModel_mtsi3d import TFModel
import requests

from darknet.python.darknet import *

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

def pred_action_orig(frames):
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


def calc_framediff(clip):
    value_list = []
    for idx in range(len(clip)-1):
        prev_frame = clip[idx]
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_frame = clip[idx+1]
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(prev_frame, curr_frame)
        frame_diff = np.array(frame_diff, dtype=np.int32)

        value_list.append(np.sum(frame_diff))

    # print(max(value_list), min(value_list))

    # scaling values to [0-1] range using formula
    for i in range(len(value_list)):
        value_list[i] = float(value_list[i] - np.min(value_list)) / (np.max(value_list) - np.min(value_list))

    # value_diff = sum(value_list)/len(value_list)
    # print(value_diff)
    # print(sum(value_list))
    return sum(value_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--caption_video_length', type=int, default=64)
    parser.add_argument('--action_video_length', type=int, default=64)
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
    cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_01_643.avi')
    # cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/200206/demo_recogtest-JH/2020-02-06/12_reading- talking on a phone-leaving the office/2020-02-06-15-09-17_01_391.avi')

    cap.set(3, args.width)
    cap.set(4, args.height)
    cap.set(5, args.fps)

    # # to save the video result
    # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    # writer = cv2.VideoWriter('/home/pjh/Videos/test-arbitrary.avi', fourcc, 30.0, (args.width, args.height))

    action_model = TFModel()

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
    cnt = 0

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while cap.isOpened():

        # prev_time = time.time()
        action_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
        # stream = jpeg.tobytes()
        #
        # # for ROI streaming
        # requests.post('http://127.0.0.1:5000/update_stream', data=stream)

        display_frame = copy.deepcopy(frame)
        display_frame = cv2.resize(display_frame, (224, 224))

        # cv2.imshow('frame', display_frame)
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.flip(frame, 1)

        frames.append(frame)
        # print(frame_num, len(frames))

        # detect
        r = np_detect(yolo, meta, frame)

        if len(r) >= 1:
            # cv2.circle(display_frame, (50, 50), 20, (255, 0, 0), -1)

            if len(frames) >= 5:
                c_frame = frames[-1]
                c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
                b_frame = frames[-2]
                b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)
                a_frame = frames[-3]
                a_frame = cv2.cvtColor(a_frame, cv2.COLOR_BGR2GRAY)

                cb_frame_diff = cv2.absdiff(c_frame, b_frame)
                ba_frame_diff = cv2.absdiff(b_frame, a_frame)

                cba_frame_diff = cv2.absdiff(cb_frame_diff, ba_frame_diff)
                _, cba_frame_diff = cv2.threshold(cba_frame_diff, 30, 255, cv2.THRESH_BINARY)

                cb_diff_mask = np.array(cb_frame_diff > 10, dtype=np.int32)
                ba_diff_mask = np.array(ba_frame_diff > 10, dtype=np.int32)
                cba_diff_mask = np.array(cba_frame_diff > 10, dtype=np.int32)

                try:
                     diff_thresh = float(1.0*np.sum(cba_diff_mask)/max(np.sum(cb_diff_mask), np.sum(ba_diff_mask)))

                except:
                    diff_thresh = 0
                # print('(threshold : {})'.format(diff_thresh))

                if diff_thresh >= args.frame_diff_thresh and not motion_detect:
                    start_frame = frame_num - 2
                    motion_detect = True
                    cnt += 1
                    print("{}\tstart frame : {}\n".format(cnt, start_frame))
                # elif diff_thresh < 0.3 :
                #     motion_detect = False
                #     # sampled_frames = []

                if motion_detect:
                    cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)    # 12
                    # print(start_frame)

                    action_time = time.time()   # reset action_time

                    if diff_thresh < args.frame_diff_thresh:#frame_num >= start_frame + args.action_video_length:

                        # print(diff_thresh, start_frame, frame_num, len(frames), len(frames[start_frame:]))  #########

                        # small_diff += 1
                        # if small_diff >= 5:
                        #     motion_detect = False

                        if len(frames[start_frame:]) >= args.frame_thresh:
                            # print('SHOW !')
                            # for f in frames[start_frame:]:
                            #     # cv2.imshow('frame', frames[-(args.action_video_length-i)])
                            #     cv2.imshow('frame', f)
                            #     cv2.waitKey(500)
                            print('total frames : {}'.format(len(frames[start_frame:])))

                            sampled_frames = sampling_frames(frames[start_frame:], args.action_video_length)
                            for num, f in enumerate(sampled_frames):
                                if not os.path.exists('/home/pjh/Videos/test-arbitrary_parts/{}'.format(len(action_list))):
                                    os.makedirs('/home/pjh/Videos/test-arbitrary_parts/{}'.format(len(action_list)))
                                cv2.imwrite('/home/pjh/Videos/test-arbitrary_parts/{}/{}.jpg'.format(len(action_list), num), f)
                            #     # cv2.imshow('frame', frames[-(args.action_video_length-i)])
                            #     cv2.imshow('frame', f)
                            #     cv2.waitKey(300)
                            print('number of sampled frames : {}'.format(len(sampled_frames)))

                            result, confidence, top_3 = pred_action_orig(sampled_frames)
                            print("\torig  : {}, {}, {}".format(result, confidence, top_3))

                            # requests.post(
                            #     'http://127.0.0.1:5000/post/{}'.format(action))


                            cv2.putText(display_frame, str(result), (100, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                            event_end_frame = frame_num

                            action_time = time.time()  # reset action_time

                            if True:#result != None:
                                # action_end_frame = frame_num
                                action_list.append(result)

                                if len(action_list) >= args.action_thresh:
                                    motion_detect = False
                                    break
                            else:
                                print("Do the previous action again ..\n")

                            motion_detect = False
                            sampled_frames = []

                    waiting_time = time.time() - action_time
                    if waiting_time > args.waiting_time:
                        print("waiting time : {}".format(waiting_time))
                        print("Finish detecting actions .\n")
                        break

                    # if (frame_num > start_frame + args.action_video_length) and (result == None):
                    #     motion_detect = False
                    #     action_list = []

        cv2.imshow('frame', display_frame)
        cv2.waitKey(50)
        # writer.write(display_frame)
        cv2.imwrite('/home/pjh/Videos/test-arbitrary_frames/{}.jpg'.format(frame_num), display_frame)

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
    #     # requests.get(
    #     #     'http://192.168.0.4:3001/api/v1/actions/action/{}/{}'.format('home', action_list[-1]))
    #     requests.post('http://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     print('send action')
    # except:
    #     pass


