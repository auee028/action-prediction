#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import cv2
import copy
import numpy as np

import time
import argparse

from TFModel import TFModel

import requests

from darknet.python.darknet import *

def pred_action(frames, frame_num):
    result, confidence, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence > 0.7 and result != 'Doing other things':
        #print(result, confidence, top_3)

        # return result, confidence, top_3
        if result == 'working' or result == 'reading' or result == 'coming' or result == 'standing' or result == 'stretching':
            print(frame_num, result, confidence, top_3)
            return result, confidence, top_3
        else:
            result = None

    else:
        result = None

    return result, confidence, top_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--caption_video_length', type=int, default=64)
    parser.add_argument('--action_video_length', type=int, default=16)
    parser.add_argument('--action_thresh', type=int, default=3)

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60

    args = parser.parse_args()

    action_model = TFModel()

    # cap = cv2.VideoCapture('sample/200205/2020-02-05-17-49-01_00_415.avi')
    # cap = cv2.VideoCapture('/home/wonhee/event_detector/sample/200206/demo_samples/2020-02-06/10_reading-blowing nose-reading-blowing nose-reading-blowing nose/2020-02-06-15-01-50_00_1024.avi')
    # cap = cv2.VideoCapture(args.cam)
    cap = cv2.VideoCapture('sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_00_642.avi')
    cap.set(3, args.width)
    cap.set(4, args.height)
    cap.set(5, args.fps)

    frames = []
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

    while cap.isOpened():

        prev_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        display_frame = copy.deepcopy(frame)
        display_frame = cv2.resize(display_frame, (224, 224))

        # cv2.imshow('frame', display_frame)
        frame = cv2.resize(frame, (224, 224))

        frames.append(frame)

        # detect
        r = np_detect(yolo, meta, frame)

        if len(r) >= 1:

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
                # print(diff_thresh)

                if diff_thresh >= 0.3 and not motion_detect:
                    start_frame = frame_num
                    motion_detect = True
                # elif diff_thresh < 0.3 :
                #     motion_detect = False

                if motion_detect:
                    cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)

                    if motion_detect and (frame_num >= start_frame + args.action_video_length):
                        result, confidence, top_3 = pred_action(frames[-args.action_video_length:], frame_num)
                        print(result)
                        cv2.putText(display_frame, str(result), (100, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255, 2))
                        event_end_frame = frame_num

                        if len(action_list) >= args.action_thresh:
                            for th in range(1, args.action_thresh - 1):
                                th = -1 * th
                                if action_list[th] == result:
                                    event_checker = True
                                else:
                                    event_checker = False
                                if event_checker:
                                    print('event time:', start_frame, event_end_frame)
                        # print(result, confidence, top_3)

                        if result != None:
                            action_end_frame = frame_num
                            if len(action_list) >= args.action_thresh:
                                for th in range(1, args.action_thresh-1):
                                    th = -1*th
                                    if action_list[th] != result:
                                        action_list = []
                                        motion_detect = False
                                        break

                                    if len(intent_list) >= 1 and len(action_list) != 0:
                                        if intent_list[-1] != result:
                                            intent_list.append(result)
                                            try:
                                                # requests.post('http://155.230.24.109:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
                                                requests.get(
                                                    'http://192.168.0.4:3001/api/v1/actions/action/{}/{}'.format('home', intent_list[-1]))
                                                # requests.post('http://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
                                                print('send action')
                                            except:
                                                pass

                                            print('action list:', intent_list)
                                            print('action time:', start_frame, action_end_frame)
                                            action_list = []
                                            motion_detect = False

                                            if len(intent_list) >= 2:
                                                # input intent tree. output: intent_result
                                                if intent_list[-1] == 'reading':
                                                    if intent_list[-2] == 'coming':
                                                        intent_result = 'one comes to the office and the other keeps reading.'

                                                elif intent_list[-1] == 'working':
                                                    if intent_list[-2] == 'coming':
                                                        intent_result = 'one comes to the office and the other keeps working.'

                                                elif intent_list[-1] == 'standing':
                                                    if intent_list[-2] == 'working':
                                                        intent_result =  'a person finished working.'
                                                    elif intent_list[-2] == 'reading':
                                                        intent_result = 'a person finished reading.'

                                                elif intent_list[-1] == 'stretching':
                                                    if intent_list[-2] == 'working':
                                                        intent_result =  'a person finished working.'
                                                    elif intent_list[-2] == 'reading':
                                                        intent_result = 'a person finished reading.'

                                                elif len(action_list) >= 5:
                                                    intent_list.pop(0)


                                                if intent_result != '':
                                                    try:
                                                        # send action label
                                                        #requests.post('http://155.230.24.109:50001/api/v1/actions/intent/{}/{}'.format('home',intent_list[-1]))
                                                        requests.get('http://192.168.0.4:3001/api/v1/actions/intent/{}/{}'.format('home',intent_result))
                                                        #requests.post('http://ceslea.ml:50001/api/v1/actions/intent/{}/{}'.format('home',intent_list[-1]))
                                                        print('send captioning')
                                                    except:
                                                        pass
                                                    print('captioning:', intent_result)

                                                    # intent_list = []
                                                    intent_result = ''

                                    else:
                                        intent_list.append(result)
                                        try:
                                            # requests.post('http://155.230.24.109:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
                                            requests.get(
                                                'http://192.168.0.4:3001/api/v1/actions/action/{}/{}'.format('home', intent_list[-1]))
                                            # requests.post('http://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
                                            print('send action')
                                        except:
                                            pass

                                        print('actions:', intent_list)
                                        print('action time:', start_frame, action_end_frame)
                                        action_list = []
                                        motion_detect = False

                            else:
                                action_list.append(result)

                    if (frame_num > start_frame + args.action_video_length) and (result == None):
                        motion_detect = False
                        action_list = []

        cv2.imshow('frame', display_frame)

        # print(time.time() - prev_time)
        frame_num = frame_num + 1

        if len(frames) > args.caption_video_length:
            frames.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            # cv2.destroyAllWindows()
            break
