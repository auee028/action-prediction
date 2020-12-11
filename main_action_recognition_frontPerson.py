#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import cv2
import copy
import numpy as np
import math
import os

import time
import argparse

from crop_frames_frontPerson import CropFrames
from TFModel_mtsi3d import TFModel

import requests
import socket
import threading
#import multiprocessing

from darknet.python.darknet import *

def sampling_frames(input_frames, sampling_num):
    total_num = len(input_frames)
    
    interval = 1
    if len(input_frames) > sampling_num:
        interval = math.floor(float(total_num) / sampling_num)

    # print("sampling interval : {}".format(interval))
    interval = int(interval)
    out_frames = []
    
    if interval > 200:
        return out_frames

    for n in range(min(len(input_frames), sampling_num)):
        out_frames.append(input_frames[n*interval])

    # # padding
    # if len(out_frames) < sampling_num:
    #     print("before padding : {}".format(len(out_frames)))
    #     for k in range(sampling_num - len(out_frames)):
    #         out_frames.append(input_frames[-1])

    return out_frames

def pred_action(frames):
    result, confidence, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence > 0.4:

        if result == 'leaving':
            result = 'None'

        return result, confidence, top_3
    else:
        result = 'None'

    return result, confidence, top_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--action_video_length', type=int, default=64)
    parser.add_argument('--action_thresh', type=int, default=4)
    parser.add_argument('--frame_thresh', type=int, default=20)
    parser.add_argument('--frame_diff_thresh', type=int, default=0.4)
    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--port', type=str, default="3009")

    args = parser.parse_args()

    action_model = TFModel()


    ############# added by WoongJae ###################
    #s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #s.connect(('8.8.8.8', 0))
    #ip = s.getsockname()[0]
    #rint("ip : ", ip, " ,port : ", args.port)
    mills = lambda: int(round(time.time() * 1000))
    cam_address = 'http://192.168.0.5:3009/?action=stream'
    ## docker 실행시에 --link 명령어를 이용해 카메라 스트리밍 도커 컨테이너와 연결 필수
    cwd_path = os.getcwd()
    ###################################################

    #cap = cv2.VideoCapture(args.cam)
    cap = cv2.VideoCapture(cam_address)
    #cap = cv2.VideoCapture(cam_address) # fix
    # cap.set(3, args.width)
    # cap.set(4, args.height)

    frames = []
    result = 'None'

    print("load yolo model....")
    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    prev_time = mills() # 현재 시간 ms 단위로 읽기
    wait_time = prev_time

    print("\nDo an action!")
    # time.sleep(1)

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    while True:
        now_time = mills()

        if (now_time - wait_time) > 60000:
            print("No Motion for 1 min - reset frame buffer")
            frames = []
            prev_time = mills()
            wait_time = prev_time

        # 스트리밍 서버는 20FPS로 동작중이므로, 현 모델이 학습된 10FPS의 영상을 얻기위해,
        # 100ms 에 한 번 이미지를 가지고 오도록 설정한다.
        # 
        #ret, frame = cap.read()
        # if not ret:
        #     break
        # if(now_time - prev_time) >= 100:
        ret, frame = cap.read()

        if not ret:
            try:
                print("LostContact")
                try:
                    requests.post(request_ip_address + 'action', headers = headers, data = json.dumps({'flags':'LostContact'}))
                except:
                    print("Flask server closed..")
                time.sleep(5)
                cap.release()
                print("Trying to Reconnect")
                cap = cv2.VideoCapture(cam_address)
            except:
                print("trying to reconnect cam server but failed.")
                continue

        prev_time = now_time

        try:
            # frame = cv2.resize(frame, (224, 224))
            assert frame is not None
        except:
            continue

        frames.append(frame)

        # when the movement stops
        if len(frames) >= args.action_video_length:
            print("Waiting the result..")

            # sampled_frames = sampling_frames(frames, args.action_video_length)
            # if len(sampled_frames)==0:
            #     frames=[]
            #     motion_detect = False
            #     continue

            # crop all images
            cropped_frames = np.array(CropFrames(yolo, meta, frames))

            if len(cropped_frames) == 0:
                frames = []
                prev_time = mills()
                wait_time = prev_time

                print("Do an action!")
                # time.sleep(1)

                continue

            cv2.imwrite('tmp/0.jpg', cropped_frames[0])

            # zero padding in time-axis
            maxlen = 64
            preprocessed = np.array(cropped_frames.tolist() + [np.zeros_like(cropped_frames[0])] * (maxlen - len(cropped_frames)))

            result, confidence, top_3 = pred_action(preprocessed)
            print("{}, {}, {}\n".format(result, confidence, top_3))
            ##### send to flask here
            if confidence > 0.3:
                try:
                    requests.post(request_ip_address + 'action', headers = headers, data = json.dumps({'flags':result}))
                except:
                    print("Flask Server Closed")
            
            # 한가지 액션에 대한 예측 완료 및 세팅 초기화
            frames=[]
            prev_time = mills()
            wait_time = prev_time

            print("Do an action!")
            # time.sleep(1)


        # if not ret || (prev - now)>100 end
    # while end
    cap.release()
    cv2.destroyAllWindows()

    # try:
    #     # requests.post('http://155.230.24.109:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     requests.get(
    #         'http://192.168.0.4:3001/api/v1/actions/action/{}/{}'.format('home', action_list[-1]))
    #     # requests.post('http://ceslea.ml:50001/api/v1/actions/action/{}/{}'.format('home',action_list[-1]))
    #     print('send action')
    # except:
    #     pass


