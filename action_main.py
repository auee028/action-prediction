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

# from crop_frames import CropFrames
from crop_frames_frontPerson import CropFrames
from TFModel import MultiscaleI3D

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

def pred_action(frames):
    result, confidence, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence > 0.7:

        if result == 'playing instrument':
            result = 'playing an uculele'
        elif result == 'waving a hand':
            result = 'waving hands'
        elif result == 'stretching':
            result = 'stretching arm'

        if result not in ['stretching arm', 'brushing', 'reading', 'drinking', 'waving hands']:
            result = None

        return result, confidence, top_3

    else:
        result = None

    return result, confidence, top_3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--action_video_length', type=int, default=64)
    parser.add_argument('--action_thresh', type=int, default=3)
    parser.add_argument('--frame_thresh', type=int, default=20)
    parser.add_argument('--frame_diff_thresh', type=int, default=0.4)
    parser.add_argument('--waiting_time', type=int, default=8)

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60
    parser.add_argument('--name', type=str, default="user")

    parser.add_argument('--mtsi3d_weights', type=str, default="mtsi3d")
    
    args = parser.parse_args()

    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, time.strftime("%Y-%m-%d"))
    with open(log_file, 'a') as f:
        f.write("---{}---\n".format(args.name))

    action_model = MultiscaleI3D(args.mtsi3d_weights)

    cap = cv2.VideoCapture(args.cam)

    cap.set(3, args.width)
    cap.set(4, args.height)
    cap.set(5, args.fps)


    frames = []
    sampled_frames = []
    frame_num = 1
    # start_frame = 1
    action_end_frame = 1
    # motion_detect = False
    result = None
    action_list = []

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        stream = jpeg.tobytes()

        requests.post("http://127.0.0.1:5001/update_stream", data=stream)

        frame = cv2.resize(frame, (224, 224))

        frames.append(frame)

        if len(frames) >= 64:
            sampled_frames = sampling_frames(frames, args.action_video_length)
            print('number of sampled frames : {}'.format(len(sampled_frames)))

            # crop all images
            cropped_frames = np.array(CropFrames(yolo, meta, sampled_frames))

            cv2.imwrite('tmp/img.jpg', cropped_frames[0])

            result, confidence, top_3 = pred_action(cropped_frames)
            print("{}, {}, {}\n".format(result, confidence, top_3))
                            
            if result == None:
                result = 'Waiting...'

            if not result == 'Waiting...':
                with open(log_file, 'a') as f:
                    f.write("{} | result: {} | confidence: {}\n".format(time.strftime("%Y-%m-%d_%H-%M-%S"), result, confidence))

            requests.get('http://127.0.0.1:5001/state/set/action', params={'action': result})

            # initialization
            frames = []
            sampled_frames = []
            frame_num = 0
            action_end_frame = 1

        frame_num = frame_num + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()




