#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import argparse
import cv2
import numpy as np
import time
import datetime
import tensorflow as tf

from TFModel_socket import Camera
from crop_frames import CropFrames
from TFModel_socket import TFModel

import requests
from socket import *

from darknet.python.darknet import *


def normalize_probs(x):
    normalized = np.exp(x)/np.sum(np.exp(x))
    return normalized.tolist()

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
    # action_model = TFModel()
    action_model = TFModel(os.path.join('save_model', 'i3d'))

    result, confidence_3, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence_3[0] > 0.7:
        print(result, confidence_3[0], top_3)
        # return result, confidence_3, top_3

        # if result == 'working' or result == 'reading' or result == 'coming' or result == 'standing' or result == 'stretching':
        #     # print(frame_num, result, confidence, top_3)
        #     return result, confidence, top_3
        # else:
        #     result = None
    else:
        result = 'None'

    return result, confidence_3, top_3

def pred_action_crop(frames):
    action_model = TFModel(os.path.join('save_model', 'i3d_crop'))

    result, confidence_3, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence_3[0] > 0.7:
        print(result, confidence_3[0], top_3)
        # return result, confidence_3, top_3

        # if result == 'working' or result == 'reading' or result == 'coming' or result == 'standing' or result == 'stretching':
        #     # print(frame_num, result, confidence, top_3)
        #     return result, confidence, top_3
        # else:
        #     result = None

    else:
        result = 'None'

    return result, confidence_3, top_3

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
        try:
            value_list[i] = float(value_list[i] - np.min(value_list)) / (np.max(value_list) - np.min(value_list))
        except:
            value_list[i] = .0

    # value_diff = sum(value_list)/len(value_list)
    # print(value_diff)
    # print(sum(value_list))
    return sum(value_list)

def recv(csoc, count):
    buf = b''
    while count:
        newbuf = csoc.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main_loop():
    ssoc = socket(AF_INET, SOCK_STREAM)
    ssoc.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    ssoc.bind(('127.0.0.1', 5050))
    ssoc.listen(1)
    csoc=None

    try:
        while True:
            if csoc is None:
                print("waiting for connection")
                csoc, addr_info = ssoc.accept()
                print("got connection from {}".format(addr_info))
            else:
                while(cap.isOpened()):
                    ret, frame = cap.read()

                    ret_jpeg, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
                    stream = jpeg.tobytes()

                    # for ROI streaming
                    requests.post('http://127.0.0.1:5000/update_stream', data=stream)

                    frames = camera.get_frames(frame)
                    if len(frames) == 0:
                        continue

                    msg = "frames collected"
                    csoc.send(str(len(msg)).ljust(4))
                    csoc.send(msg)
                    print(msg)

                    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter('frames.avi', fourcc, 30, (224, 224))
                    count = 0
                    # print('frame length: ', len(frames))

                    for i in range(0, len(frames)):
                        out.write(frames[i])

                        count = count + 1

                    out.release()

                    print('writing video... (total frame: {})'.format(count))

                    start_send = time.time()
                    with open('frames.avi', 'rb') as file:
                        sendfile = file.read()
                    csoc.send(str(len(sendfile)).ljust(16))
                    csoc.send(sendfile)
                    end_send = time.time()

                    print('video send time: {}'.format(end_send - start_send))

                    time.sleep(0.01)


                    pred_start_time = time.time()
                    # get the value of frame variances(abs difference)
                    if calc_framediff(frames) < 34.25:  # Avg. diff of 100 samples : 34.2453916243 -> 34.25
                        # crop all images
                        cropped_frames = CropFrames(yolo, meta, frames)

                        # I3D model for cropped frames
                        tf.reset_default_graph()
                        result, confidence_3, top_3 = pred_action_crop(cropped_frames)
                        print("\tcrop : {}, {}, {}\n".format(result, confidence_3[0], top_3))
                        tf.reset_default_graph()

                    else:
                        # I3D model for original frames
                        tf.reset_default_graph()
                        result, confidence_3, top_3 = pred_action_orig(frames)
                        print("\torig  : {}, {}, {}".format(result, confidence_3[0], top_3))
                        tf.reset_default_graph()

                    pred_end_time = time.time()
                    print("action pred time: ", pred_end_time - pred_start_time)
                    #time.sleep(0.01)

                    if result == None:
                        result = 'None'


                    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    total_frames = len(frames)
                    name = "profLee"

                    file_name = '{}_{}_{}_{}_{}.avi'.format(date, total_frames, name, result, confidence_3[0])
                    print(file_name)

                    start_send = time.time()

                    csoc.send(str(len(file_name)).ljust(4))
                    csoc.send(file_name)

                    end_send = time.time()
                    print('filename send time: {}'.format(end_send - start_send))

                    # csoc.send(result)
                    # print('send result complete', result)

                    #time.sleep(0.01)
                    # csoc.close()
    except Exception as e:
        csoc.close()
        print("ERROR:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data collection server")

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--frame_width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--frame_height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=15)  # 6, 15(1920~424), 30(1280~320), 60

    args = parser.parse_args()


    cap = cv2.VideoCapture(args.cam)
    # self.cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_01_643.avi')
    cap.set(3, args.frame_width)
    cap.set(4, args.frame_height)
    cap.set(5, args.fps)

    camera = Camera()

    model = TFModel()

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while True:
        main_loop()

