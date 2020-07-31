#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this

import cv2
import numpy as np
import time
import tensorflow as tf

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
    action_model = TFModel()

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

    requests.get('http://127.0.0.1:5000/state/set/action', params={'action': result})
    requests.get('http://127.0.0.1:5000/state/set/action_panel', params={'labels': str(top_3),
                                                                         'probs': str(normalize_probs(confidence_3))})

    return result, confidence_3, top_3

def pred_action_crop(frames):
    action_model = TFModel(
        os.path.join('/media/pjh/HDD2/Dataset/save_model', 'i3d-ABR_action_augmented-{}'.format(4)))

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

    requests.get('http://127.0.0.1:5000/state/set/action', params={'action': result})
    requests.get('http://127.0.0.1:5000/state/set/action_panel', params={'labels': str(top_3),
                                                                         'probs': str(normalize_probs(confidence_3))})

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
                print("got connection from".format(addr_info))
            else:
                start_recv = time.time()
                # print('file receive start')

                length = recv(csoc, 16)

                if length == None:
                    continue

                print(length, type(length))
                print('get length', int(length))
                recvfile = recv(csoc, int(length))

                with open('frames_recv.avi', 'wb') as file:
                    file.write(recvfile)
                end_recv = time.time()
                print('file receive complete', end_recv - start_recv)

                start_read = time.time()
                cap = cv2.VideoCapture('frames_recv.avi')
                frames = []
                count = 0
                ret = True
                while ret:
                    ret, frame = cap.read()
                    # cv2.imshow('frame', frame)
                    # cv2.waitKey(1)
                    if ret:
                        frames.append(frame)
                        count = count + 1
                # print('frame read time:', time.time() - start_read)
                # print('total frame', len(frames))
                # print('frame read complete, total frame:', count)
                cap.release()

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

                print('data length : ', len(result.encode()))

                # csoc.send(result)
                # print('send result complete', result)

                #time.sleep(0.01)
                # csoc.close()
    except Exception as e:
        csoc.close()
        print("ERROR:", e)


if __name__ == '__main__':
    model = TFModel()

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while True:
        main_loop()

