#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this
import os
import cv2
import numpy as np
import time
import tensorflow as tf

from crop_frames import CropFrames
from TFModel_gesture import TFModel as GestureModel
from TFModel_action import TFModel as ActionModel
import support_function as sf

import requests
from socket import *
import struct
import cPickle

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

# def pred_action(frames):
#     result, confidence_3, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))
#
#     if confidence_3[0] > 0.8:
#         print(result, confidence_3[0], top_3)
#
#     else:
#         result = 'None'
#
#     requests.get('http://155.230.104.191:5000/state/set/action', params={'action': result})
#     requests.get('http://155.230.104.191/state/set/action_panel', params={'labels': str(top_3),
#                                                                          'probs': str(normalize_probs(confidence_3))})
#
#     return result, confidence_3, top_3

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

    # scaling values to [0-1] range using formula
    for i in range(len(value_list)):
        try:
            value_list[i] = float(value_list[i] - np.min(value_list)) / (np.max(value_list) - np.min(value_list))
        except:
            value_list[i] = .0

    return sum(value_list)

def recv(csoc, count):
    buf = b''
    while count:
        newbuf = csoc.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main_loop(model_flag):
    ssoc = socket(AF_INET, SOCK_STREAM)
    ssoc.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    ssoc.bind(('127.0.0.1', 5050))
    # ssoc.bind(('155.230.104.171', 5050))
    ssoc.listen(1)
    csoc=None

    data = ''
    payload_size = struct.calcsize('I')


    while True:
        if csoc is None:
            print("waiting for connection")
            csoc, addr_info = ssoc.accept()
            print("got connection from : {}".format((addr_info)))
            print("=============================================\n")
        else:
            start_recv = time.time()
            # print('file receive start')

            length = recv(csoc, 16)

            if length == None:
                continue

            # print(length, type(length))
            # print('get length', int(length))
            print('*** received data : VIDEO (length : {}) ***'.format(int(length)))

            recvfile = recv(csoc, int(length))

            with open('frames_recv.avi', 'wb') as file:
                file.write(recvfile)
            end_recv = time.time()
            print('file receive complete ({} sec)'.format(end_recv - start_recv))

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
            cap.release()

            '''
            start_recv = time.time()
    
            frames = []
            while len(data) < payload_size:
                data += csoc.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack('I', packed_msg_size)[0]
            while len(data) < msg_size:
                data += csoc.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            if frame_data=='':
                break
            frame = cPickle.loads(frame_data)
            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
            frames.append(frame)
            
            end_recv = time.time()
            print('file receive complete ({} sec)'.format(end_recv - start_recv))
            # print(np.shape(frames))
            '''

            pred_start_time = time.time()
            # get the value of frame variances(abs difference)
            # if calc_framediff(frames) < 34.25:  # Avg. diff of 100 samples : 34.2453916243 -> 34.25
            # crop all images
            frame_batch = CropFrames(yolo, meta, frames)
            frame_batch = sampling_frames(frame_batch, maxlen)
            frame_batch = np.expand_dims(frame_batch, 0)
            # print(np.shape(frame_batch))

            # I3D model for cropped frames
            # result, confidence_3, top_3 = pred_action(cropped_frames)
            print("gesture waiting ...")
            gesture_result, gesture_confidence_3, gesture_top_3 = gesture_model.run_demo_wrapper(frame_batch)
            print("G", gesture_result, gesture_confidence_3, gesture_top_3)
            print("action waiting ...")
            action_result, action_confidence_3, action_top_3 = action_model.run_demo_wrapper(frame_batch)
            print("A", action_result, action_confidence_3, action_top_3)

            if gesture_result == "Thumb Up":
                model_flag = "G"
            elif (model_flag == "G") & (gesture_result == "Stop"):
                model_flag = "A"

            if model_flag == "G":
                result = gesture_result
                conf_3 = gesture_confidence_3
                top_3 = gesture_top_3
            else:
                result = action_result
                conf_3 = action_confidence_3
                top_3 = action_top_3

            if conf_3 > 0.8:
                pass

            else:
                result = 'None'

            requests.get('http://127.0.0.1:5000/state/set/action', params={'action': result})
            requests.get('http://127.0.0.1:5000/state/set/action_panel', params={'labels': str(top_3),
                                                                                  'probs': str(normalize_probs(conf_3))})

            print("\taction results : {}, {}, {}".format(result, conf_3, top_3))

            pred_end_time = time.time()
            print("action recog time : {}\n".format(pred_end_time - pred_start_time))
            #time.sleep(0.01)

            if result == None:
                result = 'None'

            # csoc.send(str(len(result)).ljust(4))
            # csoc.send(result)
            # print('send result complete')
            # print('data : {} (length : {})\n'.format(result, len(result.encode())))

            # else:
            #     result = 'Large amount of frame difference'
            #     print(result)
            #
            #     csoc.send(str(len(result)).ljust(4))
            #     csoc.send(result)
            #     print('send result complete')
            #     print('data : {} (length : {})\n'.format(result, len(result.encode())))


    # except Exception as e:
    #     csoc.close()
    #     print("ERROR:", e)


if __name__ == '__main__':
    #action_classifier = I3DClassifier()

    gesture_model = GestureModel()
    action_model = ActionModel()

    maxlen = 64

    model_flag = "A"        # G; gesture / A; action
    result, conf_3, top_3 = None, None, None

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while True:
        main_loop(model_flag)

