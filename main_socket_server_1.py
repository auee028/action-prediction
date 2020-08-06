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
from TFModel_socket_1 import I3DClassifier,LSTMPredictor

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
    action_model = I3DClassifier()

    result, confidence_3, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence_3[0] > 0.8:
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
    action_model = I3DClassifier(
        os.path.join('/media/pjh/HDD2/Dataset/save_model', 'i3d-ABR_action_augmented-{}'.format(4)))

    result, confidence_3, top_3 = action_model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence_3[0] > 0.8:
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

def pred_next_action(action_list):
    action_predictor = LSTMPredictor()

    with open('action_map.txt', 'r') as f:
        action_labels = [line.strip() for line in f.readlines()]
    ix2label = {i: n for i, n in enumerate(action_labels)}
    label2ix = {n: i for i, n in enumerate(action_labels)}

    action_seq = [label2ix[a] for a in action_list]
    action_seq = np.expand_dims(action_seq, 0)

    result = action_predictor.run_demo_wrapper(action_seq)      # <type 'numpy.ndarray'> , shape: (1, X)
    result = ix2label[result.tolist()[0]]

    # print([ix2label[a] for a in result])

    return result

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
                print("got connection from : {}".format((addr_info)))
                print("=============================================\n")
            else:
                start_recv = time.time()
                # print('file receive start')

                # receive a flag (frames / actions)
                flag = recv(csoc, 1)

                if flag == None:
                    continue

                # print(flag, type(flag))
                # print('get flag', int(flag))
                flag = int(flag)

                if flag == 0:   # when received data is 'frames'
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
                        print("\tcrop : {}, {}, {}".format(result, confidence_3[0], top_3))
                        tf.reset_default_graph()

                    else:
                        # I3D model for original frames
                        tf.reset_default_graph()
                        result, confidence_3, top_3 = pred_action_orig(frames)
                        print("\torig  : {}, {}, {}".format(result, confidence_3[0], top_3))
                        tf.reset_default_graph()

                    pred_end_time = time.time()
                    print("action recog time : {}".format(pred_end_time - pred_start_time))
                    #time.sleep(0.01)

                    if result == None:
                        result = 'None'

                    csoc.send(str(len(result)).ljust(4))
                    csoc.send(result)
                    print('send result complete')
                    print('data : {} (length : {})\n'.format(result, len(result.encode())))

                else:
                    pred_start_time = time.time()

                    tf.reset_default_graph()

                    length = recv(csoc, 4)

                    if length == None:
                        continue

                    # print(length, type(length))
                    # print('get length : {}'.format(int(length)))
                    print('*** received data: STRING (length : {}) ***'.format(int(length)))

                    recvdata = recv(csoc, int(length))

                    action_list = recvdata.split(',')

                    result = pred_next_action(action_list)

                    print("\t\tnext action : {}".format(result))

                    pred_end_time = time.time()
                    print("\taction pred time : {}".format(pred_end_time - pred_start_time))

                    if result == None:
                        result = 'None'

                    csoc.send(str(len(result)).ljust(4))
                    csoc.send(result)
                    print('\tsend result complete')
                    print('\tdata : {} (length : {})\n'.format(result, len(result.encode())))

                #time.sleep(0.01)
                # csoc.close()

    except Exception as e:
        csoc.close()
        print("ERROR:", e)


if __name__ == '__main__':
    action_classifier = I3DClassifier()

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    while True:
        main_loop()

