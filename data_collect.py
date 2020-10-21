#-*- coding:utf-8 -*-
import sys
# from importlib import reload            # python 3
reload(sys)
sys.setdefaultencoding('utf-8')       # python 3 doesn't need this
import os
import argparse
import cv2
import numpy as np
import time
import copy
import math
import datetime

from TFModel_data import Camera


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

def main_loop(video_root, file_count):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ret_jpeg, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
        # stream = jpeg.tobytes()
        #
        # # for ROI streaming
        # # requests.post('http://127.0.0.1:5000/update_stream', data=stream)
        # # requests.post('http://155.230.104.191:5011/update_stream', data=stream)

        display_frame = copy.deepcopy(frame)
        cv2.putText(display_frame, str(file_count), (100, 50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7, (255, 255, 255, 2))
        cv2.imshow('img', display_frame)
        cv2.waitKey(10)

        frames = camera.get_frames(frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if len(frames) == 0:        # no action yet
            continue

        # action detected !

        # video name : "DATE_NUM-FRAMES_USER-NAME_V-COUNT.avi"
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        total_frames = len(frames)

        file_name = '{}_{}_{}_{}.avi'.format(date, total_frames, usrname, file_count)
        file_name = os.path.join(video_root, file_name)
        print('Saving a video..', file_name)

        file_count += 1

        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_name, fourcc, args.fps, (224, 224))
        count = 0
        # print('frame length: ', len(frames))

        for i in range(0, len(frames)):
            out.write(frames[i])

            count = count + 1

        out.release()

        print('Done! (total frame: {})'.format(count))

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data collection server")

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--frame_width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--frame_height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=15)  # 6, 15(1920~424), 30(1280~320), 60

    parser.add_argument('--ip', type=str, default='155.230.104.171')
    parser.add_argument('--port', type=int, default=5051)
    parser.add_argument('--usrname', type=str, default="profLee")

    args = parser.parse_args()

    usrname = args.usrname

    cap = cv2.VideoCapture(args.cam)
    # self.cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_01_643.avi')
    cap.set(3, args.frame_width)
    cap.set(4, args.frame_height)
    cap.set(5, args.fps)

    camera = Camera()

    # model = TFModel()

    # yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    # meta = load_meta("./darknet/cfg/coco.data")

    video_root = "/media/pjh/HDD2/Dataset/data_collection_final_demo"
    video_root = os.path.join(video_root, datetime.datetime.now().strftime("%m%d"), args.usrname)
    if not os.path.exists(video_root):
        os.makedirs(video_root)

    file_count = 0

    main_loop(video_root, file_count)

