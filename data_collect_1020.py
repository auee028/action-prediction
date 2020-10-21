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
import datetime
import re
import math

from TFModel_data import Camera

with open('categories.txt', 'r') as f:
    classes = f.readlines()
class_dict = {}
for i, c in enumerate(classes):
    class_dict[i+1] = c.strip().replace(' ', '-')
print(class_dict)

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


def print_actions():
    print("================ ACTION LIST ================")
    print("1  : sitting")
    print("2  : standing")
    print("3  : drinking")
    print("4  : brushing")
    print("5  : dusting off(<- playing an instrument)")
    print("6  : speaking")
    print("7  : waving a hand")
    print("8  : working (= typing on laptop)")
    print("9  : coming")
    print("10 : leaving")
    print("11 : talking on the phone (= picking up and answering the phone)")
    print("12 : stretching")
    print("13 : nodding off")
    print("14 : reading")
    print("15 : blowing the nose\n")

def print_how2use():
    print("\n================ HOW TO USE ================")
    print(" (1) Looking at the action list above, enter an index number of action you want to record.")
    print(" (2) Then the count down number will appear.")
    print("     It is time for you to prepare an action.")
    print(" (3) After camera is loaded, you can take an action.")
    print("     (Please take an action for 3~5 seconds.)\n")

def print_main(saved_dir):
    print("\n###########################################################")
    print("##        *** RUNNING DATA COLLECTION PROGRAM ***        ##")
    print("## - Data is saved at : '{}' {} ##".format(saved_dir, ' '*14))
    print("## - Type in 'exit()' if you want to exit the program.   ##")
    print("###########################################################\n")

    print_actions()
    print_how2use()


def run(video_root):
    class_idx = int(video_root.split('/')[-1])
    class_name = class_dict[class_idx]
    # print(class_idx, class_name)

    print('Loading camera ..')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = copy.deepcopy(frame)
        # cv2.putText(display_frame, str(file_count), (100, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 5, (255, 255, 255, 2))
        cv2.imshow('img', display_frame)
        cv2.waitKey(10)

        frames = camera.get_frames(frame)

        if cv2.waitKey(1) == ord('q'):
            break

        if len(frames) == 0:        # no action yet
            continue

        # action detected !

        # video name : "ClassName_UserName_Date_NumFrames.avi"
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        total_frames = len(frames)
        print(np.shape(frames))

        file_name = '{}_{}_{}_{}.avi'.format(class_name, usrname, date, total_frames)
        file_name = os.path.join(video_root, file_name)
        print('Saving a video..', file_name)

        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        #out = cv2.VideoWriter(file_name, fourcc, args.fps, (args.frame_width, args.frame_height))
        out = cv2.VideoWriter(file_name, fourcc, args.fps, (frame.shape[1], frame.shape[0]))
        count = 0
        # print('frame length: ', len(frames))

        for i in range(0, len(frames)):
            out.write(frames[i])

            count = count + 1

        out.release()

        print('Done! (total frame: {})'.format(count))
        break

    cap.release()
    cv2.destroyAllWindows()



def main(video_root):
    print_main(saved_dir=video_root.replace('/home/', ''))

    while True:
        try:
            key_in = input("input key : ")

            if type(key_in) == int:
                if key_in in range(1, 16):
                    video_root = os.path.join(video_root, str(key_in))

                    if not os.path.exists(video_root):
                        os.makedirs(video_root)

                    run(video_root)

                else:
                    print("** CHECK THE ACTION INDEX AGAIN !\n")
                    print_actions()

            else:
                print_how2use()
                print("** Type in 'exit()' if you want to exit the program.\n")

                continue

        except NameError:
            print_how2use()
            print("** Type in 'exit()' if you want to exit the program.\n")

            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data collection server")

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--frame_width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--frame_height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
    parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60

    # parser.add_argument('--ip', type=str, default='155.230.14.96')
    # parser.add_argument('--port', type=int, default=5051)
    parser.add_argument('--usrname', type=str, default="profLee")

    args = parser.parse_args()

    usrname = args.usrname

    cap = cv2.VideoCapture(args.cam)
    cap.set(5, args.fps)
    #fps = int(cap.get(5))
    
    camera = Camera()

    # video_root = "/home/DB"
    video_root = '/media/pjh/HDD2/Dataset/data_collection_final_demo/DB'
    video_root = os.path.join(video_root, datetime.datetime.now().strftime("%m%d"), args.usrname)

    main(video_root)
