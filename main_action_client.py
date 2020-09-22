from TFModel_socket_1 import Camera

import os
import time
import argparse

import numpy as np
import cv2
import requests
from socket import *
import struct
import cPickle

def recv(csoc, count):
    buf = b''
    while count:
        newbuf = csoc.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def main_loop():
    while True:
        send_video()

def send_video():
    csoc = socket(AF_INET, SOCK_STREAM)
    csoc.connect(('127.0.0.1', 5050))

    # csoc = socket(AF_INET, SOCK_STREAM)
    # csoc.connect(('127.0.0.1', 5051))

    start_save = time.time()

    while True:
        try:
            retval, frame = cap.read()
            if retval == False:
                continue

            # check if frames are gathered enough
            frames, move_detect = model.get_frames(frame)
            # print(move_detect)
            if move_detect:
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)

            ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
            stream = jpeg.tobytes()

            # for ROI streaming
            requests.post('http://127.0.0.1:5000/update_stream', data=stream)

            if move_detect:
                # if (start_save - time.time()) < 5:
                #     print(start_save - time.time())
                #     continue
                # print("START TIME : ", start_save)
                print('*** send a VIDEO of current action ***')
                # print(np.shape(frames))

                # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('frames.avi', fourcc, 30, (224, 224))
                count = 0
    
                for i in range(0, len(frames)):
                    out.write(frames[i])
    
                    count = count + 1
    
                out.release()
                
                print('writing video...')
                print('total frame number: {}'.format(count))
    
                start_save = time.time()
                with open('frames.avi', 'rb') as file:
                    frames = file.read()
                csoc.send(str(len(frames)).ljust(16))           # add info of file length
                csoc.send(frames)                               # add info of data
                end_save = time.time()
    
                print('video send time : {}'.format(end_save - start_save))
    
                time.sleep(0.01)
    
                # print('waiting result...')
                # length = recv(csoc, 4)
                # frames_result = recv(csoc, int(length))
                # print('\tframes result : {}\n'.format(frames_result))

                '''
                count = len(frames)
                print('total frame number: {}'.format(count))

                start_save = time.time()

                for cur_frame in frames:
                    cur_frame = cPickle.dumps(cur_frame)
                    cur_frame_size = len(cur_frame)
                    p = struct.pack('I', cur_frame_size)
                    cur_frame = p + cur_frame
                    csoc.sendall(cur_frame)

                    # time.sleep(0.01)

                cur_frame_size = 10
                p = struct.pack('I', cur_frame_size)
                csoc.send(p)
                csoc.send('')

                end_save = time.time()
                print('video send time : {}'.format(end_save - start_save))
                '''


        except Exception as e :
            print(e)
            csoc.close()
            exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="test TF on a single video")
    parser.add_argument('--video_length', type=int, default=35)

    parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60

    parser.add_argument('--cam', type=int, default=0)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
    parser.add_argument('--frame_width', type=int,
                        default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
    parser.add_argument('--frame_height', type=int,
                        default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240

    parser.add_argument('--test_mode', type=bool, default=True)  # True, False
    parser.add_argument('--debug', type=str, default=False)

    parser.add_argument('--action_video_length', type=int, default=16)
    parser.add_argument('--action_thresh', type=int, default=20)
    parser.add_argument('--frame_thresh', type=int, default=10)
    parser.add_argument('--frame_diff_thresh', type=int, default=0.4)
    parser.add_argument('--waiting_time', type=int, default=8)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(3, args.frame_width)
    cap.set(4, args.frame_height)
    cap.set(5, args.fps)

    model = Camera()

    main_loop()
