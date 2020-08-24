from TFModel_socket_1 import Camera

import os
import time
import argparse

import cv2
import requests
from socket import *

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

    action_list = []

    START_TIME = time.time()
    FIRST_ACTION = True

    notice = ''

    while True:
        try:
            retval, frame = cap.read()
            if retval == False:
                continue

            # check if frames are gathered enough
            frames, move_detect = model.get_frames(frame)
            if move_detect:
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)

                notice = "Movement detected !"
                requests.get('http://127.0.0.1:5000/state/set/alert_user', params={'alert_user': notice})

            ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
            stream = jpeg.tobytes()

            # for ROI streaming
            requests.post('http://127.0.0.1:5000/update_stream', data=stream)

            if len(frames) == 0:
                if ((time.time() - START_TIME) > 8):     # no action in 8 secs
                    if FIRST_ACTION == True:    # if any action has not been taken(or recognized)
                        continue

                    # remove redundant arguments
                    while True:
                        try:
                            action_list.remove('None')
                        except:
                            # print('No more None in result list')
                            break

                    # print(time.time() - START_TIME)
                    if len(action_list) < 1:
                        # print('Take an action again ... !\n')
                        notice = 'Take an action again ... !'
                        requests.get('http://127.0.0.1:5000/state/set/alert_user', params={'alert_user': notice})

                        START_TIME = time.time()

                        continue

                    print('*** send a sequence of ACTIONS ***')

                    start_send = time.time()
                    actions = ','.join(action_list)

                    csoc.send('1')                              # add info of flag (actions in string)
                    csoc.send(str(len(actions)).ljust(4))       # add info of file length
                    csoc.send(actions)                          # add info of dataprint('video send time:', end_save - start_save)
                    end_send = time.time()

                    print('\tstring send time : {}'.format(end_send - start_send))

                    time.sleep(0.01)

                    print('\twaiting result...')
                    length = recv(csoc, 4)
                    actions_result = recv(csoc, int(length))
                    print('\t\tactions result : {}\n'.format(actions_result))

                    # initialize action_list and START_TIME
                    action_list = []
                    START_TIME = time.time()

                    continue

                else:
                    # keep gathering a frame
                    continue

            print('*** send a VIDEO of current action ***')

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
            csoc.send('0')                                  # add info of flag (frames in video)
            csoc.send(str(len(frames)).ljust(16))           # add info of file length
            csoc.send(frames)                               # add info of data
            end_save = time.time()

            print('video send time : {}'.format(end_save - start_save))

            time.sleep(0.01)

            print('waiting result...')
            length = recv(csoc, 4)
            frames_result = recv(csoc, int(length))
            print('\tframes result : {}\n'.format(frames_result))

            if frames_result != 'None':
                FIRST_ACTION = False

            # # just for test
            # action_list.append('reading')
            # FIRST_ACTION = False
            # action_list.append(frames_result)

            START_TIME = time.time()

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
    # self.cap = cv2.VideoCapture('/home/pjh/PycharmProjects/action-prediction/sample/200206/demo_recogtest-JH/2020-02-06/14_coming in-sitting-reading-nodding off-standing-sitting/2020-02-06-15-11-05_01_643.avi')
    cap.set(3, args.frame_width)
    cap.set(4, args.frame_height)
    cap.set(5, args.fps)

    model = Camera()

    main_loop()
