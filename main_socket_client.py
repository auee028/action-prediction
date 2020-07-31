from TFModel_socket import Camera

import time
import argparse

import cv2
import requests
from socket import *


def main_loop():
    while True:
        send_video()

def send_video():
    csoc = socket(AF_INET, SOCK_STREAM)
    csoc.connect(('127.0.0.1', 5050))

    while True:
        try:
            retval, frame = cap.read()
            if retval == False:
                continue

            ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)))
            stream = jpeg.tobytes()

            # for ROI streaming
            requests.post('http://127.0.0.1:5000/update_stream', data=stream)

            frames = model.get_frames(frame)
            if len(frames) == 0:
                continue

            # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('frames.avi', fourcc, 30, (224, 224))
            count = 0
            print('frame length: ', len(frames))

            for i in range(0, len(frames)):
                out.write(frames[i])

                count = count + 1

            out.release()

            print('writing video...total frame:', count)

            start_save = time.time()
            with open('frames.avi', 'rb') as file:
                sendfile = file.read()
            csoc.send(str(len(sendfile)).ljust(16))
            csoc.send(sendfile)
            end_save = time.time()

            print('video send time:', end_save - start_save)

            time.sleep(0.01)
            # print('waiting result...')
            #
            # result = csoc.recv(4)
            #
            # print(result)

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
