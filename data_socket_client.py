import os
import time
import datetime
import argparse

import cv2
# import requests
from socket import *


def recv(csoc, count):
    buf = b''
    while count:
        newbuf = csoc.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def send_video(ip, port):
    csoc = socket(AF_INET, SOCK_STREAM)
    # csoc.connect(('127.0.0.1', 5050))
    csoc.connect((ip, port))

    video_root = "/media/pjh/HDD2/Dataset/ces-demo-5th"
    video_root = os.path.join(video_root, datetime.datetime.now().strftime("%m%d"), "profLee")
    if not os.path.exists(video_root):
        os.makedirs(video_root)

    file_count = 0

    while True:
        try:
            length = recv(csoc, 4)
            if length == None:
                continue

            msg = recv(csoc, int(length))
            print("{}  waiting a video file ...".format(msg))

            length = recv(csoc, 16)
            video = recv(csoc, int(length))                  # VIDEO file in .avi format

            length = recv(csoc, 4)
            file_name = recv(csoc, int(length))              # name : "DATE_TOTAL-FRAMES_NAME_ACTION-CLASS_CONFIDENCE.avi"

            with open(os.path.join(video_root, file_name), 'wb') as file:
                file.write(video)

            file_count += 1

            print("{}\t{}".format(file_count, file_name))

            if file_count == 100:
                csoc.close()
                break

        except Exception as e :
            print(e)
            csoc.close()
            exit()

def main_loop(ip, port):
    while True:
        send_video(ip=ip, port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data collection client")
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5050)

    args = parser.parse_args()

    # main_loop()
    main_loop(args.ip, args.port)
