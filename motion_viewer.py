import cv2
import copy
import numpy as np

import time
import argparse


parser = argparse.ArgumentParser(description="test TF on a single video")
parser.add_argument('--video_length', type=int, default=64)


parser.add_argument('--cam', type=int, default=13)  # 0~9 / 10: color / 11: ir1 / 12: ir2 / 13: ir1 + ir2
parser.add_argument('--width', type=int,
                    default=640)  # RGB(YUY2): 1920x1080, 1280x720, 960x540, 848x480, 640x480, 640x360, 424x240, 320x240, 320x180
parser.add_argument('--height', type=int,
                    default=480)  # DEPTH : 1280x720, 848x480, 640x480, 640x360, 480x270, 424x240
parser.add_argument('--fps', type=int, default=10)  # 6, 15(1920~424), 30(1280~320), 60

args = parser.parse_args()


cap = cv2.VideoCapture('sample/2020-02-05-17-49-01_00_415.avi')

cap.set(3, args.width)
cap.set(4, args.height)
cap.set(5, args.fps)

frames = []
frame_num = 1
start_frame = 1
motion_detect = False

while cap.isOpened():

    prev_time = time.time()

    ret, frame = cap.read()

    display_frame = copy.deepcopy(frame)

    display_frame = cv2.resize(display_frame, (224, 224))

    # cv2.imshow('frame', display_frame)
    frame = cv2.resize(frame, (224, 224))

    frames.append(frame)

    if len(frames) >= 5:
        c_frame = frames[-1]
        c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
        b_frame = frames[-2]
        b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)
        a_frame = frames[-3]
        a_frame = cv2.cvtColor(a_frame, cv2.COLOR_BGR2GRAY)

        cb_frame_diff = cv2.absdiff(c_frame, b_frame)
        ba_frame_diff = cv2.absdiff(b_frame, a_frame)

        cba_frame_diff = cv2.absdiff(cb_frame_diff, ba_frame_diff)
        _, cba_frame_diff = cv2.threshold(cba_frame_diff, 30, 255, cv2.THRESH_BINARY)

        cb_diff_mask = np.array(cb_frame_diff > 10, dtype=np.int32)
        ba_diff_mask = np.array(ba_frame_diff > 10, dtype=np.int32)
        cba_diff_mask = np.array(cba_frame_diff > 10, dtype=np.int32)

        cb_frame_diff = cv2.cvtColor(cb_frame_diff, cv2.COLOR_GRAY2BGR)
        ba_frame_diff = cv2.cvtColor(ba_frame_diff, cv2.COLOR_GRAY2BGR)
        cba_frame_diff = cv2.cvtColor(cba_frame_diff, cv2.COLOR_GRAY2BGR)

        extend_frame = np.hstack((display_frame, cb_frame_diff))
        extend_frame = np.hstack((extend_frame, ba_frame_diff))
        extend_frame = np.hstack((extend_frame, cba_frame_diff))

        # print(np.sum(cb_diff_mask), np.sum(ba_diff_mask), np.sum(cba_diff_mask))
        diff_thresh = (np.sum(cba_diff_mask)/max(np.sum(cb_diff_mask), np.sum(ba_diff_mask)))

        if diff_thresh >= 0.3 and not motion_detect:
            start_frame = frame_num
            motion_detect = True

        if motion_detect:
            cv2.circle(extend_frame, (50, 50), 20, (0, 0, 255), -1)

        if frame_num > start_frame + args.video_length:
            motion_detect = False

        cv2.imshow('frame', extend_frame)

        # print(time.time() - prev_time)
    frame_num = frame_num + 1

    # if len(frames) > args.video_length:
    #     frames.pop(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        # cv2.destroyAllWindows()
        break
