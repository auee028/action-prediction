import os
import cv2
import copy

from darknet.python.darknet import *


if __name__=="__main__":

    yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    meta = load_meta("./darknet/cfg/coco.data")

    video_path = "/home/pjh/Videos/Alley-39837.mp4"    # Office-39890.mp4
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # print(frame.shape)
        width, height, _ = frame.shape

        frame = cv2.resize(frame, (int(height / 10), int(width / 10)))

        r = np_detect(yolo, meta, frame)
        print(r)        # [('person', 0.996508777141571, (1558.762451171875, 1205.1583251953125, 1076.8004150390625, 1801.0845947265625)), ('person', 0.9982682466506958, (559.3720092773438, 1253.2493896484375, 919.6573486328125, 1749.1456298828125)), ('person', 0.999303936958313, (2674.815185546875, 1268.972412109375, 759.91650390625, 1704.06787109375)), ('person', 0.9766971468925476, (3631.482177734375, 1327.2301025390625, 402.9732666015625, 1550.7791748046875))]
        if not r:
            cv2.imshow('bbox', frame)

            continue

        d_frame = copy.deepcopy(frame)

        c_x, c_y, w, h = r[0][2]

        p_start = (int(round(c_x - w/2)), int(round(c_y - h/2)))
        p_end = (int(round(c_x + w/2)), int(round(c_y + h/2)))

        d_frame = cv2.rectangle(d_frame, p_start, p_end, (255, 255, 0), 3)

        cv2.imshow('bbox', d_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
