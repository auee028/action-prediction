import os
import cv2
import numpy as np
import copy
import glob
from natsort import natsorted

from darknet.python.darknet import *


# angle_num = '00'
# data_path = "/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action/1/2020-01-13-09-33-36_{}_76/1.png".format(angle_num)
# data_path = "/media/pjh/HDD2/Dataset/ces-demo-4th/original_video/0109/Juhee/1/2020-01-09-17-24-38_00_69.avi"
#
# cap = cv2.VideoCapture(data_path)

# YOLO
yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
meta = load_meta("./darknet/cfg/coco.data")

w_margin = 20
h_margin = 10
CROP_SIZE = 170

dir_root = "/media/pjh/HDD2/Dataset/ces-demo-4th"
dirname = "ABR_action"
savename = "ABR_action-aug"
if not os.path.exists(os.path.join(dir_root, savename)):
    os.makedirs(os.path.join(dir_root, savename))

# log files directory
log_dir = "data_aug_log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# dir_list = natsorted(glob.glob(os.path.join(dir_root, dirname, '*/*')))
# # print(dir_list[:10])

dir_list = ["/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action/6/2020-01-09-09-59-51_02_117"]

# class_num = '1'      # 1 ~ 15
# dir_list = natsorted(glob.glob(os.path.join(dir_root, dirname, '{}/*'.format(class_num))))
# dir_list = dir_list[12:15]

video_NoPersonAtAll = 0
for f_num, imgs_dir in enumerate(dir_list):
    save_dir = imgs_dir.replace(dirname, savename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    imgs_dir = natsorted(glob.glob(imgs_dir + '/*'))

    # no_person_frames = []
    # first_person_flag = False
    # init_x, init_y, init_w, init_h = (0, 0, 0, 0)
    # prev_x, prev_y, prev_w, prev_h = (0, 0, 0, 0)
    '''
    Step 1. go through all frames in one video
    '''
    x_list = []
    y_list = []
    for image_path in imgs_dir:
        # print(image_path)
        image = cv2.imread(image_path)

        r = np_detect(yolo, meta, image)  # (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h))

        if len(r) >= 2:
            area = -1
            area_idx = -1
            for r_idx, r_info in enumerate(r):
                x, y, w, h = r_info[-1]
                if w * h > area:
                    area = w * h
                    area_idx = r_idx
            r = [r[area_idx]]

        if len(r) == 1:
            x, y, w, h = r[0][-1]

            x0 = int(x - w / 2)
            y0 = int(y - h / 2)
            x1 = int(x + w / 2)
            y1 = int(y + h / 2)

            x_list.append([x0, x1])
            y_list.append([y0, y1])

    # if any person is not detected in all frames
    if len(x_list) == 0:
        video_NoPersonAtAll += 1

        print("*** OUCH ! ANY PERSON IS NOT DETECTED IN ALL FRAMES ! ***")

        with open(os.path.join(log_dir, 'no_person_at_all.txt'), 'a+') as f:
            f.write("{}\n".format(save_dir))

        continue

    '''
    Step 2. Get min, max of overall moving area in x, y axis
    '''
    # print(len(x_list))
    x_min = min(np.array(x_list)[:, 0])
    y_min = min(np.array(y_list)[:, 0])
    x_max = max(np.array(x_list)[:, 1])
    y_max = max(np.array(y_list)[:, 1])

    # MARGIN = 10
    # x_min -= MARGIN
    # y_min -= MARGIN
    # x_max += MARGIN
    # y_max += MARGIN

    # for image_path in imgs_dir:
    #     image = cv2.imread(image_path)
    #
    #     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
    #     cv2.imshow('result', image)
    #     cv2.waitKey(50)
    # cv2.destroyAllWindows()

    sample_img = cv2.imread(imgs_dir[0])
    org_x = sample_img.shape[1] - 1     # x coordinate
    org_y = sample_img.shape[0] - 1     # y coordinate

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, org_x)
    y_max = min(y_max, org_y)

    '''
    Step 3. adjust frame location
    '''
    # make width and height to be same
    width = x_max - x_min + 1   # plus 1 because it is length
    height = y_max - y_min + 1
    if height > width:
        margin = height - width

        x_min -= margin / 2
        x_max += margin / 2

        # if margin is odd, extend 1 pixel to the left side
        if margin % 2 == 1:
            x_min -= 1

        if x_min < 0:
            x_max -= x_min
            x_min = 0

            if x_max >= org_x:   # same as org_x
                x_max = org_x

                # pass the current video because adjusted size is same as original video frame size
                print("================width: {}, height: {}=================".format((x_max - x_min + 1), (y_max - y_min + 1)))

                # log of passed video
                with open(os.path.join(log_dir, 'crop_passed.txt'), 'a+') as f:
                    f.write("path: {}\twidth: {}, height: {}\n".format(save_dir, (x_max - x_min + 1), (y_max - y_min + 1)))

                continue

        if x_max > org_x:
            x_min -= (x_max - org_x)
            x_max = org_x

            if x_min <= 0:   # same as org_x
                x_min = 0

                # pass the current video because adjusted size is same as original video frame size
                print("================width: {}, height: {}=================".format((x_max - x_min + 1), (y_max - y_min + 1)))

                # log of passed video
                with open(os.path.join(log_dir, 'crop_passed.txt'), 'a+') as f:
                    f.write("path: {}\twidth: {}, height: {}\n".format(save_dir, (x_max - x_min + 1), (y_max - y_min + 1)))

                continue


    else:
        margin = width - height

        y_min -= margin / 2
        y_max += margin / 2

        # if margin is odd, extend 1 pixel (default: up side)
        if margin % 2 == 1:
            y_min -= 1

        if y_min < 0:
            y_max -= y_min
            y_min = 0

            if y_max >= org_y:
                y_max = org_y

                # pass the current video because adjusted size is same as original video frame size
                print("================width: {}, height: {}=================".format((x_max - x_min + 1), (y_max - y_min + 1)))

                # log of passed video
                with open(os.path.join(log_dir, 'crop_passed.txt'), 'a+') as f:
                    f.write("path: {}\twidth: {}, height: {}\n".format(save_dir, (x_max - x_min + 1), (y_max - y_min + 1)))

                continue

        if y_max > org_y:
            y_min -= (y_max - org_y)
            y_max = org_y

            if y_min <= 0:  # same as org_x
                y_min = 0

                # pass the current video because adjusted size is same as original video frame size
                print("================width: {}, height: {}=================".format((x_max - x_min + 1), (y_max - y_min + 1)))

                # log of passed video
                with open(os.path.join(log_dir, 'crop_passed.txt'), 'a+') as f:
                    f.write("path: {}\twidth: {}, height: {}\n".format(save_dir, (x_max - x_min + 1), (y_max - y_min + 1)))

                continue

    # if adjusted width & height are not same
    if (x_max - x_min) != (y_max - y_min):
        # raw_input("Press Enter to continue...")
        print("\tNew Width & New Height Are Not Same !")
        print("\t{}\twidth: {}, height: {}".format(f_num, (x_max - x_min), (y_max - y_min)))

        # log of failed video
        with open(os.path.join(log_dir, 'crop_failed.txt'), 'a+') as f:
            f.write("path: {}\twidth: {}, height: {}\n".format(save_dir, (x_max - x_min), (y_max - y_min)))

        # break
        continue

    '''
    Step 4. crop frames as the area above
    '''
    # print(c_x - CROP_SIZE / 2, c_x + CROP_SIZE / 2, c_y - CROP_SIZE / 2, c_y + CROP_SIZE)
    for image_path in imgs_dir:
        frame = cv2.imread(image_path)

        # print("width: {}, height: {}".format(x_max - x_min, y_max - y_min))
        #
        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)
        # cv2.imshow('bbox', frame)
        # cv2.waitKey(50)

        cropped = frame[y_min:y_max, x_min:x_max, :]

        save_path = image_path.replace(dirname, savename)
        cv2.imwrite(save_path, cropped)

    # log of succeeded video
    with open(os.path.join(log_dir, 'crop_succeeded.txt'), 'a+') as f:
        f.write("path: {}\twidth: {}, height: {}\n".format(save_dir, x_max - x_min + 1, y_max - y_min + 1))

    print("{}\twidth: {}, height: {}".format(f_num, x_max - x_min + 1, y_max - y_min + 1))
    # print(x_min, y_min, x_max, y_max)
    # cv2.destroyAllWindows()

print("the number of videos that any person was not detected in all frames\n\t: {}".format(video_NoPersonAtAll))

