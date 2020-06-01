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


dir_list = natsorted(glob.glob(os.path.join(dir_root, dirname, '*/*')))
# # print(dir_list[:10])

# dir_list = ["/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action/1/2020-01-09-09-51-48_00_77"]

# class_num = '1'      # 1 ~ 15
# dir_list = natsorted(glob.glob(os.path.join(dir_root, dirname, '{}/*'.format(class_num))))
# dir_list = dir_list[12:15]


for f_num, imgs_dir in enumerate(dir_list[1719:]):
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

        if len(r) == 1:
            x, y, w, h = r[0][-1]

            x0 = int(x - w / 2)
            y0 = int(y - h / 2)
            x1 = int(x + w / 2)
            y1 = int(y + h / 2)

            x_list.append([x0, x1])
            y_list.append([y0, y1])

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




    '''
        if (width <= CROP_SIZE) & (height <= CROP_SIZE):    # moving in crop area
            crop_x0 = c_x - CROP_SIZE / 2
            crop_x1 = c_x + CROP_SIZE / 2
            crop_y0 = c_y - CROP_SIZE / 2
            crop_y1 = c_y + CROP_SIZE / 2

            if crop_x0 < 0:
                crop_x1 -= crop_x0
                crop_x0 = 0
            if crop_y0 < 0:
                crop_y1 -= crop_y0
                crop_y0 = 0
            if crop_x1 > frame.shape[0]:
                crop_x0 -= (crop_x1 - frame.shape[0])
                crop_x1 = frame.shape[0]
            if crop_y1 > frame.shape[1]:
                crop_y0 -= (crop_y1 - frame.shape[1])
                crop_y1 = frame.shape[1]

            cv2.rectangle(frame, (crop_x0, crop_y0), (crop_x1, crop_y1), (255, 0, 255), 2)
            cv2.imshow('result', frame)
            cv2.waitKey(50)

        else:
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, frame.shape[0])
            y_max = min(y_max, frame.shape[1])

            frame = cv2.resize(frame[y_min:y_max, x_min:x_max, :], (CROP_SIZE, CROP_SIZE))

            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

            cropped = frame[y_min:y_max, x_min:x_max, :]
            cv2.putText(cropped, 'xxxxx', (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0, 2))
            cv2.imshow('result', cropped)
            cv2.waitKey(50)
    
    print(x_min, y_min, x_max, y_max)
    cv2.destroyAllWindows()
    '''





'''
        if len(r) == 1:
            x, y, w, h = r[0][-1]
            new_x0 = max(int(x - CROP_SIZE / 2), 0)
            new_y0 = max(int(y - CROP_SIZE / 2), 0)
            new_x1 = min(int(x + CROP_SIZE / 2), image.shape[0])
            new_y1 = min(int(y + CROP_SIZE / 2), image.shape[1])

            display_frame = copy.deepcopy(image)
            cv2.rectangle(display_frame, (new_x0, new_y0), (new_x1, new_y1), (255, 0, 0), 2)
            cv2.rectangle(display_frame, (int(x - CROP_SIZE / 2), int(y - CROP_SIZE / 2)), (int(x + CROP_SIZE / 2), int(y + CROP_SIZE / 2)),
                          (255, 255, 255), 2)
            print(int(x - CROP_SIZE / 2), int(y - CROP_SIZE / 2), int(x + CROP_SIZE / 2), int(y + CROP_SIZE / 2))
            print(new_x0, new_x1, new_y0, new_y1)
            cv2.imshow('display bbox', display_frame)
            cv2.waitKey(500)

            save_path = image_path.replace(dirname, savename)
            # crop_dest_frame(image, new_x0, new_x1, new_y0, new_y1, CROP_SIZE, save_path)

            prev_x, prev_y, prev_w, prev_h = (x, y, w, h)

            # check if it is initial frame with person
            if first_person_flag == False:      # initial person
                init_x, init_y, init_w, init_h = (x, y, w, h)
                first_person_flag = True


        elif len(r) == 0:     # no person
            print(image_path, first_person_flag)

            if first_person_flag:
                save_path = image_path.replace(dirname, savename)
                crop_dest_frame(image, prev_x, prev_y, CROP_SIZE, save_path)

            else:
                save_path = image_path.replace(dirname, savename)
                no_person_frames.append([save_path, image])      # frames before any person is detected
        
    # process gathered no-person frames
    print(len(no_person_frames))
    for info in no_person_frames:
        save_p = info[0]
        img = info[1]

        crop_dest_frame(img, init_x, init_y, CROP_SIZE, save_p)

cv2.destroyAllWindows()
'''
'''
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    r = np_detect(yolo, meta, frame)        # (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h))
    x, y, w, h = r[0][-1]
    print(x, y, w, h)

    display_frame = copy.deepcopy(frame)
    cv2.rectangle(display_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 255, 255), 2)
    # # cv2.rectangle(display_frame, (int(x - (w / 2 + w_margin)), int(y - (h / 2 + h_margin))), (int(x + (w / 2 + w_margin)), int(y + (h / 2 + h_margin))), (255, 255, 255), 2)
    # cv2.rectangle(display_frame, (int(x - margin / 2), int(y - margin / 2)), (int(x + margin / 2), int(y + margin / 2)), (255, 0, 0), 2)
    #
    # cv2.imshow('display_frame', display_frame)
    # cv2.waitKey(0)

    save_path = "aug_test.png"
    crop_dest_frame(frame, x, y, CROP_SIZE, save_path)

    """

    # image = remove_background(frame, threshold=100)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=500, detectShadows=0)

    width = frame.shape[1]
    height = frame.shape[0]
    frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))

    fgMask = backSub.apply(frame)

    # cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    #
    # cv2.imshow('image', frame)
    # cv2.imshow('FG Mask', fgMask)
    # # cv2.waitKey(0)
    # k = cv2.waitKey(30) & 0xFF
    # if k == ord('q'):
    #     break

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask)

    for index, centroid in enumerate(centroids):
        if stats[index][0] == 0 and stats[index][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[index]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 10:
            cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))

    cv2.imshow('mask', fgMask)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    """



cap.release()
cv2.destroyAllWindows()
'''

