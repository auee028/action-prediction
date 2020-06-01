import os
import cv2
import numpy as np
import copy
import glob
from natsort import natsorted

from darknet.python.darknet import *


def remove_background(img, threshold):
    """
    This method removes background from your image

    :param img: cv2 image
    :type img: np.array
    :param threshold: threshold value for cv2.threshold
    :type threshold: float
    :return: RGBA image
    :rtype: np.ndarray
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    images, cnts, hierachy = cv2.findContours(morphed,
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)

    mask = cv2.drawContours(threshed, cnts, 0, (0, 255, 0), 0)
    masked_data = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(cnts)
    dst = masked_data[y: y + h, x: x + w]

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(dst)

    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)

    return dst

def crop_dest_frame(frame, x0, x1, y0, y1, CROP_SIZE, filename):
    # resize scale is smaller than original image size
    cropped = cv2.resize(frame, (CROP_SIZE, CROP_SIZE))
    cropped[:y1 - y0, :x1 - x0, :] = frame[y0:y1, x0:x1, :]
    print(cropped.shape, x0, x1, y0, y1)
    # cv2.imshow('test', frame[new_y0:new_y1, new_x0:new_x1, :])
    # cv2.waitKey(500)
    # # cv2.destroyAllWindows()
    # cv2.imwrite(filename, cropped)

    # convert dtype uint8 --> float32
    return cropped.astype(np.float32)



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


# dir_list = natsorted(glob.glob(os.path.join(dir_root, dirname, '*/*')))
# # print(dir_list[:10])
# dir_list = ["/media/pjh/HDD2/Dataset/ces-demo-4th/ABR_action/1/2020-01-09-09-51-48_00_77"]
class_num = '1'      # 1 ~ 15
dir_list = natsorted(glob.glob(os.path.join(dir_root, dirname, '{}/*'.format(class_num))))
dir_list = dir_list[:3]


for imgs_dir in dir_list:
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

    print(x_min, y_min, x_max, y_max)

    width = x_max - x_min
    height = y_max - y_min
    # print("width: {}, height: {}".format(width, height))

    c_x = int(x_min + width / 2)
    c_y = int(y_min + height / 2)
    # print("center x: {}, center y: {}".format(c_x, c_y))

    # compare width & height
    longer = height
    if width > height:
        longer = width

    '''
    Step 3. crop each frame along to the longer axe
    '''
    # print(c_x - CROP_SIZE / 2, c_x + CROP_SIZE / 2, c_y - CROP_SIZE / 2, c_y + CROP_SIZE)
    for image_path in imgs_dir:
        frame = cv2.imread(image_path)

        x0 = 0
        y0 = 0
        x1 = 0
        y1 = 0

        r = np_detect(yolo, meta, frame)  # (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h))

        if len(r) == 1:
            x, y, _, _ = r[0][-1]       # x, y, w, h

            x0 = int(x - longer / 2)
            y0 = int(y - longer / 2)
            x1 = int(x + longer / 2)
            y1 = int(y + longer / 2)

            if x0 < 0:
                x1 -= x0
                x0 = 0
            if y0 < 0:
                y1 -= y0
                y0 = 0

            if x1 > frame.shape[1]:
                x1 = frame.shape[1]
            if y1 > frame.shape[0]:
                y1 = frame.shape[0]

            print("width: {}, height: {}".format(x1 - x0, y1 - y0))

            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
            cv2.imshow('bbox', frame)
            cv2.waitKey(50)

            # cropped = frame[y0:y1, x0:x1, :]






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
    '''
    print(x_min, y_min, x_max, y_max)
    cv2.destroyAllWindows()






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

