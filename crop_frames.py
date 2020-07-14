import os
import cv2
import numpy as np
import copy

from darknet.python.darknet import *


w_margin = 20
h_margin = 10
CROP_SIZE = 170

def CropFrames(yolo, meta, frames):
    '''
    Step 1. go through all frames in one video
    '''
    x_list = []
    y_list = []
    for image in frames:

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

    '''
    Step 2. Get min, max of overall moving area in x, y axis
    '''
    # print(len(x_list))
    x_min = min(np.array(x_list)[:, 0])
    y_min = min(np.array(y_list)[:, 0])
    x_max = max(np.array(x_list)[:, 1])
    y_max = max(np.array(y_list)[:, 1])

    sample_img = frames[0]
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
                return frames

        if x_max > org_x:
            x_min -= (x_max - org_x)
            x_max = org_x

            if x_min <= 0:   # same as org_x
                x_min = 0

                # pass the current video because adjusted size is same as original video frame size
                return frames


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
                return frames

        if y_max > org_y:
            y_min -= (y_max - org_y)
            y_max = org_y

            if y_min <= 0:  # same as org_x
                y_min = 0

                # pass the current video because adjusted size is same as original video frame size
                return frames

    # if adjusted width & height are not same
    if (x_max - x_min) != (y_max - y_min):
        # raw_input("Press Enter to continue...")
        print("\t*** New Width & New Height Are Not Same ! ***")
        print("\t\twidth: {}, height: {}".format((x_max - x_min), (y_max - y_min)))

        # break
        return frames

    '''
    Step 4. crop frames as the area above
    '''
    new_frames = []

    # crop a frame
    for image in frames:
        new_image = image[y_min:y_max, x_min:x_max, :]
        new_image = cv2.resize(new_image, (224, 224))
        new_frames.append(new_image)

    return np.array(new_frames)
