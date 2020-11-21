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
    height, width, _ = frames[0].shape
    center_x_range = (int(width / 3), int(2 * width / 3))

    x_list = []
    y_list = []
    for image in frames:

        r = np_detect(yolo, meta, image)  # (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h))

        """
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
        """

        if len(r) > 0:
            biggest_person = r[0]
        else:
            continue

        x, y, w, h = biggest_person[-1]

        if (x > center_x_range[1]) or (x < center_x_range[0]):
            continue

        x0 = int(x - w / 2)
        y0 = int(y - h / 2)
        x1 = int(x + w / 2)
        y1 = int(y + h / 2)

        ratio = 1.2
        if h > (w * ratio):
            # y1 = int(y + w * ratio / 2)
            y1 = y0 + int(w * ratio)

        image = cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 0), 3)
        # cv2.imshow("image", image)
        # cv2.waitKey(50)


        # if bbox is too big, just set it as the criteria
        if w > width / 0.5:
            x_list = [[x0, x1]]
            y_list = [[y0, y1]]

            break


        x_list.append([x0, x1])
        y_list.append([y0, y1])

    if len(x_list) == 0:
        return []


    '''
    Step 2. Get min, max of overall moving area in x, y axis
    '''
    # print(len(x_list))
    # print(x_list)
    x_min = min(np.array(x_list)[:, 0])
    y_min = min(np.array(y_list)[:, 0])
    x_max = max(np.array(x_list)[:, 1])
    y_max = max(np.array(y_list)[:, 1])

    org_x = width - 1     # x coordinate
    org_y = height - 1     # y coordinate

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, org_x)
    y_max = min(y_max, org_y)

    print(x_min, x_max, y_min, y_max)

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
                try:
                    assert (y_max - y_min) == (x_max - x_min)

                    return frames

                except:
                    # process one more adjustment at Step 4.
                    pass

        if x_max > org_x:
            x_min -= (x_max - org_x)
            x_max = org_x

            if x_min <= 0:   # same as org_x
                x_min = 0

                # pass the current video because adjusted size is same as original video frame size
                try:
                    assert (y_max - y_min) == (x_max - x_min)

                    return frames

                except:
                    # process one more adjustment at Step 4.
                    pass


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
                try:
                    assert (y_max - y_min) == (x_max - x_min)

                    return frames

                except:
                    # process one more adjustment at Step 4.
                    pass

        if y_max > org_y:
            y_min -= (y_max - org_y)
            y_max = org_y

            if y_min <= 0:  # same as org_x
                y_min = 0

                # pass the current video because adjusted size is same as original video frame size
                try:
                    assert (y_max - y_min) == (x_max - x_min)

                    return frames

                except:
                    # process one more adjustment at Step 4.
                    pass


    '''
    Step 4. make sure a bbox to have the same length of width & height
    '''
    # if adjusted width & height are not same
    if (x_max - x_min) != (y_max - y_min):
        # raw_input("Press Enter to continue...")
        # print("\t*** New Width & New Height Are Not Same ! ***")
        # print("\t\twidth: {}, height: {}".format((x_max - x_min), (y_max - y_min)))
        #
        # # break
        # return frames


        print("Adjust again ...")

        if (x_max - x_min) > (y_max - y_min):
            # reduce the length of width of bbox
            x_min = (org_x - (y_max - y_min)) / 2
            x_max = x_min + (y_max - y_min)

        else:
            # reduce the length of height of bbox
            y_min = (org_y - (x_max - x_min)) / 2
            y_max = y_min + (x_max - x_min)


    assert (y_max - y_min) == (x_max - x_min)

    # print("\t*** New Width & New Height Are Not Same ! ***")
    # print("\t\twidth: {}, height: {}".format((x_max - x_min), (y_max - y_min)))

    '''
    Step 5. crop frames as the area above
    '''
    new_frames = []

    # crop a frame
    for image in frames:
        new_image = image[y_min:y_max, x_min:x_max, :]
        new_image = cv2.resize(new_image, (224, 224))
        # cv2.imshow("show", new_image)
        # cv2.waitKey(50)
        new_frames.append(new_image)

    return np.array(new_frames)


