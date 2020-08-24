# -*- coding: utf-8 -*-

import cv2, os
from natsort import natsorted
import numpy as np
import pandas as pd
import pickle
import random
import time
from natsort import natsorted


def get_frames(videos, n_frames, channel = 0):
    frames_list = []    
    for i in range(len(videos)):
        frames = np.array([ cv2.resize(cv2.imread(j, channel), (224,224)) for j in videos[i] ])
        if len(frames) > n_frames:
            frame_indices = np.linspace(0, len(frames), num=n_frames, endpoint=False).astype(int)
            frames = frames[frame_indices]
        frames_list.append(frames)
    return np.array(frames_list)


def get_frames_data(dirname):
    ''' Given a directory containing extracted(sampled) frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    ret_arr = []

    filenames = natsorted(os.listdir(dirname))

    is_color = cv2.IMREAD_COLOR # np.random.binomial(1, 0.5)

    for filename in filenames:
        image_name = str(dirname) + '/' + str(filename)
        img = cv2.imread(image_name, is_color)
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_data = np.array(img)
        ret_arr.append(img_data)

    return ret_arr


def get_HRI(root, text_path):
    txt = pd.read_table(text_path, dtype=str, sep='\t', names={'path', 'label'})
    txt['path'] = txt['path'].map( lambda x: os.path.join(root, x) )
    txt = txt[txt['path'].map(lambda x: os.path.exists(x))]

    label_set = natsorted(set(txt['label']))

    label_to_ix = dict(zip(list(label_set), range(len(label_set))))
    print(label_set)

    # with open('categories.txt', 'w') as txt:
    #     txt.write(str(label_set))

    # save label_to_ix as pkl
    pickle.dump(label_to_ix, file('label_to_ix.pkl','wb'))

    n_intention = len(label_set)

    return txt['path'].values, \
           np.array(map(lambda lab: label_to_ix[lab], txt['label'].values)),\
           int(n_intention)


def get_HRI_v2(list_root, list_text_path):
    filename_lb2ix = 'label_to_ix-ABR_action-aug.pkl'

    len_root = len(list_root)
    len_text_path = len(list_text_path)

    if len_root != len_text_path:
        print("ROOT NUMBER & TEXT FILE NUMBER are not same !\n")
        print("len_root: {}, len_text_path: {}".format(len_root, len_text_path))
        return 1

    # when the number of data is one
    if len_root == 1:
        root = list_root
        text_path = list_text_path

        txt = pd.read_table(text_path, dtype=str, sep='\t', names={'path', 'label'})
        txt['path'] = txt['path'].map(lambda x: os.path.join(root, x))
        txt = txt[txt['path'].map(lambda x: os.path.exists(x))]

        label_set = natsorted(set(txt['label']))

        label_to_ix = dict(zip(list(label_set), range(len(label_set))))
        print(label_set)

        # with open('categories.txt', 'w') as txt:
        #     txt.write(str(label_set))

        # save label_to_ix as pkl
        pickle.dump(label_to_ix, file(filename_lb2ix, 'wb'))

        n_intention = len(label_set)

        return txt['path'].values, \
               np.array(map(lambda lab: label_to_ix[lab], txt['label'].values)), \
               int(n_intention)

    # when the number of data are more than two(augmented data -> same labels)
    for i in range(len_root):
        root = list_root[i]
        text_path = list_text_path[i]

        if i == 0:
            txt = pd.read_table(text_path, dtype=str, sep='\t', names={'path', 'label'})
            txt['path'] = txt['path'].map(lambda x: os.path.join(root, x))
            txt = txt[txt['path'].map(lambda x: os.path.exists(x))]
            # print(type(txt))
            # print(len(txt['path'].values))
        else:
            df_tmp = pd.read_table(text_path, dtype=str, sep='\t', names={'path', 'label'})
            df_tmp['path'] = df_tmp['path'].map(lambda x: os.path.join(root, x))
            df_tmp = df_tmp[df_tmp['path'].map(lambda x: os.path.exists(x))]
            # print(len(df_tmp['path'].values))

            txt = pd.concat([txt, df_tmp])
    print("number of videos : {}".format(len(txt)))

    label_set = natsorted(set(txt['label']))

    label_to_ix = dict(zip(list(label_set), range(len(label_set))))
    # print(label_set)

    # with open('categories.txt', 'w') as txt:
    #     txt.write(str(label_set))

    # save label_to_ix as pkl
    pickle.dump(label_to_ix, file(filename_lb2ix, 'wb'))

    n_intention = len(label_set)

    np.random.shuffle(txt['path'].values)
    # print(txt['path'].values[:100])


    return txt['path'].values, \
           np.array(map(lambda lab: label_to_ix[lab], txt['label'].values)), \
           int(n_intention)


def crop_frame(frame, CROP_SIZE):
    height, width, channel = frame.shape

    if (width > height):
        scale = float(CROP_SIZE) / float(height)
        frame = cv2.resize(frame, (int(width * scale + 1), CROP_SIZE)).astype(np.float32)
    elif (width == height):
        frame = cv2.resize(frame, (CROP_SIZE, CROP_SIZE)).astype(np.float32)
    else:
        scale = float(CROP_SIZE) / float(width)
        frame = cv2.resize(frame, (CROP_SIZE, int(height * scale + 1))).astype(np.float32)

    crop_x = int((frame.shape[1] - CROP_SIZE) / 2)
    crop_y = int((frame.shape[0] - CROP_SIZE) / 2)

    cropped = frame[crop_y:crop_y + CROP_SIZE, crop_x:crop_x + CROP_SIZE, :]

    # convert dtype uint8 --> float32
    return cropped.astype(np.float32)

# def change_contrast(frame, random_var):"
#     # 160 seconds per batch
#     alpha = random_var*0.01
#
#     new_frame = np.zeros(frame.shape, frame.dtype)
#
#     for y in range(frame.shape[0]):
#         for x in range(frame.shape[1]):
#             for c in range(frame.shape[2]):
#                 new_frame[y, x, c] = np.clip(alpha * frame[y, x, c], 0, 255)
#
#     return new_frame


def change_contrast(frame, random_var):

    alpha = random_var * 0.01

    # src1, alpha, src2, beta, gamma > new_frame = alpha * src1 + beta * src2 + gamma
    new_frame = cv2.addWeighted(frame, alpha, frame, 0, 0)

    return new_frame


def add_s_p_noise(frame, amount):
        # start_time = time.time()

        s_vs_p = 0.5
        # amount = 0.2 #0.08 #0.004
        out = np.copy(frame)

        # Salt mode
        num_salt = np.ceil(amount * frame.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in frame.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * frame.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in frame.shape]
        out[coords] = 0

        # end_time = time.time()
        # print('process time: ', end_time - start_time)

        return out


def preprocess_frame(video_batch):

    # start_time = time.time()
    frame_batch = []
    for index, video in enumerate(video_batch):
        frame_sampled = get_frames_data(dirname=video)
        frame_list = []
        random_var = random.randint(1, 200)
        random_amount = random.randint(10, 200)
        for frame in frame_sampled:
            frame = crop_frame(frame, CROP_SIZE=224)
            # TODO: add color augmentation
            # ex: brightness, contrast, saturation, hue, lightingAug
            # frame = change_contrast(frame, random_var)
            # frame = add_s_p_noise(frame, random_amount*0.001)

            frame_list.append(frame)

        frame_batch.append(frame_list)

    # zero-padding through time axis
    # maxlen = max(map(lambda frame: len(frame), frame_batch))
    maxlen = 64
    frame_batch = np.array(map(lambda frame: frame + [np.zeros_like(frame[0])]*(maxlen-len(frame)), frame_batch))

    # end_time = time.time()
    # print('preprocess time:', end_time-start_time)
    return frame_batch


def normalize(_array, lower_b, upper_b):
    scale = upper_b - lower_b
    offset = lower_b
    min_v = np.min(_array)
    max_v = np.max(_array)
    return scale*((_array-min_v)/float(max_v-min_v)) + offset


def cross_validation(videos, intention_data, k_fold, k_fold_status, n_class):
    #### k-class
    total_class_analysis = np.zeros(n_class).tolist()

    train_videos = []
    valid_videos = []

    train_intention_data = []
    valid_intention_data = []

    videos.tolist()
    intention_data.tolist()

    for cls in range(n_class):
        total_videos = []
        total_intention_data = []
        for idx in range(len(videos)):
            if intention_data[idx] == cls:
                total_class_analysis[cls] = total_class_analysis[cls] + 1

                total_videos.append(videos[idx])
                total_intention_data.append(intention_data[idx])

        split_ind = int(len(total_videos) * 1/k_fold)

        valid_start_split = (k_fold_status%k_fold)*split_ind
        valid_end_split = ((k_fold_status+1)%k_fold)*split_ind

        if k_fold_status == 0:
            train_videos = train_videos + total_videos[valid_end_split:]
            train_intention_data = train_intention_data + total_intention_data[valid_end_split:]
        elif k_fold_status == k_fold-1:
            train_videos = train_videos + total_videos[:valid_start_split]
            train_intention_data = train_intention_data + total_intention_data[:valid_start_split]
        else:
            train_videos = train_videos + total_videos[:valid_start_split] + total_videos[valid_end_split:]
            train_intention_data = train_intention_data + total_intention_data[:valid_start_split] + total_intention_data[valid_end_split:]

        valid_videos = valid_videos + total_videos[valid_start_split:valid_end_split]
        valid_intention_data = valid_intention_data + total_intention_data[valid_start_split:valid_end_split]

    for cls in range(n_class):
        print('class:' + str(cls) + ' total:' + str(total_class_analysis[cls]))
    print('train:' + str(len(train_videos)) + ' valid:' + str(len(valid_videos)))

    ### list to array
    train_videos = np.array(train_videos)
    train_intention_data = np.array(train_intention_data)
    valid_videos = np.array(valid_videos)
    valid_intention_data = np.array(valid_intention_data)

    return train_videos, train_intention_data, valid_videos, valid_intention_data