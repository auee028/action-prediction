import os
import glob
import time
import cv2


video_root = "/media/pjh/HDD2/Dataset/Breakfast/BreakfastII_15fps_qvga_sync"
videos = []
for person in glob.glob(video_root + "/*"):
    # print(person)
    for camera in glob.glob(person + "/*"):
        # print(camera)
        for file_path in glob.glob(camera + "/*"):
            # print(file_path)        # .avi's & .labels(.txt)'s
            if file_path.split('.')[-1] == 'avi':
                videos.append(file_path.replace(video_root+'/', ''))
print(len(videos))      # .avi files : total 3832
# print(videos[0])

anno_root = "/media/pjh/HDD2/Dataset/Breakfast/segmentation_coarse"
cnt = 1
failed_cnt = 0
failed_dict = {}
diff_dict = {}
for clss in glob.glob(anno_root + "/*"):
    # print(clss)
    for anno_file in glob.glob(clss + "/*"):     # .txt's & .xml's
        # print(anno_file)
        if anno_file.split('.')[-1] == "xml": continue

        person_txt, camera_txt, _, label_txt = anno_file.split('/')[-1].split('.')[0].split('_')
        if camera_txt == "stereo01":
            camera_txt = "stereo"
        # print(person_txt, camera_txt, label_txt)

        cur_video = "{}/{}/{}_{}.avi".format(person_txt, camera_txt, person_txt, label_txt)
        cur_video_ch0 = "{}/{}/{}_{}_ch0.avi".format(person_txt, camera_txt, person_txt, label_txt)
        cur_video_ch1 = "{}/{}/{}_{}_ch1.avi".format(person_txt, camera_txt, person_txt, label_txt)

        if cur_video in videos:
            print("{}\t{}".format(cnt, cur_video))
            with open(anno_file, 'r') as f:
                info = f.readlines()
                # print(info)
                # print(info[-1].strip().split(' '))
                timestamp, label = info[-1].strip().split(' ')
                the_last_start = int(timestamp.split('-')[0])
                the_last_end = int(timestamp.split('-')[1])

                cap = cv2.VideoCapture(os.path.join(video_root, cur_video))
                num_frame = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    num_frame += 1

                diff = the_last_end - the_last_start
                # print(the_last_end, num_frame, diff, the_last_end - num_frame)
                if diff in diff_dict:
                    diff_dict[diff] += 1
                else:
                    diff_dict[diff] = 1

                # for seg in info:
                #     # print(seg.split(' '))
                #     timestamp, label = seg.strip().split(' ')
                #     start_point =int(timestamp.split('-')[0])       # start from 1st frame
                #     end_point = int(timestamp.split('-')[1])
                #     print(start_point, end_point, end_point - start_point)

        elif cur_video_ch0 in videos:
            print("{}\t{}".format(cnt, cur_video_ch0))
            with open(anno_file, 'r') as f:
                info = f.readlines()
                # print(info)
                # print(info[-1].strip().split(' '))
                timestamp, label = info[-1].strip().split(' ')
                the_last_start = int(timestamp.split('-')[0])
                the_last_end = int(timestamp.split('-')[1])

                cap = cv2.VideoCapture(os.path.join(video_root, cur_video_ch0))
                num_frame = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    num_frame += 1

                diff = the_last_end - the_last_start
                # print(the_last_end, num_frame, diff, the_last_end - num_frame)
                if diff in diff_dict:
                    diff_dict[diff] += 1
                else:
                    diff_dict[diff] = 1

        elif cur_video_ch1 in videos:
            print("{}\t{}".format(cnt, cur_video_ch1))
            with open(anno_file, 'r') as f:
                info = f.readlines()
                # print(info)
                # print(info[-1].strip().split(' '))
                timestamp, label = info[-1].strip().split(' ')
                the_last_start = int(timestamp.split('-')[0])
                the_last_end = int(timestamp.split('-')[1])

                cap = cv2.VideoCapture(os.path.join(video_root, cur_video_ch1))
                num_frame = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    num_frame += 1

                diff = the_last_end - the_last_start
                # print(the_last_end, num_frame, diff, the_last_end - num_frame)
                if diff in diff_dict:
                    diff_dict[diff] += 1
                else:
                    diff_dict[diff] = 1

        else:
            # print("{}\t{}\t\t( X )".format(cnt, cur_video))
            with open(anno_file, 'r') as f:
                info = f.readlines()
            print("{}\t{}\t\t( {} )".format(cnt, cur_video_ch1, len(info)))
            failed_cnt += 1
            failed_dict[failed_cnt] = cur_video_ch1

        cnt += 1
# print(cnt)
print(failed_cnt)
print(diff_dict)