import os
import glob
import natsort
import itertools
import shutil
import json

import argparse


def gen_action_seq_data(video_root, seq_map, new_vseq_root, new_anno_path):
    # action sequence map
    with open(seq_map, 'r') as f:
        seqs = f.readlines()

    # video list for each action class
    video_dict = {}
    for person in glob.glob(os.path.join(video_root, '*')):
        cur_person = person.split('/')[-1]#.lower()
        for action_clss in glob.glob(os.path.join(person, '*')):
            cur_clss = int(action_clss.split('/')[-1])

            for video in glob.glob(os.path.join(action_clss, '*')):
                cur_date = '-'.join(video.split('/')[-1].split('-')[:3])
                cur_angle = video.split('/')[-1].split('_')[-2]

                if cur_person in video_dict.keys():
                    if cur_date in video_dict[cur_person].keys():
                        if cur_angle in video_dict[cur_person][cur_date].keys():
                            if cur_clss in video_dict[cur_person][cur_date][cur_angle].keys():
                                v_list = video_dict[cur_person][cur_date][cur_angle][cur_clss]
                                v_list.append(video)
                                video_dict[cur_person][cur_date][cur_angle][cur_clss] = v_list
                            else:
                                video_dict[cur_person][cur_date][cur_angle][cur_clss] = [video]
                        else:
                            video_dict[cur_person][cur_date][cur_angle] = {cur_clss: [video]}
                    else:
                        video_dict[cur_person][cur_date] = {cur_angle: {cur_clss: [video]}}
                else:
                    video_dict[cur_person] = {cur_date: {cur_angle: {cur_clss: [video]}}}

    # print(video_dict.keys())

    # generate action sequences
    with open(seq_map, 'r') as f:
        label2ix = f.readlines()
    label2ix = [line.strip() for line in label2ix]
    # print(label2ix)

    anno = {}
    for seq_id, cur_seq in enumerate(seqs[:1]):
        cur_seq = cur_seq.replace('\t', ',')
        actions = cur_seq.strip().split(',')
        actions_idx = [label2ix.index(a) for a in actions]
        print(actions_idx)

        for person in video_dict.keys()[:1]:
            gender = "she" if person in ['Gwena', 'Juhee', 'kanchen', 'safaa', 'SeungYeon'] else "he"

            dates = video_dict[person]
            for date in dates.keys()[:1]:
                angles = video_dict[person][date]
                for angle in angles.keys()[:1]:
                    video_clss_list = video_dict[person][date][angle]

                    # get all of separated videos for each class for the current action sequence
                    video_table = []
                    for cur_idx in actions_idx:
                        cur_action_list = video_clss_list[cur_idx]
                        video_table.append(cur_action_list)
                    # print(video_table)

                    for comb_id, cur_comb in enumerate(itertools.product(*video_table)):
                        # print(cur_comb)

                        new_video = "{}_{}_{}_seq{}-{}".format(person, date, angle, seq_id, comb_id)
                        save_dir = os.path.join(new_vseq_root, new_video)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        accumulated = 0
                        timestamps = []
                        sentences = []
                        for vid_idx, vid in enumerate(cur_comb):
                            # save frames in a combination
                            frames = natsort.natsorted(glob.glob(os.path.join(vid, '*')))

                            for frame in frames:
                                assert frame.split('/')[-1].split('.')[-1] == 'png'
                                org_f_num = int(frame.split('/')[-1].split('.')[0])
                                new_f_path = os.path.join(save_dir, "{}.png".format(accumulated + org_f_num))
                                # print("{}.png".format(accumulated + org_f_num))

                                shutil.copyfile(frame, new_f_path)

                            # update current annotation
                            timestamp = [accumulated + 1, accumulated + len(frames)]
                            timestamps.append(timestamp)

                            label = label2ix[actions_idx[vid_idx]]
                            sentence = "{} is {}".format(gender, label)
                            sentences.append(sentence)

                            accumulated += len(frames)
                        # print(comb_id, accumulated, timestamps, sentences)

                        # write annotations
                        anno[new_video] = {"duration": accumulated,
                                          "timestamps": timestamps,
                                          "sentences": sentences}

    # print(natsort.natsorted(anno.keys()))
    # print(len(anno.keys()))

    with open(new_anno_path, 'w') as f:
        json.dump(anno, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='action sequence data generation')

    parser.add_argument('--video_root', type=str, default="/media/pjh/HDD2/Dataset/ces-demo-4th/frames", help='input_dir')
    parser.add_argument('--seq_map', type=str, default="actionseq_dataset.txt", help='sequence samples to generate data')
    parser.add_argument('--new_vseq_root', type=str, default="/media/pjh/HDD2/Dataset/vseq_ABRaction", help='dir of new input seq of video to save')
    parser.add_argument('--new_anno_path', type=str, default="abr_action_seq.json", help='path of new anno file')

    args = parser.parse_args()

    video_root = args.video_root
    seq_map = args.seq_map
    new_vseq_root = args.new_vseq_root
    new_anno_path = args.new_anno_path

    gen_action_seq_data(video_root, seq_map, new_vseq_root, new_anno_path)
