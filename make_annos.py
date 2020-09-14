import os
import glob
import natsort


# jester_v1_root = ""
abr_gesture_root = "/media/pjh/HDD2/Dataset/ces_gesture/videos"
abr_action_4th_root = "/media/pjh/HDD2/Dataset/ces-demo-5th/ABR_action_partdet"
abr_action_5th_root = "/media/pjh/HDD2/Dataset/ces-demo-5th/ABR_action_5th_partdet"
abr_action_3th_root = "/media/pjh/HDD2/Dataset/ces-demo-5th/ABR-3th_partdet_slided"

anno_root = "/media/pjh/HDD2/Dataset/ces-demo-5th/annotations"

abr_gesture_annofile = "ces_gesture_{}.txt"
abr_action_annofile = "abr_action_{}.txt"
abr_action_5th_annofile = "abr_action_5th_{}.txt"
abr_action_3th_annofile = "abr_action_3th_{}.txt"

'''
with open("categories_ceslea.txt", 'r') as f:
    categories = f.readlines()
categories = [line.strip() for line in categories]
# print(categories)

abr_gesture_labels = categories[:12]        # 0-11 (12 classes)
abr_action_labels = categories[12:18]       # 12-17 (6 classes)
abr_action_3th_labels = categories[18:]     # 18-19 (2 classes)
# print(abr_gesture_labels)
# print(abr_action_labels)
# print(abr_action_3th_labels)

train_ratio = 0.9     # train:val = 9:1
'''

with open('categories_newactions.txt', 'r') as f:
    categories = f.readlines()
categories = [line.strip() for line in categories]

abr_action_labels = categories[:7]       # 0-6 (7 classes)
abr_action_3th_labels = categories[7:]     # 7 (1 classes)
print(abr_action_labels)
print(abr_action_3th_labels)


'''
"""
# ABR gesture dataset
"""
for clss in glob.glob(os.path.join(abr_gesture_root, '*')):
    label = clss.split('/')[-1].replace('_', ' ')
    if not label in abr_gesture_labels:
        continue
    label2ix = categories.index(label)
    # print(clss)

    videos = natsort.natsorted(glob.glob(os.path.join(clss, '*')))
    # print(clss, len(videos))
    v_len = len(videos)

    train_list = videos[:int(v_len * train_ratio)]
    val_list = videos[int(v_len * train_ratio):]
    # print(v_len, len(train_list), len(val_list))

    with open(os.path.join(anno_root, abr_gesture_annofile.format("train"), 'a') as f:
        for path in train_list:
            f.write("{}\t{}\n".format(path, label2ix))
    with open(os.path.join(anno_root, abr_gesture_annofile.format("val), 'a') as f:
        for path in val_list:
            f.write("{}\t{}\n".format(path, label2ix))

'''


"""
# ABR action (4th) dataset
"""
with open("/home/pjh/PycharmProjects/action-prediction/categories.txt", 'r') as f:
    org_actions = f.readlines()
org_actions = [line.strip() for line in org_actions]
# print(org_actions)

for clss in glob.glob(os.path.join(abr_action_4th_root, '*')):
    # print(clss)
    label = org_actions[int(clss.split('/')[-1]) - 1]
    # print(label)
    if not label in abr_action_labels:
        continue
    label2ix = categories.index(label)

    videos = natsort.natsorted(glob.glob(os.path.join(clss, '*')))
    # print(clss, len(videos))
    v_len = len(videos)

    # train_list = videos[:int(v_len * train_ratio)]
    # val_list = videos[int(v_len * train_ratio):]
    # # print(v_len, len(train_list), len(val_list))
    #
    # # with open(os.path.join(anno_root, abr_action_annofile.format("train")), 'a') as f:
    # #     for path in train_list:
    # #         f.write("{}\t{}\n".format(path, label2ix))
    # # with open(os.path.join(anno_root, abr_action_annofile.format("val")), 'a') as f:
    # #     for path in val_list:
    # #         f.write("{}\t{}\n".format(path, label2ix))

    with open(os.path.join(anno_root, abr_action_annofile.format("train").replace('action', 'newaction')), 'a') as f:
        for path in videos:
            f.write("{}\t{}\n".format(path, label2ix))

"""
# ABR action 5th dataset
"""
with open("/home/pjh/PycharmProjects/action-prediction/categories.txt", 'r') as f:
    org_actions = f.readlines()
org_actions = [line.strip() for line in org_actions]
# print(org_actions)

for clss in glob.glob(os.path.join(abr_action_5th_root, '*')):
    # print(clss)
    label = org_actions[int(clss.split('/')[-1]) - 1]
    # print(label)
    if not label in abr_action_labels:
        continue
    label2ix = categories.index(label)

    videos = natsort.natsorted(glob.glob(os.path.join(clss, '*')))
    # print(clss, len(videos))
    v_len = len(videos)

    # train_list = videos[:int(v_len * train_ratio)]
    # val_list = videos[int(v_len * train_ratio):]
    # # print(v_len, len(train_list), len(val_list))
    #
    # # with open(os.path.join(anno_root, abr_action_5th_annofile.format("train")), 'a') as f:
    # #     for path in train_list:
    # #         f.write("{}\t{}\n".format(path, label2ix))
    # # with open(os.path.join(anno_root, abr_action_5th_annofile.format("val")), 'a') as f:
    # #     for path in val_list:
    # #         f.write("{}\t{}\n".format(path, label2ix))

    with open(os.path.join(anno_root, abr_action_5th_annofile.format("train")).replace("action", "newaction"), 'a') as f:
        for path in videos:
            f.write("{}\t{}\n".format(path, label2ix))




"""
# ABR action 3th dataset ( lookar , search(x) )
"""
for clss in glob.glob(os.path.join(abr_action_3th_root, '*')):
    # print(clss)
    label = clss.split('/')[-1]
    # print(label)
    if not label in abr_action_3th_labels:
        continue
    label2ix = categories.index(label)

    videos = natsort.natsorted(glob.glob(os.path.join(clss, '*')))
    # print(clss, len(videos))
    v_len = len(videos)

    # train_list = videos[:int(v_len * train_ratio)]
    # val_list = videos[int(v_len * train_ratio):]
    # # print(v_len, len(train_list), len(val_list))
    #
    # # with open(os.path.join(anno_root, .format("train"), 'a') as f:
    # #     for path in train_list:
    # #         f.write("{}\t{}\n".format(path, label2ix))
    # # with open(os.path.join(anno_root, .format("val"), 'a') as f:
    # #     for path in val_list:
    # #         f.write("{}\t{}\n".format(path, label2ix))

    with open(os.path.join(anno_root, abr_action_3th_annofile).format("train").replace("action", "newaction"), 'a') as f:
        for path in videos:
            f.write("{}\t{}\n".format(path, label2ix))



def check_datanum():
    # anno_files = [abr_gesture_annofile.format("train"), abr_gesture_annofile.format("val"),
    #               abr_action_annofile.format("train"), abr_action_annofile.format("val"),
    #               abr_action_5th_annofile.format("train"), abr_action_5th_annofile.format("val"),
    #               abr_action_3th_annofile.format("train"), abr_action_3th_annofile.format("val")]
    anno_files = [abr_action_annofile.format("train"),
                  abr_action_5th_annofile.format("train"),
                  abr_action_3th_annofile.format("train")]

    for fl in anno_files:
        file_path = os.path.join(anno_root, fl.replace("action", "newaction"))
        with open(file_path, 'r') as f:
            lines = f.readlines()
        print(fl, len(lines))


def change_labels():
    with open("/home/pjh/PycharmProjects/action-prediction/categories_ceslea.txt", 'r') as f:
        org_labels = f.readlines()
    org_labels = [line.strip() for line in org_labels]
    new_labels = org_labels[12:]
    # print(org_labels)
    # print(new_labels)
    # for i, l in enumerate(org_labels):
    #     print(i, l)
    # print('\n')
    # for i, l in enumerate(new_labels):
    #     print(i, l)

    anno_files = [abr_action_annofile.format("train"), abr_action_annofile.format("val"),
                  abr_action_5th_annofile.format("train"), abr_action_5th_annofile.format("val"),
                  abr_action_3th_annofile.format("train"), abr_action_3th_annofile.format("val")]

    for f_name in anno_files:
        with open(os.path.join(anno_root, f_name), 'r') as f:
            data = f.readlines()

        new_data = []
        for d in data:
            new_l = new_labels.index(org_labels[int(d.strip().split('\t')[-1])])
            new_data.append("{}\t{}\n".format(d.strip().split('\t')[0], new_l))

        with open(os.path.join(anno_root, "new-{}".format(f_name)), 'w') as f:
            f.writelines(new_data)



if __name__=="__main__":
    check_datanum()
    # change_labels()

