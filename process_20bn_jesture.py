import os
import csv
import shutil


# output: categories of jestures
categories = []
with open("annotation/jester-v1-labels.csv") as f:
    lines = csv.reader(f)
    for line in lines:
        clss = line[0]
        # print(clss)
        categories.append(clss)
print(categories)
# with open('annotation/category-12_jestures.txt') as f:
#     lines = f.readlines()
# categories = []
# for line in lines:
#     line = line.rstrip()
#     categories.append(line)
# categories = sorted(categories)
# print(categories)

def check_csv():
    csv_list = ["annotation/jester-v1-train.csv", "annotation/jester-v1-validation.csv"]
    for csv_file in csv_list:
        with open(csv_file) as f:
            lines = csv.reader(f)

            dict_folders = {}
            for line in lines:
                items = line[0].split(';')

                if items[1].replace('_', ' ') in categories:
                    dict_folders[items[0]] = items[1].replace('_', ' ')
        print(len(dict_folders))        # train: 118562 / val: 14787

def load_category_dict():
    """
    # EXAMPLE ::
    #   output: dictionary of categories of 9 gestures
    #   dict_categories = {0: 'Doing_other_things', 1: 'Rolling_Hand_Backward', 2: 'Sliding_Two_Fingers_Down', 3: 'Sliding_Two_Fingers_Left',
    #                       4: 'Sliding_Two_Fingers_Right', 5: 'Sliding_Two_Fingers_Up', 6: 'Stop_Sign', 7: 'Thumb_Down', 8: 'Thumb_Up'}
    """
    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[i] = category

    return dict_categories

def mv_datadir(mode, csv_file, old_datadir, new_datadir):
    if mode == "test":
        with open(csv_file) as f:
            lines = csv.reader(f)

            cnt = 0
            for line in lines:
                data_idx = line[0]

                save_dir = os.path.join(new_datadir, mode)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                shutil.move(os.path.join(old_datadir, data_idx), save_dir)

                cnt += 1
                print("{}\t{}".format(cnt, os.path.join(save_dir, data_idx)))

    else:
        with open(csv_file) as f:
            lines = csv.reader(f)

            cnt = 0
            for line in lines:

                items = line[0].split(';')

                data_idx = items[0]
                data_label = items[1]

                save_dir = os.path.join(new_datadir, mode, data_label)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # print(save_dir)

                cnt += 1
                if not os.path.exists(os.path.join(old_datadir, data_idx)):
                    print("{}\t{}\t\t : NO DIR)".format(cnt, save_dir))
                    continue
                else:
                    print("{}\t{}".format(cnt, save_dir))

                shutil.move(os.path.join(old_datadir, data_idx), save_dir)


def check_newdir(mode, csv_file, new_datadir):
    if mode == "test":
        with open(csv_file) as f:
            lines = csv.reader(f)

            cnt = 1
            for line in lines:
                data_idx = line[0]

                new_dir = os.path.join(new_datadir, mode, data_idx)
                if not os.path.exists(new_dir):
                    print("{}\tNo dir: {}".format(cnt, os.path.join(data_idx)))
                else:
                    print("{}".format(cnt))

                cnt += 1

    else:
        with open(csv_file) as f:
            lines = csv.reader(f)

            cnt = 1
            for line in lines:

                items = line[0].split(';')

                data_idx = items[0]
                data_label = items[1]

                new_dir = os.path.join(new_datadir, mode, data_label, data_idx)
                if not os.path.exists(new_dir):
                    print("{}\tNo dir: {}, {}".format(cnt, data_idx, data_label))
                else:
                    print("{}".format(cnt))

                cnt += 1


# # write in trainlist-jester-9.txt
# with open('./jester-v1-train.csv') as f:
#     lines = csv.reader(f)
#     folders = []
#     labels = []
#     for line in lines:
#         items = line[0].split(';')      # type(line): list -> extract a word(string type) for .split() function
#         if items[1].replace(' ', '_') in categories:
#             folders.append(items[0].lstrip())       # .lstrip() : delete left spaces
#             labels.append((items[1].rstrip()).replace(' ', '_'))        # .rstrip() : delete right spaces
#     contents = []
#     for i in range(len(folders)):
#         contents.append('%s\t%s'%(str(folders[i]), labels[i]))
#     with open('./trainlist-jester-9', 'w') as f:
#         f.write('\n'.join(contents))
#
# # write in vallist-jester-9.txt
# with open('./jester-v1-validation.csv') as f:
#     lines = csv.reader(f)
#     folders = []
#     labels = []
#     for line in lines:
#         items = line[0].split(';')
#         if items[1].replace(' ', '_') in categories:
#             folders.append(items[0].lstrip())
#             labels.append((items[1].rstrip()).replace(' ', '_'))
#     contents = []
#     for i in range(len(folders)):
#         contents.append('%s\t%s'%(str(folders[i]), labels[i]))
#     with open('./vallist-jester-9', 'w') as f:
#         f.write('\n'.join(contents))

def make_txtfile():
    # write in train_videofolder-9.txt
    with open('annotation/jester-v1-train.csv') as f:
        lines = csv.reader(f)
        #
        folders = []
        frames = []
        labels = []
        dict_folders = {}
        for line in lines:
            items = line[0].split(';')

            if items[1].replace('_', ' ') in categories:
                dict_folders[items[0]] = items[1].replace('_', ' ')
        print(dict_folders)
        dir = '/home/pjh/Downloads/20bn-jester-v1'
        for key in dict_folders:
            try:
                new_dir = os.path.join(dir, key)
                frames.append(str(len(os.listdir(new_dir))))
                folders.append(key)
                label = [str(n) for n, category in enumerate(categories) if dict_folders[key] == category]
                labels.append(label[0])
                # for n, category in enumerate(categories):
                #     if dict_folders[key] == category: labels.append(str(n))
            except:
                continue
        # print (len(folders), len(frames), len(labels))
        output = []
        for i in range(len(folders)):
            output.append('%s %s %s'%(folders[i], frames[i], labels[i]))
        with open('annotation/train_videofolder-9', 'w') as k:
            k.write('\n'.join(output))

        print(min(frames))
        # for i in range(len(folders)):
        #     if frames[i] == max(frames):
        #         print folders[i]

    # write in val_videofolder-9.txt
    with open('annotation/jester-v1-validation.csv') as f:
        lines = csv.reader(f)
        #
        folders = []
        frames = []
        labels = []
        dict_folders = {}
        for line in lines:
            items = line[0].split(';')

            if items[1].replace(' ', '_') in categories:
                dict_folders[items[0]] = items[1].replace(' ', '_')
        #print(dict_folders)
        dir = '/home/pjh/Downloads/20bn-jester-v1'
        for key in dict_folders:
            try:
                new_dir = os.path.join(dir, key)
                frames.append(str(len(os.listdir(new_dir))))
                folders.append(key)
                for n, category in enumerate(categories):
                    if dict_folders[key] == category: labels.append(str(n))
            except:
                continue
        # print (len(folders), len(frames), len(labels))
        output = []
        for i in range(len(folders)):
            output.append('%s %s %s'%(folders[i], frames[i], labels[i]))
        with open('annotation/val_videofolder-9', 'w') as k:
            k.write('\n'.join(output))

        print(min(frames))
        # for i in range(len(folders)):
        #     if frames[i] == max(frames):
        #         print folders[i]

    # filename number class
    files_input = ['./train_videofolder-9', './val_videofolder-9']
    files_output = ['./trainlist-jester-9', './vallist-jester-9']

    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(' ')
            folders.append(items[0])
            idx_categories.append(dict_categories[int(items[2])])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            #print(curFolder)
            output.append('%s\t%s'%(str(curFolder), curIDX))
            # print('%d/%d'%(i, len(folders)))
        with open(filename_output,'w') as f:
            f.write('\n'.join(output))

if __name__=="__main__":
    # check_csv()

    train_csv = "annotation/jester-v1-train.csv"
    val_csv = "annotation/jester-v1-validation.csv"
    test_csv = "annotation/jester-v1-test.csv"

    old_dataroot = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/dataset_jester/20bn-jester-v1"
    new_dataroot = "/media/pjh/2e8b4b6c-7754-4bf3-b610-7e52704614af/Dataset/dataset_jester/20bn-jester-v1"

    # mv_datadir("train", train_csv, old_dataroot, new_dataroot)
    # mv_datadir("val", val_csv, old_dataroot, new_dataroot)
    # mv_datadir("test", test_csv, old_dataroot, new_dataroot)

    # check_newdir("train", train_csv, new_dataroot)
    # check_newdir("validation", val_csv, new_dataroot)
    check_newdir("test", test_csv, new_dataroot)
