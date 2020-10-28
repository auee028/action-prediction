import os
import glob
import natsort

frames_root = "/home/pjh/Videos/test-arbitrary_frames/0"
save_root = "/home/pjh/Videos/test-arbitrary_frames/0_flip"
if not os.path.exists(save_root):
    os.makedirs(save_root)

frames = natsort.natsorted(glob.glob(os.path.join(frames_root, '*')))

for frame in frames:
    print(frame)
