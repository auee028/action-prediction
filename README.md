# action-prediction

## Dependencies
* python 2.7
* CUDA 8.0, cuDNN 6.0

## Packages
* tensorflow-gpu==1.4.0
* numpy==1.16.2
* opencv-python==3.4.4.19
* requests==2.24.0
* natsort==6.2.1
* pillow==6.2.2
* stn==1.0.1
* Flask==1.0.2

## Dataset
* STAIR dataset
* 16 action classes among whole STAIR dataset
```
bowing
entering_room
going_out_of_room
listening_to_music_with_headphones
nodding
reading_book
reading_newspaper
setting_hair
shaking_hands
sitting_down
standing_up
studying
using_computer
using_smartphone
walking
writing

```
* training and validation video files lists under 'annotation/' folder<br>
  (trainlist-STAIR.txt, vallist-STAIR.txt)
```
Train dataset : 12806 videos
Val dataset : 1430 videos
(ratio of train/val : 8.96)
```

* Breakfast dataset
* 10 action classes in the kitchen
```
coffee (n=200)
orange juice (n=187)
chocolate milk (n=224)
tea (n=223)
bowl of cereals (n=214)
fried eggs (n=198)
pancakes (n=173)
fruit salad (n=185)
sandwich (n=197)
scrambled eggs (n=188).
```

## Requirements
Download darknet and move the folder of 'darknet/' below the root directory of 'action_anticipation/'. (reference : [here](https://pgmrlsh.tistory.com/4))

# step 1. Download darknet
```
cd action_anticipation/
git clone https://github.com/pjreddie/darknet

```

# step 2. Edit Makefile
```
vi Makefile

```
If you use CUDA, set GPU=1.
If you use OpenCV, set OPENCV=1.

# step 3. Make
```
cd darknet
make

```

## Training/Validation
train_script.py

## Test(including real-time)
action_viewer_webcam.py
action_viewer_webcam-cropped.py

## Data augmentation(to crop frames)
data_aug_2.py
