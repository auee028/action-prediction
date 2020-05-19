# action-prediction

## Dependencies
* python 2.7
* CUDA 8.0, cuDNN 6.0

## Packages
* tensorflow-gpu 1.4.0
* numpy==1.16.2
* opencv-python==3.4.4.19

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

## Training/Validation
train_script.py

## Test(including real-time)
action_viewer_webcam.py

