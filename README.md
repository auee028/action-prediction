# action-prediction

## Prerequisites
* python 2.7
* CUDA 8.0, cuDNN 6.0

## Packages to install
```
pip install -r requirements.txt
```

## ABR-actionDataset
* saved in NAS (/homes/admin/Lab Dataset/action_data)
```
action_data/
        |____ ABR_action-cropped          # videos cropped by part_detector
        |____ ceslea_videos               # original videos collected in 2018?
        |____ ceslea_videos_2020          # original videos collected in 2020.2(4th year of Ceslea)
        |____ ceslea_videos_2020_cropped  # cropped frames of ceslea_videos_2020/ with crop_frames.py (used for training)
```

* 15 action classes
```

sitting
standing
drinking
brushing
playing uculele
speaking
waving hands
working
coming
leaving
talking on the phone
stretching
nodding off
reading 
blowing nose

```
* You should locate training and validation video list files under 'annotation/' folder<br>
  (trainlist-ABR.txt, vallist-ABR.txt)


## Requirements

### Darknet
* step 1. Download darknet if you don't have
Download darknet and move the folder of 'darknet/' below the root directory of this project. (reference : [here](https://pgmrlsh.tistory.com/4))
```
cd action_anticipation/
git clone https://github.com/pjreddie/darknet

```

* step 2. Edit Makefile
```
vi Makefile

```
If you use CUDA, set GPU=1.
If you use OpenCV, set OPENCV=1.

* step 3. Make
```
cd darknet
make

```

## Training/Validation
1. training mtsi3d with pretrained i3d weights
```
python train_mtsi3d.py --mode_pretrained=i3d --pretrained_model_path=pretrained/i3d-tensorflow/kinetics-i3d/data/kinetics_i3d/model --scope=v/SenseTime_I3D
```

2. training mtsi3d with pretrained(or paused) mtsi3d weights
```
python train_mtsi3d.py --mode_pretrained=mtsi3d --pretrained_model_path=pretrained/mtsi3d_ABR-action_finetune --scope=v/SenseTime_I3D
```
or
```
python train_mtsi3d.py --mode_pretrained=mtsi3d --pretrained_model_path=_your model path_ --scope=v/MultiScale_I3D
```
* If you set cross-validation as True, only train_text_path is used for training.
* If you set cross-validation as False, both train_text_path and val_text_path are used for training.

3. check for loss and accuracy saved at ./analysis/ (default)

4. check for trained model at ./save_model (default)


## Test(including real-time) ==> for more details, see HOWtoUSE.md
```
python action_app.py
python action_main.py
```
or
```
python action_app.py
python action_pred.py
```
