# action-prediction

## Prerequisites
* python 2.7
* CUDA 8.0, cuDNN 6.0

## Packages to install
```
pip install -r requirements.txt
```

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

## Test
1. export variables for CUDA 8.0 if it is needed
```
$ export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
$ export PATH=/usr/local/cuda-8.0/bin:$PATH

```
2. Activate a virtual environment with Anaconda
```
$ conda activate action

```
3. First run the flask app
```
$ python action_app.py

```
4. Second run the main code 
```
$ python action_main.py

```

If you want to get an action sequence and predict the next action, run the action_pred.py file instead of action_main.py 
```
$ python action_pred.py

```


## NOTICE
### Action list
* drinking
* brushing
* waving hands
* stretching arm
* reading

### Prediction list
* reading-reading ==> stretching arm
* reading-stretching arm ==> drinking

