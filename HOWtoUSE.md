## ACTION DEMO

### SETTINGS
* ubuntu 16.04
* CUDA 8.0
* CuDNN 6.0

### HOW TO RUN
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



