#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import math
import re
import copy

import model_zoo_train as model_zoo
import support_function as sf

import datetime

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d_%H-%M-%S')


print(tf.__version__)

from tensorflow.python import debug as tf_debug

def sampling_frames(input_frames, sampling_num):
    total_num = len(input_frames)

    interval = 1
    if len(input_frames) > sampling_num:
        interval = math.floor(float(total_num) / sampling_num)
    print("sampling interval : {}".format(interval))
    interval = int(interval)

    out_frames = []
    if interval == 1:
        out_frames = copy.deepcopy(input_frames)
    else:
        for n in range(min(len(input_frames), sampling_num)):
            out_frames.append(input_frames[n*interval])

    # padding
    if len(out_frames) < sampling_num:
        print("before padding : {}".format(len(out_frames)))
        for k in range(sampling_num - len(out_frames)):
            out_frames.append(input_frames[-1])

    return out_frames



# configuration
tf.app.flags.DEFINE_integer("batch_size", 8, "batch size") # 8

tf.app.flags.DEFINE_string("video_root_path", "/media/pjh/HDD2/Dataset/ces-demo-5th/ABR_action_partdet", "video root path")
tf.app.flags.DEFINE_string("valid_text_path", "/media/pjh/HDD2/Dataset/ces-demo-5th/annotations/abr_4th_trainval/ABR_action_val.txt", "valid text path")
tf.app.flags.DEFINE_string("which", 'add_i3d-i3d-ABR_action_partdet_2020-10-09', "which annotation to use")

tf.app.flags.DEFINE_string("saved_model_path", '/media/pjh/HDD2/Dataset/save_model/i3d-i3d-ABR_action_partdet_2020-10-09', "path of the pretrained model to load")
# tf.app.flags.DEFINE_string("saved_model_path", '/media/pjh/HDD2/Dataset/save_model/mtsi3d-add_v-MultiScale_I3D_ABR-action-partdet_2020-09-26_22-30-25', "path of the pretrained model to load")
tf.app.flags.DEFINE_string("result_dir", None, 'result directory')

tf.app.flags.DEFINE_string("gpu", '0', 'which gpu to use')

# -------------------------------------------------------------------------------------------------------------------

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

video_root_path = FLAGS.video_root_path # os.path.join(FLAGS.video_root_path, FLAGS.which)

batch_size = FLAGS.batch_size

cwd = os.getcwd()

if not FLAGS.result_dir:
    result_path = os.path.join(cwd, 'analysis', 'validate-{}.txt'.format(FLAGS.which))
    if not os.path.exists(os.path.join(cwd, 'analysis')):
        os.makedirs(os.path.join(cwd, 'analysis'))
else:
    result_path = os.path.join(FLAGS.result_dir, 'validate-{}.txt'.format(FLAGS.which))
    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)


# load validation video
video_root_path = FLAGS.video_root_path
valid_text_path = FLAGS.valid_text_path

valid_videos, valid_intention_data, n_class = sf.get_HRI(video_root_path, valid_text_path)
# valid_videos, valid_intention_data, n_class = sf.get_HRI_v2(frame_path, valid_text_path)

print('valid_video:', FLAGS.which, len(valid_videos))
# print(valid_videos[:3], valid_intention_data[:3])

########################################


# inputs: [batch_size, num_frames, h, w, c], outputs: [batch_size, dim_features]
inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3])
targets = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)
dropout_keep_prob = tf.placeholder(dtype=tf.float32)

# build net
net = model_zoo.I3DNet(inps=inputs, n_class=n_class, batch_size=batch_size,
                           pretrained_model_path=None,
                           final_end_point='Logits', dropout_keep_prob=dropout_keep_prob,
                           is_training=is_training, scope='v/SenseTime_I3D')
# net = model_zoo.multiscaleI3DNet(inps=inputs, n_class=n_class, batch_size=batch_size,
#                            pretrained_model_path=None,
#                            final_end_point='Logits', dropout_keep_prob=dropout_keep_prob,
#                            is_training=is_training, scope='v/MultiScale_I3D')

logits = net(inps=inputs)

predictions = tf.cast(tf.argmax(logits, 1), tf.int32)

# accuracy
correct_prediction = tf.equal(predictions, targets)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


######################
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True

summary = tf.summary.merge_all()

# batching and train #

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if hasattr(net, 'assign_ops'):
    # init all variables with pre-trained i3d
    sess.run(net.assign_ops)


# summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)


saver = tf.train.Saver()

# ckpt = tf.train.latest_checkpoint(FLAGS.saved_model_path)

ckpt_num = 44
ckpt_state = tf.train.get_checkpoint_state(FLAGS.saved_model_path)
ckpt = ckpt_state.all_model_checkpoint_paths[ckpt_num]
ckpt = ckpt.replace('/mnt/juhee/action-prediction/save_model', '/media/pjh/HDD2/Dataset/save_model')

# ckpt_num = 47
# ckpt_state = tf.train.get_checkpoint_state(FLAGS.saved_model_path)
# ckpt = ckpt_state.all_model_checkpoint_paths[ckpt_num]

if ckpt:
    print( 'restore from {}...'.format(ckpt))
    saver.restore(sess, ckpt)
#
# elif FLAGS.mode_pretrained == 'i3d':
#     # variables = tf.contrib.slim.get_variables_to_restore()
#     # print(variables)
#
#     variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v/SenseTime_I3D')
#
#     # variables_to_restore = []
#     # var_dict = {re.sub(r':\d*', '', v.name): v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v/SenseTime_I3D')}
#     # for var_name, var_shape in tf.contrib.framework.list_variables(FLAGS.pretrained_model_path):
#     #     if var_name.startswith('v/SenseTime_I3D/Logits'):
#     #         continue
#     #     # load variable
#     #     var = tf.contrib.framework.load_variable(FLAGS.pretrained_model_path, var_name)
#     #     assign_op = var_dict[var_name].assign(var)
#     #     variables_to_restore.append(assign_op)
#
#     variables_to_restore = []
#     ckpt_vars = [name for name, shape in tf.train.list_variables(FLAGS.pretrained_model_path)]
#     # print(ckpt_vars[:3])
#     for v in variables:
#         if (v.name.split(':')[0] not in ckpt_vars) | (v.name.startswith('v/SenseTime_I3D/Logits')):
#             # print(v.name)
#             continue
#         # print(v)
#         variables_to_restore.append(v)
#
#     if variables_to_restore:
#         print('restore from {}...'.format(FLAGS.pretrained_model_path))
#         saver_pretrained = tf.train.Saver(variables_to_restore)
#         saver_pretrained.restore(sess, FLAGS.pretrained_model_path)


val_acc = 0
val_len = 0

y_true = []
y_pred = []

val_cnt = 0

print('Start validation')
for start, end in zip(range(0, len(valid_videos), batch_size),
                      range(batch_size, len(valid_videos) + 1, batch_size)):
    val_frames = sf.preprocess_frame(valid_videos[start:end])
    val_intention = valid_intention_data[start:end]

    feed_dic = {inputs: val_frames,
                targets: val_intention,
                is_training: False,
                dropout_keep_prob: 1.0}

    pred, acc = sess.run([predictions, accuracy], feed_dict=feed_dic)

    y_true += list(val_intention)
    y_pred += list(pred)

    val_len = val_len + batch_size
    print('validation:(%d/%d) batch_accuracy:%.6f'%(val_len, len(valid_videos), acc))
    val_acc += acc
    val_cnt += 1

val_acc = val_acc / float(val_cnt)
print("validation accuracy : %f" % val_acc)
with open(result_path, 'a') as txt:
    txt.write('val. accuracy : %f \n' % (val_acc))

    for tar, out in zip(y_true, y_pred):
        txt.write('target: {}\toutput: {}\n'.format(tar, out))
#######################################
import pandas as pd

pd_true = pd.Series(y_true)
pd_pred = pd.Series(y_pred)

print(pd.crosstab(pd_true, pd_pred, rownames=['True'], colnames=['Predicted'], margins=True))


