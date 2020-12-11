import tensorflow as tf
import numpy as np
import os
import math

import model_zoo_train as model_zoo
import support_function as sf

import datetime


now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d_%H-%M-%S')

print(tf.__version__)


# configuration
# example: --which=action --video_root_path=/home/user/dataset --batch_size=4
tf.app.flags.DEFINE_integer("batch_size", 4, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")

tf.app.flags.DEFINE_integer("k_fold", 5, "data ratio")
tf.app.flags.DEFINE_bool("enable_k_fold", False, "enable k fold")

# # (1) when loading i3d as pretrained model
# tf.app.flags.DEFINE_string("mode_pretrained", 'i3d', "which model to load")      # 'i3d' / 'mtsi3d'/ None
# tf.app.flags.DEFINE_string("pretrained_model_path", 'pretrained/i3d-tensorflow/kinetics-i3d/data/kinetics_i3d/model', "path of the pretrained model to load")
# (2) when loading mtsi3d as pretrained model
tf.app.flags.DEFINE_string("mode_pretrained", 'mtsi3d', "which model to load")      # 'i3d' / 'mtsi3d'/ None
tf.app.flags.DEFINE_string("pretrained_model_path", 'pretrained/mtsi3d', "path of the pretrained model to load")

# pretrained/mtsi3d_ABR-action_finetune 's scope : v/SenseTime_I3D // after running this code, model var scope : v/MultiScale_I3D
tf.app.flags.DEFINE_string("scope", 'v/SenseTime_I3D', 'scope name of pretrained mtsi3d model')     # v/SenseTime_I3D, v/MultiScale_I3D

tf.app.flags.DEFINE_string("video_root_path", "ceslea_videos_2020_cropped", "video root path")
tf.app.flags.DEFINE_string("train_text_path", "annotation/trainlist_ABR-action.txt", "video root path")
tf.app.flags.DEFINE_string("val_text_path", "annotation/vallist_ABR-action.txt", "video root path")

tf.app.flags.DEFINE_string("save_root_path", './save_model', "save root path")
tf.app.flags.DEFINE_string("which", 'ABR-action_{}'.format(nowDatetime), "which annotation to use")

tf.app.flags.DEFINE_string("max_to_keep", 150, "number of checkpoints to save")

tf.app.flags.DEFINE_string("gpu", '0', 'which gpu to use')

# -------------------------------------------------------------------------------------------------------------------

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

video_root_path = FLAGS.video_root_path

batch_size = FLAGS.batch_size
k_fold = FLAGS.k_fold

cwd = os.getcwd()

saved_model_path = os.path.join(FLAGS.save_root_path, 'mtsi3d_'+FLAGS.which)
print('model_path: '.format(saved_model_path))
if not os.path.exists(saved_model_path):
    os.makedirs(saved_model_path)

loss_path = os.path.join(cwd, 'analysis', 'loss-{}.txt'.format('mtsi3d_'+FLAGS.which))
accuracy_path = os.path.join(cwd, 'analysis', 'accuracy-{}.txt'.format('mtsi3d_'+FLAGS.which))
if not os.path.exists(os.path.join(cwd, 'analysis')):
    os.makedirs(os.path.join(cwd, 'analysis'))

############################# k-fold ###########################################
#### load train + validation video
if FLAGS.enable_k_fold:
    videos, intention_data, n_class = sf.get_HRI(video_root_path, FLAGS.train_text_path)
    # videos, intention_data, n_class = sf.get_HRI_v2(video_root_path, FLAGS.train_text_path)

    # load and shuffle data
    index = np.arange(len(videos))
    np.random.shuffle(index)
    intention_data = intention_data[index]
    videos = videos[index]

    k_fold_status = 0

    train_videos, train_intention_data, valid_videos, valid_intention_data = sf.cross_validation(videos, intention_data, k_fold, k_fold_status, n_class)
########################################################################

#### load train video and validation
else:
    train_videos, train_intention_data, n_class = sf.get_HRI(video_root_path, FLAGS.train_text_path)
    # train_videos, train_intention_data, n_class = sf.get_HRI_v2(video_root_path, FLAGS.train_text_path)

    print('train_video:', FLAGS.which, len(train_videos))
    print('class:', n_class)

    valid_videos, valid_intention_data, _ = sf.get_HRI(video_root_path, FLAGS.val_text_path)
    # valid_videos, valid_intention_data, _ = sf.get_HRI_v2(video_root_path, FLAGS.val_text_path)

    print('valid_video:', FLAGS.which, len(valid_videos))

########################################


# inputs: [batch_size, num_frames, h, w, c], outputs: [batch_size, dim_features]
inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3])
targets = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)
dropout_keep_prob = tf.placeholder(dtype=tf.float32)

# build net
net = model_zoo.multiscaleI3DNet(inps=inputs, n_class=n_class, batch_size=batch_size,
                           pretrained_model_path=FLAGS.pretrained_model_path,
                           final_end_point='Logits', dropout_keep_prob=dropout_keep_prob,
                           is_training=is_training, scope=FLAGS.scope)

logits = net(inps=inputs)

# Make saver at the end of weights to save - not to save useless variables after the model
# saver = tf.train.Saver(max_to_keep=20)
saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)      # max_to_keep default value is 5

# loss
one_hot_targets = tf.one_hot(targets, n_class, dtype='float32')
crossent = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(targets, n_class),
                                                   logits=logits)

loss = tf.reduce_mean(crossent)

# accuracy
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), targets)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# gradient clip
global_step = tf.Variable(0, trainable=False)

starter_learning_rate = FLAGS.learning_rate

# learning_rate = tf.train.exponential_decay(starter_learning_rate,global_step,decay_steps=len(train_videos)*5,decay_rate=0.1,staircase=True,name='Learning_rate')
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                     global_step,
                                                     decay_steps=len(train_videos)*5,
                                                     decay_rate=0.1,
                                                     staircase=True,
                                                     name='Learning_rate')  # 30 epochs expected. 10^-3=>10^-5

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

######################
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True


# batching and train #

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

if hasattr(net, 'assign_ops'):
    # init all variables with pre-trained i3d
    sess.run(net.assign_ops)


if FLAGS.mode_pretrained == 'mtsi3d':
    ckpt = tf.train.latest_checkpoint(FLAGS.pretrained_model_path)

    if ckpt:
        print( 'restore from {}...'.format(ckpt))
        saver.restore(sess, ckpt)

elif FLAGS.mode_pretrained == 'i3d':
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v/SenseTime_I3D')

    # variables_to_restore = []
    # var_dict = {re.sub(r':\d*', '', v.name): v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v/SenseTime_I3D')}
    # for var_name, var_shape in tf.contrib.framework.list_variables(FLAGS.pretrained_model_path):
    #     if var_name.startswith('v/SenseTime_I3D/Logits'):
    #         continue
    #     # load variable
    #     var = tf.contrib.framework.load_variable(FLAGS.pretrained_model_path, var_name)
    #     assign_op = var_dict[var_name].assign(var)
    #     variables_to_restore.append(assign_op)

    variables_to_restore = []
    ckpt_vars = [name for name, shape in tf.train.list_variables(FLAGS.pretrained_model_path)]
    for v in variables:
        if (v.name.split(':')[0] not in ckpt_vars) | (v.name.startswith('v/SenseTime_I3D/Logits')):
            continue
        variables_to_restore.append(v)

    if variables_to_restore:
        print('restore from {}...'.format(FLAGS.pretrained_model_path))
        saver_pretrained = tf.train.Saver(variables_to_restore)
        saver_pretrained.restore(sess, FLAGS.pretrained_model_path)

else:
    pass

# start = -1 if not ckpt else int(os.path.basename(ckpt).split('-')[-1])
start = -1
epochs = 120

crossval_flag = 0
prev_acc = 0

for epoch in range(start+1, epochs):
    index = np.arange(len(train_videos))
    np.random.shuffle(index)
    train_intention_data = train_intention_data[index]
    train_paths = train_videos[index]
    train_videos = train_videos[index]
    train_acc = 0
    train_cur_acc = 0
    for start, end in zip(range(0, len(train_videos), batch_size), range(batch_size, len(train_videos), batch_size)):
        if (end - start) != batch_size:
            print("start batch: {}, end batch: {}".format(start, end))
            continue
        train_frames = sf.preprocess_frame(train_videos[start:end])
        # print(train_paths[start:end])
        if train_frames.shape[1] != 64:
            print(train_frames.shape, train_paths[start:end])
        train_intention = train_intention_data[start:end]

        feed_dic = {inputs: train_frames,
                    targets: train_intention,
                    is_training: True,
                    dropout_keep_prob: 0.5}

        _, loss_value, acc, lr = sess.run([optimizer, loss, accuracy, learning_rate], feed_dict=feed_dic)

        local_step = start / batch_size

        train_acc += acc

        print('Epoch : %d, lr : %.8f, current batch : %d, loss : %f, batch_acc:%.6f' % (epoch, lr, start, loss_value, acc))
        # with open(loss_path, 'a+') as lossfile:
        #     lossfile.write('Epoch : %d, lr : %f, current batch : %d, loss : %f, batch_acc:%.6f' % (epoch, lr, start, loss_value, acc))

        if math.isnan(loss_value):
            print('NaN error')
    train_acc = train_acc / (len(train_videos) / float(batch_size))
    print("train accuracy : %f" % train_acc)
    with open(accuracy_path, 'a+') as txt:
        # txt.write('train: epoch : %d, lr : %f, accuracy : %f \n' % (epoch, lr, train_acc))
        txt.write('train: epoch : %d, lr : %.8f, loss : %f, accuracy : %f \n' % (epoch, lr, loss_value, train_acc))

    print("Epoch ", epoch, " is finished. I'm going to save the model ...")
    saver.save(sess, os.path.join(saved_model_path, 'step'), global_step=epoch)

    print('validation is being processed')
    val_acc = 0
    val_len = 0
    for start, end in zip(range(0, len(valid_videos), batch_size),
                          range(batch_size, len(valid_videos) + 1, batch_size)):
        val_frames = sf.preprocess_frame(valid_videos[start:end])

        val_intention = valid_intention_data[start:end]

        feed_dic = {inputs: val_frames,
                    targets: val_intention,
                    is_training: False,
                    dropout_keep_prob: 1.0}

        acc = sess.run(accuracy, feed_dict=feed_dic)
        val_len=val_len+batch_size
        print('validation:(%d/%d) batch_accuracy:%.6f'%(val_len, len(valid_videos), acc))
        val_acc += acc

    val_acc = val_acc / (len(valid_videos) / float(batch_size))
    print("validation accuracy : %f" % val_acc)
    with open(accuracy_path, 'a+') as txt:
        txt.write('val: epoch : %d, accuracy : %f \n' % (epoch, val_acc))
    ###########################################################################

    # learning_rate *= 0.9
    # learning_rate *= 0.98

    if FLAGS.enable_k_fold:
        if prev_acc > val_acc:
            if crossval_flag == 1:
                prev_acc = 0
                crossval_flag = 0

                # cross validation
                if k_fold_status == k_fold:
                    k_fold_status = 0
                else:
                    k_fold_status = k_fold_status + 1
                    print('change validation ', prev_acc, val_acc)
                    train_videos, train_intention_data, valid_videos, valid_intention_data = sf.cross_validation(videos,
                                                                                                              intention_data,
                                                                                                              k_fold,
                                                                                                              k_fold_status, n_class)

            else:
                prev_acc = val_acc
                crossval_flag = 1

