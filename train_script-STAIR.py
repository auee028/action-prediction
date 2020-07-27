import tensorflow as tf
import numpy as np
import os
import math

import model_zoo_train as model_zoo
import support_function as sf


print(tf.__version__)

from tensorflow.python import debug as tf_debug

# configuration
# example: --which=jester --video_root_path=/home/wonhee/2-dataset --batch_size=8
tf.app.flags.DEFINE_integer("batch_size", 8, "batch size") # 8
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "learning rate")

tf.app.flags.DEFINE_integer("k_fold", 5, "data ratio")
tf.app.flags.DEFINE_bool("enable_k_fold", True, "enable k fold")

tf.app.flags.DEFINE_string("model", 'i3d', "resnetl10 / i3d")

tf.app.flags.DEFINE_string("video_root_path", '/media/pjh/HDD2/Dataset/STAIR-actions-master/STAIR_Actions_v1.1-frames', "video root path")
tf.app.flags.DEFINE_string("which", 'STAIR', "which annotation to use")

tf.app.flags.DEFINE_string("gpu", '0', 'which gpu to use')

# -------------------------------------------------------------------------------------------------------------------

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

video_root_path = FLAGS.video_root_path

batch_size = FLAGS.batch_size
k_fold = FLAGS.k_fold
# learning_rate = FLAGS.learning_rate

model = FLAGS.model

cwd = os.getcwd()

model_path = os.path.join(cwd, 'save_model', FLAGS.model+'-'+FLAGS.which+'-finetune')
print('model_path: ', model_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)

accuracy_path = os.path.join(cwd, 'analysis', 'accuracy-{}.txt'.format(FLAGS.model+'-'+FLAGS.which+'-finetune'))

############################# k-fold ###########################################
#### load train + validation video
if FLAGS.enable_k_fold:
    train_text_path = os.path.join(cwd, 'annotation/trainlist-{}.txt'.format(FLAGS.which))
    print(train_text_path, video_root_path)

    videos, intention_data, n_class = sf.get_HRI(video_root_path, train_text_path)

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
    frame_path = video_root_path
    train_text_path = os.path.join(cwd, 'annotation/trainlist-{}.txt'.format(FLAGS.which))
    train_videos, train_intention_data, n_class = sf.get_HRI(frame_path, train_text_path)

    print('train_video:', FLAGS.which, len(train_videos))
    print('class:', n_class)

    valid_text_path = os.path.join(cwd, 'annotation/vallist-{}.txt'.format(FLAGS.which))
    valid_videos, valid_intention_data, _ = sf.get_HRI(frame_path, valid_text_path)

    print('valid_video:', FLAGS.which, len(valid_videos))

########################################


# inputs: [batch_size, num_frames, h, w, c], outputs: [batch_size, dim_features]
inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, 224, 224, 3])
targets = tf.placeholder(dtype=tf.int32, shape=[None])
is_training = tf.placeholder(dtype=tf.bool)
dropout_keep_prob = tf.placeholder(dtype=tf.float32)

# build net
if model == 'resnetl10':
    net = model_zoo.ResNetl10Detector(inps=inputs, batch_size=batch_size, n_class=n_class, is_training=is_training,
                           final_end_point='Logits', dropout_keep_prob=dropout_keep_prob,
                           scope='v/ResNetl10', filters=[16, 16, 32, 64, 128], block_num=[1, 1, 1, 1])

elif model == 'i3d':
    net = model_zoo.I3DNet(inps=inputs, n_class=n_class, batch_size=batch_size,
                           pretrained_model_path='pretrained/i3d-tensorflow/kinetics-i3d/data/kinetics_i3d/model',
                           final_end_point='Logits', dropout_keep_prob=dropout_keep_prob,
                           is_training=is_training, scope='v/SenseTime_I3D')

else:
    raise NameError('can not find the model')

logits = net(inps=inputs)

# loss
# one_hot_targets = tf.one_hot(targets, n_class, dtype='float32')
crossent = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(targets, n_class),
                                                   logits=logits)

loss = tf.reduce_mean(crossent)

# accuracy
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), targets)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# gradient clip
global_step = tf.Variable(0, trainable=False)

starter_learning_rate = FLAGS.learning_rate

learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                     global_step,
                                                     decay_steps=len(train_videos)*5,
                                                     decay_rate=0.1,
                                                     staircase=True,
                                                     name='Learning_rate')


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
    # init all variables with pre-trained c3d
    sess.run(net.assign_ops)

saver = tf.train.Saver(max_to_keep=20)

ckpt = tf.train.latest_checkpoint(model_path)

if ckpt:
    print( 'restore from {}...'.format(ckpt))
    saver.restore(sess, ckpt)

start = -1 if not ckpt else int(os.path.basename(ckpt).split('-')[-1])
epochs = 120

crossval_flag = 0
prev_acc = 0

for epoch in range(start+1, epochs):
    index = np.arange(len(train_videos))
    np.random.shuffle(index)
    train_intention_data = train_intention_data[index]
    train_videos = train_videos[index]
    train_acc = 0
    train_cur_acc = 0
    for start, end in zip(range(0, len(train_videos), batch_size), range(batch_size, len(train_videos), batch_size)):
        train_frames = sf.preprocess_frame(train_videos[start:end])
        train_intention = train_intention_data[start:end]

        feed_dic = {inputs: train_frames,
                    targets: train_intention,
                    is_training: True,
                    dropout_keep_prob: 0.5}

        _, loss_value, acc, lr = sess.run([optimizer, loss, accuracy, learning_rate], feed_dict=feed_dic)

        train_acc += acc

        print('Epoch : %d, lr : %f, current batch : %d, loss : %f, batch_acc:%.6f' % (epoch, lr, start, loss_value, acc))
        if math.isnan(loss_value):
            print('NaN error')
    train_acc = train_acc / (len(train_videos) / float(batch_size))
    print("train accuracy : %f" % train_acc)
    with open(accuracy_path, 'a+') as txt:
        txt.write('train: epoch : %d, lr : %f, accuracy : %f \n' % (epoch, lr, train_acc))

    print("Epoch ", epoch, " is finished. I'm going to save the model ...")
    saver.save(sess, os.path.join(model_path, 'step'), global_step=epoch)

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
