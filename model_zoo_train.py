import tensorflow as tf
import i3d
import re
from stn import spatial_transformer_network as transformer

from resnet3d import ResNet


def preprocess(inps, batch_size, is_training):
    _shape = tf.shape(inps)
    l, h, w = tf.unstack(_shape[1:-1])

    # first, crop it randomly
    crop_inputs = tf.cond(is_training,
                          lambda: tf.random_crop(inps, tf.unstack(_shape[:2]) + [224, 224, 3]),
                          lambda: inps[:, :, (h - 224) / 2:(h + 224) / 2, (w - 224) / 2:(w + 224) / 2])
    crop_inputs = tf.reshape(crop_inputs, (batch_size, -1, 224, 224, 3))  # for channel dimension restore

    processed_inputs_trans = []

    for _in in tf.unstack(crop_inputs):
        scale_x = tf.random_uniform([], 1.0, 5.0)
        scale_y = tf.random_uniform([], 1.0, scale_x)

        def body(t, seq_img_trans):
            img = tf.image.per_image_standardization(_in[t])  # standardization
            img_trans = transformer(tf.expand_dims(img, 0), tf.stack([[scale_x, 0., 0., 0., scale_y, 0.]]))

            seq_img_trans = seq_img_trans.write(t, img_trans[0])

            return t + 1, seq_img_trans

        t = tf.constant(0)
        seq_img_trans = tf.TensorArray(dtype=tf.float32, size=l)

        _, seq_img_trans = tf.while_loop(cond=lambda t, *_: t < l,
                                         body=body, loop_vars=(t, seq_img_trans))

        processed_inputs_trans.append(seq_img_trans.stack())

    return processed_inputs_trans


class I3DNet:
    def __init__(self, inps, n_class, batch_size,
                 pretrained_model_path, final_end_point, dropout_keep_prob,
                 is_training, scope='v/SenseTime_I3D'):

        self.final_end_point = final_end_point
        self.n_class = n_class
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.scope = scope

        # build entire pretrained networks (dummy operation!)
        i3d.I3D(preprocess(inps,batch_size,is_training), num_classes=n_class,
            final_endpoint=final_end_point, scope=scope,
            dropout_keep_prob=dropout_keep_prob, is_training=is_training)

        var_dict = { re.sub(r':\d*','',v.name):v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope) }
        self.assign_ops = []
        for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
            if var_name.startswith('v/SenseTime_I3D/Logits'):
                continue
            # load variable
            var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
            assign_op = var_dict[var_name].assign(var)
            self.assign_ops.append(assign_op)

    def __call__(self, inps):
        out, _ = i3d.I3D(preprocess(inps, self.batch_size,self.is_training), num_classes=self.n_class,
                        final_endpoint=self.final_end_point, scope=self.scope,
                        dropout_keep_prob=self.dropout_keep_prob, is_training=self.is_training, reuse=True)

        return out


class ResNetl10Detector:
    def __init__(self, inps, n_class, batch_size,
                 final_end_point, dropout_keep_prob,
                 is_training, filters, block_num, scope='v/ResNetl10'):

        self.final_end_point = final_end_point
        self.n_class = n_class
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.is_training = is_training
        self.scope = scope
        self.filters = filters
        self.block_num = block_num
        self.inps = inps

    def __call__(self, inps):
        resnet = ResNet(preprocess(inps, self.batch_size, self.is_training), num_classes=self.n_class,
                                 final_endpoint=self.final_end_point, scope=self.scope,
                                 dropout_keep_prob=self.dropout_keep_prob, is_training=self.is_training,
                                 filters=self.filters, block_num=self.block_num)

        out, _ = resnet._build_model()

        return out


class C3DNet:
    def __init__(self, pretrained_model_path, scope=None, trainable=False, finetune=False):
        if scope == None:
            self.scope = 'C3D'
        # load weights
        self.load_weights(pretrained_model_path, trainable)
        self.finetune = finetune

    def __call__(self, *args, **kwargs):
        def conv3d(name, l_input, w, b):
            return tf.nn.bias_add(
                tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
                b, name=name)

        def max_pool(name, l_input, k):
            return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='VALID', name=name)

        # Convolution Layer
        conv1 = conv3d('conv1', kwargs['inputs'], self._weights['wc1'], self._biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = max_pool('pool1', conv1, k=1)

        # Convolution Layer
        conv2 = conv3d('conv2', pool1, self._weights['wc2'], self._biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = max_pool('pool2', conv2, k=2)

        # Convolution Layer
        conv3 = conv3d('conv3a', pool2, self._weights['wc3a'], self._biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = conv3d('conv3b', conv3, self._weights['wc3b'], self._biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = max_pool('pool3', conv3, k=2)

        # Convolution Layer
        conv4 = conv3d('conv4a', pool3, self._weights['wc4a'], self._biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = conv3d('conv4b', conv4, self._weights['wc4b'], self._biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = max_pool('pool4', conv4, k=2)

        # Convolution Layer
        conv5 = conv3d('conv5a', pool4, self._weights['wc5a'], self._biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = conv3d('conv5b', conv5, self._weights['wc5b'], self._biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        # zero padding
        conv5 = tf.pad(conv5, [[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]], name='zeropad5')
        pool5 = max_pool('pool5', conv5, k=2)

        # Fully connected layer
        # pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3]) # only for ucf
        fc6 = tf.reshape(pool5, [-1, self._weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input
        fc6 = tf.matmul(fc6, self._weights['wd1']) + self._biases['bd1']
        fc6 = tf.nn.relu(fc6, name='fc6')  # Relu activation
        if self.finetune:
            fc6 = tf.nn.dropout(fc6, 0.5)

        fc7 = tf.matmul(fc6, self._weights['wd2']) + self._biases['bd2']
        fc7 = tf.nn.relu(fc7, name='fc7')  # Relu activation
        if self.finetune:
            fc7 = tf.nn.dropout(fc7, 0.5)

        net = dict(conv1=conv1, pool1=pool1,
                   conv2=conv2, pool2=pool2,
                   conv3=conv3, pool3=pool3,
                   conv4=conv4, pool4=pool4,
                   conv5=conv5, pool5=pool5,
                   fc6=fc6, fc7=fc7)

        return net[kwargs['layer']]

    def load_weights(self, pretrained_model_path, trainable):
        with tf.variable_scope(self.scope):
            # load pre-trained weights(C3D)
            self._weights = {}
            self._biases = {}
            for var_name, var_shape in tf.contrib.framework.list_variables(pretrained_model_path):
                # load variable
                var = tf.contrib.framework.load_variable(pretrained_model_path, var_name)
                var_dict = self._biases if len(var_shape) == 1 else self._weights

                var_dict[var_name.split('/')[-1]] = tf.get_variable(var_name,
                                                                    var_shape,
                                                                    initializer=tf.constant_initializer(var),
                                                                    dtype='float32',
                                                                    trainable=trainable)


class LSTM_Action:
    def __init__(self, n_hidden, n_class, batch_size):
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, None, 3],
                                     name='inputs')  # (batch, time, h, w, c)
        self.targets = tf.placeholder(tf.int32, [batch_size], name='targets')
        self.training = tf.placeholder(tf.bool, name='training')  # for BN

        _shape = tf.shape(self.inputs)
        l, h, w = tf.unstack(_shape[1:-1])

        # first, crop it randomly
        crop_inputs = tf.cond(self.training,
                              lambda: tf.random_crop(self.inputs, tf.unstack(_shape[:2]) + [100, 100, 3]),
                              lambda: self.inputs[:, :, (h - 100) / 2:(h + 100) / 2, (w - 100) / 2:(w + 100) / 2])
        crop_inputs = tf.reshape(crop_inputs, (batch_size, -1, 100, 100, 3))  # for channel dimension restore

        processed_inputs = []
        for _in in tf.unstack(crop_inputs):
            def preprocess(t, seq_img):
                img = tf.image.per_image_standardization(_in[t])  # standardization
                seq_img = seq_img.write(t, img)

                return t + 1, seq_img

            t = tf.constant(0)
            seq_img = tf.TensorArray(dtype=tf.float32, size=l)

            _, seq_img = tf.while_loop(cond=lambda t, *_: t < l,
                                       body=preprocess, loop_vars=(t, seq_img))

            processed_inputs.append(seq_img.stack())

        def batch_norm(inputs, training, name, device):
            with tf.device(device):
                inputs_norm = tf.layers.batch_normalization(inputs=inputs,
                                                            training=training, name=name)
            return inputs_norm

        # img_summary = tf.summary.image(name='img_summary', tensor=tf.stack(processed_inputs)[:,0])
        # self.summary_op = tf.summary.merge_all()

        # conv1
        conv1 = tf.layers.conv3d(inputs=tf.stack(processed_inputs), filters=64, kernel_size=(3, 3, 3), padding='same',
                                 name='conv1')
        conv1 = batch_norm(inputs=conv1, training=self.training, name='conv1_BN', device='/gpu:0')
        conv1 = tf.nn.relu(conv1, name='conv1_relu')

        # conv2
        conv2 = tf.layers.conv3d(inputs=conv1, filters=32, kernel_size=(1, 1, 1), padding='same', name='conv2')
        conv2 = batch_norm(inputs=conv2, training=self.training, name='conv2_BN', device='/gpu:0')
        conv2 = tf.nn.relu(conv2, name='conv2_relu')

        # pool1
        pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool1')

        # conv3
        conv3 = tf.layers.conv3d(inputs=pool1, filters=128, kernel_size=(3, 3, 3), padding='same', name='conv3')
        conv3 = batch_norm(inputs=conv3, training=self.training, name='conv3_BN', device='/gpu:0')
        conv3 = tf.nn.relu(conv3, name='conv3_relu')

        # conv4
        conv4 = tf.layers.conv3d(inputs=conv3, filters=64, kernel_size=(1, 1, 1), padding='same', name='conv4')
        conv4 = batch_norm(inputs=conv4, training=self.training, name='conv4_BN', device='/gpu:0')
        conv4 = tf.nn.relu(conv4, name='conv4_relu')

        # pool2
        pool2 = tf.layers.max_pooling3d(inputs=conv4, pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool2')

        # conv5
        conv5 = tf.layers.conv3d(inputs=pool2, filters=256, kernel_size=(3, 3, 3), padding='same', name='conv5')
        conv5 = batch_norm(inputs=conv5, training=self.training, name='conv5_BN', device='/gpu:0')
        conv5 = tf.nn.relu(conv5, name='conv5_relu')

        # conv6
        conv6 = tf.layers.conv3d(inputs=conv5, filters=128, kernel_size=(1, 1, 1), padding='same', name='conv6')
        conv6 = batch_norm(inputs=conv6, training=self.training, name='conv6_BN', device='/gpu:0')
        conv6 = tf.nn.relu(conv6, name='conv6_relu')

        # pool3
        pool3 = tf.layers.max_pooling3d(inputs=conv6, pool_size=(1, 2, 2), strides=(1, 2, 2), name='pool3')

        input_sequence = tf.reduce_mean(pool3, axis=(2, 3))

        cell = tf.nn.rnn_cell.LSTMCell(n_hidden)

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=input_sequence,
                                           dtype=tf.float32)

        self.softmax = tf.nn.softmax(tf.layers.dense(outputs, n_class), name='softmax')
        self.softmax_avg = tf.reduce_mean(self.softmax, axis=1)

