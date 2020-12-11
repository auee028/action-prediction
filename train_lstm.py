import tensorflow as tf
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

with open('action_map.txt', 'r') as f:
    action_labels = [line.strip() for line in f.readlines()]

# action_dic = sorted({n: i for i, n in enumerate(action_labels)}.items(), key=lambda x:x[1])
# print(action_dic)


def load_data(path):
    dataset = []
    for line in open(path,'r'):
        action_list, next_action = line.strip().split('\t')
        action_list = list(map(lambda x: action_labels.index(x), action_list.split(',')))
        next_action = action_labels.index(next_action)
        dataset.append({'action_list': action_list,
                        'next_action': next_action})

    return dataset


def next_batch(dataset, batch_size):
    start = 0
    while True:
        if start >= len(dataset)-batch_size:
            start = 0
            dataset = sorted(dataset, key=lambda x: np.random.rand())
            # print('Shuffle...')

        slices = dataset[start:start+batch_size]
        maxlen = max(map(lambda x: len(x.get('action_list')), slices))
        cur_batch = {'action_list': map(lambda x: x.get('action_list')+[0]*(maxlen-len(x.get('action_list'))), slices),
                     'next_action': map(lambda x: x.get('next_action'), slices)}

        start += batch_size

        yield cur_batch

# build model
def build(n_actions, n_hidden, dropout, learning_rate):
    action_list_labels = tf.placeholder(dtype=tf.int32, shape=[None,None], name='action_list_labels')
    next_action_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='next_action_labels')

    action_one_hot = tf.one_hot(action_list_labels, n_actions + 1)
    embedding_action = tf.get_variable('embedding_action', shape=[n_actions + 1, n_hidden])

    with tf.variable_scope('ActionLSTM'):
        # lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden)
        # lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=dropout)
        # lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden)
        #
        # multi_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])
        #
        # lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell=multi_cell,
        #                                     inputs=action_one_hot,
        #                                     sequence_length=tf.reduce_sum(tf.sign(action_list_labels),axis=1),
        #                                     dtype=tf.float32)

        lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=action_one_hot,
                                            dtype=tf.float32)

        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        lstm_outputs = lstm_outputs[-1]

        logits = tf.matmul(lstm_outputs, embedding_action, transpose_b=True)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(next_action_labels, n_actions+1),
                                                                  logits=logits))

    correct = tf.cast(tf.equal(next_action_labels, tf.cast(tf.argmax(logits,axis=-1), tf.int32)), tf.float32)
    acc = tf.reduce_mean(correct)

    global_step = tf.Variable(0, trainable=False)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    return {
        'action_list_labels': action_list_labels,
        'next_action_labels': next_action_labels,
        'loss': loss,
        'train_op': train_op,
        'logits': logits,
        'acc': acc,
        'global_step': global_step
    }



def train(data_path, batch_size):
    # train script
    m = build(n_actions=15,
              n_hidden=128,
              dropout=1.0,
              learning_rate=1e-3)

    print(tf.trainable_variables())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=50, var_list=tf.trainable_variables())

    # load dataset & define generator for batching
    dataset = load_data(path=data_path)

    # initial shuffling
    dataset = sorted(dataset, key=lambda x: np.random.rand())
    print('Initial Shuffle...')

    gen = next_batch(dataset, batch_size=batch_size)

    history = {'loss': [],
               'acc': []}

    epoch = 0
    while True:
        try:
            b = next(gen)
            _action_list_labels = b['action_list']
            _next_action_labels = b['next_action']

            acc, _, loss = sess.run([m['acc'], m['train_op'], m['loss']],
                                    feed_dict={m['action_list_labels']: _action_list_labels,
                                               m['next_action_labels']: _next_action_labels})

            if m['global_step'].eval(sess) % data_len == 0:
            # if m['global_step'].eval(sess) % 100 == 0:
                history['loss'].append(loss)
                history['acc'].append(acc)

                epoch += 1
                print('epoch : {}, loss: {}, acc : {}'.format(epoch, loss, acc))
                # print('step : {}, loss: {}, acc : {}'.format(m['global_step'].eval(sess), loss, acc))

                with open('result_lstm_{}.txt'.format(args.which), 'a') as f:
                    f.write('epoch : {}, loss: {}, acc : {}\n'.format(epoch, loss, acc))

                # print('saving ckpt...')
                # saver.save(sess, save_path='./actionseq_model/model-{}.ckpt'.format(epoch))
                if not os.path.exists(os.path.join(args.save_root, 'lstm_{}'.format(args.which))):
                    os.makedirs(os.path.join(args.save_root, 'lstm_{}'.format(args.which)))
                saver.save(sess, save_path=os.path.join(args.save_root, 'lstm_{}'.format(args.which), 'model-{}.ckpt'.format(epoch)))

            if epoch == args.epochs:
                break


        except KeyboardInterrupt:
            break

    # plt.plot(range(len(history['loss'])), history['loss'], 'r',
    #          range(len(history['acc'])), history['acc'], 'b')

    # plt.subplot(2, 1, 1)  # nrows=2, ncols=1, index=1
    plt.plot(range(len(history['loss'])), history['loss'], 'r')#, 'o-')
    plt.title('Training LSTM')
    plt.ylabel('loss')

    plt.savefig('train_seq-loss.png')
    plt.show()

    # plt.subplot(2, 1, 2)  # nrows=2, ncols=1, index=2
    plt.plot(range(len(history['acc'])), history['acc'], 'b')#, 'o-')
    plt.xlabel('epoch')
    plt.ylabel('acc.')

    plt.savefig('train_seq-acc.png')
    plt.show()



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_root', type=str, default="save_model")
    parser.add_argument('--which', type=str, default="ABR-action")
    parser.add_argument('--data_text_path', type=str, default="actionseq_dataset.txt")

    parser.add_argument('--epochs', type=int, default=20)

    args = parser.parse_args()

    with open(args.data_text_path, 'r') as f:
        data_len = len(f.readlines())

    train(data_path=args.data_text_path, batch_size=data_len)

