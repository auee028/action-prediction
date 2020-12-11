import tensorflow as tf

def LSTM(inputs,
        num_classes=15,
        num_hidden=128,
        scope=None,
		reuse=None):

    action_one_hot = tf.one_hot(inputs, num_classes)
    embedding_action = tf.get_variable('embedding_action', shape=[num_classes + 1, num_hidden])

    with tf.variable_scope(scope, 'ActionLSTM', [inputs], reuse=reuse):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                      inputs=action_one_hot,
                                                      dtype=tf.float32)

        lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])
        lstm_outputs = lstm_outputs[-1]

        logits = tf.matmul(lstm_outputs, embedding_action, transpose_b=True)

        return logits


if __name__ == '__main__':
    # inputs: [batch_size, num_seqs], outputs: [batch_size, num_classes]
    inps = tf.placeholder(dtype=tf.int32, shape=[4, 5])
    pred = LSTM(inps, num_classes=15, num_hidden=128, scope='ActionLSTM', reuse=False)
    print(pred)

