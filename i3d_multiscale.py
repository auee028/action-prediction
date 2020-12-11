""" This implementation based on naive tensorflow framework
Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.
The model is introduced in:
  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
  Joao Carreira, Andrew Zisserman
  https://arxiv.org/pdf/1705.07750v1.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from snets.net_utils import unit3D


# fc
def _dense(x, out_dim):
	x = tf.cast(x, dtype=tf.float16)

	h1 = tf.layers.dense(x, 4096)
	h2 = tf.layers.dense(h1, 1000)
	h3 = tf.layers.dense(h2, 1000)
	y = tf.layers.dense(h3, out_dim)
	return y

# 2d cnn
def _2dcnn(x, name):
	_shape = tf.shape(x)
	l, h, w = tf.unstack(_shape[1:-1])

	processed_inputs = []
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		for _in in tf.unstack(x):
			with tf.variable_scope('Conv2d_1a'):
				cnv = _conv2d(_in, 512)
			with tf.variable_scope('Conv2d_2a'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3a'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_a')

			with tf.variable_scope('Conv2d_1b'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2b'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3b'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_b')

			processed_inputs.append(pool)

	return tf.stack(processed_inputs)

# embedding for feat_1 (2d cnn)
def embedding_feat_1(x, out_dim, name, dropout_keep_prob=1.0):
	_shape = tf.shape(x)
	l, h, w = tf.unstack(_shape[1:-1])

	processed_inputs = []
	with tf.variable_scope(name, 'conv2d', reuse=tf.AUTO_REUSE):
		for _in in tf.unstack(x):
			with tf.variable_scope('Conv2d_1a'):
				cnv = _conv2d(_in, 512)
			with tf.variable_scope('Conv2d_2a'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3a'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_a')

			with tf.variable_scope('Conv2d_1b'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2b'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3b'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_b')

			with tf.variable_scope('Conv2d_1c'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2c'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3c'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_c')

			with tf.variable_scope('Conv2d_1d'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2d'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3d'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_d')

			processed_inputs.append(pool)

		out = tf.squeeze(tf.stack(processed_inputs), [2, 3])
		out = tf.layers.flatten(out)

		with tf.variable_scope('fc_layers'):
			# out = tf.layers.dense(out, 4096)
			# out = tf.nn.dropout(out, dropout_keep_prob)
			out = tf.layers.dense(out, 2048)
			# out = tf.nn.dropout(out, dropout_keep_prob)
			out = tf.layers.dense(out, 1000)
			out = tf.nn.dropout(out, dropout_keep_prob)
			# out = tf.layers.dense(out, 1000)
			out = tf.layers.dense(out, out_dim)

		return out

# embedding for feat_2 (2d cnn)
def embedding_feat_2(x, out_dim, name, dropout_keep_prob=1.0):
	_shape = tf.shape(x)
	l, h, w = tf.unstack(_shape[1:-1])

	processed_inputs = []
	with tf.variable_scope(name, 'conv2d', reuse=tf.AUTO_REUSE):
		for _in in tf.unstack(x):
			with tf.variable_scope('Conv2d_1a'):
				cnv = _conv2d(_in, 512)
			with tf.variable_scope('Conv2d_2a'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3a'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_a')

			with tf.variable_scope('Conv2d_1b'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2b'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3b'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_b')

			with tf.variable_scope('Conv2d_1c'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2c'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3c'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_c')

			processed_inputs.append(pool)

		out = tf.squeeze(tf.stack(processed_inputs), [2, 3])
		out = tf.layers.flatten(out)

		with tf.variable_scope('fc_layers'):
			# out = tf.layers.dense(out, 4096)
			# out = tf.nn.dropout(out, dropout_keep_prob)
			out = tf.layers.dense(out, 2048)
			# out = tf.nn.dropout(out, dropout_keep_prob)
			out = tf.layers.dense(out, 1000)
			out = tf.nn.dropout(out, dropout_keep_prob)
			# out = tf.layers.dense(out, 1000)
			out = tf.layers.dense(out, out_dim)

		return out

# embedding for feat_3 (2d cnn)
def embedding_feat_3(x, out_dim, name, dropout_keep_prob=1.0):
	_shape = tf.shape(x)
	l, h, w = tf.unstack(_shape[1:-1])

	processed_inputs = []
	with tf.variable_scope(name, 'conv2d', reuse=tf.AUTO_REUSE):
		for _in in tf.unstack(x):
			with tf.variable_scope('Conv2d_1a'):
				cnv = _conv2d(_in, 512)
			with tf.variable_scope('Conv2d_2a'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3a'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_a')

			with tf.variable_scope('Conv2d_1b'):
				cnv = _conv2d(pool, 512)
			with tf.variable_scope('Conv2d_2b'):
				cnv = _conv2d(cnv, 512)
			with tf.variable_scope('Conv2d_3b'):
				cnv = _conv2d(cnv, 512)
			pool = _max_pool(cnv, 'Emb_MaxPool_b')

			processed_inputs.append(pool)

			# 		def conv(t, seq_img):
			# 			cnv = _conv2d(_in, 512, 'Emb_Conv2d_1')
			# 			cnv = _conv2d(cnv, 512, 'Emb_Conv2d_2')
			# 			cnv = _conv2d(cnv, 512, 'Emb_Conv2d_3')
			# 			pool = _max_pool(cnv, 'Emb_MaxPool')
			# 			seq = seq_img.write(t, pool)
			#
			# 			return t + 1, seq
			#
			# 		t = tf.constant(0)
			# 		seq = tf.TensorArray(dtype=tf.float32, size=l)
			#
			# 		_, seq = tf.while_loop(cond=lambda t, *_: t < l,
			# 								   body=conv, loop_vars=(t, seq))
			#
			# 		processed_inputs.append(seq.stack())
			#
		out = tf.squeeze(tf.stack(processed_inputs), [2, 3])
		out = tf.layers.flatten(out)

		with tf.variable_scope('fc_layers'):
			# out = tf.nn.dropout(out, dropout_keep_prob)
			out = tf.layers.dense(out, 1000)
			out = tf.nn.dropout(out, dropout_keep_prob)
			# out = tf.layers.dense(out, 1000)
			out = tf.layers.dense(out, out_dim)

		return out

def _conv2d(x, num_filters, filter_height=3, filter_width=3, stride=1, padding='SAME'):
	input_channels = int(x.get_shape()[-1])

	W = tf.get_variable('w', shape=[filter_height, filter_width, input_channels, num_filters],
						initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
	x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
	x = tf.nn.relu(x)
	return x

def _max_pool(x, name, filter_height=2, filter_width=2, stride=2, padding='VALID'):
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride, stride, 1], padding=padding, name=name)

# dot product
def _dot_product(x1, x2):
	# x1 = tf.cast(x1, dtype=tf.float16)
	# x2 = tf.cast(x2, dtype=tf.float16)

	y = tf.reduce_sum(tf.multiply(x1, x2), axis=1, keep_dims=True)  # shape : (batch_size, 1)
	return y

'''
def FeatureEmbedding(inputs,
					 num_classes=8,
					 batch_size=4,
					 final_endpoint='Predictions'):
	end_points = {}

	# hidden_feat_1
	hidden_feat_1 = inputs[0]
	hidden_feat_1.set_shape((batch_size, 32, 28, 28, 192))
	with tf.variable_scope('EmbeddingFeats'):
		embedding_1 = embedding_feat_1(hidden_feat_1, num_classes, name='Embedding_feat_1')

	# hidden_feat_2
	hidden_feat_2 = inputs[1]
	hidden_feat_2.set_shape((batch_size, 16, 14, 14, 480))
	with tf.variable_scope('EmbeddingFeats'):
		embedding_2 = embedding_feat_2(hidden_feat_2, num_classes, name='Embedding_feat_2')

	# hidden_feat_3
	hidden_feat_3 = inputs[2]
	hidden_feat_3.set_shape((batch_size, 8, 7, 7, 832))
	with tf.variable_scope('EmbeddingFeats'):
		embedding_3 = embedding_feat_3(hidden_feat_3, num_classes, name='Embedding_feat_3')

	# last logits
	averaged_logits = inputs[3]

	# dot product
	similarity_score_1 = _dot_product(averaged_logits, embedding_1)
	similarity_score_2 = _dot_product(averaged_logits, embedding_2)
	similarity_score_3 = _dot_product(averaged_logits, embedding_3)

	similarity_scores = tf.concat([similarity_score_1, similarity_score_2, similarity_score_3], axis=1)
	attn_distribs = tf.nn.softmax(similarity_scores, dim=1)

	_shape = tf.shape(attn_distribs)
	_, num_embs = tf.unstack(_shape)

	embs = tf.stack([embedding_1, embedding_2, embedding_3], axis=0)

	def body(t, seq_weighted_emb):
		embedding = embs[t]
		emb_weight = tf.gather(attn_distribs, t, axis=1)
		weighted_emb = tf.add(tf.expand_dims(emb_weight, axis=1), embedding)

		seq_weighted_emb = seq_weighted_emb.write(t, weighted_emb)

		return t + 1, seq_weighted_emb

	t = tf.constant(0)
	seq_weighted_emb = tf.TensorArray(dtype=tf.float32, size=num_embs)

	_, seq_weighted_emb = tf.while_loop(cond=lambda t, *_: t < num_embs,
										body=body, loop_vars=(t, seq_weighted_emb))

	# processed_weighted_embs.append(seq_weighted_emb.stack())
	#
	# attn_value = tf.reduce_sum(tf.concat(processed_weighted_embs, axis=0), axis=1)
	attn_value = tf.reduce_sum(seq_weighted_emb.stack(), axis=0)

	end_point = 'Logits'
	logits = tf.layers.dense(tf.concat([attn_value, averaged_logits], axis=1), num_classes)
	end_points[end_point] = logits
	if end_point == final_endpoint: return logits, end_points  # , [hidden_feat_1, hidden_feat_2, hidden_feat_3]

	end_point = 'Predictions'
	predictions = tf.nn.softmax(averaged_logits)
	end_points[end_point] = predictions
	if end_point == final_endpoint: return predictions, end_points
'''


def MultiscaleI3D(inputs,
				  num_classes=400,
				  is_training=True,
				  batch_size=4,
				  final_endpoint='Predictions',
				  data_format='NHWC',
				  dropout_keep_prob=1.0,
				  min_depth=16,
				  depth_multiplier=1.0,
				  scope=None,
				  reuse=None):
	# bgr -> rgb
	b,g,r = tf.split(inputs, 3, axis=4)
	inputs = tf.squeeze(tf.stack([r, g, b], axis=4), axis=5)

	# (min,max) => (0,255)
	inputs /= 255.0

	end_points = {}
	if depth_multiplier <= 0:
		raise ValueError('depth_multiplier is not greater than zero.')
	depth = lambda d: max(int(d * depth_multiplier), min_depth)

	concat_axis = 2 if data_format == 'NCHW' else -1
	with tf.variable_scope(scope, 'I3D', [inputs],reuse=reuse):
		end_point = 'Conv3d_1a_7x7x7'
		net = unit3D(inputs, depth(64), [7,7,7], 2, is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_2a_1x3x3'
		net = tf.nn.max_pool3d(net, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME', name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Conv3d_2b_1x1x1'
		net = unit3D(net, depth(64), [1, 1, 1], is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Conv3d_2c_3x3x3'
		net = unit3D(net, depth(192), [3, 3, 3], is_training=is_training, name=end_point)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_3a_1x3x3'
		net = tf.nn.max_pool3d(net, [1, 1, 3, 3, 1], [1, 1, 2, 2, 1], padding='SAME', name=end_point)
		end_points[end_point] = net

		# hidden_feat_1
		net.set_shape((batch_size, 32, 28, 28, 192))
		with tf.variable_scope('EmbeddingFeats'):
			embedding_1 = embedding_feat_1(net, num_classes, name='Embedding_feat_1')

		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_3b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(96), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(16), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(32), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, [1, 3, 3, 3, 1], strides=[1, 1, 1, 1, 1],
				                            padding='SAME', name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)

		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_3c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(192), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(96), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
		net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_4a_3x3x3'
		net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name=end_point)
		end_points[end_point] = net

		# hidden_feat_2
		net.set_shape((batch_size, 16, 14, 14, 480))
		with tf.variable_scope('EmbeddingFeats'):
			embedding_2 = embedding_feat_2(net, num_classes, name='Embedding_feat_2')

		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_4b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(192), kernel_shape=[1, 1, 1],
                           		  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(96), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(208), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(16), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(48), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
		net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_4c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(160), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(112), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(224), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(24), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(64), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4d'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(256), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(24), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(64), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')

			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net

		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4e'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(112), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(144), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(288), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(64), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')

			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(64), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_4f'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(256), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(160), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(320), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points
		end_point = 'MaxPool3d_5a_2x2x2'
		net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
							   padding='SAME', name=end_point)
		end_points[end_point] = net

		# hidden_feat_3
		net.set_shape((batch_size, 8, 7, 7, 832))
		with tf.variable_scope('EmbeddingFeats'):
			embedding_3 = embedding_feat_3(net, num_classes, name='Embedding_feat_3')

		if end_point == final_endpoint: return net, end_points
		end_point = 'Mixed_5b'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(256), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(160), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(320), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(32), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0a_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'Mixed_5c'
		with tf.variable_scope(end_point):
			with tf.variable_scope('Branch_0'):
				branch_0 = unit3D(net, depth(384), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
			with tf.variable_scope('Branch_1'):
				branch_1 = unit3D(net, depth(192), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_1 = unit3D(branch_1, depth(384), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_2'):
				branch_2 = unit3D(net, depth(48), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0a_1x1x1')
				branch_2 = unit3D(branch_2, depth(128), kernel_shape=[3, 3, 3],
								  is_training=is_training, name='Conv3d_0b_3x3x3')
			with tf.variable_scope('Branch_3'):
				branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
											strides=[1, 1, 1, 1, 1], padding='SAME',
											name='MaxPool3d_0a_3x3x3')
				branch_3 = unit3D(branch_3, depth(128), kernel_shape=[1, 1, 1],
								  is_training=is_training, name='Conv3d_0b_1x1x1')
			net = tf.concat([branch_0, branch_1, branch_2, branch_3], concat_axis)
		end_points[end_point] = net
		if end_point == final_endpoint: return net, end_points

		end_point = 'FeatureExtraction'
		with tf.variable_scope(end_point):
			net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
								   strides=[1, 1, 1, 1, 1], padding='VALID')
			net = tf.nn.dropout(net, dropout_keep_prob)
			feats = tf.reduce_mean(net, axis=[1,2,3])
			end_points[end_point] = feats
			if final_endpoint=='FeatureExtraction': return tf.reduce_mean(net, axis=[1,2,3]), end_points

		end_point = 'Logits'
		with tf.variable_scope(end_point):
			squatial_logits = unit3D(net, num_classes,
							kernel_shape=[1, 1, 1],
							activation_fn=None,
							is_training=is_training,
							use_batch_norm=False,
							use_bias=True,
							name='Conv3d_0c_1x1x1')
			squatial_logits = tf.squeeze(squatial_logits, [2, 3], name='SpatialSqueeze')
			averaged_logits = tf.reduce_mean(squatial_logits, axis=1)
		# end_points[end_point] = averaged_logits

		"""
		hidden_feat_1.set_shape((batch_size, 32, 28, 28, 192))
		hidden_feat_2.set_shape((batch_size, 16, 14, 14, 480))
		hidden_feat_3.set_shape((batch_size, 8, 7, 7, 832))

		'''
		# embedding_1 = self._dense(tf.reshape(hidden_feat_1, (self.batch_size, tf.reduce_prod(shape_h1[1:-1]))), self.n_class)
		# embedding_2 = self._dense(tf.reshape(hidden_feat_2, (self.batch_size, tf.reduce_prod(shape_h2[1:-1]))), self.n_class)
		# embedding_3 = self._dense(tf.reshape(hidden_feat_3, (self.batch_size, tf.reduce_prod(shape_h3[1:-1]))), self.n_class)
		embedding_1 = _dense(tf.layers.flatten(hidden_feat_1), num_classes)
		embedding_2 = _dense(tf.layers.flatten(hidden_feat_2), num_classes)
		embedding_3 = _dense(tf.layers.flatten(hidden_feat_3), num_classes)
		'''

		# embedding_1 = _2dcnn(hidden_feat_1, 'FeatEmbedding_1')
		# embedding_2 = _2dcnn(hidden_feat_2, 'FeatEmbedding_2')
		# embedding_3 = _2dcnn(hidden_feat_3, 'FeatEmbedding_3')
		with tf.variable_scope('EmbeddingFeats'):
			embedding_1 = embedding_feat_1(hidden_feat_1, num_classes, name='Embedding_feat_1')
			embedding_2 = embedding_feat_2(hidden_feat_2, num_classes, name='Embedding_feat_2')
			embedding_3 = embedding_feat_3(hidden_feat_3, num_classes, name='Embedding_feat_3')
		"""

		similarity_score_1 = _dot_product(averaged_logits, embedding_1)
		similarity_score_2 = _dot_product(averaged_logits, embedding_2)
		similarity_score_3 = _dot_product(averaged_logits, embedding_3)

		similarity_scores = tf.concat([similarity_score_1, similarity_score_2, similarity_score_3], axis=1)
		attn_distribs = tf.nn.softmax(similarity_scores, dim=1)

		_shape = tf.shape(attn_distribs)
		_, num_embs = tf.unstack(_shape)

		embs = tf.stack([embedding_1, embedding_2, embedding_3], axis=0)

		def body(t, seq_weighted_emb):
			embedding = embs[t]
			emb_weight = tf.gather(attn_distribs, t, axis=1)
			weighted_emb = tf.add(tf.expand_dims(emb_weight, axis=1), embedding)

			seq_weighted_emb = seq_weighted_emb.write(t, weighted_emb)

			return t + 1, seq_weighted_emb

		t = tf.constant(0)
		seq_weighted_emb = tf.TensorArray(dtype=tf.float32, size=num_embs)

		_, seq_weighted_emb = tf.while_loop(cond=lambda t, *_: t < num_embs,
											body=body, loop_vars=(t, seq_weighted_emb))

		# processed_weighted_embs.append(seq_weighted_emb.stack())
		#
		# attn_value = tf.reduce_sum(tf.concat(processed_weighted_embs, axis=0), axis=1)
		attn_value = tf.reduce_sum(seq_weighted_emb.stack(), axis=0)

		logits = tf.layers.dense(tf.concat([attn_value, averaged_logits], axis=1), num_classes)

		end_points[end_point] = logits
		if end_point == final_endpoint: return logits, end_points#, averaged_logits#, [hidden_feat_1, hidden_feat_2, hidden_feat_3]

		if final_endpoint == 'SequatialLogits': return squatial_logits, end_points

		end_point = 'Predictions'
		predictions = tf.nn.softmax(averaged_logits)
		end_points[end_point] = predictions
		if end_point == final_endpoint: return predictions, end_points




if __name__ == '__main__':
	# inputs: [batch_size, num_frames, h, w, c], outputs: [batch_size, dim_features]
	inps = tf.placeholder(dtype=tf.float32, shape=[3, 64, 224, 224, 3])
	si3d, hiddens = MultiscaleI3D(inps, num_classes=15, batch_size=3,
								   final_endpoint='Logits', scope='v/SenseTime_I3D',
								   dropout_keep_prob=0.5, is_training=True)
	# si3d, hiddens, logits = MultiscaleI3D(inps, num_classes=15, batch_size=3,
	# 							   final_endpoint='Logits', scope='v/SenseTime_I3D',
	# 							   dropout_keep_prob=0.5, is_training=True)
	print(si3d)  # (4, 15)
	print(hiddens)
	# print(logits)

	# tvar = tf.trainable_variables()
	# for i in tvar:
	# 	print(i)

	'''
	inps = tf.placeholder(dtype=tf.float32, shape=[4, 64, 224, 224, 3])
	si3d, _, hddns = MultiscaleI3D(inps, num_classes=15,
				  final_endpoint='Logits', scope='v/SenseTime_I3D',
				  dropout_keep_prob=0.5, is_training=True)
	print(si3d)		# (4, 15)
	print(hddns)
	print(tf.shape(hddns[0]), tf.shape(hddns[1]), tf.shape(hddns[2]))	# [<tf.Tensor 'v/SenseTime_I3D/MaxPool3d_3a_1x3x3:0' shape=(4, 32, 28, 28, 192) dtype=float32>, <tf.Tensor 'v/SenseTime_I3D/MaxPool3d_4a_3x3x3:0' shape=(4, 16, 14, 14, 480) dtype=float32>, <tf.Tensor 'v/SenseTime_I3D/MaxPool3d_5a_2x2x2:0' shape=(4, 8, 7, 7, 832) dtype=float32>]

	x = tf.reshape(hddns[0], [4, -1])		# (4, 4816896)
	y = tf.reshape(hddns[1], [4, -1])		# (4, 1505280)
	z = tf.reshape(hddns[2], [4, -1])		# (4, 326144)
	print(x, y, z)
	# dot_product = tf.reduce_sum(tf.multiply(x, x), axis=1)		# (4,)
	dot_product_1 = tf.reduce_sum(tf.multiply(x, x), axis=1, keep_dims=True)		# (4, 1)
	dot_product_2 = tf.reduce_sum(tf.multiply(x, x), axis=1, keep_dims=True)
	print(dot_product_1)
	concat = tf.concat([dot_product_1, dot_product_2], axis=1)
	print(concat)
	softmax = tf.nn.softmax(concat)
	print(softmax)
	x_prime = concat
	attention_value = tf.multiply(softmax, x_prime)
	print(attention_value)
	'''

	'''
	input_data = [
		[1, 2, 1], [1, 4, 1], [1, 6, 1], [1, 8, 1]
	]

	x = tf.placeholder(dtype=tf.int8, shape=[4, 3])
	w = tf.Variable([1, 2, 3], dtype=tf.int8)
	b = tf.Variable([4], dtype=tf.float32)
	y = tf.add(x, w)
	shape = tf.shape(x)
	yy = tf.unstack(y, axis=1)
	us = tf.unstack(shape)
	z = tf.nn.softmax(tf.cast(x, tf.float32), dim=1)

	print(x.get_shape())

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)
	result = sess.run(y, feed_dict={x: input_data})
	print(result)
	result = sess.run(tf.reduce_prod(shape), feed_dict={x: input_data})
	print(result)
	result = sess.run(yy, feed_dict={x: input_data})
	print(result)
	result = sess.run(us, feed_dict={x: input_data})
	print(result)
	result = sess.run(z, feed_dict={x: input_data})
	print(result)
	result = sess.run(tf.stack([tf.cast(x, tf.float32), z], axis=0), feed_dict={x: input_data})
	print(result)
	result = sess.run(tf.reduce_sum(tf.concat([tf.cast(x, tf.float32), z], axis=0), axis=1), feed_dict={x: input_data})
	print(result)


	def test():
		for i in yy:
			zz = tf.add(tf.expand_dims(i, axis=1), x)
	'''
