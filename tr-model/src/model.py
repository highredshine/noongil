from __future__ import division, print_function

import sys, os
import numpy as np
import tensorflow as tf


class Model: 
	def __init__(self, charList, restore=False):
		self.id = 0
		self.training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.float32, shape=(None, 128, 32)) #image has shape (128,32)
		self.learningRate = tf.placeholder(tf.float32, shape=[])

		self.charList = charList
		self.restore = restore

		self.buildCNN()
		self.buildRNN()
		self.buildCTC()

		self.trainedbatches = 0

		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 

		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		self.session, self.saver = self.buildTF()


	def buildTF(self):
		session = tf.Session()
		saver = tf.train.Saver(max_to_keep=1)
		path = '../model/'
		if self.restore and not tf.train.latest_checkpoint(path):
			raise Exception('Model not saved in: ' + path)

		if tf.train.latest_checkpoint(path):
			saver.restore(session, tf.train.latest_checkpoint(path))
		else:
			session.run(tf.global_variables_initializer())

		return session, saver


	def sparseTensor(self, texts):
		idxs = []
		vals = []
		shape = 0

		for (batchElement, text) in enumerate(texts):
			labels = []
			for c in text:
				labels.append(self.charList.index(c))
			if len(labels) > shape:
				shape = len(labels)
			for (idx, label) in enumerate(labels):
				idxs.append([batchElement, idx])
				vals.append(label)

		return idxs, vals, (len(texts), shape)


	def save(self):
		self.id += 1
		self.saver.save(self.session, '../model/interm', global_step=self.id)


	def decoderOutputToText(self, ctcOutput, batchSize):
		encoded = []
		decoded = ctcOutput[0][0] 
		for i in range(batchSize):
			encoded.append([])
		for (idx, idx2) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2[0]
			encoded[batchElement].append(label)

		outputs = []
		for labels in encoded:
			output = []
			for label in labels:
				output.append(self.charList[label])
			outputs.append(str().join(output))

		return outputs


	def buildCNN(self):
		kernel1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
		conv1 = tf.nn.conv2d(tf.expand_dims(self.inputs, axis=3), kernel1, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm1 = tf.layers.batch_normalization(conv1, training=self.training)
		relu1 = tf.nn.relu(conv_batch_norm1)
		pool1 = tf.nn.max_pool(relu1, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

		kernel2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
		conv2 = tf.nn.conv2d(pool1, kernel2, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm2 = tf.layers.batch_normalization(conv2, training=self.training)
		relu2 = tf.nn.relu(conv_batch_norm2)
		pool2 = tf.nn.max_pool(relu2, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

		kernel3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
		conv3 = tf.nn.conv2d(pool2, kernel3, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm3 = tf.layers.batch_normalization(conv3, training=self.training)
		relu3 = tf.nn.relu(conv_batch_norm3)
		pool3 = tf.nn.max_pool(relu3, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

		kernel4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
		conv4 = tf.nn.conv2d(pool3, kernel4, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm4 = tf.layers.batch_normalization(conv4, training=self.training)
		relu4 = tf.nn.relu(conv_batch_norm4)
		pool4 = tf.nn.max_pool(relu4, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

		kernel5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
		conv5 = tf.nn.conv2d(pool4, kernel5, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm5 = tf.layers.batch_normalization(conv5, training=self.training)
		relu5 = tf.nn.relu(conv_batch_norm5)
		pool5 = tf.nn.max_pool(relu5, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

		self.cnnOutput = pool5


	def buildRNN(self):
		# for 2 layers
		stacked = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(num_units=256, state_is_tuple=True), tf.contrib.rnn.LSTMCell(num_units=256, state_is_tuple=True)], state_is_tuple=True)
		(fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=tf.squeeze(self.cnnOutput, axis=[2]), dtype=self.cnnOutput.dtype)
		self.rnnOutput = tf.squeeze(tf.nn.atrous_conv2d(value=tf.expand_dims(tf.concat([fw, bw], 2), 2), filters=tf.Variable(tf.truncated_normal([1, 1, 256 * 2, len(self.charList) + 1], stddev=0.1)), rate=1, padding='SAME'), axis=[2])
		

	def buildCTC(self):
		self.ctcRes = tf.transpose(self.rnnOutput, [1, 0, 2])
		# ground truth text -> sparse tensor
		self.texts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.texts, inputs=self.ctcRes, sequence_length=self.seqLen, ctc_merge_repeated=True))

		self.savedCtcInput = tf.placeholder(tf.float32, shape=[32, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.texts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)
		self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcRes, sequence_length=self.seqLen)


	def trainBatch(self, batch):
		numBatchElements = len(batch.imgs)
		sparse = self.sparseTensor(batch.texts)
		if self.trainedbatches < 10:
			rate = 0.01
		elif self.trainedbatches < 10000:
			rate = 0.001
		else:
			rate = 0.0001
				
		dictionary = {self.inputs : batch.imgs, self.texts : sparse, self.seqLen : [32] * numBatchElements, self.learningRate : rate, self.training: True}
		_, lossVal = self.session.run([self.optimizer, self.loss], dictionary)

		self.trainedbatches += 1

		return lossVal


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		if calcProbability:
			a = [self.ctcRes]
		else:
			a = []
		
		dict1 = {self.inputs : batch.imgs, self.seqLen : [32] * numBatchElements, self.training: False}
		out = self.session.run([self.decoder] + a, dict1)
		texts = self.decoderOutputToText(out[0], len(batch.imgs))
		
		probs = None
		if calcProbability:
			if probabilityOfGT:
				sparse = self.sparseTensor(batch.texts)
			else:
				sparse = self.sparseTensor(texts)
			dictionary = {self.savedCtcInput : out[1], self.texts : sparse, self.seqLen : [32] * numBatchElements, self.training: False}
			
		return texts, np.exp(-1 * self.session.run(self.lossPerElement, dictionary))
 
