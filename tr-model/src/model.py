from __future__ import division, print_function

import sys, os
import numpy as np
import tensorflow as tf


class Model: 
	def __init__(self, charList, restore=False):
		self.id = 0
		self.training = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.float32, shape=(None, 128, 32)) #image has shape (128,32)

		self.charList = charList
		self.restore = restore

		self.buildCNN()
		self.buildRNN()
		self.buildCTC()

		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
		self.sess, self.saver = self.buildTF()


	def buildTF(self):
		session = tf.Session()
		saver = tf.train.Saver(max_to_keep=1)
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir)

		if self.restore and not latestSnapshot:
			raise Exception('Model not saved in: ' + modelDir)

		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(session, latestSnapshot)
		else:
			print('Init with new values')
			session.run(tf.global_variables_initializer())

		return session, saver


	def sparseTensor(self, texts):
		indices = []
		values = []
		shape = [len(texts), 0]

		for (batchElement, text) in enumerate(texts):
			labelStr = [self.charList.index(c) for c in text]
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return indices, values, shape


	def save(self):
		self.id += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.id)


	def decoderOutputToText(self, ctcOutput, batchSize):
		encodedLabelStrs = [[] for i in range(batchSize)]
		decoded = ctcOutput[0][0] 
		idxDict = { b : [] for b in range(batchSize) }

		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0]
			encodedLabelStrs[batchElement].append(label)

		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


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

		self.cnnOut4d = pool5


	def buildRNN(self):
		# for 2 layers
		cells = [tf.contrib.rnn.LSTMCell(num_units=256, state_is_tuple=True), tf.contrib.rnn.LSTMCell(num_units=256, state_is_tuple=True)]
		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		(fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=tf.squeeze(self.cnnOut4d, axis=[2]), dtype=self.cnnOut4d.dtype)
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
		kernel = tf.Variable(tf.truncated_normal([1, 1, 256 * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def buildCTC(self):
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
		# ground truth text -> sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		self.savedCtcInput = tf.placeholder(tf.float32, shape=[32, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)
		self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)


	def trainBatch(self, batch):
		numBatchElements = len(batch.imgs)
		sparse = self.sparseTensor(batch.gtTexts)
		if self.batchesTrained < 10:
			rate = 0.01
		elif self.batchesTrained < 10000:
			rate = 0.001
		else:
			rate = 0.0001
				
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputs : batch.imgs, self.gtTexts : sparse, self.seqLen : [32] * numBatchElements, self.learningRate : rate, self.training: True}
		_, lossVal = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1

		return lossVal


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):		
		numBatchElements = len(batch.imgs)
		evalRnnOutput = calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputs : batch.imgs, self.seqLen : [32] * numBatchElements, self.training: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		probs = None
		if calcProbability:
			sparse = self.sparseTensor(batch.gtTexts) if probabilityOfGT else self.sparseTensor(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [32] * numBatchElements, self.training: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)

		return texts, probs
 
