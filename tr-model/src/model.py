from __future__ import division, print_function

import sys, os
import numpy as np
import tensorflow as tf


class Model: 
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, restore=False):
		self.charList = charList
		self.restore = restore
		self.snapID = 0
		self.is_train = tf.placeholder(tf.bool)
		self.inputs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))

		self.buildCNN()
		self.buildRNN()
		self.buildCTC()

		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
		self.sess, self.saver = self.buildTF()

			
	def buildCNN(self):
		kernel1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
		conv1 = tf.nn.conv2d(tf.expand_dims(self.inputs, axis=3), kernel1, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm1 = tf.layers.batch_normalization(conv1, training=self.is_train)
		relu1 = tf.nn.relu(conv_batch_norm1)
		pool1 = tf.nn.max_pool(relu1, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

		kernel2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
		conv2 = tf.nn.conv2d(pool1, kernel2, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm2 = tf.layers.batch_normalization(conv2, training=self.is_train)
		relu2 = tf.nn.relu(conv_batch_norm2)
		pool2 = tf.nn.max_pool(relu2, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID')

		kernel3 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
		conv3 = tf.nn.conv2d(pool2, kernel3, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm3 = tf.layers.batch_normalization(conv3, training=self.is_train)
		relu3 = tf.nn.relu(conv_batch_norm3)
		pool3 = tf.nn.max_pool(relu3, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

		kernel4 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
		conv4 = tf.nn.conv2d(pool3, kernel4, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm4 = tf.layers.batch_normalization(conv4, training=self.is_train)
		relu4 = tf.nn.relu(conv_batch_norm4)
		pool4 = tf.nn.max_pool(relu4, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

		kernel5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
		conv5 = tf.nn.conv2d(pool4, kernel5, padding='SAME', strides=(1,1,1,1))
		conv_batch_norm5 = tf.layers.batch_normalization(conv5, training=self.is_train)
		relu5 = tf.nn.relu(conv_batch_norm5)
		pool5 = tf.nn.max_pool(relu5, (1, 1, 2, 1), (1, 1, 2, 1), 'VALID')

		self.cnnOut4d = pool5


	def buildRNN(self):
		# for 2 layers
		cells = [tf.contrib.rnn.LSTMCell(num_units=256, state_is_tuple=True) for i in range(2)]

		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		# bidirectional RNN
		# BxTxF -> BxTx2H
		# dtype=rnnIn3d.dtype
		(fw, bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=tf.squeeze(self.cnnOut4d, axis=[2]), dtype=self.cnnOut4d.dtype)
									
		# BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
									
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, 256 * 2, len(self.charList) + 1], stddev=0.1))
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def buildCTC(self):
		# BxTxC -> TxBxC
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		# calc loss for batch
		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		# calc loss for each element to compute label probability
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		# decoder: best path decoding
		self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)


	def buildTF(self):
		print('Python: ' + sys.version)
		print('Tensorflow: ' + tf.__version__)

		sess = tf.Session() # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.restore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return sess, saver


	# put ground truth texts into sparse tensor for ctc_loss
	def toSparse(self, texts):
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])

		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return indices, values, shape


	# extract texts from output of CTC decoder
	def decoderOutputToText(self, ctcOutput, batchSize):
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# ctc returns tuple, first element is SparseTensor 
		decoded=ctcOutput[0][0] 

		# go over all indices and save mapping: batch -> values
		idxDict = { b : [] for b in range(batchSize) }
		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0] # index according to [b,t]
			encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		if self.batchesTrained < 10:
			rate = 0.01
		else:
			if self.batchesTrained < 10000:
				rate = 0.001
			else:
				rate = 0.0001
				
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputs : batch.imgs, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		_, lossVal = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1

		return lossVal


	# feed a batch into the NN to recognize the texts
	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):		
		numBatchElements = len(batch.imgs)
		evalRnnOutput = calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)

		return texts, probs
	

	def save(self):
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
