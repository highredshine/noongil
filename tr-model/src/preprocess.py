from __future__ import division, print_function

import cv2, random, os
import numpy as np


class LoadData:
	def __init__(self, filePath, batchSize, imageSize):
		self.augmentData = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imageSize = imageSize
		self.samples = []
		self.trainDataRatio = 0.95
		self.maxTextLength = 32
	
		chars = set()
		file_list = open(filePath + 'words.txt')
		
		for line in file_list:
			if not line or line[0]=='#':
				continue
			
			# format filename
			lineSplit = line.strip().split(' ')
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

			c = 0
			ground_truth_text = ' '.join(lineSplit[8:])
			for i in range(len(ground_truth_text)):
				if (not i) and (ground_truth_text[i] == ground_truth_text[i-1]):
					c += 2
				else:
					c += 1

				if c > self.maxTextLength:
					ground_truth_text = ground_truth_text[:i]

			chars = chars.union(set(list(ground_truth_text)))
			self.samples.append(Sample(ground_truth_text, fileName))

		splitIdx = int(self.trainDataRatio * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		self.trainWords = []
		self.validationWords = []
		for train_samp in self.trainSamples:
			self.trainWords.append(train_samp.ground_truth_text)
		for validation_samp in self.validationSamples:
			self.trainWords.append(validation_samp.ground_truth_text)

		self.numTrainSamplesPerEpoch = 25000 
		self.trainSet()
		chars = list(chars)
		self.charList = sorted(chars)

	def trainSet(self):
		self.augmentData = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	def validationSet(self):
		self.augmentData = False
		self.currIdx = 0
		self.samples = self.validationSamples

	def getBatchInfo(self):
		return self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize
		
	def getNext(self):
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		texts = [self.samples[i].ground_truth_text for i in batchRange]
		imgs = [preprocess_data(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imageSize, self.augmentData) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(texts, imgs)

	def hasNext(self):
		return self.currIdx + self.batchSize <= len(self.samples)

# transpose and normalize
def preprocess_data(img, imageSize, augmentData=False):
	if img is None:
		img = np.zeros([imageSize[1], imageSize[0]]) # black image if image data is damaged

	if augmentData:
		stretch = random.random() - 0.5
		stretched = max(int(32 * (1 + stretch)), 1)
		img = cv2.resize(img, (stretched, 128))
	
	size1, size2 = imageSize
	h, w = img.shape
	res1 = w / size1
	res2 = h / size2
	bigger = max(res2, res1)
	reSize1 = max(min(size1, int(w / bigger)), 1)
	reSize2 = max(min(size2, int(h / bigger)), 1)
	target = np.ones([size2, size1]) * 255
	target[0:reSize2, 0:reSize1] = cv2.resize(img, (reSize1, reSize2))
	img = cv2.transpose(target)
	mean, stddev = cv2.meanStdDev(img)
	img -= mean[0][0]

	if stddev[0][0] > 0:
		img /= stddev[0][0]

	return img

# Wrapper classes for convenience
class Batch:
	def __init__(self, texts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.texts = texts

class Sample:
	def __init__(self, ground_truth_text, filePath):
		self.ground_truth_text = ground_truth_text
		self.filePath = filePath