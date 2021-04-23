from __future__ import division, print_function

import cv2, random, os
import numpy as np


class Sample:
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


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
			gtText = ' '.join(lineSplit[8:])
			for i in range(len(gtText)):
				if (not i) and (gtText[i] == gtText[i-1]):
					c += 2
				else:
					c += 1

				if c > self.maxTextLength:
					gtText = gtText[:i]

			chars = chars.union(set(list(gtText)))
			self.samples.append(Sample(gtText, fileName))

		splitIdx = int(self.trainDataRatio * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		self.trainWords = []
		self.validationWords = []
		for train_samp in self.trainSamples:
			self.trainWords.append(train_samp.gtText)
		for validation_samp in self.validationSamples:
			self.trainWords.append(validation_samp.gtText)

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
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess_data(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imageSize, self.augmentData) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)

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
	img = cv2.resize(img, (reSize1, reSize2))
	target = np.ones([size2, size1]) * 255
	target[0:reSize2, 0:reSize1] = img
	img = cv2.transpose(target)
	mean, stddev = cv2.meanStdDev(img)
	img -= mean[0][0]

	if stddev[0][0] > 0:
		img /= stddev[0][0]

	return img