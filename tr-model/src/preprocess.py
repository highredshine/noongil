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
	def __init__(self, filePath, batchSize, imgSize, maxTextLen):
		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []
		self.trainDataRatio = 0.95
	
		chars = set()
		file_list = open(filePath + 'words.txt')
		
		for line in file_list:
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			
			# format filename
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

			c = 0
			gtText = ' '.join(lineSplit[8:])
			for i in range(len(gtText)):
				if (not i) and (gtText[i] == gtText[i-1]):
					c += 2
				else:
					c += 1

				if c > maxTextLen:
					gtText = gtText[:i]

			chars = chars.union(set(list(gtText)))
			self.samples.append(Sample(gtText, fileName))

		splitIdx = int(self.trainDataRatio * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]

		# number of randomly chosen samples / epoch
		self.numTrainSamplesPerEpoch = 25000 
		
		self.trainSet()

		# list of all chars in dataset
		self.charList = sorted(list(chars))

	def trainSet(self):
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	def validationSet(self):
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples

	def getBatchInfo(self):
		return self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize

	def hasNext(self):
		return self.currIdx + self.batchSize <= len(self.samples)
		
	def getNext(self):
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess_data(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)

# transpose and normalize
def preprocess_data(img, imgSize, dataAugmentation=False):
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]]) # black image if image data is damaged

	# randomly stretch images
	if dataAugmentation:
		stretch = random.random() - 0.5 # -0.5 .. +0.5
		wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
		img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
	
	# create target image and copy sample image into it
	wt, ht = imgSize
	h, w = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
	img = cv2.resize(img, newSize)
	target = np.ones([ht, wt]) * 255
	target[0:newSize[1], 0:newSize[0]] = img

	img = cv2.transpose(target)
	mean, stddev = cv2.meanStdDev(img)
	img -= mean[0][0]

	if stddev[0][0] > 0:
		img /= stddev[0][0]

	return img