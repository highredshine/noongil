from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from preprocess_data import Load, preprocess
from Model import Model


def train(model, loader):
	"train NN"
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement

	# Accuracy stops increasing after 25 epochs
	# while True
	while True:
		epoch += 1
		print('Epoch:', epoch)

		# train
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getBatch()
			gtTexts, imgs = loader.getNext()
			loss = model.trainBatch(gtTexts, imgs)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)

		# validate
		charErrorRate = validate(model, loader)
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open('../model/accuracy.txt', 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getBatch()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		gtTexts, imgs = loader.getNext()
		(recognized, _) = model.inferBatch(imgs)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], gtTexts[i])
			numCharErr += dist
			numCharTotal += len(gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


def main():
	"main function"
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--validate', action='store_true')
	# parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	# parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	# dump output of NN to CSV file(s)'
	parser.add_argument('--dump', action='store_true')

	args = parser.parse_args()

	# decoderType = DecoderType.BestPath
	# if args.beamsearch:
	# 	decoderType = DecoderType.BeamSearch
	# elif args.wordbeamsearch:
	# 	decoderType = DecoderType.WordBeamSearch

	# train or validate on IAM dataset	
	if args.train or args.validate:
		# load training data, create TF model
		loader = Load('../data/', Model.batchSize, Model.imgSize, Model.maxTextLen)

		# save characters of model for inference mode
		open('../model/charList.txt', 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open('../data/corpus.txt', 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, mustRestore=True)
			validate(model, loader)

	# infer text on test image
	else:
		print(open('../model/accuracy.txt').read())
		model = Model(open('../model/charList.txt').read(), mustRestore=True, dump=args.dump)

		"recognize text in image provided by file path"
		img = preprocess(cv2.imread('../data/test.png', cv2.IMREAD_GRAYSCALE), Model.imgSize)
		# batch = Batch(None, [img])
		imgs = np.stack([img], axis=0)
		(recognized, probability) = model.inferBatch(imgs, True)
		print('Recognized:', '"' + recognized[0] + '"')
		print('Probability:', probability[0])


if __name__ == '__main__':
	main()

