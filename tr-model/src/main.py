from __future__ import division, print_function
from preprocess import LoadData, preprocess_data
from model import Model
import cv2, sys, argparse, editdistance


def train(model, processed_data):
	"""
	This function trains our neural network
	PARAMETERS
	model: class that represents our model
	processed_data: class that represents the preprocessed data
	RETURNS: none
	"""
	
	epoch = 0 
	min_error = float('inf') 

	while epoch <= 50:
		processed_data.trainSet()
		while processed_data.hasNext():
			batch = processed_data.getNext()
			loss = model.trainBatch(batch)
			print("Loss:", loss)

		curr_error = test(model, processed_data)
		if curr_error < min_error:
			min_error = curr_error
			model.save()
			open('../model/accuracy.txt', 'w').write('Testing error: %f%%' % (curr_error*100.0))
		epoch += 1

def test(model, processed_data):
	"""
	This function tests our neural network
	PARAMETERS
	model: class that represents our model
	processed_data: class that represents the preprocessed data
	RETURNS: error rate
	"""

	processed_data.validationSet()

	curr_words = 0
	total_words = 0
	curr_chars = 0
	total_chars = 0
	
	while processed_data.hasNext():
		batch = processed_data.getNext()
		texts = batch.gtTexts
		recognized, _ = model.inferBatch(batch)

		for i in range(len(recognized)):
			if texts[i] == recognized[i]:
				curr_words += 1
			else:
				curr_words += 0
			
			curr_chars += editdistance.eval(recognized[i], texts[i])
			total_chars += len(texts[i])
			total_words += 1
	
	return curr_chars / total_chars


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--test', action='store_true')
	a = parser.parse_args()

	processed_data = LoadData('../data/', Model.batchSize, Model.imgSize, Model.maxTextLen)
	open('../model/charList.txt', 'w').write(str().join(processed_data.charList))
	open('../data/corpus.txt', 'w').write(str(' ').join(processed_data.trainWords + processed_data.validationWords))
	if a.train:
		model = Model(processed_data.charList)
		train(model, processed_data)
	elif a.test:
		model = Model(processed_data.charList, restore=True)
		test(model, processed_data)

if __name__ == '__main__':
	main()

