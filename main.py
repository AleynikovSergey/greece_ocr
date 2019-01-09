from __future__ import division
from __future__ import print_function

import sys
import io
import argparse
import codecs
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model
from SamplePreprocessor import preprocess


class FilePaths:
	"filenames and paths to data"
	fnCharList = 'model/charList.txt'
	fnTrain = 'data/'
	fnCorpus = 'data/corpus.txt'


def train(model, loader):
	epoch = 0
	bestCharErrorRate = float('inf')
	noImprovementSince = 0
	earlyStopping = 5
	while True:
		epoch += 1
		print('Epoch:', epoch)

		
		print('Train NN')
		loader.trainSet()
		while loader.hasNext():
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss:', loss)
		
		charErrorRate = validate(model, loader)
		
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

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
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
	
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate


def infer(model, fnImg):
	"""Распознает текс на изображении по пути fnImg"""
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:' + recognized[0] + '; Probability:' + str(probability[0]))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", help="train the NN", action="store_true")
	parser.add_argument("--validate", help="validate the NN", action="store_true")
	args = parser.parse_args()

	
	if args.train or args.validate:
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)

		with io.open(FilePaths.fnCharList, "w", encoding="utf-8") as f:
			f.write(str().join(loader.charList))
	
		with io.open(FilePaths.fnCorpus, "w", encoding="utf-8") as f:
			f.write(str(' ').join(loader.trainWords + loader.validationWords))

		if args.train:
			model = Model(loader.charList)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, mustRestore=True)
			validate(model, loader)


if __name__ == '__main__':
	main()

