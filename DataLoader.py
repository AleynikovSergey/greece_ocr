from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess
import codecs


class Sample:
	def __init__(self, gtText, filePath):
		self.gtText = gtText
		self.filePath = filePath


class Batch:
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"""Загружает данные для тренировки"""

	def __init__(self, filePath, batchSize, imgSize, maxTextLen):		
		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []
	
		f = codecs.open(filePath+'words.txt', encoding='utf-8')
		chars = set()
		clt = 0
		for line in f:
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split('\r')
			
			fileName = filePath + 'words/' + str(clt) + '.png'
			clt += 1

			gtText = lineSplit[0]
			chars = chars.union(set(list(gtText)))

			self.samples.append(Sample(gtText, fileName))

		splitIdx = int(0.95 * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]

		self.numTrainSamplesPerEpoch = 25000 
		
		self.trainSet()

		self.charList = sorted(list(chars))


	def truncateLabel(self, text, maxTextLen):
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text


	def trainSet(self):
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	
	def validationSet(self):
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples


	def getIteratorInfo(self):
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


	def hasNext(self):
		return self.currIdx + self.batchSize <= len(self.samples)
		
		
	def getNext(self):
		batchRange = range(self.currIdx, self.currIdx + self.batchSize)
		gtTexts = [self.samples[i].gtText for i in batchRange]
		imgs = [preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation) for i in batchRange]
		self.currIdx += self.batchSize
		return Batch(gtTexts, imgs)


