from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

class Model: 
 
    batchSize = 50
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, mustRestore=False):
        """Инициализация модели"""
        self.charList = charList
        self.mustRestore = mustRestore
        self.snapID = 0
            
        self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
    
        # инициализация CNN, RNN и CTC
        self.setupCNN()
        self.setupRNN()
        self.setupCTC()

        self.batchesTrained = 0
        self.learningRate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)
    
        (self.sess, self.saver) = self.setupTF()

            
    def setupCNN(self):
        cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
    
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
        numLayers = len(strideVals)

        pool = cnnIn4d
        for i in range(numLayers):
            kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i],
                                                      featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')

        self.cnnOut4d = pool


    def setupRNN(self):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
    
        # basic cells which is used to build RNN
        numHidden = 256
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)
                                    
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
                                    
        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
        

    def setupCTC(self):
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) ,
                                       tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        self.seqLen = tf.placeholder(tf.int32, [None])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC,
                                                  sequence_length=self.seqLen, ctc_merge_repeated=True))

        self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
        self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput,
                                             sequence_length=self.seqLen, ctc_merge_repeated=True)

        self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        

    def setupTF(self):
        sess=tf.Session()

        saver = tf.train.Saver(max_to_keep=1)
        modelDir = 'model/'
        latestSnapshot = tf.train.latest_checkpoint(modelDir)

        if self.mustRestore and not latestSnapshot:
            raise Exception('No saved model found in: ' + modelDir)

        if latestSnapshot:
            print('Init with stored values from ' + latestSnapshot)
            saver.restore(sess, latestSnapshot)
        else:
            print('Init with new values')
            sess.run(tf.global_variables_initializer())

        return (sess,saver)


    def toSparse(self, texts):
        """Преобразование метки (слова) объекта в sparse tensor"""
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

        return (indices, values, shape)


    def decoderOutputToText(self, ctcOutput, batchSize):
        
        encodedLabelStrs = [[] for i in range(batchSize)]
        
        decoded=ctcOutput[0][0] 

        idxDict = { b : [] for b in range(batchSize) }
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0]
            encodedLabelStrs[batchElement].append(label)

        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def trainBatch(self, batch):
        numBatchElements = len(batch.imgs)
        sparse = self.toSparse(batch.gtTexts)
        rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001) # decay learning rate
        evalList = [self.optimizer, self.loss]
        feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate}
        (_, lossVal) = self.sess.run(evalList, feedDict)
        self.batchesTrained += 1
        return lossVal

    def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
        """Получает batch изображений, на которых необходимо распознать текст"""
        
        numBatchElements = len(batch.imgs)
        evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability else [])
        feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements}
        evalRes = self.sess.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoderOutputToText(decoded, numBatchElements)
        
        probs = None
        if calcProbability:
            sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
            ctcInput = evalRes[1]
            evalList = self.lossPerElement
            feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements}
            lossVals = self.sess.run(evalList, feedDict)
            probs = np.exp(-lossVals)
        return (texts, probs)
    

    def save(self):
        "save model to file"
        self.snapID += 1
        self.saver.save(self.sess, 'model/snapshot', global_step=self.snapID)



