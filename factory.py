# handler for all th other object

from model import Model
from handleData import *
from common import *
import numpy as np
import tensorflow as tf


class Factory(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.modelML = Model(sess, 'pornEye')
        self.totalEpoch = 3000
        self.batchSize = 100
        self.modelFilePath = 'pornEye.model'
        
        logging.info("Factory Init")

    def train(self, iteration, dropoutRate=0.25):
        x_train, y_train, x_test, y_test, sizeOfTraining, sizeOfTest = getDataSet()
        for epoch in range(totalEpoch):
            averageCost = 0
            totalBatch = int(sizeOfTraining/batchSize)

            for idx in range(totalBatch):
                x_batch = x_train[idx*self.batchSize:(idx+1)*self.batchSize]
                y_batch = y_train[idx*self.batchSize:(idx+1)*self.batchSize]
                cost, _ = self.modelML.train(x_batch, y_batch)
                averageCost += cost / totalBatch
            
            logging.info("Epoch: {}, Cost: {}".format(epoch, averageCost))

        logging.info("Learning Done")


    def saveModel(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.modelFilePath)
        logging.info("Done")

    def loadModel(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, self.modelFilePath)
        logging.info("Done")
    
    def predict(self, filePath):
        packedData = np.ndarray(1, RESIZED_HEIGHT*RESIZED_WIDTH*3)
        imgData = preprocessImg(filePath)
        packedData[0] = imgData
        self.modelML.getAccuracy(packedData)
