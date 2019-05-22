# handler for all th other object

from model import Model
from handleData import *
from common import *
import numpy as np
import tensorflow as tf
import getopt
import os
import sys

class Factory(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.modelML = Model(self.sess, 'pornEye')
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init_g)
        self.sess.run(init_l)
        self.batchSize = 30
        self.modelFilePath = './pornEye.model'
        
        logging.info("Factory Init")

    def train(self, iteration, dropoutRate=0.25):
        x_train, y_train, x_test, y_test, sizeOfTraining, sizeOfTest = getDataSet()
        for epoch in range(iteration):
            averageCost = 0
            totalBatch = int(sizeOfTraining/self.batchSize)

            for idx in range(totalBatch):
                x_batch = x_train[idx*self.batchSize:(idx+1)*self.batchSize]
                y_batch = y_train[idx*self.batchSize:(idx+1)*self.batchSize]
                
                cost, _ = self.modelML.train(x_batch, y_batch, 0.1)
                logging.debug("batch idx : {0}".format(idx))
                

                averageCost += cost / totalBatch

            acc = self.modelML.getAccuracy(x_test, y_test)
            logging.info("Epoch: {}, accuracy: {}, Cost: {}".format(epoch, averageCost, acc))

        logging.info("Learning Done")


    def saveModel(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.modelFilePath)
        logging.info("Done")

    def loadModel(self):
        if os.path.exists(self.modelFilePath): 
            saver = tf.train.Saver()
            saver.restore(self.sess, self.modelFilePath)
            logging.info("Done")
        else:
            logging.info("There is no model. This is first try")
    
    def predict(self, filePath):
        packedData = np.ndarray((1, RESIZED_HEIGHT*RESIZED_WIDTH*3))
        imgData = preprocessImg(filePath)
        packedData[0] = imgData
        return self.modelML.predict(packedData)

def help():
    print("Help :")
    print("-t             train by specified folder")
    print("-c=filePath    check the image file is porn or not")
    print("-f=folerPath   check the image files in folder")


if __name__ == '__main__':
    setupLogging("./logfile")
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'c:f:th')
    except getopt.GetoptError as err:
        logging.error(err)
        sys.exit(1)

    executer = Factory()
    for opt, param in opts:
        if opt == '-h':
            help()
            exit(1)
        elif opt == '-c':
            filePath = param
            executer.loadModel()
            result = executer.predict(filePath)
            logging.info("result is {}".format(result))
        elif opt == '-f':
            executer.loadModel()
            folderPath = param
            imgPathList = getImagePathList(folderPath)
            for imgPath in imgPathList:
                result = executer.predict(imgPath)
                logging.info("{0: <20} : {1}".format(os.path.basename(imgPath), result))
                


        elif opt == '-t':
            executer.loadModel()
            executer.train(iteration=4)
            executer.saveModel()
        
        else:
            help()
            exit(1)

    