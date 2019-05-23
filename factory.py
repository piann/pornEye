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
    def __init__(self, numOfEnsemble=5):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        self.numOfEnsemble = numOfEnsemble
        self.modelList = []
        for idx in range(self.numOfEnsemble):
            modelML = Model(self.sess, 'pornEye_{}'.format(idx))
            self.modelList.append(modelML)

        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        self.sess.run(init_g)
        self.sess.run(init_l)
        self.batchSize = 25
        self.modelFilePath = './modelFolder/model.ckpt'
        
        logging.info("Factory Init")

    def train(self, iteration, dropoutRate=0.05):
        x_train, y_train, x_test, y_test, sizeOfTraining, sizeOfTest = getDataSet()
        for epoch in range(iteration):
            averageCostList = np.zeros(self.numOfEnsemble)
            totalBatch = int(sizeOfTraining/self.batchSize)

            for idx in range(totalBatch+1):
                
                if (idx+1)*self.batchSize < sizeOfTraining:
                    x_batch = x_train[idx*self.batchSize:(idx+1)*self.batchSize]
                    y_batch = y_train[idx*self.batchSize:(idx+1)*self.batchSize]
                else:
                    x_batch = x_train[sizeOfTraining-self.batchSize:sizeOfTraining]
                    y_batch = y_train[sizeOfTraining-self.batchSize:sizeOfTraining]
            

                for modelIdx, modelML in enumerate(self.modelList):
                    cost, _ = modelML.train(x_batch, y_batch, dropoutRate)
                    logging.debug("batch idx : {}, model idx: {}".format(idx, modelIdx))
                

                averageCostList[modelIdx] += cost / totalBatch

            for modelIdx, modelML in enumerate(self.modelList):
                acc = modelML.getAccuracy(x_test, y_test)
                logging.info("Model #{}, Epoch: {}, accuracy: {:.2f}%, Cost: {}".format(modelIdx, epoch,acc*100,averageCostList[modelIdx]))

        logging.info("Learning Done")


    def saveModel(self):
        saver = tf.train.Saver()
        saver.save(self.sess, self.modelFilePath)
        logging.info("Done")

    def loadModel(self): 
        try: 
            saver = tf.train.Saver()
            saver.restore(self.sess, self.modelFilePath)
            logging.info("Done")
        
        except:
            logging.info("Fail to load model")
    
    def predict(self, filePath):
        packedData = np.ndarray((1, RESIZED_HEIGHT*RESIZED_WIDTH*3))
        imgData = preprocessImg(filePath)
        packedData[0] = imgData
        resultList = [modelML.predict(packedData)[0] for modelML in self.modelList]    
        return np.argmax(np.bincount(resultList))

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
            executer.saveModel()
        elif opt == '-f':
            executer.loadModel()
            folderPath = param
            imgPathList = getImagePathList(folderPath)
            for imgPath in imgPathList:
                result = executer.predict(imgPath)
                logging.info("{0: <20} : {1}".format(os.path.basename(imgPath), result))
            executer.saveModel()


        elif opt == '-t':
            executer.loadModel()
            executer.train(iteration=10)
            executer.saveModel()
        
        else:
            help()
            exit(1)

    