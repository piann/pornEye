import random
from common import *
import cv2
import os
import numpy as np
import logging


def preprocessImg(imgPath):

    try:
        imgObj = cv2.imread(imgPath)
        imgObj = cv2.resize(imgObj, (RESIZED_HEIGHT, RESIZED_WIDTH))
        raveledData = imgObj.ravel()

        return raveledData
    except:
        logging.error("Error in handling : {0}")
        return None

def denseToOneHot(y_data, num_classes=2):
    # just imported from other source

    num_labels = y_data.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + y_data.ravel()] = 1
    return labels_one_hot


def getDataSet():
    TEST_RATIO = 0.1

    pornImagePathList = getImagePathList(PORN_TRAINING_DIR)
    nonPornImagePathList = getImagePathList(NON_PORN_TRAINING_DIR)
    
    pornImageObjList = [preprocessImg(p) for p in pornImagePathList]
    nonPornImageObjList = [preprocessImg(p) for p in nonPornImagePathList]
    
    pornImageObjList = [img for img in pornImageObjList if img is not None]
    nonPornImageObjList = [img for img in nonPornImageObjList if img is not None]

    # 1 means porn, 0 means non-porn
    sampleList = [(data, 1) for data in pornImageObjList] + [(data, 0) for data in nonPornImageObjList]
    
    random.shuffle(sampleList)

    trainSet = sampleList[int(TEST_RATIO*len(sampleList)):]
    sizeOfTraining = len(trainSet)
    
    testSet = sampleList[:int(TEST_RATIO*len(sampleList))]
    sizeOfTest = len(testSet)

    logging.debug(sizeOfTraining)
    logging.debug(sizeOfTest)

    x_train = np.array([data[0] for data in trainSet])
    y_train = np.array([data[1] for data in trainSet])
    x_test = np.array([data[0] for data in testSet])
    y_test = np.array([data[1] for data in testSet])

    y_train = denseToOneHot(y_train)
    y_test = denseToOneHot(y_test)

    return x_train, y_train, x_test, y_test, sizeOfTraining, sizeOfTest 


    