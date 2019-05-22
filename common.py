# methods for general use

import logging


PORN_TRAINING_DIR = "pornTrainingDirectory"
NON_PORN_TRAINING_DIR = "nonPornTrainingDirectory"
TEST_DIR = "testDriectory"
RESIZED_WIDTH = 256
RESIZED_HEIGHT = 256
TRAINING_EPOCH = 3000



def setupLogging(fileName):
    # setup log file and log depth 

    fileHandler = logging.FileHandler(fileName)
    fileHandler.setLevel(logging.DEBUG)
    fileHandler.setFormatter( logging.Formatter('%(asctime)s:%(levelname)s:[%(filename)s.%(funcName)s]%(message)s', '%m-%d %H:%M:%S'))

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    streamHandler.setFormatter( logging.Formatter('%(asctime)s:%(levelname)s:[%(filename)s.%(funcName)s]%(message)s', '%m-%d %H:%M:%S'))

    logger = logging.getLogger('')

    logger.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

def getImagePathList(dirPath, targetExt=["jpg","jpeg","png"]):
    # get File Path List in "dirPath"

    filePathList = []
    filenames = os.listdir(dirPath)
    for filename in filenames:
        fileExt = filename.split('.')[-1]
        if not (fileExt.lower() in targetExt):
            continue
        fullPath = os.path.join(dirPath,filename)
        filePathList.append(fullPath)
    
    return filePathList