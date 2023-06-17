from utils import *
from models import *

if __name__ == '__main__':
    dataFile_mask = 'data/mask'
    dataFile_test = 'data/test'
    tempFolder = 'temp'
    modelFile_tmp = 'temp/model_e5'
    confMatImgPath = 'temp/cm.png'
    ''' Generate data file to use. '''
    # fileNames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
    #              'data_batch_5']
    # testNames = ['test_batch']
    # outOrigName = 'data/origin'
    # outMaskName = 'data/mask'
    # outTestName = 'data/test'
    # genData(fileNames, outOrigName)
    # genData(fileNames, outMaskName, doMask=True)
    # genData(testNames, outTestName)
    ''' Do training. '''
    # model = CNN_2L()
    # dataFile = "data/mask"
    # learning_rate, weight_decay = 1e-3, 1e-4
    # optimizer = nn.Adam(model.parameters(), learning_rate, weight_decay)
    # loss_func = nn.CrossEntropyLoss()
    # train_files(model, dataFile, loss_func, optimizer, 40, tempFolder)
    ''' Get performance. '''
    model = CNN_2L()
    testGenData(model, modelFile_tmp, dataFile_test, confMatImgPath)
    pass