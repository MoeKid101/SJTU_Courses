from utils import *
from models import *

def ROS(dataArr:np.ndarray, labelArr:np.ndarray, rho:float=1.0):
    from numpy import random
    # check labelArr and calc rho.
    itemCnt = np.ndarray(10, dtype=np.int32)
    for label in range(10):
        itemCnt[label] = np.sum(labelArr == label)
    reqMinCnt = int(rho * np.max(itemCnt))
    dataItems, labelItems = list(), list()
    for label in range(10):
        localItems = dataArr[labelArr == label]
        if itemCnt[label] >= reqMinCnt:
            localLabels = labelArr[labelArr == label]
            dataItems.append(localItems), labelItems.append(localLabels)
            continue
        reqNum = reqMinCnt - itemCnt[label] # get required number of items.
        newData = np.ndarray([reqMinCnt, 3, 32, 32], dtype=dataArr.dtype)
        newLabel = np.ones([reqMinCnt], dtype=labelArr.dtype) * label
        newData[0:localItems.shape[0]], newDataPtr = localItems, localItems.shape[0]
        addedIdxes = random.randint(0, localItems.shape[0], size=reqNum)
        for itr in range(addedIdxes.shape[0]):
            newData[newDataPtr] = localItems[addedIdxes[itr]]
            newDataPtr += 1
        dataItems.append(newData), labelItems.append(newLabel)
    dataROS, labelROS = np.concatenate(dataItems, axis=0), np.concatenate(labelItems, axis=0)
    return dataROS, labelROS

if __name__ == '__main__':
    dataFile_ros = 'data/ros1.0'
    ''' Do ROS on masked dataset. '''
    # dataArr, labelArr = jt.load('data/mask')
    # dataROS, labelROS = ROS(dataArr, labelArr)
    # jt.save((dataROS, labelROS), dataFile_ros)
    ''' Do training. '''
    # model = CNN_3L_Dropout_2()
    # optimizer = nn.Adam(model.parameters(), 1e-3, 1e-4)
    # loss_func = nn.CrossEntropyLoss()
    # train_files(model, dataFile_ros, loss_func, optimizer, 40, 'temp')
    ''' Get performance. '''
    modelFile = 'temp/model_e5'
    dataFile_test = 'data/test'
    confMatImgPath = 'temp/cm.png'
    model = CNN_3L_Dropout_2()
    testGenData(model, modelFile, dataFile_test, confMatImgPath)