import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def genMNIST(root_folder:str='data/'):
    '''
    Download MNIST dataset from Internet and store as (dataArr, labelArr)
    with dataArr.shape=[-1, 28, 28] and labelArr.shape=[-1,]
    '''
    trainSet = datasets.MNIST(root='data/', train=True,
                              transform=transforms.ToTensor(),
                              download=True)
    testSet = datasets.MNIST(root='data/', train=False,
                             transform=transforms.ToTensor(),
                             download=True)
    
    trainDataList, trainLabelList = list(), list()
    train_loader = DataLoader(trainSet, batch_size=128, shuffle=False)
    for data, label in iter(train_loader):
        data = data.reshape((-1, 28, 28))
        trainDataList.append(data.numpy()), trainLabelList.append(label.numpy())
    trainDataArr, trainLabelArr = (np.concatenate(trainDataList, axis=0),
                                   np.concatenate(trainLabelList, axis=0))
    torch.save((trainDataArr, trainLabelArr), f'{root_folder}/mnist_train')

    testDataList, testLabelList = list(), list()
    test_loader = DataLoader(testSet, batch_size=128, shuffle=False)
    for data, label in iter(test_loader):
        data = data.reshape((-1, 28, 28))
        testDataList.append(data.numpy()), testLabelList.append(label.numpy())
    testDataArr, testLabelArr = (np.concatenate(testDataList, axis=0),
                                 np.concatenate(testLabelList, axis=0))
    torch.save((testDataArr, testLabelArr), f'{root_folder}/mnist_test')

def genCIFAR(root_folder:str='data/'):
    trainSet = datasets.CIFAR10(root='data/', train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    testSet = datasets.CIFAR10(root='data/', train=False,
                               transform=transforms.ToTensor(),
                               download=True)
    
    trainDataList, trainLabelList = list(), list()
    train_loader = DataLoader(trainSet, batch_size=128, shuffle=False)
    for data, label in iter(train_loader):
        data = data.reshape((-1, 3, 32, 32))
        trainDataList.append(data.numpy()), trainLabelList.append(label.numpy())
    trainDataArr, trainLabelArr = (np.concatenate(trainDataList, axis=0),
                                   np.concatenate(trainLabelList, axis=0))
    torch.save((trainDataArr, trainLabelArr), f'{root_folder}/cifar_train')

    testDataList, testLabelList = list(), list()
    test_loader = DataLoader(testSet, batch_size=128, shuffle=False)
    for data, label in iter(test_loader):
        data = data.reshape((-1, 3, 32, 32))
        testDataList.append(data.numpy()), testLabelList.append(label.numpy())
    testDataArr, testLabelArr = (np.concatenate(testDataList, axis=0),
                                 np.concatenate(testLabelList, axis=0))
    torch.save((testDataArr, testLabelArr), f'{root_folder}/cifar_test')

def genSubData(origData_path:str, target_path:str, subClass_size:int=500):
    import torch
    train_data, train_label = torch.load(origData_path)
    shuffle_idx = np.arange(train_data.shape[0])
    import numpy.random as rdm
    rdm.shuffle(shuffle_idx)
    train_data, train_label = train_data[shuffle_idx], train_label[shuffle_idx]
    subTrain_data, subTrain_label = list(), list()
    num_classes = 10
    for class_idx in range(num_classes):
        class_data = train_data[train_label == class_idx]
        subTrain_data.append(class_data[:subClass_size])
        subTrain_label.append(np.ones(subClass_size, dtype=np.int32)*class_idx)
    subTrain_data, subTrain_label = np.concatenate(subTrain_data, axis=0), np.concatenate(subTrain_label, axis=0)
    torch.save((subTrain_data, subTrain_label), target_path)