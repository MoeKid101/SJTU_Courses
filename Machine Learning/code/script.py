import load_data
import torch
from svm import *

def mnist_test(trainDataPath:str, testDataPath:str):
    ''' define kernels and parameters '''
    lin_kernel = {'type': 'linear', 'mtp':0.03}
    rbf_kernel = {'type': 'rbf', 'gamma':0.04, 'mtp':1.0}
    poly_kernel = {'type': 'poly', 'c':10, 'd':1.5}
    mix_kernel = {'type':'mix_mtp', 'c':0, 'd':1.0, 'gamma':0.022}
    ''' load data '''
    trainData, trainLabel = torch.load(trainDataPath)
    testData, testLabel = torch.load(testDataPath)
    trainData = trainData.reshape((-1, 28*28)).astype(np.float32)
    testData = testData.reshape((-1, 28*28)).astype(np.float32)
    ''' perform test '''
    svm_model = SVM_OAO(max_iter=10, kernel=lin_kernel)
    svm_model.fit(trainData, trainLabel)
    predLabel = svm_model.predict(testData)
    print(f'err rate={val(predLabel, testLabel)}')

def cifar_test(trainDataPath:str, testDataPath:str):
    ''' define kernels and parameters '''
    lin_kernel = {'type': 'linear', 'mtp':100}
    rbf_kernel = {'type': 'rbf', 'gamma':0.01, 'mtp':1.0}
    poly_kernel = {'type': 'poly', 'c':10, 'd':2.5}
    mix_kernel = {'type':'mix_sum', 'c':10, 'd':2.5, 'gamma':0.01,
                  'mtp_lin':0, 'mtp_rbf':200, 'mtp_poly':2e-6}
    ''' load data '''
    trainData, trainLabel = torch.load(trainDataPath)
    testData, testLabel = torch.load(testDataPath)
    trainData = trainData.reshape((-1, 3*32*32)).astype(np.float32)
    testData = testData.reshape((-1, 3*32*32)).astype(np.float32)
    ''' perform test '''
    svm_model = SVM_OAO(max_iter=10, kernel=mix_kernel)
    svm_model.fit(trainData, trainLabel)
    predLabel = svm_model.predict(testData)
    print(f'err rate={val(predLabel, testLabel)}')

if __name__ == '__main__':
    ''' Load data (download from Internet). '''
    load_data.genMNIST()
    load_data.genCIFAR()
    load_data.genSubData('data/cifar_train', 'data/cifar_train_sub')
    load_data.genSubData('data/mnist_train', 'data/mnist_train_sub')
    ''' Do test on MNIST and CIFAR. '''
    mnist_test('data/mnist_train_sub', 'data/mnist_test')
    cifar_test('data/cifar_train_sub', 'data/cifar_test')
    pass