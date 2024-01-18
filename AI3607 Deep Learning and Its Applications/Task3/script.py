from models import *
from utils import *

if __name__ == '__main__':
    ''' Load data. '''
    load_data('data/train', 'data/test')
    genMaskData('data/train', 'data/mask')

    maskData, maskLabel = jt.load('data/mask')
    maskAugData, maskAugLabel = augment(maskData, maskLabel)
    jt.save((maskAugData, maskAugLabel), 'data/mask_aug')

    rosAugData, rosAugLabel = ROS(maskData, maskLabel)
    rosAugData, rosAugLabel = augment(rosAugData, rosAugLabel)
    jt.save((rosAugData, rosAugLabel), 'data/ros1_aug')
    ''' Perform test using CNN with ROS. '''
    cnn_model = CNN()
    trainDataPath = 'data/ros1_aug'
    testDataPath = 'data/test'
    optimizer = nn.SGD(cnn_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    train_batch(cnn_model, trainDataPath, optimizer, 'temp')
    val(cnn_model, 'temp/model_e30', testDataPath, '')
    val_d(cnn_model, 'temp/model_e30', testDataPath)
    ''' Perform test using CNN with weight. '''
    cnn_model = CNN()
    trainDataPath = 'data/mask_aug'
    testDataPath = 'data/test'
    optimizer = nn.SGD(cnn_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    train_batch(cnn_model, trainDataPath, optimizer, 'temp', weight_high=10)
    val(cnn_model, 'temp/model_e30', testDataPath, '')
    val_d(cnn_model, 'temp/model_e30', testDataPath)
    ''' Perform test using RNN with ROS. '''
    rnn_model = RNN2()
    trainDataPath = 'data/ros1_aug'
    testDataPath = 'data/test'
    optimizer = nn.SGD(rnn_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    train_batch(rnn_model, trainDataPath, optimizer, 'temp')
    val(rnn_model, 'temp/model_e30', testDataPath, '')
    val_d(rnn_model, 'temp/model_e30', testDataPath)
    ''' Perform test using CNN with weight. '''
    rnn_model = RNN2()
    trainDataPath = 'data/mask_aug'
    testDataPath = 'data/test'
    optimizer = nn.SGD(rnn_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    train_batch(rnn_model, trainDataPath, optimizer, 'temp', weight_high=10)
    val(rnn_model, 'temp/model_e30', testDataPath, '')
    val_d(rnn_model, 'temp/model_e30', testDataPath)
    pass