from models import *
from utils import *
import nat_util

def show(img, path, size:int=32):
    '''
    The function which outputs a certain image. Remember that it accepts only integer
    type arrays.
    '''
    import matplotlib.pyplot as plt
    from PIL import Image
    img = img.reshape((3, size, size)) * 256
    x = img
    x = np.zeros((size, size, 3), dtype=int)
    for i in range(3):
        x[:,:,i] = img[i,:,:]
    plt.imshow(x)
    plt.savefig(path)
    plt.clf()

if __name__ == '__main__':
    ''' Generate data file to use. '''
    fileNames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                 'data_batch_5']
    testNames = ['test_batch']
    outOrigName = 'data/train'
    outTestName = 'data/test'
    genData(fileNames, outOrigName)
    genData(testNames, outTestName)
    ''' Generate splitted dataset. '''
    data_file = 'data/test'
    split_file = 'data/test_split'
    split_image(data_file, split_file)
    data_file = 'data/train'
    split_file = 'data/train_split'
    std_split(data_file, split_file)
    ''' Training. '''
    model = MPerm()
    trainDataPath = 'data/train_split'
    testDataPath = 'data/val_split'
    tempFolder = 'temp'
    optimizer = nn.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_func = HVCrossEntropy()
    train_2(model, trainDataPath, loss_func, optimizer, max_epoch=40,
            temp_folder=tempFolder, genDataFunc=genDataLabel, batch_size=128,
            num_copy=2, from_epoch=0)
    optimizer.lr = 1e-4
    train_2(model, trainDataPath, loss_func, optimizer, max_epoch=70,
            temp_folder=tempFolder, genDataFunc=genDataLabel, batch_size=128,
            num_copy=2, from_epoch=40)
    ''' Test. '''
    val_2(model, 'temp/model_e0', trainDataPath, useSinkHorn=False)
    ''' Train to judge whether a picture is natural. '''
    model = JudgeNatPic()
    trainDataPath = 'data/train'
    testDataPath = 'data/test'
    tempFolder = 'temp2'
    optimizer = nn.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_func = nn.BCELoss()
    nat_util.train_2(model, trainDataPath, loss_func, optimizer, max_epoch=50,
                     temp_folder=tempFolder, from_epoch=0)
    ''' Test. '''
    nat_util.val_2(model, 'temp/model_e0', trainDataPath, useSinkHorn=False)
    pass