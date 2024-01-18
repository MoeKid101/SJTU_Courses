import jittor as jt
import numpy as np
from jittor.nn import Module

jt.flags.use_cuda = True

def show(img, path, size:int=(32, 32)):
    '''
    The function which outputs a certain image. Remember that it accepts only integer
    type arrays.
    '''
    import matplotlib.pyplot as plt
    from PIL import Image
    img = img.reshape((3, size[0], size[1])) * 256
    x = img
    x = np.zeros((size[0], size[1], 3), dtype=int)
    for i in range(3):
        x[:,:,i] = img[i,:,:]
    plt.imshow(x)
    plt.axis('off')
    plt.savefig(path)
    plt.clf()

def genDataLabel2(data:np.ndarray, num_copy:int=2, erase:bool=False):
    from numpy import random
    # expected data.shape=[-1, 4, 3, 16, 16]
    num_pieces = data.shape[1]
    dataGen, labelGen = (np.zeros((num_copy*data.shape[0], num_pieces, 3, 16, 16), dtype=data.dtype),
                         np.zeros((num_copy*data.shape[0], num_pieces), dtype=np.int16))
    genDataPtr = 0
    for idx in range(data.shape[0]):
        localSpt = data[idx]
        for copy_idx in range(num_copy):
            shuffle_idx = np.arange(localSpt.shape[0])
            random.shuffle(shuffle_idx)
            dataGen[genDataPtr] = localSpt[shuffle_idx]
            if erase:
                erase_idx = np.random.randint(0, 4)
                dataGen[genDataPtr, erase_idx] = np.zeros((3, 16, 16), dtype=dataGen.dtype)
            labelGen[genDataPtr] = shuffle_idx
            genDataPtr += 1
    return dataGen, labelGen

def OUTtoLABEL(input:jt.Var, useSinkHorn:bool=False):
    import pygmtools as pgm
    # Expected input.shape=[-1, 4, 4]
    SKresult:jt.Var = pgm.sinkhorn(input) if useSinkHorn else input
    argmax = jt.argmax(SKresult.reshape((-1, 4)), dim=1)[0]
    result = jt.nn.one_hot(argmax)
    return result.reshape((-1, 4, 4))

def genJudgeNatData(data:np.ndarray, num_copy:int=8):
    from numpy import random
    num = data.shape[0] * num_copy
    labels = random.randint(0, 2, size=num)
    splitDir = random.randint(0, 2, size=num)
    DIR_UD, DIR_LR = 0, 1
    dataGen = np.zeros((num, 3, 16, 16), dtype=data.dtype)
    for idx in range(num):
        img = data[idx % data.shape[0]]
        if labels[idx] == 1:
            horBegin, verBegin = random.randint(0, 17), random.randint(0, 17)
            dataGen[idx] = img[:, verBegin:verBegin+16, horBegin:horBegin+16]
            continue
        if splitDir[idx] == DIR_UD:
            verBegin = random.randint(0, 17)
            dataGen[idx, :, :, 8:16] = img[:, verBegin:verBegin+16, 0:8]
            dataGen[idx, :, :, 0:8] = img[:, verBegin:verBegin+16, 24:32]
            continue
        if splitDir[idx] == DIR_LR:
            horBegin = random.randint(0, 17)
            dataGen[idx, :, 8:16, :] = img[:, 0:8, horBegin:horBegin+16]
            dataGen[idx, :, 0:8, :] = img[:, 8:16, horBegin:horBegin+16]
    return dataGen, labels

def train_2(model:Module, data_file:str, loss_func, optimizer:jt.optim.Optimizer,
            max_epoch:int, temp_folder:str, from_epoch:int=0,
            shuffle:bool=True, batch_size:int=128, num_copy:int=2, testOnTrain:bool=True):
    import time, os
    model.train()
    epoch_losses = list()
    LossSavePath = f'{temp_folder}/loss'
    fin_losses = (jt.load(LossSavePath) if (os.path.exists(LossSavePath) and 
                  from_epoch > 0) else list())
    ModelLoadPath = f'{temp_folder}/model_e{from_epoch}'
    if from_epoch > 0: model.load_state_dict(jt.load(ModelLoadPath))
    dataOrig = jt.load(data_file)
    for epoch in range(from_epoch, max_epoch+1):
        genData_start = time.time()
        ''' generate train data from data file (splitted images). '''
        dataGen, labelGen = genJudgeNatData(dataOrig, num_copy)
        dataVar, labelVar = jt.Var(dataGen), jt.Var(labelGen)
        if shuffle:
            shuffle_idx = jt.randperm(dataVar.shape[0])
            dataVar, labelVar = dataVar[shuffle_idx], labelVar[shuffle_idx]
        ''' generate batches. '''
        batch_num = dataVar.shape[0] // batch_size
        total_num = batch_num * batch_size
        dataVar, labelVar = (dataVar[:total_num].reshape((-1, batch_size, 3, 16, 16)),
                             labelVar[:total_num].reshape((-1, batch_size,)))
        genData_end = time.time()
        print(f'Epoch {epoch}. Generate data cost {round(genData_end-genData_start, 4)}s.')
        Training_start = time.time()
        ''' perform training. '''
        for batch_idx in range(batch_num):
            prediction = model(dataVar[batch_idx])
            # print(prediction.shape, labelVar[batch_idx].shape)
            loss = loss_func(prediction, labelVar[batch_idx])
            optimizer.step(loss)
            epoch_losses.append(loss.item())
        Training_end = time.time()
        epoch_loss_avg = jt.mean(jt.Var(epoch_losses)).item()
        epoch_losses.clear()
        fin_losses.append(epoch_loss_avg)
        ''' output necessary information. '''
        print(f'Training cost {round(Training_end-Training_start, 4)}s with average ' +
              f'loss {np.round(epoch_loss_avg, 5)}.')
        if epoch % 5 == 0:
            jt.save(model.state_dict(), f'{temp_folder}/model_e{epoch}')
            jt.save(fin_losses, LossSavePath)
        ''' test results. '''
        if not testOnTrain: continue
        model.eval()
        wrong_sum = 0
        for batch_idx in range(batch_num):
            outputs = model(dataVar[batch_idx])
            prediction = jt.where(outputs > 0.5, 1, 0)
            # prediction = jt.argmax(outputs, dim=1)[0]
            wrong_sum += jt.sum(prediction != labelVar[batch_idx]).item()
        print(f'Got error rate {round(wrong_sum/total_num,4)} on training set.')
        model.train()

def val_2(model:Module, model_file:str, data_file:str, batch_size:int=128,
          useSinkHorn:bool=False):
    model.eval()
    model.load_state_dict(jt.load(model_file))
    dataArr = jt.load(data_file)
    dataGen, labelGen = genJudgeNatData(dataArr)
    dataVar, labelVar = jt.Var(dataGen), jt.Var(labelGen)
    batch_num = dataVar.shape[0] // batch_size
    total_num = batch_num * batch_size
    dataVar, labelVar = (dataVar[:total_num].reshape((-1, batch_size, 3, 16, 16)),
                         labelVar[:total_num].reshape((-1, batch_size,)))
    # wrong_sum = 0
    pred_lst = list()
    for batch_idx in range(batch_num):
        outputs = model(dataVar[batch_idx])
        # prediction = jt.where(outputs > 0.5, 1, 0)
        # prediction = jt.argmax(outputs, dim=1)[0]
        # wrong_sum += jt.sum(prediction != labelVar[batch_idx]).item()
        pred_lst.append(outputs.numpy())
    predArr = np.concatenate(pred_lst, axis=0)
    labelArr = labelVar.numpy().reshape((-1,))
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labelArr, predArr)
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr)
    plt.savefig('Permutation/judgeNat_perf.png', dpi=600)
    plt.clf()

def cropToBatchVar(data:np.ndarray, batch_size:int=128)->jt.Var:
    data = jt.Var(data)
    batch_num = data.shape[0] // batch_size
    total_num = batch_num * batch_size
    data = data[:total_num]
    return data

def stdPredPerm(data:np.ndarray, JudgeNatModel:Module, out_folder:str,
                batch_size:int=128,
                direction:str='UD'):
    JudgeNatModel.eval()
    if direction=='UD':
        data1, data2 = data[:, :, :, 0:8, :], data[:, :, :, 8:16, :]
    elif direction=='LR':
        data1, data2 = data[:, :, :, :, 0:8], data[:, :, :, :, 8:16]
    for idx in range(16):
        i,j=idx//4, idx%4
        if i==j: continue
        dataCrop = np.zeros((data.shape[0], 3, 16, 16), dtype=data.dtype)
        if direction=='UD':
            dataCrop[:,:,0:8,:], dataCrop[:,:,8:16,:] = data1[:,i], data2[:,j]
        elif direction=='LR':
            dataCrop[:,:,:,0:8], dataCrop[:,:,:,8:16] = data1[:,i], data2[:,j]
        dataCrop = cropToBatchVar(dataCrop, batch_size)
        dataCrop = dataCrop.reshape((-1, batch_size, 3, 16, 16))
        cache = list()
        for batch_idx in range(dataCrop.shape[0]):
            cache.append(JudgeNatModel(dataCrop[batch_idx]))
        resVar = jt.concat(cache, dim=0)
        print(resVar.shape)
        jt.save(resVar, f'{out_folder}/{i}_{j}_{direction}')

def genPermTrain(data:np.ndarray, JudgeNatModel:Module, temp_folder:str,
                 batch_size:int=128):
    numSamples = (data.shape[0] // batch_size) * batch_size
    stdPredPerm(data, JudgeNatModel, temp_folder, direction='UD')
    stdPredPerm(data, JudgeNatModel, temp_folder, direction='LR')
    resultMat = jt.zeros((numSamples, 2, 4, 4), dtype=jt.float32)
    for idx in range(16):
        i, j = idx//4, idx%4
        if i==j: continue
        resultMat[:,0,i,j] = jt.load(f'{temp_folder}/{i}_{j}_UD')
        resultMat[:,1,i,j] = jt.load(f'{temp_folder}/{i}_{j}_LR')
    jt.save(resultMat, f'{temp_folder}/stdMatTest')

def genPMLabel(data_file:str, num_copy:int=8):
    num_pieces = 4
    dataVar:np.ndarray = jt.load(data_file) # [-1, 2, 4, 4]
    num = num_copy * dataVar.shape[0]
    dataGen, labelGen = (np.zeros((num, 2, 4, 4), dtype=np.float32),
                         np.zeros((num, 4, 4), dtype=np.float32))
    for idx in range(num):
        locSample = dataVar[idx % dataVar.shape[0]]
        shuffle_idx = np.arange(4)
        np.random.shuffle(shuffle_idx)
        locSample = locSample[:,:,shuffle_idx]
        locSample = locSample[:,shuffle_idx,:]
        dataGen[idx] = locSample
        for piece_idx in range(num_pieces):
            labelGen[idx, piece_idx, shuffle_idx[piece_idx]] = 1.0
    finGen = np.zeros((num, 24), dtype=np.float32)
    finPtr = 0
    for idx in range(16):
        i, j = idx//4, idx%4
        if i==j: continue
        finGen[:, finPtr] = dataGen[:,0,i,j]
        finPtr += 1
        finGen[:, finPtr] = dataGen[:,1,i,j]
    finGen, labelGen = jt.Var(finGen), jt.Var(labelGen)
    return finGen, labelGen

def val_3(permModel:Module,
          data_file:str, useSinkHorn=False, batch_size:int=128):
    permModel.eval()
    dataGen, labelGen = genPMLabel(data_file)
    dataGen, labelGen = (dataGen.reshape((-1, batch_size, 24)),
                         labelGen.reshape((-1, batch_size, 4, 4)))
    wrong_sum = 0
    for batch_idx in range(dataGen.shape[0]):
        outputs = permModel(dataGen[batch_idx])
        prediction = OUTtoLABEL(outputs, useSinkHorn)
        diff = (prediction != labelGen[batch_idx]).reshape((-1, 16))
        wrong_sum += jt.sum(jt.any(diff, dim=1)).item()
    return wrong_sum / dataGen.shape[0] / batch_size

def train_3(permModel:Module,
            trainFile:str, testFile:str, loss_func, 
            optimizer:jt.optim.Optimizer, max_epoch:int, temp_folder:str,
            shuffle:bool=True, batch_size:int=128, 
            testOnTrain:bool=True, testOnTest:bool=True):
    for epoch in range(max_epoch):
        dataGen, labelGen = genPMLabel(trainFile)
        if shuffle:
            shuffle_idx = jt.randperm(dataGen.shape[0])
            dataGen, labelGen = dataGen[shuffle_idx], labelGen[shuffle_idx]
        dataGen, labelGen = (dataGen.reshape((-1, batch_size, 24)),
                             labelGen.reshape((-1, batch_size, 4, 4)))
        dataGen = jt.float32(dataGen > 0.5)
        # print(dataGen.shape, labelGen.shape)
        permModel.train()
        losses = list()
        for batch_idx in range(dataGen.shape[0]):
            outputs = permModel(dataGen[batch_idx])
            loss = loss_func(outputs, labelGen[batch_idx])
            optimizer.step(loss)
            losses.append(loss)
        print(np.average(np.array(losses)))
        losses.clear()
        if epoch % 2 == 0:
            jt.save(permModel.state_dict(), f'{temp_folder}/pm_e{epoch}')
        if not testOnTrain: continue
        permModel.eval()
        wrong_sum = 0
        print(dataGen.shape, labelGen.shape)
        for batch_idx in range(dataGen.shape[0]):
            outputs = permModel(dataGen[batch_idx])
            prediction = OUTtoLABEL(outputs, False)
            diff = (prediction != labelGen[batch_idx]).reshape((-1, 16))
            wrong_sum += jt.sum(jt.any(diff, dim=1)).item()
        print(wrong_sum/dataGen.shape[0]/batch_size)
        print(val_3(permModel, testFile))

def testResult(data:np.ndarray, jnmodel:jt.nn.Module):
    dataGen, labelGen = genDataLabel2(data)
    restoreLabel = np.zeros_like(labelGen)
    for i in range(4):
        restoreLabel[:,i] = np.where(labelGen==i)[1]
    