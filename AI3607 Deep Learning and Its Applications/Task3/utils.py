import jittor as jt
from jittor import nn, Module
import numpy as np

jt.flags.use_cuda = True
save_int=2

def load_data(train_path:str, test_path:str):
    from jittor.dataset.mnist import MNIST
    import jittor.transform as trans
    batch_size=128
    train_loader = MNIST(train=True, transform=trans.Resize(28)).set_attrs(batch_size=batch_size, shuffle=True)
    test_loader = MNIST(train=False, transform=trans.Resize(28)).set_attrs(batch_size=batch_size, shuffle=True)
    trainDataList, trainLabelList = list(), list()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        trainDataList.append(inputs[:,[0,]].numpy())
        trainLabelList.append(targets)
    trainDataArr, trainLabelArr = np.concatenate(trainDataList,axis=0), np.concatenate(trainLabelList,axis=0)
    jt.save((trainDataArr, trainLabelArr), train_path)

    testDataList, testLabelList = list(), list()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        testDataList.append(inputs[:,[0,]].numpy())
        testLabelList.append(targets)
    testDataArr, testLabelArr = np.concatenate(testDataList,axis=0), np.concatenate(testLabelList,axis=0)
    jt.save((testDataArr, testLabelArr), test_path)

def genMaskData(orig_path:str, target_path:str):
    data, label = jt.load(orig_path)
    label_cond = (label >= 5)
    index_cond = (np.arange(label.shape[0]) % 10 == 0)
    final_cond = np.bitwise_or(label_cond, index_cond)
    data, label = data[final_cond], label[final_cond]
    jt.save((data, label), target_path)

def show(img, path):
    '''
    The function which outputs a certain image. Remember that it accepts only integer
    type arrays.
    '''
    import matplotlib.pyplot as plt
    from PIL import Image
    img = img.reshape((28,28))
    plt.imshow(Image.fromarray(img))
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

def train_batch(model:jt.nn.Module, dataPath:str, optimizer:jt.optim.Optimizer,
                saveFolder:str, batch_size:int=128, from_epoch:int=0, max_epoch:int=40,
                weight_high:float=1.0):
    import time
    dataARR, labelARR = jt.load(dataPath)
    model.train()
    fin_losses, epoch_losses = list(), list()
    if weight_high != 1.0:
        weights = [weight_high * jt.ones((5,), dtype=jt.float32),
                jt.ones((5,), dtype=jt.float32)]
        weights = jt.concat(weights, dim=0)
        loss_func = nn.CrossEntropyLoss(weights)
    else:
        loss_func = nn.CrossEntropyLoss()
    modelFile = saveFolder+f'/model_e{from_epoch}'
    if from_epoch > 0 and not modelFile is None:
        model.load_state_dict(jt.load(modelFile))
        fin_losses = jt.load(saveFolder+f'/loss')
    for epoch in range(from_epoch, max_epoch+1):
        start_time = time.time()
        # generate data with shuffle and reform into batches
        dataVAR, labelVAR = jt.Var(dataARR), jt.Var(labelARR)
        shuffle_idx = jt.randperm(dataVAR.shape[0])
        dataVAR, labelVAR = dataVAR[shuffle_idx], labelVAR[shuffle_idx]
        batch_num = dataVAR.shape[0] // batch_size
        dataVAR, labelVAR = (dataVAR[:batch_num*batch_size],
                             labelVAR[:batch_num*batch_size])
        dataVAR, labelVAR = (dataVAR.reshape((-1, batch_size, 1, 28, 28)),
                             labelVAR.reshape((-1, batch_size)))
        # train
        for batch in range(dataVAR.shape[0]):
            batch_out = model(dataVAR[batch])
            loss = loss_func(batch_out, labelVAR[batch])
            optimizer.step(loss)
            epoch_losses.append(loss.item())
        end_time = time.time()
        # output necessary information
        epoch_loss = np.average(np.array(epoch_losses))
        epoch_losses.clear()
        fin_losses.append(epoch_loss)
        print(f'Epoch {epoch}, time {round(end_time-start_time, 4)}s, loss {epoch_loss}.')
        if epoch % save_int == 0:
            jt.save(model.state_dict(), saveFolder+f'/model_e{epoch}')
            jt.save(fin_losses, saveFolder+f'/loss')

def val(model:jt.nn.Module, modelPath:str, dataPath:str, cm_path:str,
        batch_size:int=128, plot_cm:bool=False):
    model.load_state_dict(jt.load(modelPath))
    dataARR, labelARR = jt.load(dataPath)
    dataVAR, labelVAR = jt.Var(dataARR), jt.Var(labelARR)
    batch_num = dataVAR.shape[0] // batch_size
    dataVAR, labelVAR = (dataVAR[:batch_size*batch_num].reshape((-1, batch_size, 1, 28, 28)),
                         labelVAR[:batch_size*batch_num].reshape((-1, batch_size)))
    wrong_sum, total_sum = 0, 0
    pred_lst = list()
    for batch in range(dataVAR.shape[0]):
        batch_out = model(dataVAR[batch])
        prediction = jt.argmax(batch_out, dim=1)[0]
        pred_lst.append(prediction)
        diff = (prediction != labelVAR[batch]).sum()
        wrong_sum += diff.item()
        total_sum += labelVAR[batch].shape[0]
    print(f'{modelPath} result {wrong_sum} / {total_sum}.')
    if not plot_cm: return wrong_sum/total_sum
    predVAR, labelVAR = jt.concat(pred_lst, dim=0), labelVAR.reshape((-1,))
    label_names = [i for i in range(10)]
    plot_conf_mat(labelVAR, predVAR, label_names, cm_path)
    return wrong_sum/total_sum

def plot_conf_mat(actual:np.ndarray, pred:np.ndarray, label_names:list, path:str):
    '''
    The function to plot a confusion matrix according to given labels of prediction and
    ground truth.
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import itertools
    axis_font = {'family':'Consolas', 'color':'darkred', 'size':18}
    cm = confusion_matrix(actual, pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=1.0)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(x=j, y=i, s=cm[i,j], va='center', ha='center',
                size='x-large', color=('white' if i==j else 'black'))
    plt.xticks(range(10), label_names)
    plt.yticks(range(10), label_names)
    ax.set_xticklabels(label_names, rotation = 50)
    ax.set_yticklabels(label_names)
    plt.text(3.0, -1.5, 'Prediction', fontdict=axis_font)
    plt.ylabel('Ground Truth', fontdict=axis_font)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

def val_d(model:jt.nn.Module, modelPath:str, dataPath:str, batch_size:int=128):
    model.load_state_dict(jt.load(modelPath))
    dataARR, labelARR = jt.load(dataPath)
    dataVAR, labelVAR = jt.Var(dataARR), jt.Var(labelARR)
    batch_num = dataVAR.shape[0] // batch_size
    dataVAR, labelVAR = (dataVAR[:batch_size*batch_num].reshape((-1, batch_size, 1, 28, 28)),
                         labelVAR[:batch_size*batch_num].reshape((-1, batch_size)))
    pred_lst = list()
    for batch in range(dataVAR.shape[0]):
        batch_out = model(dataVAR[batch])
        prediction = jt.argmax(batch_out, dim=1)[0]
        pred_lst.append(prediction)
    predVAR, labelVAR = jt.concat(pred_lst, dim=0), labelVAR.reshape((-1,))
    wPredVAR, wLabelVAR = predVAR[predVAR != labelVAR], labelVAR[predVAR != labelVAR]
    jt.save((wPredVAR, wLabelVAR), modelPath+'_wVAR')
    predA = wPredVAR >= 5
    predI = jt.bitwise_not(predA)
    labelA = wLabelVAR >= 5
    labelI = jt.bitwise_not(labelA)
    num_ItoA = jt.bitwise_and(labelI, predA).sum().item()
    num_ItoI = jt.bitwise_and(labelI, predI).sum().item()
    num_AtoI = jt.bitwise_and(labelA, predI).sum().item()
    num_AtoA = jt.bitwise_and(labelA, predA).sum().item()
    num_A = (labelVAR >= 5).sum().item()
    num_I = labelVAR.shape[0] - num_A
    p_ItoI, p_ItoA, p_AtoI, p_AtoA = (round(num_ItoI / num_I, 4), round(num_ItoA/num_I,4),
                                      round(num_AtoI / num_A, 4), round(num_AtoA/num_A,4))
    print(f'ItoI:{p_ItoI}, ItoA:{p_ItoA}.')
    print(f'AtoI:{p_AtoI}, AtoA:{p_AtoA}.')
    return [p_ItoI, p_ItoA, p_AtoI, p_AtoA]

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
        newData = np.ndarray([reqMinCnt, 1, 28, 28], dtype=dataArr.dtype)
        newLabel = np.ones([reqMinCnt], dtype=labelArr.dtype) * label
        newData[0:localItems.shape[0]], newDataPtr = localItems, localItems.shape[0]
        addedIdxes = random.randint(0, localItems.shape[0], size=reqNum)
        for itr in range(addedIdxes.shape[0]):
            newData[newDataPtr] = localItems[addedIdxes[itr]]
            newDataPtr += 1
        dataItems.append(newData), labelItems.append(newLabel)
    dataROS, labelROS = np.concatenate(dataItems, axis=0), np.concatenate(labelItems, axis=0)
    return dataROS, labelROS

def augment(data:np.ndarray, label:np.ndarray, copy_times:int=2):
    from PIL import Image
    import matplotlib.pyplot as plt
    from numpy import random
    data = np.uint8(data*256).reshape((-1, 28, 28))
    aug_data_lst = list()
    for _ in range(copy_times):
        tmp_data = np.zeros_like(data)
        for idx in range(data.shape[0]):
            angle = random.randint(-25, 25+1)
            img = Image.fromarray(data[idx]).rotate(angle)
            tmp_data[idx] = np.array(img)
        aug_data_lst.append(tmp_data)
    aug_data_arr = np.concatenate(aug_data_lst, axis=0).reshape((-1, 1, 28, 28))
    aug_data_arr = aug_data_arr.astype(np.float32) / 256
    aug_label_arr = np.tile(label, (copy_times,))
    return aug_data_arr, aug_label_arr