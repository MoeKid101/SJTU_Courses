import numpy as np
import time

class SVM:
    def __init__(self, kernel:dict, max_iter:int=10, const_marg:float=1.0,
                 eps:float=1e-3):
        self._c = const_marg
        self._kernel = kernel
        self._max_iter = max_iter
        self._eps = eps
        self._valid = False
        self._data = None
        self._label = None
        # execution parameters
        self.numSamples = None
        self.dim = None
        self.K = None
        self.alpha = None
        self.w = None
        self.b = None
        self.errors = None
    
    def getKernel(self, x:np.ndarray, y:np.ndarray)->np.ndarray:
        result = None
        if self._kernel['type'] == 'linear':
            result = self._kernel['mtp'] * np.matmul(x, y.T)
        elif self._kernel['type'] == 'poly':
            result = np.power(np.matmul(x, y.T)+self._kernel['c'], self._kernel['d'])
        elif self._kernel['type'] == 'rbf':
            norm_x:np.ndarray = np.sum(np.square(x), axis=1)
            norm_y:np.ndarray = np.sum(np.square(y), axis=1)
            logMat = norm_x.reshape((-1, 1)) + norm_y - 2*np.matmul(x, y.T)
            result = np.exp(-self._kernel['gamma']*logMat)
        elif self._kernel['type'] == 'mix_mtp':
            poly_kernel = np.power(np.matmul(x, y.T)+self._kernel['c'], self._kernel['d'])
            norm_x:np.ndarray = np.sum(np.square(x), axis=1)
            norm_y:np.ndarray = np.sum(np.square(y), axis=1)
            logMat = norm_x.reshape((-1, 1)) + norm_y - 2*np.matmul(x, y.T)
            rbf_kernel = np.exp(-self._kernel['gamma']*logMat)
            result = poly_kernel*rbf_kernel
        elif self._kernel['type'] == 'mix_sum':
            lin_kernel = np.matmul(x, y.T)
            poly_kernel = np.power(np.matmul(x, y.T)+self._kernel['c'], self._kernel['d'])
            norm_x:np.ndarray = np.sum(np.square(x), axis=1)
            norm_y:np.ndarray = np.sum(np.square(y), axis=1)
            logMat = norm_x.reshape((-1, 1)) + norm_y - 2*np.matmul(x, y.T)
            rbf_kernel = np.exp(-self._kernel['gamma']*logMat)
            result = (self._kernel['mtp_lin']*lin_kernel + self._kernel['mtp_poly']*poly_kernel
                      + self._kernel['mtp_rbf']*rbf_kernel)
        return result
    
    def approx(self, v1, v2):
        return np.abs(v1-v2) < self._eps
    
    def update(self, idx1:int, idx2:int)->bool:
        if not self._valid: return False
        if (idx1 == idx2): return False
        a1_old, a2_old = self.alpha[idx1].copy(), self.alpha[idx2].copy()
        x1, x2 = self._data[idx1].copy(), self._data[idx2].copy()
        y1, y2 = self._label[idx1].copy(), self._label[idx2].copy()
        e1_old, e2_old = self.errors[idx1].copy(), self.errors[idx2].copy()
        e_old = self.errors.copy()
        w_old = self.w.copy()
        b_old = self.b
        L, H = .0, .0
        if y1*y2 > 0:
            L, H = max(0, a1_old + a2_old - self._c), min(self._c, a1_old + a2_old)
        else:
            L, H = max(0, a2_old - a1_old), min(self._c, self._c + a2_old - a1_old)
        if L == H: return False
        eta = self.K[idx1, idx1] + self.K[idx2, idx2] - 2*self.K[idx1, idx2]
        if eta <= 0: return False

        a1_new, a2_new, a1_clip, a2_clip = .0, .0, .0, .0
        a2_new = a2_old + y2 * (e1_old-e2_old) / eta
        if a2_new <= L: a2_clip = L
        elif a2_new >= H: a2_clip = H
        else: a2_clip = a2_new
        if abs(a2_clip - a2_old) < self._eps * (a2_clip + a2_old + self._eps): return False
        a1_clip = a1_old - y1*y2*(a2_clip - a2_old)

        wnew = w_old + (a1_clip-a1_old)*y1*x1 + (a2_clip-a2_old)*y2*x2

        b1 = b_old - e1_old - y1*self.K[idx1,idx1]*(a1_clip-a1_old) - y2*self.K[idx1,idx2]*(a2_clip-a2_old)
        b2 = b_old - e2_old - y1*self.K[idx1,idx2]*(a1_clip-a1_old) - y2*self.K[idx2,idx2]*(a2_clip-a2_old)
        bnew = .0
        if 0<a1_clip and a1_clip < self._c: bnew = b1
        elif 0<a2_clip and a2_clip < self._c: bnew = b2
        else: bnew = 0.5*(b1+b2)

        deltaErrors = bnew - b_old + y1*(a1_clip-a1_old)*self.K[idx1] + y2*(a2_clip-a2_old)*self.K[idx2]
        newErrors = e_old + deltaErrors

        self.alpha[idx1], self.alpha[idx2] = a1_clip, a2_clip
        self.w, self.b, self.errors = wnew, bnew, newErrors
        return True
    
    def examine(self, idx:int):
        if not self._valid: return False
        import numpy.random as rdm
        yi, alphai = self._label[idx], self.alpha[idx]
        ri = yi * self.errors[idx]
        if ((ri < -self._eps and alphai < self._c) or (ri > self._eps and alphai > 0)):
            alphaNZ:np.ndarray = np.bitwise_and(self.alpha != 0, self.alpha != self._c)
            if alphaNZ.any():
                # heuristic
                idx2 = 0
                if self.errors[idx] > 0: idx2 = np.argmin(self.errors)
                if self.errors[idx] <= 0: idx2 = np.argmax(self.errors)
                if self.update(idx, idx2): return True
                # loop
                NZidxes = np.where(alphaNZ)[0]
                for idx2 in np.roll(NZidxes, rdm.randint(0, NZidxes.shape[0])):
                    if self.update(idx, idx2): return True
            for idx2 in np.roll(np.arange(self.numSamples), rdm.randint(0, self.numSamples)):
                if self.update(idx, idx2): return True
        return False
    
    def fit(self, data:np.ndarray, label:np.ndarray):
        '''
        Expected data.shape=[#samples, dim], label.shape=[#samples,] take value
        {-1, +1}.
        '''
        # define necessary parameters
        self._valid = True
        self._data = data
        self._label = label
        self.numSamples = data.shape[0]
        self.dim = data.shape[1]
        self.K = self.getKernel(self._data, self._data)
        self.alpha = np.zeros((self.numSamples,), dtype=np.float32)
        self.w = np.zeros((self.dim,), dtype=np.float32)
        self.b = .0
        self.errors = - self._label
        # start training
        start_time = time.time()
        numChanged, examineAll = 0, True
        totalNumChanged = 0
        iter = 0
        while ((numChanged > 0) or examineAll) and iter < self._max_iter:
            numChanged = 0
            if examineAll:
                for a1 in range(self.numSamples):
                    locChange = self.examine(a1)
                    numChanged += locChange
                    totalNumChanged += locChange
            else:
                nonBounds = np.nonzero((self.alpha > 0)*(self.alpha < self._c))[0]
                for a1 in nonBounds:
                    locChange = self.examine(a1)
                    numChanged += locChange
                    totalNumChanged += locChange
            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True
            iter += 1
        # print(iter)
        end_time = time.time()
        # print(f'fit time {end_time-start_time}.')

    def predict(self, data:np.ndarray):
        if not self._valid: return None
        test_samples = data.shape[0]
        kernels = self.getKernel(self._data, data) # [n,m] where n training, m test.
        aiyi = np.broadcast_to(self.alpha*self._label, (test_samples, self.numSamples)).T
        result = np.sum(aiyi*kernels, axis=0) + self.b
        predictions = np.where(result > 0, 1, -1)
        return predictions.reshape((-1,))
    
    def predict_prob(self, data:np.ndarray):
        if not self._valid: return None
        test_samples = data.shape[0]
        kernels = self.getKernel(self._data, data) # [n,m] where n training, m test.
        aiyi = np.broadcast_to(self.alpha*self._label, (test_samples, self.numSamples)).T
        results = np.sum(aiyi*kernels, axis=0) + self.b
        return results.reshape((-1,))

class SVM_OAO():
    def __init__(self, kernel, max_iter:int=30, const_marg:float=1.0,
                 eps:float=1e-9):
        self._max_iter = max_iter
        self._const_marg = const_marg
        self._kernel = kernel
        self._eps = eps
        self.models:dict = {}
        self.numClass = None
        self.class_labels:np.ndarray = None
        self.data:np.ndarray = None
        self.label:np.ndarray = None

    def extractData(self, data:np.ndarray, label:np.ndarray, idx_i:int, idx_j:int,
                    shuffle:bool=True):
        ''' class_labels[idx_i] has label +1 while class_labels[j] has label -1. '''
        classi = data[label == self.class_labels[idx_i]].astype(np.float32)
        classj = data[label == self.class_labels[idx_j]].astype(np.float32)
        labeli, labelj = (np.ones([classi.shape[0],], dtype=np.float32),
                          -np.ones([classj.shape[0],], dtype=np.float32))
        fin_data, fin_label = (np.concatenate([classi, classj], axis=0),
                               np.concatenate([labeli, labelj], axis=0))
        if shuffle:
            import numpy.random as rdm
            shuffle_idx = np.arange(fin_data.shape[0])
            rdm.shuffle(shuffle_idx)
            fin_data, fin_label = fin_data[shuffle_idx], fin_label[shuffle_idx]
        return fin_data, fin_label

    def fit(self, data:np.ndarray, label:np.ndarray):
        '''
        Expected data.shape=[#samples, dim], label.shape=[#samples,] take integer values.
        '''
        self.data = data
        self.label = label
        self.class_labels = np.unique(label)
        self.numClass = self.class_labels.shape[0]
        print(f'Fit start.')
        start_time = time.time()
        for idx_i in range(self.numClass):
            for idx_j in range(idx_i+1, self.numClass):
                locData, locLabel = self.extractData(self.data, self.label, idx_i, idx_j)
                locModel = SVM(self._kernel, self._max_iter, self._const_marg, self._eps)
                locModel.fit(locData, locLabel)
                # print(np.average(locModel.alpha))
                self.models[idx_i*self.numClass+idx_j] = locModel
        end_time = time.time()
        print(f'Fit finished with {round(end_time-start_time, 4)}s.')
    
    def predict(self, data:np.ndarray):
        numSamples = data.shape[0]
        vote_box = np.zeros((self.numClass, numSamples), dtype=np.int32)
        print(f'Prediction start.')
        start_time = time.time()
        for key in self.models.keys():
            idx_i, idx_j = key // self.numClass, key % self.numClass
            model:SVM = self.models[key]
            locResult:np.ndarray = model.predict(data) # [#samples,]
            vote_box[idx_i] = vote_box[idx_i] + (locResult > 0).astype(np.int32)
            vote_box[idx_j] = vote_box[idx_j] + (locResult < 0).astype(np.int32)
        fin_pred = np.argmax(vote_box, axis=0)
        label_pred = self.class_labels[fin_pred]
        end_time = time.time()
        print(f'Prediction finished with {round(end_time-start_time, 4)}s.')
        return label_pred
    
    def val_by_class(self, data:np.ndarray, label:np.ndarray):
        for idx_i in range(self.numClass):
            for idx_j in range(idx_i+1, self.numClass):
                locData, locLabel = self.extractData(data, label, idx_i, idx_j)
                locModel:SVM = self.models[idx_i*self.numClass + idx_j]
                locPred = locModel.predict(locData)
                err_rate = val(locPred, locLabel)
                print(f'{idx_i}, {idx_j}: {err_rate}')
        pass
    
    def save(self, path:str):
        import torch
        save_list = list()
        for key in self.models.keys():
            idx_i, idx_j = key // self.numClass, key % self.numClass
            model:SVM = self.models[key]
            locParams:dict = {}
            locParams['idx_i'] = idx_i
            locParams['idx_j'] = idx_j
            locParams['model'] = model
            save_list.append(locParams)
        torch.save((self.class_labels, save_list), path)
    
    def load(self, path:str):
        import torch
        self.models.clear()
        class_labels, save_list = torch.load(path)
        self.class_labels = class_labels
        self.numClass = class_labels.shape[0]
        for param_dict in save_list:
            param_dict:dict
            idx_i, idx_j = param_dict['idx_i'], param_dict['idx_j']
            self.models[idx_i*self.numClass+idx_j] = param_dict['model']

class SVM_OAR():
    def __init__(self, kernel:dict, max_iter:int=30, const_marg:float=1.0,
                 eps:float=1e-9):
        self._max_iter = max_iter
        self._const_marg = const_marg
        self._kernel = kernel
        self._eps = eps
        self.models:dict = {}
        self.numClass = None
        self.class_labels:np.ndarray = None
        self.data:np.ndarray = None
        self.label:np.ndarray = None

    def extractData(self, data:np.ndarray, label:np.ndarray, idx_i:int,
                    shuffle:bool=True):
        ''' class_labels[idx_i] has label +1 while other classes have label -1. '''
        genData, genLabel = data, np.where(label==idx_i, +1, -1)
        if shuffle:
            shuffle_idx = np.arange(genData.shape[0])
            import numpy.random as rdm
            rdm.shuffle(shuffle_idx)
            genData, genLabel = genData[shuffle_idx], genLabel[shuffle_idx]
        return genData, genLabel

    def fit(self, data:np.ndarray, label:np.ndarray):
        '''
        Expected data.shape=[#samples, dim], label.shape=[#samples,] take integer values.
        '''
        self.data = data
        self.label = label
        self.class_labels = np.unique(label)
        self.numClass = self.class_labels.shape[0]
        for idx_i in range(self.numClass):
            locData, locLabel = self.extractData(self.data, self.label, idx_i)
            locModel = SVM(self._kernel, self._max_iter, self._const_marg, self._eps)
            locModel.fit(locData, locLabel)
            self.models[idx_i] = locModel
            # print(f'class {idx_i} fit complete.')
    
    def predict(self, data:np.ndarray):
        numSamples = data.shape[0]
        prob_box = np.zeros((self.numClass, numSamples), dtype=np.float32)
        for idx_i in range(self.numClass):
            model:SVM = self.models[idx_i]
            locResult = model.predict_prob(data)
            prob_box[idx_i] = locResult
        fin_pred = np.argmax(prob_box, axis=0)
        label_pred = self.class_labels[fin_pred]
        return label_pred
    
    def save(self, path:str):
        import torch
        save_list = list()
        for key in self.models.keys():
            idx_i, idx_j = key // self.numClass, key % self.numClass
            locParams:dict = {}
            locParams['idx'] = key
            locParams['model'] = self.models[key]
            save_list.append(locParams)
        torch.save((self.class_labels, save_list), path)
    
    def load(self, path:str):
        import torch
        self.models.clear()
        class_labels, save_list = torch.load(path)
        self.class_labels = class_labels
        self.numClass = class_labels.shape[0]
        for param_dict in save_list:
            param_dict:dict
            self.models[param_dict['idx']] = param_dict['model']

def val(pred:np.ndarray, label:np.ndarray):
    wrong_sum = np.sum(pred != label)
    total_sum = pred.shape[0]
    return wrong_sum / total_sum

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
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    plt.text(3.5, -1.2, 'Prediction', fontdict=axis_font)
    plt.ylabel('Ground Truth', fontdict=axis_font)
    plt.tight_layout()
    plt.savefig(path, dpi=800)
    plt.clf()
