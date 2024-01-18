import jittor as jt
from jittor import nn
from jittor.nn import Module

class CNN_2L(Module):
    def __init__(self, batch_size:int=128):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.Pool(2,2)
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.dropout_2 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 10)
    
    def execute(self, x):
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.dropout_2(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.dropout_3(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN_2L_2(Module):
    def __init__(self, batch_size:int=128):
        super().__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.Pool(2,2)
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout(p=0.3)
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.dropout_3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def extract_feat(self, x):
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.dropout_2(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        return x
    
    def classify(self, x):
        x = self.dropout_3(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    def execute(self, x):
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.dropout_2(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.dropout_3(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNN_3L_Dropout_2(Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.Relu()
        self.softmax = nn.Softmax()
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        self.max_pool_1 = nn.Pool(2,2)
        self.dropout_1 = nn.Dropout(p=0.2, is_train=True)
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        self.max_pool_2 = nn.Pool(2,2)
        self.dropout_2 = nn.Dropout(p=0.4, is_train=True)
        self.conv31 = nn.Conv(64, 128, 3, 1, padding=1)
        self.conv32 = nn.Conv(128, 128, 3, 1, padding=1)
        self.dropout_3 = nn.Dropout(p=0.4, is_train=True)
        self.max_pool_3 = nn.Pool(2,2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 10)

    def execute(self, x):
        x = self.relu(self.conv12(self.relu(self.conv11(x))))
        x = self.dropout_1(self.max_pool_1(x))
        x = self.relu(self.conv22(self.relu(self.conv21(x))))
        x = self.dropout_2(self.max_pool_2(x))
        x = self.relu(self.conv32(self.relu(self.conv31(x))))
        x = self.dropout_3(self.max_pool_3(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc2(self.relu(self.fc1(x)))
        return x

class CNN_3L_Dropout_3(Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.Relu()
        self.softmax = nn.Softmax()
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        self.max_pool_1 = nn.Pool(2,2)
        self.dropout_1 = nn.Dropout(p=0.3, is_train=True)
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        self.max_pool_2 = nn.Pool(2,2)
        self.dropout_2 = nn.Dropout(p=0.5, is_train=True)
        self.conv31 = nn.Conv(64, 128, 3, 1, padding=1)
        self.conv32 = nn.Conv(128, 128, 3, 1, padding=1)
        self.dropout_3 = nn.Dropout(p=0.5, is_train=True)
        self.max_pool_3 = nn.Pool(2,2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 10)

    def execute(self, x):
        x = self.relu(self.conv12(self.relu(self.conv11(x))))
        x = self.dropout_1(self.max_pool_1(x))
        x = self.relu(self.conv22(self.relu(self.conv21(x))))
        x = self.dropout_2(self.max_pool_2(x))
        x = self.relu(self.conv32(self.relu(self.conv31(x))))
        x = self.dropout_3(self.max_pool_3(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc2(self.relu(self.fc1(x)))
        return x