import jittor as jt
from jittor import nn, Module
import pygmtools as pgm
pgm.BACKEND = 'jittor'
NUM_FC_INTERFACE = 32
NUM_PERM = 4

class PermNet(Module):
    def __init__(self):
        super().__init__()
        ''' CNN base model part '''
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
        self.fc1 = nn.Linear(64*4*4, NUM_FC_INTERFACE)
        ''' Final classification part. '''
        self.fc_fin1 = nn.Linear(NUM_PERM * NUM_FC_INTERFACE, 64)
        self.fc_fin2 = nn.Linear(64, NUM_PERM*NUM_PERM)
    
    def CNNBase(self, x):
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        x = self.relu(x)
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.relu(x)
        x = self.dropout_2(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        return x
    
    def execute(self, x:jt.Var):
        # expected x.shape=[batch_size, 4, 3, 16, 16]
        x = x.reshape((-1, 3, 16, 16))
        x = self.CNNBase(x) # x.shape=[batch_size*4, NUM_FC_INTERFACE]
        x = x.reshape((-1, NUM_PERM*NUM_FC_INTERFACE))
        x = self.fc_fin1(x)
        x = self.relu(x)
        x = self.fc_fin2(x)
        x = x.reshape((-1, NUM_PERM, NUM_PERM))
        # x = pgm.sinkhorn(x)
        # x = x.reshape((-1, NUM_PERM*NUM_PERM))
        return x

class PosPred(Module):
    def __init__(self, num_fc_interface:int=64):
        super().__init__()
        self.num_fc_interface = num_fc_interface
        ''' CNN base model part '''
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.Pool(2,2)
        # block 1
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout(p=0.2)
        # block 2
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.dropout_2 = nn.Dropout(p=0.4)
        # block 3
        self.conv31 = nn.Conv(64, 128, 3, 1, padding=1)
        self.conv32 = nn.Conv(128, 128, 3, 1, padding=1)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(128*2*2, num_fc_interface)
        ''' Final classification part. '''
        self.fc_fin1 = nn.Linear(2 * num_fc_interface, 32)
        self.dropout_4 = nn.Dropout(p=0.4)
        self.fc_fin2 = nn.Linear(32, 5)

    def CNNBase(self, x):
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        x = self.relu(x)
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.relu(x)
        x = self.dropout_2(self.max_pool(x))
        x = self.relu(self.conv31(x))
        x = self.conv32(x)
        x = self.relu(x)
        x = self.dropout_3(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        return x
    
    def execute(self, x:jt.Var):
        # expected x.shape=[batch_size, 4, 3, 16, 16]
        x = x.reshape((-1, 3, 16, 16))
        x = self.CNNBase(x) # x.shape=[batch_size*4, NUM_FC_INTERFACE]
        x = x.reshape((-1, 2 * self.num_fc_interface))
        x = self.fc_fin1(x)
        x = self.relu(x)
        x = self.dropout_4(x)
        x = self.fc_fin2(x)
        return x

class MPerm(Module):
    def __init__(self, num_fc_interface:int=64, num_perm:int=4):
        super().__init__()
        self.num_fc_interface = num_fc_interface
        self.num_perm = num_perm
        ''' CNN base model part '''
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.Pool(2,2)
        # block 1
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout(p=0.2)
        # block 2
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.dropout_2 = nn.Dropout(p=0.4)
        # block 3
        self.conv31 = nn.Conv(64, 128, 3, 1, padding=1)
        self.conv32 = nn.Conv(128, 128, 3, 1, padding=1)
        self.dropout_3 = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(64*4*4, num_fc_interface)
        ''' Final classification part. '''
        self.fc_fin1 = nn.Linear(num_perm * num_fc_interface, 64)
        self.dropout_4 = nn.Dropout(p=0.4)
        self.fc_fin2 = nn.Linear(64, num_perm * num_perm)

    def CNNBase(self, x):
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        x = self.relu(x)
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.relu(x)
        x = self.dropout_2(self.max_pool(x))
        # x = self.relu(self.conv31(x))
        # x = self.conv32(x)
        # x = self.relu(x)
        # x = self.dropout_3(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        return x
    
    def execute(self, x:jt.Var):
        # expected x.shape=[batch_size, 4, 3, 16, 16]
        x = x.reshape((-1, 3, 16, 16))
        x = self.CNNBase(x) # x.shape=[batch_size*4, NUM_FC_INTERFACE]
        x = x.reshape((-1, self.num_perm * self.num_fc_interface))
        x = self.fc_fin1(x)
        x = self.relu(x)
        # x = self.dropout_4(x)
        x = self.fc_fin2(x)
        x = x.reshape((-1, self.num_perm, self.num_perm))
        return x

class JudgeNatPic(Module):
    def __init__(self):
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.max_pool = nn.Pool(2,2)
        # block 1
        self.conv11 = nn.Conv(3, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        # self.bn1 = nn.BatchNorm2d(32)
        self.dropout_1 = nn.Dropout(p=0.2)
        # block 2
        self.conv21 = nn.Conv(32, 64, 3, 1, padding=1)
        self.conv22 = nn.Conv(64, 64, 3, 1, padding=1)
        # self.bn2 = nn.BatchNorm2d(64)
        self.dropout_2 = nn.Dropout(p=0.4)
        # block 3
        self.conv31 = nn.Conv(64, 128, 3, 1, padding=1)
        self.conv32 = nn.Conv(128, 128, 3, 1, padding=1)
        self.dropout_3 = nn.Dropout(p=0.4)
        # prediction
        self.fc1 = nn.Linear(64*4*4, 64)
        self.dropout_4 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def execute(self, x):
        x = self.relu(self.conv11(x))
        x = self.conv12(x)
        x = self.relu(x)
        x = self.dropout_1(self.max_pool(x))
        x = self.relu(self.conv21(x))
        x = self.conv22(x)
        x = self.relu(x)
        x = self.dropout_2(self.max_pool(x))
        # x = self.relu(self.conv31(x))
        # x = self.conv32(x)
        # x = self.relu(x)
        # x = self.dropout_3(self.max_pool(x))
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_4(x)
        x = self.fc2(x)
        x = self.sigmoid(x).reshape((-1,))
        return x

class GenPerm(Module):
    def __init__(self):
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(24, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, 24)
        self.fc4 = nn.Linear(24, 16)
    
    def execute(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x).reshape((-1, 4, 4))
        return x