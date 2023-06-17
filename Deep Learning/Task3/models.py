from jittor import nn, Module
import jittor as jt

jt.flags.use_cuda = 1

class CNN(Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv(1, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 32, 3, 1, padding=1)
        self.bn = nn.BatchNorm(32)
        self.max_pool = nn.Pool(2,2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32*14*14, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def execute(self,x):
        x = self.relu(self.conv11(x))
        x = self.relu(self.bn(self.conv12(x)))
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class CNN2(Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv(1, 32, 3, 1, padding=1)
        self.conv12 = nn.Conv(32, 64, 3, 1, padding=1)
        self.bn = nn.BatchNorm(64)
        self.max_pool = nn.Pool(2,2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64*14*14, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def execute(self, x):
        x = self.relu(self.conv11(x))
        x = self.relu(self.bn(self.conv12(x)))
        x = self.max_pool(x)
        x = self.dropout1(x)
        x = jt.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = self.dropout2(self.relu(x))
        x = self.fc2(x)
        return x

class RNN(Module):
    def __init__(self, in_feat:int=28, hidden_feat:int=100, num_layer:int=1):
        super().__init__()
        self._in_feat = in_feat
        self._hidden_feat = hidden_feat
        self._num_layer = num_layer
        self.lstm = nn.LSTM(in_feat, hidden_feat, num_layer)
        self.linear = nn.Linear(hidden_feat, 10)
    
    def execute(self, x:jt.Var):
        '''
        x: [-1, 1, 28, 28] should be reformed into [28, -1, 28].
        '''
        L, N, Hin, Hout = 28, x.shape[0], 28, self._hidden_feat
        x = x.reshape((-1, L, Hin)).transpose((1,0,2))
        # here x.shape=[L, batch_size, Hin] where L=28, batch_size=-1, Hin=28
        h0, c0 = (jt.zeros((self._num_layer, N, Hout), dtype=jt.float32),
                  jt.zeros((self._num_layer, N, Hout), dtype=jt.float32))
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # here lstm_out.shape=[L, batch_size, Hout]
        lstm_out = lstm_out[-1]
        result = self.linear(lstm_out)
        return result

class RNN2(Module):
    def __init__(self, in_feat:int=28, hidden_feat:int=100, num_layer:int=2):
        super().__init__()
        self._in_feat = in_feat
        self._hidden_feat = hidden_feat
        self._num_layer = num_layer
        self.lstm = nn.LSTM(in_feat, hidden_feat, num_layer)
        self.linear = nn.Linear(hidden_feat, 10)
    
    def execute(self, x:jt.Var):
        '''
        x: [-1, 1, 28, 28] should be reformed into [28, -1, 28].
        '''
        L, N, Hin, Hout = 28, x.shape[0], 28, self._hidden_feat
        x = x.reshape((-1, L, Hin)).transpose((1,0,2))
        # here x.shape=[L, batch_size, Hin] where L=28, batch_size=-1, Hin=28
        h0, c0 = (jt.zeros((self._num_layer, N, Hout), dtype=jt.float32),
                  jt.zeros((self._num_layer, N, Hout), dtype=jt.float32))
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # here lstm_out.shape=[L, batch_size, Hout]
        lstm_out = lstm_out[-1]
        result = self.linear(lstm_out)
        return result