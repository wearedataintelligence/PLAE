import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, data_sequence, flag='train', size=None, 
                 scale=True, inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        self.inverse = inverse
#         self.freq = freq
#         self.cols=cols
        self.data_sequence = data_sequence
        self.__read_data__()

    def __read_data__(self):
        ## 归一化处理数据工具
        self.scaler = StandardScaler()
        df_raw = pd.DataFrame(self.data_sequence)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''

        num_train = int(len(df_raw)*1)
        num_test = int(len(df_raw)*0)
        num_vali = len(df_raw) - num_train - num_test
        ## train+vali-sequence_len
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        # train 0, vali 1, test 2
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
      
        df_data = df_raw

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values) 
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    
class Dataset_Pred(Dataset):
    def __init__(self, data_sequence, flag='pred', size=None,  
                scale=True, inverse=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.scale = scale
        self.inverse = inverse
        self.data_sequence = data_sequence
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.DataFrame(self.data_sequence)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        df_data = df_raw
        
        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values       

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]

    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
