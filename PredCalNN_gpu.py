from PredCal import PredicabilityTest
import numpy as np
import pandas as pd
from time import time
from torch import nn
import torch
import torch.utils.data.dataloader as DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error as mape
from pandarallel import pandarallel
import pickle
import math




pandarallel.initialize()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device : ", device)


EPOCH = 400
GPU_TRAIN = True




class Pred_CAL(nn.Module):
    def __init__(self, INPUT_LENGTH):
        super().__init__()
        self.CRNN = nn.Sequential(nn.Conv1d(in_channels= 1,         
                                            out_channels= 32,       
                                            kernel_size= 11,
                                            stride= 1,
                                            padding= 5),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2,
                                               stride=1),
                                  nn.Conv1d(in_channels= 32,       
                                            out_channels= 128,      
                                            kernel_size= 11,
                                            stride= 1,
                                            padding= 5),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2,
                                               stride=1),
                                  nn.Conv1d(in_channels= 128,       
                                            out_channels= 256,      
                                            kernel_size= 11,
                                            stride= 1,
                                            padding= 5),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(256),
                                  nn.Conv1d(in_channels= 256,       
                                            out_channels= 256,      
                                            kernel_size= 11,
                                            stride= 1,
                                            padding= 5),
                                  nn.ReLU(),
                                  nn.MaxPool1d(kernel_size=2,       
                                               stride=1),
                                  nn.LSTM(input_size= INPUT_LENGTH - 3,
                                          hidden_size= 256,
                                          num_layers= 2,
                                          batch_first= True,
                                          bidirectional= True))
        self.out = nn.Sequential(nn.Linear(512, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))
    
    def forward(self, x):
        r_out, (h_n, h_c) = self.CRNN(x)
        y = self.out(r_out[:, -1, :])
        return y




class trainDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return self.Data.shape[0]
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index, :, :])
        label = torch.Tensor(self.Label[index])
        return data, label




def dataSplit(inputArray, interval= 0.22):
    _min = inputArray.min()
    _max = inputArray.max()
    _num = int(np.ceil((_max - _min) / interval))
    for i in range(inputArray.shape[0]):
        interval_num = int(np.ceil((inputArray[i] - _min) / interval ))
        inputArray[i] = (interval_num - 1) + interval / 2
    return inputArray




def dataGeneration_random(seqLength, sample_num= 9999):
    
    start_time = time()
    seqList = []
    label = []
    
    while len(seqList) <= sample_num :
        seq_i = np.random.randn(seqLength)
        seq_i = dataSplit(seq_i)
        label_i = 1 - PredicabilityTest(seq_i)
        seqList.append(seq_i)
        label.append(label_i)
    
    data = np.array(seqList).reshape(-1, 1, seqLength)
    label = np.array(label).reshape(-1, 1)
    
    end_time = time()
    timecost = round(end_time - start_time, 4)

    print(" ****  Data generation accomplished.  ****  time cost : ", timecost, " seconds  **** ")
    print("data shape : ", data.shape)
    print("label shape : ", label.shape)
    
    return data, label




def dataGeneration_pattern(seqLength, sample_num= 9999):
    
    start_time = time()
    seqList = []
    label = []
    season_type = ["sine", "square", "triangle"]
    
    while len(seqList) <= sample_num :
        weight_S = np.random.rand()
        weight_T = np.random.rand()
        weight_R = np.random.rand()
        period_S = np.random.randint(int(seqLength / 20), int(seqLength / 3))
        seq_S = np.array([math.sin(2 * math.pi * i / period_S) for i in range(seqLength)])
        org_T = np.random.rand(seqLength)
        seq_T = np.convolve(np.ones(10) / 10, org_T)[5 : -4]
        seq_R = np.random.rand(seqLength)
        seq_i = weight_S * seq_S + weight_T * seq_T + weight_R * seq_R
        
        seq_i = dataSplit(seq_i)
        label_i = 1 - PredicabilityTest(seq_i)
        seqList.append(seq_i)
        label.append(label_i)
    
    data = np.array(seqList).reshape(-1, 1, seqLength)
    label = np.array(label).reshape(-1, 1)
    
    end_time = time()
    timecost = round(end_time - start_time, 4)

    print(" ****  Data generation accomplished.  ****  time cost : ", timecost, " seconds  **** ")
    print("data shape : ", data.shape)
    print("label shape : ", label.shape)
    
    return data, label




def trainNN(seqLength, sample_num= 9999, EPOCH= 2000, datasource= "aimaster"):

    if datasource == "generator" :
        data, label = dataGeneration_pattern(seqLength, sample_num)
    
    elif datasource == "local" :
        _x = open("_X.traindata", "rb")
        data = pickle.load(_x)
        _x.close()
        _y = open("_Y.traindata", "rb")
        label = pickle.load(_y)
        _y.close()
    
    elif datasource == "aimaster" :
        _x = open("/dfs/data/CRNN/_X.traindata", "rb")
        data = pickle.load(_x)
        _x.close()
        _y = open("/dfs/data/CRNN/_Y.traindata", "rb")
        label = pickle.load(_y)
        _y.close()

    entropyModel = Pred_CAL(seqLength).to(device)
    optimizer = torch.optim.Adam(entropyModel.parameters(), lr=0.02)
    lossfunc = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    dataSet = trainDataset(data, label)
    dataLoader = DataLoader.DataLoader(dataSet, batch_size=32, shuffle=True, num_workers=1)
    for epoch in range(EPOCH):
        for step, (x, b_y) in enumerate(dataLoader):
            x = x.to(device)
            b_y = b_y.to(device)
            output = entropyModel(x)
            loss = lossfunc(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(epoch + 1, loss.item()))
    
    return entropyModel




def evaluate(model, seqLength, eval_num= 1000):

    data, label = dataGeneration_pattern(seqLength, eval_num)
    label = label.reshape((-1))
    
    pred = []
    for i in range(data.shape[0]):
        this_pred = model(torch.from_numpy(data[i, :, :].reshape(1, 1, -1)))
        pred.append(this_pred.detach().numpy())
    pred = np.array(pred).reshape((-1))

    loss = mse(label, pred)
    _mape = mape(label, pred)
    print("pred : ", pred)
    print("label : ", label)

    return np.sqrt(loss), _mape




if __name__ == "__main__" :
    
    torch.set_default_tensor_type(torch.DoubleTensor)

    
    if GPU_TRAIN == True :
        
        model = trainNN(74, 400000, EPOCH)
        model = model.to("cpu")
        
        loss, _mape = evaluate(model, 74, 999)
        print("Test RMSE : ", loss)
        print("Test MAPE : ", _mape)
        
        torch.save(model, "/dfs/data/CRNN/model.tlz")
    
    
    else :
        
        model = torch.load("/dfs/data/CRNN/model.tlz")
        loss = evaluate(model, 74, 99)
        print("Test Loss : ", loss)