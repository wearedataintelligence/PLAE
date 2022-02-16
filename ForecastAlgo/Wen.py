from Wen_utils import CNN_Wen, trainDataset_wen
from Wen_utils import Summarize_Seq, TrainDataGen, Normalization
import numpy as np
import torch
from torch import nn
import torch.utils.data.dataloader as DataLoader
torch.set_default_tensor_type(torch.DoubleTensor)




wen_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("wen device : ", wen_device)




WL= 13
SMIN= 4
SMAX= 11
MAXDIST= 7 
MAXVOC= 5
EPOCH= 400




def Wen_train(inputArray, forecast_steps, windowLen= WL, s_min= SMIN, s_max= SMAX, max_dist= MAXDIST, max_vocab= MAXVOC, epoch= EPOCH):
    
    inputArray = inputArray.reshape(1, -1)
    models, _, _, idx, _, _ = Summarize_Seq(inputArray, s_min, s_max, max_dist, max_vocab)
    for i in range(len(idx)) :
        if i == 0 :
            new_seq = models[idx[i]].reshape(-1, )
        else :
            new_seq = np.append(new_seq, models[idx[i]].reshape(-1, ))
    print("new_seq shape : ", new_seq.shape)
    print("new seq : ", new_seq)
    
    
    _data, _label = TrainDataGen(new_seq, windowLen, forecast_steps)
    print("Train data generated.")
    print("_data size : ", _data.shape)
    print("_label size : ", _label.shape)
    
    wen_cnn_model = CNN_Wen(forecast_steps, windowLen).to(wen_device)
    optimizer = torch.optim.Adam(wen_cnn_model.parameters(), lr=0.1)
    lossfunc = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    dataSet = trainDataset_wen(_data, _label)
    dataLoader = DataLoader.DataLoader(dataSet, batch_size= 8, shuffle= True, num_workers= 1)
    for eachepoch in range(epoch):
        for step, (x, b_y) in enumerate(dataLoader):
            x = x.to(wen_device)
            b_y = b_y.to(wen_device)
            output = wen_cnn_model(x)
            loss = lossfunc(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (eachepoch + 1) % 10 == 0:
            print('Epoch:{}, Loss:{:.5f}'.format(eachepoch + 1, loss.item()))
    
    print("Model traning done.")
    return wen_cnn_model.to("cpu")




def Wen_forecast(inputArray, model, windowLen= WL):
    
    inputArray = inputArray.reshape(-1, )
    testArray, _mean, _vol = Normalization(inputArray[-windowLen : ])
    testArray = testArray.reshape(1, 1, -1)
    testTensor = torch.from_numpy(testArray)
    print("test tensor size : ", testTensor.size())
    print("test tensor : ", testTensor)
    
    pred_res = model(testTensor.double())
    pred_res = pred_res.detach().numpy()
    pred_res = pred_res.reshape(-1, ) * _vol + _mean
    return pred_res.reshape(-1, )
    



if __name__ == "__main__" :
    
    inputSeq = [[1, 2, 3, 4 , 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4]]
    inputSeq = np.array(inputSeq)
    model = Wen_train(inputArray= inputSeq, forecast_steps= 2, windowLen= 4, s_min= 2, s_max= 6, max_dist= 4, max_vocab= 5, epoch= 10)
    pred_res = Wen_forecast(inputSeq, model, windowLen= 4)
    print(type(pred_res))
    print(pred_res.shape)
    print(pred_res)