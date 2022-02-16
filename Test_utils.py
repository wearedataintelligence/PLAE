from ForecastAlgo.AutoArima import AutoARIMA
from ForecastAlgo.GBDT_Regressor import GBDT_train, GBDT_predict
from ForecastAlgo.AutoTSB import Auto_TSB
from ForecastAlgo.Wen import Wen_train, Wen_forecast
from ForecastAlgo.Informer2020.interface import train_and_predict
import torch
from torch import nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pandarallel import pandarallel
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score




EPOCH = 700


pandarallel.initialize()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device : ", device)




def Normalization(inputArray):
    _mean = inputArray.mean()
    _volatility = inputArray.std()
    numpyArray = (inputArray - _mean) / _volatility
    return numpyArray, _mean, _volatility




def reverseNormalization(inputArray, _mean, _volatility):
    return inputArray * _volatility + _mean




def corrcoef(x):
    f = (x.shape[0] - 1) / x.shape[0]     
    x_reducemean = x - torch.mean(x, axis=0)
    numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
    var_ = x.var(axis=0).reshape(x.shape[1], 1)
    denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f
    corrcoef = numerator / denominator
    return corrcoef




class QuasiAE(nn.Module):

    def __init__(self, grouping_idx):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1, 16),
                                     nn.Sigmoid(),
                                     nn.LayerNorm(16),
                                     nn.Linear(16, 128),
                                     nn.Sigmoid(),
                                     nn.LayerNorm(128),
                                     nn.Linear(128, 1024))
        self.decoder = nn.Sequential(nn.Linear(128, 16),
                                     nn.Sigmoid(),
                                     nn.LayerNorm(16),
                                     nn.Linear(16, 1))
        self.grouping_idx = grouping_idx
    

    def Grouping_TensorGen(self):
        groupingTensor = []
        for each_group_idx in self.grouping_idx :
            this_col = [0 for i in range(0, each_group_idx[0])]
            this_col.extend([1 for i in range(each_group_idx[0], each_group_idx[-1] + 1)])
            this_col.extend([0 for i in range(each_group_idx[-1] + 1, self.grouping_idx[-1][-1] + 1)])
            groupingTensor.append(this_col)    
        groupingTensor = torch.from_numpy(np.array(groupingTensor).T).double()
        return groupingTensor
        

    def forward(self, x):
        encoder_output = self.encoder(x)                       
        groupingTensor = self.Grouping_TensorGen()
        groupingTensor = groupingTensor.to(device)
        encoder_groupped = torch.mm(encoder_output, groupingTensor)                       
        decoder_output = self.decoder(encoder_groupped)                     
        return encoder_groupped, decoder_output
    

    def reverso(self, encoder_groupped):
        encoder_groupped = encoder_groupped.to(device)
        decoder_output = self.decoder(encoder_groupped)
        return decoder_output




def MGR_QAE(inputArray, grouping_idx, predcal_model, predScore_weight= 1, epoch= EPOCH):
    qaeModel = QuasiAE(grouping_idx).to(device)
    predcal_model = predcal_model.to(device)
    inputArray = inputArray.to(device)
    optimizer = torch.optim.Adam(qaeModel.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    loss_func = nn.MSELoss()

    for eachepoch in range(epoch):

        encoderOutput, output = qaeModel(inputArray)

        encoderOutput_first = encoderOutput[:, 0]
        encoderOutput_first = encoderOutput_first.reshape(-1, 1, encoderOutput_first.shape[0])
        pred_loss = predcal_model(encoderOutput_first)
        for each_dim in range(1, encoderOutput.size(1)):
            encoderOutput_each = encoderOutput[:, each_dim]
            encoderOutput_each = encoderOutput_each.reshape(-1, 1, encoderOutput_each.shape[0])
            pred_loss_each = predcal_model(encoderOutput_each)
            pred_loss += pred_loss_each
        
        corr_mat = corrcoef(encoderOutput)
        corr_loss = torch.sum(torch.abs(corr_mat))
        
        range_loss = encoderOutput_first.max() - encoderOutput_first.min()
        for each_dim in range(1, encoderOutput.size(1)):
            encoderOutput_each = encoderOutput[:, each_dim]
            range_loss += encoderOutput_each.max() - encoderOutput_each.min()
        
        loss = loss_func(output, inputArray) + pred_loss * predScore_weight + corr_loss + range_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    
    # Plot
    encoderOutput = encoderOutput.to("cpu")
    f = open("/dfs/data/CRNN/encoder_output.tlz", "wb")
    pickle.dump(encoderOutput, f)
    f.close()
    for each_dim in range(1, encoderOutput.size(1)):
        encoderOutput_each = encoderOutput[:, each_dim]
        y = encoderOutput_each.clone().detach().numpy().reshape(-1, )
        x = np.arange(y.shape[0])
        plt.figure(figsize= (7, 4))
        plt.plot(x, y)
        plt.show()
        plt.savefig("/dfs/data/CRNN/encoder_output/dim=" + str(each_dim) + ".jpg")
    

    return qaeModel




class Test_Forecast():
    
    def __init__(self, model_name, forecast_step):
        self.forecast_algo = model_name
        self.steps = forecast_step
        
    
    def forecast(self, inputArray):
        
        if self.forecast_algo == "ARIMA" :
            pred_res = AutoARIMA(inputArray, self.steps).reshape(-1, )
        
        elif self.forecast_algo == "GBDT" :
            gbdt_model = GBDT_train(inputArray, 16)
            pred_res = GBDT_predict(inputArray, gbdt_model, 16, self.steps).reshape(-1, )
        
        elif self.forecast_algo == "AutoTSB" :
            auto_tsb = Auto_TSB(inputArray, 4)
            pred_res = auto_tsb.forecast_AutoTSB(inputArray, self.steps).reshape(-1, )
        
        elif self.forecast_algo == "Wen" :
            wen_model = Wen_train(inputArray, self.steps)
            pred_res = Wen_forecast(inputArray, wen_model, self.steps)
        
        elif self.forecast_algo == "informer" :
            inputArray = inputArray.astype('int')
            pred_res = train_and_predict(inputArray, 30, self.steps)

        
        return pred_res
    



class Test_PLAE(Test_Forecast):
    
    def __init__(self, model_name, forecast_step, path):
        super().__init__(model_name, forecast_step)
        if path == "aimaster" :
            self.PCNN = torch.load("/dfs/data/CRNN/model.tlz")
        else :
            self.PCNN = torch.load("model.tlz")
        torch.set_default_tensor_type(torch.DoubleTensor)
        
    
    def forecast_PLAE(self, inputArray):
        
        inputArray_normalized, input_mean, input_vol = Normalization(inputArray)
        inputArray_normalized = inputArray_normalized.reshape(-1, 1)
        inputMatrix = torch.from_numpy(inputArray_normalized)
        
        
        grouping_idx = []
        for i in range(128):    
            grouping_idx.append([i * 8, i * 8 + 7])
        
        
        _model = MGR_QAE(inputMatrix, grouping_idx, self.PCNN)
        inputMatrix = inputMatrix.to(device)
        encoderGrouped, output = _model(inputMatrix)
        encoderGrouped, output = encoderGrouped.to("cpu"), output.to("cpu")
        

        subseq_PredValue_list = []
        all_subseq = encoderGrouped.clone().detach().numpy().T
        for each_subseq_idx in range(all_subseq.shape[0]):
            this_subseq = all_subseq[each_subseq_idx, :].reshape(-1, )

            if self.forecast_algo == "ARIMA" :
                this_forecast = AutoARIMA(this_subseq, self.steps).reshape(-1, )
            
            elif self.forecast_algo == "GBDT" :
                gbdt_model = GBDT_train(this_subseq, 16)
                this_forecast = GBDT_predict(this_subseq, gbdt_model, 16, self.steps).reshape(-1, )
                
            elif self.forecast_algo == "AutoTSB" :
                auto_tsb = Auto_TSB(inputArray, 4)
                this_forecast = auto_tsb.forecast_AutoTSB(inputArray, self.steps).reshape(-1, )
            
            elif self.forecast_algo == "Wen" :
                wen_model = Wen_train(inputArray, self.steps)
                this_forecast = Wen_forecast(inputArray, wen_model)
            
            elif self.forecast_algo == "Informer" :
                inputArray = inputArray.astype('int')
                this_forecast = train_and_predict(inputArray, 30, self.steps)

            this_subseq = np.append(this_subseq, this_forecast)
            subseq_PredValue_list.append(this_subseq)
        decoderInput = torch.from_numpy(np.array(subseq_PredValue_list).T)
        pred_grouped = _model.reverso(decoderInput)
        pred_grouped = pred_grouped.to("cpu")
        pred_grouped = pred_grouped.clone().detach().numpy().reshape(-1, )[-self.steps : ]
        pred_grouped = reverseNormalization(pred_grouped, input_mean, input_vol)
        
        return pred_grouped




def Result_Analysis(pred, label, ver= True):
    
    try: 
        mape = min(1, max(0, mean_absolute_percentage_error(label, pred)))
    except :
        mape = False
    rmse = np.sqrt(mean_squared_error(label, pred))
    r2 = r2_score(label, pred)
    
    if ver == True : 
        print("MAPE : ", mape)
        print("RMSE : ", rmse)
        print("R square : ", r2)
    
    return mape, rmse, r2