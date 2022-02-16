import pandas as pd
import numpy as np
from Test_utils import Test_Forecast, Test_PLAE
from Test_utils import Result_Analysis
from pandarallel import pandarallel
pandarallel.initialize()




DATA_SOURCE = "parts"         # "traffic", "parts", "electricity"
FORECAST_ALGO = "informer"      # "ARIMA", "GBDT", "AutoTSB", "Wen", "informer"
TEST_ON = "direct"              # "direct", "PLAE", "both"
FORECAST_STEP = 3              # 24 for traffic, 3 for parts, 24 for electricity




def Direct_Forecast_Lambda(eachRow, _algo, _step):
    
    train_data, label_data = eachRow.values[ : - _step], eachRow.values[- _step : ]
    
    Origin_Forecast_Module = Test_Forecast(model_name= _algo,  forecast_step= _step)
    pred_res_origin = Origin_Forecast_Module.forecast(inputArray= train_data)
    
    mape_origin, rmse_origin, r2_origin = Result_Analysis(pred_res_origin, label_data, ver= False)
    
    print("true : ", label_data)
    print("pred : ", pred_res_origin)
    print("MAPE : ", mape_origin)
    print("RMSE : ", rmse_origin)
    print("R square : ", r2_origin)
    print()
    print()
    
    if mape_origin != False :
        return mape_origin, rmse_origin, r2_origin, 1
    else :
        return 0, rmse_origin, r2_origin, 0




def Direct_Forecast_w_Lambda(eachRow, _algo, _step):
    
    train_data, label_data = eachRow.values[ : - _step], eachRow.values[- _step : ]
    
    Origin_Forecast_Module = Test_Forecast(model_name= _algo,  forecast_step= _step)
    pred_res_origin = Origin_Forecast_Module.forecast(inputArray= train_data)
    
    mape_origin, rmse_origin, r2_origin = Result_Analysis(pred_res_origin, label_data, ver= False)
    val = np.sum(label_data)
    
    print("true : ", label_data)
    print("pred : ", pred_res_origin)
    print("MAPE : ", mape_origin)
    print("RMSE : ", rmse_origin)
    print("R square : ", r2_origin)
    print()
    print()
    
    return mape_origin * val, rmse_origin * val, r2_origin * val, val




def PLAE_Forecast_Lambda(eachRow, _algo, _step, ):
    
    train_data, label_data = eachRow.values[ : - _step], eachRow.values[- _step : ]
    
    PLAE_Forecast_Module = Test_PLAE(model_name= _algo, forecast_step= _step, path= "local")
    pred_res_PLAE = PLAE_Forecast_Module.forecast_PLAE(inputArray= train_data)
    
    mape_origin, rmse_origin, r2_origin = Result_Analysis(pred_res_PLAE, label_data, ver= False)
    
    print("true : ", label_data)
    print("pred : ", pred_res_PLAE)
    print("MAPE : ", mape_origin)
    print("RMSE : ", rmse_origin)
    print("R square : ", r2_origin)
    print()
    print()
    
    return mape_origin, rmse_origin, r2_origin, 1




def PLAE_Forecast_w_Lambda(eachRow, _algo, _step):
    
    train_data, label_data = eachRow.values[ : - _step], eachRow.values[- _step : ]
    
    PLAE_Forecast_Module = Test_PLAE(model_name= _algo, forecast_step= _step, path= "local")
    pred_res_PLAE = PLAE_Forecast_Module.forecast_PLAE(inputArray= train_data)
    
    mape_origin, rmse_origin, r2_origin = Result_Analysis(pred_res_PLAE, label_data, ver= False)
    val = np.sum(label_data)
    
    print("true : ", label_data)
    print("pred : ", pred_res_PLAE)
    print("MAPE : ", mape_origin)
    print("RMSE : ", rmse_origin)
    print("R square : ", r2_origin)
    print()
    print()
    
    return mape_origin * val, rmse_origin * val, r2_origin * val, val




def Test_PD_Main(_data, _algo, _on, _step, _source= "local"):
    
    if _data == "traffic" :
        if _source == "aimaster" : 
            test_data = pd.read_csv("/dfs/data/CRNN/data_traffic.csv", header= None).drop(columns= [0]).sample(n= 100, axis= 1).reset_index(drop= True).T
            test_data.reset_index(drop= True, inplace= True)
        else : 
            test_data = pd.read_csv("data_traffic.csv", header= None).drop(columns= [0]).sample(n= 100, axis= 1).reset_index(drop= True).T
            test_data.reset_index(drop= True, inplace= True)

    elif _data == "parts" :
        if _source == "aimaster" : 
            test_data = pd.read_csv("/dfs/data/CRNN/data_carparts.csv").drop(columns= ["Part"]).dropna().sample(n= 100).reset_index(drop= True)
        else : 
            test_data = pd.read_csv("data_carparts.csv").drop(columns= ["Part"]).dropna().sample(n= 100).reset_index(drop= True)
    
    elif _data == "electricity" :
        if _source == "aimaster" : 
            test_data = pd.read_csv("/dfs/data/CRNN/data_elec.csv", header= None).sample(n= 100).reset_index(drop= True)
        else : 
            test_data = pd.read_csv("data_elec.csv", header= None).sample(n= 100).reset_index(drop= True)

    
    if _on == "direct" or _on == "both" :
        if _algo != "Wen" and _algo != "informer" :
            test_data["direct_mape"], test_data["direct_rmse"], test_data["direct_r2"], test_data["direct_val"] = zip(*test_data.parallel_apply(lambda x : Direct_Forecast_Lambda(x, _algo, _step), axis= 1))
        else :
            direct_mape_list, direct_rmse_list, direct_r2_list, direct_val_list = [], [], [], []
            for eachRow in range(test_data.shape[0]):
                x = test_data.loc[eachRow, :]
                direct_mape, direct_rmse, direct_r2, direct_val = Direct_Forecast_Lambda(x, _algo, _step)
                direct_mape_list.append(direct_mape)
                direct_rmse_list.append(direct_rmse)
                direct_r2_list.append(direct_r2)
                direct_val_list.append(direct_val)
                print(eachRow, " done.")
            test_data["direct_mape"], test_data["direct_rmse"], test_data["direct_r2"], test_data["direct_val"] = direct_mape_list, direct_rmse_list, direct_r2_list, direct_val_list
        
    
    if _on == "PLAE" or _on == "both" :
        if _data == "traffic" :
            # test_data["PLAE_mape"], test_data["PLAE_rmse"], test_data["PLAE_r2"], test_data["PLAE_val"] = zip(*test_data.parallel_apply(lambda x : PLAE_Forecast_Lambda(x, _algo, _step), axis= 1))
            print(test_data)
            PLAE_mape_list, PLAE_rmse_list, PLAE_r2_list, PLAE_val_list = [], [], [], []
            for eachRow in range(test_data.shape[0]):
                x = test_data.loc[eachRow, :]
                PLAE_mape, PLAE_rmse, PLAE_r2, PLAE_val = PLAE_Forecast_Lambda(x, _algo, _step)
                PLAE_mape_list.append(PLAE_mape)
                PLAE_rmse_list.append(PLAE_rmse)
                PLAE_r2_list.append(PLAE_r2)
                PLAE_val_list.append(PLAE_val)
            test_data["PLAE_mape"], test_data["PLAE_rmse"], test_data["PLAE_r2"], test_data["PLAE_val"] = PLAE_mape_list, PLAE_rmse_list, PLAE_r2_list, PLAE_val_list
                
        else : 
            test_data["PLAE_mape"], test_data["PLAE_rmse"], test_data["PLAE_r2"], test_data["PLAE_val"] = zip(*test_data.parallel_apply(lambda x : PLAE_Forecast_Lambda(x, _algo, _step), axis= 1))
    
    
    if _on == "direct" or _on == "both" :
        total_val = np.sum(test_data["direct_val"].values)
        mape_origin, rmse_origin, r2_origin = np.sum(test_data["direct_mape"].values) / total_val, np.sum(test_data["direct_rmse"].values) / total_val, np.sum(test_data["direct_r2"].values) / total_val
        
    if _on == "PLAE" or _on == "both" :
        total_val = np.sum(test_data["PLAE_val"].values)
        mape_PLAE, rmse_PLAE, r2_PLAE = np.sum(test_data["PLAE_mape"].values) / total_val, np.sum(test_data["PLAE_rmse"].values) / total_val, np.sum(test_data["PLAE_r2"].values) / total_val
    
    
    print()
    print(" *** Test on ", _data, " data. *** ")
    print()
    if _on == "direct" or _on == "both" :
        print(" *** Forecast Directly *** ")
        print("MAPE : ", mape_origin)
        print("RMSE : ", rmse_origin)
        print("R square : ", r2_origin)
        print()
    if _on == "PLAE" or _on == "both" :
        print(" *** Forecast using PLAE *** ")
        print("MAPE : ", mape_PLAE)
        print("RMSE : ", rmse_PLAE)
        print("R square : ", r2_PLAE)
        print()
        print()
        



if __name__ == "__main__" : 
    
    Test_PD_Main(DATA_SOURCE, FORECAST_ALGO, TEST_ON, FORECAST_STEP)