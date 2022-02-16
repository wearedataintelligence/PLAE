import pandas as pd
import numpy as np
from Test_utils import Test_Forecast, Test_PLAE
from Test_utils import Result_Analysis




FORECAST_ALGO = "Wen"      # "ARIMA", "GBDT", "AutoTSB", "Wen", "informer"
TEST_ON = "PLAE"              # "direct", "PLAE", "both"
FORECAST_STEP = 13




def Test_CTO_Main(_algo, _on, _step):

    test_data = pd.read_csv("data_AllFam_CTO_WW.csv", keep_default_na= False)["sum"].values[4 : -4]
    input_data, label_data = test_data[ : -_step], test_data[-_step : ]


    if _on == "direct" or _on == "both" :
        Origin_Forecast_Module = Test_Forecast(model_name= _algo,  forecast_step= _step)
        pred_res_origin = Origin_Forecast_Module.forecast(inputArray= input_data)

    if _on == "PLAE" or _on == "both" :
        PLAE_Forecast_Module = Test_PLAE(model_name= _algo, forecast_step= _step, path= "local")
        pred_res_PLAE = PLAE_Forecast_Module.forecast_PLAE(inputArray= input_data)


    print(" *** Test on Lenovo CTO data. *** ")
    print("input data length : ", input_data.shape)
    print("label length : ", label_data.shape)
    print()
    print()
    if _on == "direct" or _on == "both" :
        print(" *** Forecast Directly *** ")
        print("pred_res : ", pred_res_origin)
        print("label : ", label_data)
        _ = Result_Analysis(pred_res_origin, label_data)
        print()
        print()
    if _on == "PLAE" or _on == "both" :
        print(" *** Forecast using PLAE *** ")
        print("pred_res : ", pred_res_PLAE)
        print("label : ", label_data)
        _ = Result_Analysis(pred_res_PLAE, label_data)
        print()
        print()




if __name__ == "__main__" : 
    
    Test_CTO_Main(FORECAST_ALGO, TEST_ON, FORECAST_STEP)