import math
import numpy as np
import mpmath
import pandas as pd
from sklearn.cluster import DBSCAN
from Decompose import seasonality_decomposition




def contains(small, big):
    for i in range(len(big)-len(small)+1):
        if big[i:i+len(small)] == small:
            return True
    return False




def actual_entropy(l):
    n = len(l)
    sequence = [l[0]]
    sum_gamma = 0
    for i in range(1, n):
        for j in range(i+1, n+1):
            s = l[i:j]
            if not contains(list(s), sequence): 
                sum_gamma += len(s)
                sequence.append(l[i])
                break
    ae = 1 / (sum_gamma / n ) * math.log(n)            
    return ae




def getPredictability(N, S):
    f = lambda x: (((1-x) / (N-1)) ** (1-x)) * x**x - 2**(-S)
    root = mpmath.findroot(f, 1)
    pred = float(root.real)
    return pred




def checkUnique(inputArray):
    return len(set(inputArray.tolist()))




def ResidueClustering(inputArray):
    
    if inputArray.shape == (inputArray.shape[0], ) :
        inputArray = inputArray.reshape(-1, 1)
    
    clustering = DBSCAN(eps= 3, min_samples= 2).fit(inputArray)
    res = clustering.labels_[clustering.labels_ >= 0]
    
    return res




def PredCal_resid(resdiArray):
    
    N = checkUnique(resdiArray)
    try :
        S = actual_entropy(resdiArray.tolist())
        predicability = getPredictability(N, S)
    except :
        predicability = 0
    predicability = max(0, predicability)
    return predicability




def PredicabilityTest(inputArray):
    
    input_dataframe = pd.DataFrame()
    input_dataframe["y"] = inputArray
    
    decompose_res = seasonality_decomposition(input_dataframe)
    _S = sum(decompose_res["seasonal"].values())
    _T = decompose_res["trend"]
    _R = decompose_res["resid"]
    
    weight_S = _S[-1]
    weight_T = _T[-1]
    weight_R = _R[-1]
    
    pred_R = PredCal_resid(ResidueClustering(_R))
    predictability = (weight_S + weight_T + weight_R * pred_R) / (weight_S + weight_T + weight_R)
    if predictability < 0 :
        predictability = 0
    if predictability > 1 :
        predictability = 1
    
    return predictability