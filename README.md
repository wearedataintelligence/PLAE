# PLAE

This is a simple version of README before the paper is reviewed. The purpose is only to show how the programme should be run.

The electricity data set can be found in https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014. The traffic data set can be found in https://archive.ics.uci.edu/ml/datasets/PEMS-SF. The parts data set can be found in https://robjhyndman.com/expsmooth/. The pre-processing on this three public data sets is the same as what it does in DeepTCN.
Because of the data security issues, Lenovo CTO sales data will not be available to the public.

Before utilising the PLAE, you should fit the PCNN through PredCalNN_gpu.py first. Note that the parametre seqLength in trainNN should be equivalent to the length of your original forecast subject. For example, if you want to leverage the forecast accuracy on a sequence contains 108 data points, then the seqLength shall be set as 108. Besides, if the parametre datasource is set to be "local", a pair of training data, _X.traindata and _Y.traindata, must be generated before the PCNN training. The size of _X.traindata is sample number * seqLength and it represents, and the size of _Y.traindata is sample number * 1.

After generating the PCNN model, Test_PubData.py can be run and the test is made on a chosen data set using the chosen forecast algorithm.

The function of each file is as follows:
1. PredCal.py is the quantitative predictability measuring method;
2. Decompose.py and Decompose_utils.py serve to finish the Fourier seasonality decomposition and Kalman Filter. They split the seasonality and trend from the original data.
3. The five forecast algorithms are located in the fold ForecastAlgo/.
4. The main part of PLAE can be found in Test_utils.py.