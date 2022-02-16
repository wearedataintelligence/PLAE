from sklearn.ensemble import GradientBoostingRegressor
import numpy as np




def GBDT_train(inputArray, windowLength):

    train_X, train_Y = [], []
    for i in range(inputArray.shape[0] - windowLength):
        train_X.append(inputArray[i : i + windowLength])
        train_Y.append(inputArray[i + windowLength])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    gbdt = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.25, verbose= 0)
    gbdt.fit(train_X, train_Y)

    return gbdt




def GBDT_predict(inputArray, model, windowLength, forecast_steps):

    test_Y = []
    for i in range(forecast_steps):
        
        if i == 0 :
            test_X = inputArray[- windowLength : ]
        else :
            test_X = test_X.reshape(-1, ).tolist()
            test_X.append(test_Y[-1])
            test_X = test_X[- windowLength : ]
            test_X = np.array(test_X)
        test_X = test_X.reshape(1, -1)
        test_Y.append(model.predict(test_X))

    return np.array(test_Y)




if __name__ == "__main__" :

    testArray = np.array([i for i in range(16)] * 10)
    gbdt_model = GBDT_train(testArray, 11)
    pred = GBDT_predict(testArray, gbdt_model, 11, 4)
    print(pred)