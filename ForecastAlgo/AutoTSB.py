import numpy as np




class SES():
    
    def __init__(self):
        pass

    def forecast_SES(self, inputArray, alpha, lookahead):
        X = inputArray.reshape(-1, ).tolist()
        F_0 = X[0]
        length = len(X)

        F_list = [F_0]
        for i in range(1, length + lookahead):
            F_i = alpha * X[i - 1] + (1 - alpha) * F_list[i - 1]
            F_list.append(F_i)
            if i > length - 1:
                X.append(F_i)
        return np.array(F_list[-lookahead:])




class Croston(SES):
    
    def __init__(self):
        pass

    def init_zero_cut(self, inputArray):
        X = inputArray.reshape(-1, ).tolist()
        length = len(X)
        if X[0] != 0:
            return X
        else:
            cut_idx = 0
            for i in range(length):
                if X[i] == 0:
                    cut_idx += 1
                else:
                    break
            return X[cut_idx:]

    def V_cal(self, inputArray):
        v_total = 0
        v_num = 0
        zeroFlag = False
        cut_X = self.init_zero_cut(inputArray)
        for i in range(len(cut_X)):
            if zeroFlag == False and cut_X[i] == 0:
                v_total += 1
                v_num += 1
                zeroFlag = True
            elif zeroFlag == True and cut_X[i] != 0:
                zeroFlag = False
            elif zeroFlag == True and cut_X[i] == 0:
                v_total += 1
        return v_total, v_num

    def forecast_Croston(self, inputArray, alpha, lookahead):
        X = inputArray.reshape(-1, ).tolist()
        length = len(X)
        Z_0 = X[-1]
        Y_0 = X[-1]
        if X[-1] == 0:
            q = 1
            for i in range(2, length):
                if X[-i] == 0:
                    q += 1
                else:
                    break
        else:
            q = 0

        V_total, V_num = self.V_cal(inputArray)
        if V_num == 0:
            return self.forecast_SES(inputArray, alpha, lookahead)

        else:
            Z_list = [Z_0]
            V_list = [int(V_total / V_num)]
            Y_list = [Y_0]
            for i in range(1, lookahead + 1):
                if X[i - 1] != 0:
                    Z_i = alpha * X[i - 1] + (1 - alpha) * Z_list[i - 1]
                    V_i = alpha * q + (1 - alpha) * V_list[i - 1]
                    Y_i = Z_i / V_i
                    q = 0
                else:
                    Z_i = Z_list[i - 1]
                    V_i = V_list[i - 1]
                    Y_i = Y_list[i - 1]
                    q += 1
                Z_list.append(Z_i)
                V_list.append(V_i)
                Y_list.append(Y_i)
                if i > length - 1:
                    X.append(Y_i)
            return np.array(Y_list[-lookahead:])




class TSB(Croston):
    
    def __init__(self):
        pass

    def forecast_TSB(self, inputArray, alpha, beta, lookahead):
        X = inputArray.reshape(-1, ).tolist()
        length = len(X)
        Z_0 = X[-1]
        Y_0 = X[-1]
        P_0 = X[-1]

        _, V_num = self.V_cal(inputArray)
        if V_num == 0:
            return self.forecast_SES(inputArray, alpha, lookahead)

        else:
            Z_list = [Z_0]
            P_list = [P_0]
            Y_list = [Y_0]
            for i in range(1, lookahead + 1):

                if X[-1] != 0:
                    P_next = beta + (1 - beta) * P_list[-1]
                    P_list.append(P_next)
                    Z_next = alpha * X[-1] + (1 - alpha) * Z_list[-1]
                    Z_list.append(Z_next)
                    Y_next = P_next * Z_next
                    Y_list.append(Y_next)

                else:
                    P_next = (1 - beta) * P_list[-1]
                    P_list.append(P_next)
                    Z_next = Z_list[-1]
                    Z_list.append(Z_next)
                    Y_next = P_next * Z_next
                    Y_list.append(Y_next)

                if i > length - 1:
                    X.append(Y_next)
            return np.array(Y_list[-lookahead:])




class Auto_TSB(TSB):

    def __init__(self, inputArray, backtestLength):
        self.alphaList = [each_alpha for each_alpha in np.arange(0.2, 0.8, 0.05)]
        self.betaList = [each_beta for each_beta in np.arange(0.2, 0.8, 0.05)]
        self.backtestArray = np.array(inputArray.reshape(-1, ).tolist()[: -backtestLength])
        self.answerSum = np.sum(np.array(inputArray.reshape(-1, ).tolist()[-backtestLength:]))
        self.backtestLength = backtestLength

    def mapeLoss(self, predSum):
        return np.abs(self.answerSum - predSum) / self.answerSum

    def forecast_AutoTSB(self, inputArray, lookahead):
        min_mape = 1
        best_alpha = self.alphaList[0]
        best_beta = self.betaList[0]
        for each_alpha in self.alphaList:
            for each_beta in self.betaList:
                this_pred = self.forecast_TSB(self.backtestArray, each_alpha, each_beta, self.backtestLength)
                this_predSum = np.sum(this_pred)
                this_mape = self.mapeLoss(this_predSum)

                if this_mape < min_mape:
                    best_alpha = each_alpha
                    best_beta = each_beta

        predResult = self.forecast_TSB(inputArray, best_alpha, best_beta, lookahead)
        return predResult