import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import Dataset




def DTW(testingAgainst_seq, subject_seq, max_dist):
    
    N = testingAgainst_seq.shape[1]
    M = subject_seq.shape[1]

    
    d = np.zeros((N, M))    
    for n in range(N):
        for m in range(M):
            d[n, m] = (testingAgainst_seq[0, n] - subject_seq[0, m]) ** 2
    
    
    costMat = np.ones((N, M))
    costMat.fill(float("inf"))
    
    costMat[0, 0] = d[0, 0]
    for n in range(1, N):
        costMat[n, 0] = d[n, 0] + costMat[n - 1, 0]
    for m in range(1, M):
        costMat[0, m] = d[0, m] + costMat[0, m - 1]
   
    
    encode_cost = 1
    m_cost = encode_cost * np.std(testingAgainst_seq) * math.log(M, 2)
    n_cost = encode_cost * np.std(subject_seq) + math.log(N, 2)
    
    
    for n in range(1, N):
        m_min = max(1, n - max_dist)
        m_max = min(M, n + max_dist)
        for m in range(m_min, m_max):
            choice = min(min(costMat[n - 1, m] + m_cost, costMat[n - 1, m - 1]), costMat[n, m - 1] + n_cost)
            costMat[n, m] = d[n, m] + choice
    
    
    Dist = costMat[N - 1, M - 1]
    n = N - 1
    m = M - 1
    k = 1                       
    w = [[]]
    w[0] = [N - 1, M - 1]
    while (n + m) != 0 :
        if n == 0 :
            m = m - 1
        elif m == 0 :
            n = n - 1
        else:
            choice = np.array([costMat[n - 1, m], costMat[n, m - 1], costMat[n - 1, m - 1]])
            min_index = np.argmin(choice)
            if min_index == 0 :
                n = n - 1
            elif min_index == 1 :
                m = m - 1
            else:
                n = n - 1
                m = m - 1
        k = k + 1
        w.append([n, m])
    w = np.array(w)
    
    return Dist, costMat, k, w




def New_Segment_Size(inputSeq, cur, models, s_min, s_max, max_dist):
    
    num_models = len(models)
    ave_cost = np.ones((s_max, num_models + 1, s_max))
    ave_cost.fill(float("inf"))
    
    
    for s in range(s_min, s_max):
        if cur + s >= inputSeq.shape[1]:
            continue
        for k in range(num_models + 1):
            if k < num_models:
                curr_model = models[k]
                curr_model = curr_model.reshape(1, -1)
                
            else:
                curr_model = inputSeq[:, cur : cur + s]
                
            inputSeq_curr = inputSeq[:, cur + s : min(inputSeq.shape[1], cur + s + s_max)]
            _, dtw_mat, _, _ = DTW(curr_model, inputSeq_curr, max_dist)
            dtw_cost = dtw_mat[-1, :]
            ave_cost[s, k, 0 : inputSeq_curr.shape[1]] = dtw_cost / np.arange(1, inputSeq_curr.shape[1] + 1)
            ave_cost[s, k, 0 : s_min] = float("inf")
    
    
    _min = np.min(ave_cost)
    best_s1, best_k, _ = np.where(ave_cost == _min)
    best_s1 = best_s1[0]                                                   
    best_k = best_k[0]                                                          
    return best_s1, best_k




def Summarize_Seq(inputSeq, s_min, s_max, max_dist, max_vocab):
    
    starts = []
    ends = []
    starts.append(0)
    best_initial, _ = New_Segment_Size(inputSeq, 0, [], s_min, s_max, max_dist)
    ends.append(best_initial - 1)
    models = []
    models.append(inputSeq[:, starts[0] : ends[0] + 1])
    idx = [0]
    model_momentum = 0.8
    max_vocab = max_vocab
    terminal_threshold = 0
    
    
    new_cluster_threshold = 0.04
    mean_dev = np.mean((inputSeq - np.mean(inputSeq)) ** 2)
    
    best_prefix_length = 0
    total_error = 0
    
    last_seq_flag = False
    while ends[-1] + terminal_threshold < inputSeq.shape[1] - 1 :
        curr_index = len(starts)
        cur = ends[-1] + 1
        starts.append(cur)    # starts[curr_index]
        
        num_models = len(models)
        ave_costs = np.ones((num_models, s_max))
        ave_costs.fill(float("inf"))
        
        
        curr_end = min(cur + s_max - 1, inputSeq.shape[1] - 1)
        inputSeq_curr = inputSeq[:, cur : curr_end + 1]
        for k in range(num_models):
            dtw_dist, dtw_mat, _, dtw_trace = DTW(models[k], inputSeq_curr, max_dist)
            dtw_cost = dtw_mat[-1, :]
            ave_costs[k, 0 : inputSeq_curr.shape[1]] = dtw_cost / np.arange(0, inputSeq_curr.shape[1])
            ave_costs[k, 0 : s_min] = float("inf")
            
        best_cost = np.min(ave_costs)
        best_k, best_size = np.where(ave_costs == best_cost)
        best_k = best_k[0]
        best_size = best_size[0] + 1
        
        
        if cur + s_max >= inputSeq.shape[1] - 1 :
            
            last_seq_flag = True
            good_prefix_costs = np.zeros((num_models, 1))
            good_prefix_costs.fill(float("inf"))
            good_prefix_lengths = np.zeros((num_models, 1))
            good_prefix_lengths.fill(float("inf"))
            
            for k in range(num_models):
                _, dtw_mat, _, _ = DTW(models[k], inputSeq_curr, max_dist)
                prefix_costs = dtw_mat[:, -1].T
                ave_prefix_costs = prefix_costs / np.arange(1, len(models[k]) + 1)
                good_prefix_costs[k] = np.min(ave_prefix_costs)
                # print(np.where(ave_prefix_costs == good_prefix_costs[k])[0])
                good_prefix_lengths[k] = np.where(ave_prefix_costs == good_prefix_costs[k])[0][0]
            
            best_prefix_cost = np.min(good_prefix_costs)
            best_prefix_k = np.where(good_prefix_costs == best_prefix_cost)[0]
            best_prefix_k = best_prefix_k[0]
            best_prefix_length = good_prefix_lengths[best_prefix_k]
            
            if best_prefix_cost < best_cost :
                ends.append(inputSeq.shape[1] - 1)    # ends[curr_index]
                idx.append(best_prefix_k)     # idx[curr_index]
                break
        
        
        inputSeq_best = inputSeq[:, cur : cur + best_size]
        if best_cost > new_cluster_threshold * mean_dev and len(models) < max_vocab :
            if last_seq_flag == False :
                best_s1, _ = New_Segment_Size(inputSeq, cur, models, s_min, s_max, max_dist)
            else:
                best_s1 = inputSeq.shape[1] - cur
            ends.append(cur + best_s1 - 1)    # ends[curr_index]
            idx.append(num_models)        # idx[curr_index]
            models.append(inputSeq[:, starts[curr_index] : ends[curr_index] + 1])     # models[num_models]
            total_error = total_error + new_cluster_threshold * mean_dev * best_s1
            
            
        else:
            ends.append(cur + best_size - 1)      # ends[curr_index]
            idx.append(best_k)            # idx[curr_index]
            total_error = total_error + best_cost * best_size
            _, _, _, dtw_trace = DTW(models[best_k], inputSeq_best, max_dist)
            trace_summed = np.zeros(models[best_k].shape)
            
            for t in range(0, dtw_trace.shape[0]):
                trace_summed[:, dtw_trace[t, 0]] = trace_summed[:, dtw_trace[t, 0]] + inputSeq_best[:, dtw_trace[t, 1]]
            
            unique_dtw_trace = np.unique(dtw_trace[:, 0])
            trace_counts = []
            for each in unique_dtw_trace :
                trace_counts.append(np.sum(dtw_trace[:, 0] == each))
            trace_counts = np.array(trace_counts).reshape(1, -1)
            trace_ave = trace_summed / trace_counts
            models[best_k] = model_momentum * models[best_k] + (1 - model_momentum) * trace_ave
    
    total_error = total_error / np.std(inputSeq) ** 2 + (len(idx) - 1) * np.log2(inputSeq.shape[1]) + len(idx) * np.log2(len(models))
    
    return models, starts, ends, idx, best_prefix_length, total_error




def Normalization(inputArray):
    _mean = inputArray.mean()
    _volatility = inputArray.std()
    numpyArray = (inputArray - _mean) / _volatility
    return numpyArray, _mean, _volatility




class CNN_Wen(nn.Module):
    def __init__(self, forecast_step, windowlen):
        super(CNN_Wen, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels= 1,         
                                             out_channels= 64,       
                                             kernel_size= 5,
                                             stride= 1,
                                             padding= 2),
                                   nn.Sigmoid())
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels= 64,         
                                             out_channels= 64,       
                                             kernel_size= 5,
                                             stride= 1,
                                             padding= 2),
                                   nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(64 * windowlen, 128),
                                 nn.Sigmoid(),
                                 nn.Linear(128, forecast_step))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output




def TrainDataGen(inputArray, windowLen, forecastLen):
    
    inputList = inputArray.tolist()
    dataList, labelList = [], []
    for i in range(len(inputList) - windowLen - forecastLen + 1):
        dataList.append(inputList[i : i + windowLen])
        labelList.append(inputList[i + windowLen : i + windowLen + forecastLen])

    _data, _, _ = Normalization(np.array(dataList))
    _label, _, _ = Normalization(np.array(labelList))
    
    return _data.reshape(_data.shape[0], 1, _data.shape[1]), _label




class trainDataset_wen(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return self.Data.shape[0]
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index, :, :]).double()
        label = torch.Tensor(self.Label[index, :]).double()
        return data, label




if __name__ == '__main__':
    
    inputSeq = [[1, 2, 3, 4 , 1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4]]
    inputSeq = np.array(inputSeq)
    models, starts, ends, idx, best_prefix_length, total_error = Summarize_Seq(inputSeq, 2, 6, 5, 5)
    print()
    print("RESULT : ")
    print("length : ", inputSeq.shape[1])
    print("models : ", models)
    print("starts : ", starts)
    print("ends : ", ends)
    print("idx : ", idx)
    print("best prefix length : ", best_prefix_length)
    print("total error : ", total_error)