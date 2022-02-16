import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.fft import fft
from Decompose_utils import seasonal_decompose
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

np.seterr(divide='ignore')




def seasonality_utils_fft(dataframe, target_col=None):
    if target_col is None:
        target_col = 'y'
    series = dataframe[target_col]
    ffted = fft(series)
    ffted[[0, 1]] = [0, 0]
    ffted = np.abs(ffted)
    ffted_half = ffted[:int(ffted.shape[0] / 2)]
    return ffted_half




def seasonalily_utils_detection(dataframe, target_col=None, plot=False):
    
    thre_ = 52  # max of period: 52 weeks (1 year)
    max_periods = 5  # max number of periods
    ffted_half = seasonality_utils_fft(dataframe, target_col)
    temp_df = pd.DataFrame({'y': ffted_half})
    temp_df = extreme_std_iqr(temp_df, target_col='y', hyper_param_=1.5)
    if plot:
        temp_df[['y', 'outlier']].plot(figsize=[20, 8])
        plt.title('Frequency Domain')
        plt.grid()
        plt.show()
    outlier_ind = np.where(temp_df['outlier'].values > 0)[0]


    tobe_del = []
    for i in range(outlier_ind.shape[0]):
        if outlier_ind[i] > thre_:
            tobe_del.append(i)
            continue

        if outlier_ind[i] in tobe_del:
            continue
        for ii in np.arange(i + 1, outlier_ind.shape[0]):
            if outlier_ind[ii] in tobe_del:
                continue
            if outlier_ind[ii] % outlier_ind[i] == 0:
                tobe_del.append(ii)
    tobe_del = list(set(tobe_del))
    outlier_ind = np.delete(outlier_ind, tobe_del)
    

    if outlier_ind.shape[0] > max_periods:
        outlier_ind = np.argsort(temp_df['outlier'].values)[::-1][:max_periods]
        
    elif outlier_ind.shape[0] == 0:
        outlier_ind = np.argsort(temp_df['outlier'].values[:thre_])[::-1][:1]
        
        if outlier_ind[0] > thre_:
            outlier_ind[0] = thre_
            
    return outlier_ind




# noinspection PyUnboundLocalVariable
def seasonality_decomposition(dataframe, target_col='y', periods=None, plot=False):
    
    dataframe = dataframe.copy()
    
    if periods is None:
        periods = seasonalily_utils_detection(dataframe, target_col=target_col, plot=plot)
    
    decomped_obj_dict = {}
    target_seq = dataframe[target_col].values
    counter = 0
    for period in periods:  # split seasonal
        decom = seasonal_decompose(target_seq, model='additive', period=period)
        if plot:
            print('\n periods: ', period)
            plt.rc('figure', figsize=(16, 12))
            decom.plot()
            plt.grid()
            plt.show()

        target_seq -= decom.seasonal
        decomped_obj_dict[period] = decom
        if counter == 0:
            trend_compo = decom.trend  # split trend
        counter += 1


    target_seq = dataframe[target_col].values - trend_compo
    for period in periods:  # split resid
        target_seq -= decomped_obj_dict[period].seasonal
    resid_compo = target_seq

    """
    seasonal compo
    """
    seasonal_compo = {}
    for period in periods:
        seasonal_compo[period] = decomped_obj_dict[period].seasonal

    """
    resid compo
    """
    resid_compo = resid_compo

    """
    trend compo
    """
    trend_compo = trend_compo

    return {'seasonal': seasonal_compo, 'trend': trend_compo, 'resid': resid_compo}




def iqr(data, hyper_param):
    
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    iqr_value = q75 - q25
    cut_off = iqr_value * hyper_param
    lower, upper = q25 - cut_off, q75 + cut_off
    data_nan = np.where((data > upper) | (data < lower), np.nan, data)
    ab_index_low = list(np.where(data < lower))[0].tolist()
    ab_index_high = list(np.where(data > upper))[0].tolist()
    return data_nan, ab_index_low, ab_index_high




def Kalman1D(observations, damping=2):
   
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state.reshape([-1])




def detrend(dataframe, target_col=None, hyper_param_=None):
   
    if target_col is None:
        target_col = 'y'
    dataframe = dataframe.copy()
    kalman_ed_array = Kalman1D(dataframe[target_col].values, damping=hyper_param_)
    dataframe['trend_' + target_col] = kalman_ed_array
    dataframe['detrend_' + target_col] = dataframe[target_col] - kalman_ed_array
    return dataframe




# noinspection PyUnresolvedReferences,PyTypeChecker
def extreme_std_iqr(data, preds=None, target_col=None, hyper_param_=3.0, print_flag=False):
    
    if target_col is None:
        target_col = 'y'  # default column name

    if data[target_col].sum() == 0:
        
        data['recovered'] = np.zeros(data.shape[0])
        data['outlier'] = np.zeros(data.shape[0])
        return data

    d_data = detrend(data, target_col=target_col, hyper_param_=0.1)
    de_target_col = 'detrend_' + str(target_col)
    tr_target_col = 'trend_' + str(target_col)

    if preds is not None:
        # also drop starting/ending zeros
        data_array = np.trim_zeros(d_data.iloc[:-len(preds)].loc[:, de_target_col].fillna(0).values.reshape([-1]))
        data_trend = np.trim_zeros(d_data.iloc[:-len(preds)].loc[:, tr_target_col].fillna(0).values.reshape([-1]))
    else:
        data_array = np.trim_zeros(d_data.loc[:, de_target_col].fillna(0).values.reshape([-1]))
        data_trend = np.trim_zeros(d_data.loc[:, tr_target_col].fillna(0).values.reshape([-1]))

    data_nan, ab_index_low, ab_index_high = iqr(data_array, hyper_param_)  # select a method for extract abnormal values
    ab_values_low = np.array([data_array[i] for i in ab_index_low])
    ab_values_high = np.array([data_array[i] for i in ab_index_high])

    SCALE = 0.2
    max_value = pd.DataFrame(data_nan).dropna(axis=0, how='any').values.squeeze().max()
    min_value = pd.DataFrame(data_nan).dropna(axis=0, how='any').values.squeeze().min()
    # special situation

    if print_flag:
        print('family:', target_col)
        print('low outliers values:', ab_values_low)
        print('high outliers values:', ab_values_high)
        print('IQR recover into range: [{}, {}]'.format(min_value, max_value))

    # for low outliers
    # outlier nums more than 1
    if len(ab_index_low) > 1:
        # if more than one outliers (have same values) appears
        if len(set(ab_values_low.tolist())) == 1:
            ab_values_low = min_value * np.ones(ab_values_low.shape)
        elif np.max(ab_values_low) != 0 and min_value - abs(min_value) * SCALE != min_value:
            enc = MinMaxScaler(feature_range=[min_value - abs(min_value) * SCALE, min_value])
            ab_values_low = enc.fit_transform(ab_values_low.reshape(-1, 1)).reshape([-1])  # add np.log()
    # only one outlier, need to confirm it is high outlier/ low outlier
    elif len(ab_index_low) == 1:
        ab_values_low[0] = min_value

    # for high outliers
    if len(ab_index_high) > 1:
        if len(set(ab_values_high.tolist())) == 1:
            ab_values_high = max_value * np.ones(ab_values_high.shape)
        elif np.max(ab_values_high) != 0 and max_value + abs(max_value) * SCALE != max_value:
            enc = MinMaxScaler(feature_range=[max_value, max_value + abs(max_value) * SCALE])
            ab_values_high = enc.fit_transform(ab_values_high.reshape(-1, 1)).reshape([-1])
    elif len(ab_index_high) == 1:
        ab_values_high[0] = max_value

    if print_flag:
        print('low outliers replaced to:', ab_values_low)
        print('high outliers replaced to:', ab_values_high)

    ab_index = ab_index_low + ab_index_high
    ab_values = ab_values_low.tolist() + ab_values_high.tolist()

    recovered = data.loc[:, target_col].fillna(0).values.reshape([-1])
    outlier = np.zeros(recovered.shape)
    # data_array_begin_index = (recovered != 0).argmax(axis=0)
    data_array_begin_index = (d_data.loc[:, de_target_col].fillna(0).values.reshape([-1]) != 0).argmax(axis=0)
    for i, j in enumerate(ab_index):
        recovered[data_array_begin_index + j] = ab_values[i] + data_trend[j]
        outlier[data_array_begin_index + j] = data_array[j] + data_trend[j]
    data['outlier'] = outlier
    data['recovered'] = recovered
    return data